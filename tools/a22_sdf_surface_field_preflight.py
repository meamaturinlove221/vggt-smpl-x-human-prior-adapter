from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from a2_neural_field_reconstruction_preflight import (  # noqa: E402
    build_view_payloads,
    load_mesh_vertices,
    mask_metrics,
    ray_aabb_intersection,
    ray_weight_entropy,
    write_mesh_ply,
    write_point_ply,
)
from preflight_differentiable_renderer_backend import (  # noqa: E402
    describe_cuda_device,
    normalize_depth,
    parse_view_indices,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A2.2 research-only raw RGB/mask/camera SDF surface-field preflight. "
            "This changes representation from A2's free occupancy volume to an SDF-derived thin-surface field. "
            "It never exports a teacher/candidate and never writes strict passes."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--mesh-seed", type=Path, required=True, help="A3 visual-hull/template mesh NPZ used only for bbox bounds.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--eval-view-indices", default="", help="Optional held-out/eval views. Empty means training views.")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--ray-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-ray", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pos-frequencies", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--color-weight", type=float, default=0.35)
    parser.add_argument("--background-color-weight", type=float, default=0.03)
    parser.add_argument("--surface-entropy-weight", type=float, default=0.01)
    parser.add_argument("--eikonal-weight", type=float, default=0.03)
    parser.add_argument("--sdf-l1-weight", type=float, default=0.0005)
    parser.add_argument("--sdf-beta", type=float, default=0.035)
    parser.add_argument("--density-scale", type=float, default=45.0)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--prior-radius", type=float, default=0.92)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--sdf-grid", type=int, default=64)
    parser.add_argument("--render-threshold", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class TinySDFSurfaceField(nn.Module):
    def __init__(
        self,
        frequencies: int,
        hidden_dim: int,
        residual_scale: float,
        prior_radius: float,
    ) -> None:
        super().__init__()
        self.frequencies = int(frequencies)
        self.residual_scale = float(residual_scale)
        self.prior_radius = float(prior_radius)
        in_dim = 3 * (1 + 2 * self.frequencies)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def encode(self, xyz_norm: torch.Tensor) -> torch.Tensor:
        feats = [xyz_norm]
        for freq in range(self.frequencies):
            scale = float(2**freq) * np.pi
            feats.append(torch.sin(xyz_norm * scale))
            feats.append(torch.cos(xyz_norm * scale))
        return torch.cat(feats, dim=-1)

    def forward(self, xyz_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(self.encode(xyz_norm))
        # Analytic ellipsoid prior gives the SDF a single initial zero level set.
        # The MLP learns residual detail from raw calibrated images, not from VGGT shell predictions.
        ellipsoid = torch.sqrt((xyz_norm[..., 0] / 0.62) ** 2 + (xyz_norm[..., 1] / 1.0) ** 2 + (xyz_norm[..., 2] / 0.62) ** 2 + 1e-8)
        prior_sdf = ellipsoid - self.prior_radius
        residual = torch.tanh(raw[..., 0]) * self.residual_scale
        sdf = prior_sdf + residual
        color = torch.sigmoid(raw[..., 1:4])
        return sdf, color


def sdf_to_sigma(sdf: torch.Tensor, beta: float, density_scale: float) -> torch.Tensor:
    beta_t = max(float(beta), 1e-4)
    return float(density_scale) * torch.exp(-torch.abs(sdf) / beta_t)


def render_sdf_volume(
    field: TinySDFSurfaceField,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    bbox_center: torch.Tensor,
    bbox_half: torch.Tensor,
    samples_per_ray: int,
    sdf_beta: float,
    density_scale: float,
) -> dict[str, torch.Tensor]:
    sample_count = int(samples_per_ray)
    t_lin = torch.linspace(0.0, 1.0, sample_count, dtype=origins.dtype, device=origins.device)
    z_vals = near[:, None] * (1.0 - t_lin[None, :]) + far[:, None] * t_lin[None, :]
    if field.training:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=1)
        lower = torch.cat([z_vals[:, :1], mids], dim=1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    pts = origins[:, None, :] + dirs[:, None, :] * z_vals[..., None]
    pts_norm = ((pts - bbox_center[None, None, :]) / bbox_half[None, None, :].clamp_min(1e-8)).clamp(-1.35, 1.35)
    sdf, color = field(pts_norm.reshape(-1, 3))
    sdf = sdf.reshape(origins.shape[0], sample_count)
    color = color.reshape(origins.shape[0], sample_count, 3)
    sigma = sdf_to_sigma(sdf, sdf_beta, density_scale)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, deltas[:, -1:]], dim=1).clamp_min(1e-5)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), (1.0 - alpha + 1e-8)], dim=1), dim=1)[:, :-1]
    weights = alpha * trans
    opacity = weights.sum(dim=1).clamp(0.0, 1.0)
    rgb = (weights[..., None] * color).sum(dim=1)
    rgb = rgb + (1.0 - opacity[:, None]) * torch.ones_like(rgb)
    depth = (weights * z_vals).sum(dim=1) / opacity.clamp_min(1e-6)
    return {"rgb": rgb, "opacity": opacity, "depth": depth, "weights": weights, "z_vals": z_vals, "sdf": sdf, "sigma": sigma}


def eikonal_loss(
    field: TinySDFSurfaceField,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    pts = torch.empty((batch_size, 3), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
    pts.requires_grad_(True)
    sdf, _color = field(pts)
    grad = torch.autograd.grad(
        sdf.sum(),
        pts,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def evaluate_view(
    field: TinySDFSurfaceField,
    payload: dict[str, Any],
    bbox_center: torch.Tensor,
    bbox_half: torch.Tensor,
    samples_per_ray: int,
    batch_size: int,
    height: int,
    width: int,
    render_threshold: float,
    sdf_beta: float,
    density_scale: float,
) -> dict[str, np.ndarray | dict[str, Any]]:
    field.eval()
    rgb_chunks: list[torch.Tensor] = []
    opacity_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        ray_count = payload["origins"].shape[0]
        for start in range(0, ray_count, int(batch_size)):
            sl = slice(start, min(start + int(batch_size), ray_count))
            valid = payload["valid"][sl]
            rgb = torch.ones((valid.shape[0], 3), dtype=torch.float32, device=valid.device)
            opacity = torch.zeros((valid.shape[0],), dtype=torch.float32, device=valid.device)
            depth = torch.zeros((valid.shape[0],), dtype=torch.float32, device=valid.device)
            if bool(valid.any()):
                rendered = render_sdf_volume(
                    field,
                    payload["origins"][sl][valid],
                    payload["dirs"][sl][valid],
                    payload["near"][sl][valid],
                    payload["far"][sl][valid],
                    bbox_center,
                    bbox_half,
                    samples_per_ray,
                    sdf_beta,
                    density_scale,
                )
                rgb[valid] = rendered["rgb"]
                opacity[valid] = rendered["opacity"]
                depth[valid] = rendered["depth"]
            rgb_chunks.append(rgb)
            opacity_chunks.append(opacity)
            depth_chunks.append(depth)
    rgb_np = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3).detach().cpu().numpy().astype(np.float32)
    opacity_np = torch.cat(opacity_chunks, dim=0).reshape(height, width).detach().cpu().numpy().astype(np.float32)
    depth_np = torch.cat(depth_chunks, dim=0).reshape(height, width).detach().cpu().numpy().astype(np.float32)
    target_mask = payload["target_mask_np"]
    metrics = mask_metrics(opacity_np, target_mask, threshold=render_threshold)
    metrics["opacity_p50"] = float(np.percentile(opacity_np, 50))
    metrics["opacity_p90"] = float(np.percentile(opacity_np, 90))
    metrics["opacity_p95"] = float(np.percentile(opacity_np, 95))
    metrics["opacity_p99"] = float(np.percentile(opacity_np, 99))
    metrics["opacity_max"] = float(opacity_np.max()) if opacity_np.size else 0.0
    visible = target_mask.astype(bool)
    if np.any(visible):
        residual = np.linalg.norm(rgb_np[visible] - payload["rgb_np"][visible].astype(np.float32) / 255.0, axis=1)
        metrics["foreground_rgb_residual_mean"] = float(residual.mean())
        metrics["foreground_rgb_residual_p90"] = float(np.percentile(residual, 90))
    else:
        metrics["foreground_rgb_residual_mean"] = 0.0
        metrics["foreground_rgb_residual_p90"] = 0.0
    return {"rgb": rgb_np, "opacity": opacity_np, "depth": depth_np, "metrics": metrics}


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A2.2 SDF Surface Field Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is research-only. It changes A2 from a free occupancy volume to an SDF-derived surface field; it is not a teacher, not a candidate, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## View Metrics",
        "",
        "```json",
        json.dumps(summary["view_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Eval View Metrics",
        "",
        "```json",
        json.dumps(summary["eval_view_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Loss Curve",
        "",
        "```json",
        json.dumps(summary["loss_curve"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
        "## Outputs",
        "",
    ]
    for item in summary["outputs"]:
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    cuda_info = describe_cuda_device()
    if not torch.cuda.is_available():
        summary = {
            "status": "blocked_no_cuda",
            "summary": {
                "research_only": True,
                "strict_candidate_passes": 0,
                "strict_teacher_passes": 0,
                "formal_cloud_train_infer_export": "blocked",
                "cuda": cuda_info,
            },
            "view_metrics": [],
            "eval_view_metrics": [],
            "loss_curve": [],
            "decision": "A2.2 SDF surface-field preflight requires CUDA. No teacher/candidate artifact was created.",
            "outputs": [],
        }
        summary_path = output_dir / "a22_sdf_surface_field_summary.json"
        report_path = output_dir / "a22_sdf_surface_field_summary.md"
        summary["outputs"] = [str(summary_path), str(report_path)]
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(report_path, summary)
        return 2

    device = torch.device("cuda")
    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    eval_view_indices = parse_view_indices(args.eval_view_indices, len(views)) if str(args.eval_view_indices).strip() else view_indices
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)
    height = width = int(args.target_size)
    seed_vertices = load_mesh_vertices(args.mesh_seed)
    bbox_min_np = seed_vertices.min(axis=0)
    bbox_max_np = seed_vertices.max(axis=0)
    span_np = np.maximum(bbox_max_np - bbox_min_np, 1e-6)
    pad_np = span_np * float(args.bbox_pad_ratio)
    bbox_min_np = bbox_min_np - pad_np
    bbox_max_np = bbox_max_np + pad_np
    bbox_min = torch.as_tensor(bbox_min_np, dtype=torch.float32, device=device)
    bbox_max = torch.as_tensor(bbox_max_np, dtype=torch.float32, device=device)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_half = 0.5 * (bbox_max - bbox_min)

    payloads = build_view_payloads(scene_dir, manifest, cameras, view_indices, height, device)
    for payload in payloads:
        near, far, valid = ray_aabb_intersection(payload["origins"], payload["dirs"], bbox_min, bbox_max)
        payload["near"] = near
        payload["far"] = far
        payload["valid"] = valid
        payload["valid_ray_count"] = int(valid.detach().sum().cpu())
    eval_payloads = build_view_payloads(scene_dir, manifest, cameras, eval_view_indices, height, device)
    for payload in eval_payloads:
        near, far, valid = ray_aabb_intersection(payload["origins"], payload["dirs"], bbox_min, bbox_max)
        payload["near"] = near
        payload["far"] = far
        payload["valid"] = valid
        payload["valid_ray_count"] = int(valid.detach().sum().cpu())

    field = TinySDFSurfaceField(
        int(args.pos_frequencies),
        int(args.hidden_dim),
        float(args.residual_scale),
        float(args.prior_radius),
    ).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=float(args.lr))
    rng = torch.Generator(device=device)
    rng.manual_seed(20260506)
    loss_curve: list[dict[str, Any]] = []

    for step in range(int(args.steps) + 1):
        field.train()
        payload = payloads[step % len(payloads)]
        valid_indices = torch.nonzero(payload["valid"], as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            raise RuntimeError(f"No rays intersect bbox for view {payload['view_index']}")
        sample_count = min(int(args.ray_batch_size), int(valid_indices.numel()))
        perm = torch.randperm(valid_indices.numel(), generator=rng, device=device)[:sample_count]
        ray_ids = valid_indices[perm]
        rendered = render_sdf_volume(
            field,
            payload["origins"][ray_ids],
            payload["dirs"][ray_ids],
            payload["near"][ray_ids],
            payload["far"][ray_ids],
            bbox_center,
            bbox_half,
            int(args.samples_per_ray),
            float(args.sdf_beta),
            float(args.density_scale),
        )
        target_mask = payload["mask"][ray_ids]
        target_rgb = payload["rgb"][ray_ids]
        opacity = rendered["opacity"].clamp(1e-5, 1.0 - 1e-5)
        rgb_pred = rendered["rgb"].clamp(0.0, 1.0)
        mask_loss = F.binary_cross_entropy(opacity, target_mask)
        fg = target_mask > 0.5
        if bool(fg.any()):
            color_loss = torch.sqrt(((rgb_pred[fg] - target_rgb[fg]) ** 2).sum(dim=1) + 1e-4).mean()
            entropy_loss = ray_weight_entropy(rendered["weights"][fg])
        else:
            color_loss = rgb_pred.sum() * 0.0
            entropy_loss = rgb_pred.sum() * 0.0
        bg = target_mask <= 0.5
        if bool(bg.any()):
            bg_color_loss = ((rgb_pred[bg] - 1.0) ** 2).mean()
        else:
            bg_color_loss = rgb_pred.sum() * 0.0
        eikonal = eikonal_loss(field, max(256, sample_count // 4), device)
        sdf_l1 = rendered["sdf"].abs().mean()
        loss = (
            mask_loss
            + float(args.color_weight) * color_loss
            + float(args.background_color_weight) * bg_color_loss
            + float(args.surface_entropy_weight) * entropy_loss
            + float(args.eikonal_weight) * eikonal
            + float(args.sdf_l1_weight) * sdf_l1
        )
        if step < int(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step in {0, int(args.steps)} or step % max(1, int(args.steps) // 10) == 0:
            loss_curve.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "mask_bce": float(mask_loss.detach().cpu()),
                    "foreground_color_charbonnier": float(color_loss.detach().cpu()),
                    "background_color_mse": float(bg_color_loss.detach().cpu()),
                    "surface_entropy": float(entropy_loss.detach().cpu()),
                    "eikonal": float(eikonal.detach().cpu()),
                    "sdf_l1": float(sdf_l1.detach().cpu()),
                    "train_view_index": int(payload["view_index"]),
                    "train_camera_id": payload["camera_id"],
                    "batch_foreground_fraction": float(target_mask.mean().detach().cpu()),
                    "batch_opacity_mean": float(opacity.mean().detach().cpu()),
                }
            )

    output_paths: list[str] = []

    def render_payloads_for_review(review_payloads: list[dict[str, Any]], prefix_label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for payload in review_payloads:
            result = evaluate_view(
                field,
                payload,
                bbox_center,
                bbox_half,
                int(args.samples_per_ray),
                int(args.ray_batch_size),
                height,
                width,
                float(args.render_threshold),
                float(args.sdf_beta),
                float(args.density_scale),
            )
            prefix = output_dir / f"{prefix_label}_view_{payload['view_index']:02d}_cam{payload['camera_id']}"
            save_image(prefix.with_name(prefix.name + "_target_mask.png"), payload["target_mask_np"].astype(np.float32))
            save_image(prefix.with_name(prefix.name + "_render_opacity.png"), result["opacity"])
            save_image(prefix.with_name(prefix.name + "_render_rgb.png"), result["rgb"])
            save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(result["depth"], result["opacity"] > 0.5))
            output_paths.extend(
                [
                    str(prefix.with_name(prefix.name + "_target_mask.png")),
                    str(prefix.with_name(prefix.name + "_render_opacity.png")),
                    str(prefix.with_name(prefix.name + "_render_rgb.png")),
                    str(prefix.with_name(prefix.name + "_depth.png")),
                ]
            )
            rows.append(
                {
                    "view_index": int(payload["view_index"]),
                    "camera_id": payload["camera_id"],
                    "valid_ray_count": int(payload["valid_ray_count"]),
                    **result["metrics"],
                }
            )
        return rows

    view_metrics = render_payloads_for_review(payloads, "train")
    eval_view_metrics = render_payloads_for_review(eval_payloads, "eval")

    grid_resolution = int(args.sdf_grid)
    xs = torch.linspace(float(bbox_min[0]), float(bbox_max[0]), grid_resolution, device=device)
    ys = torch.linspace(float(bbox_min[1]), float(bbox_max[1]), grid_resolution, device=device)
    zs = torch.linspace(float(bbox_min[2]), float(bbox_max[2]), grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
    sdf_chunks: list[torch.Tensor] = []
    color_chunks: list[torch.Tensor] = []
    field.eval()
    with torch.no_grad():
        for start in range(0, grid.shape[0], 65536):
            pts = grid[start : start + 65536]
            pts_norm = ((pts - bbox_center[None, :]) / bbox_half[None, :].clamp_min(1e-8)).clamp(-1.35, 1.35)
            sdf, color = field(pts_norm)
            sdf_chunks.append(sdf.detach().cpu())
            color_chunks.append(color.detach().cpu())
    sdf_grid = torch.cat(sdf_chunks, dim=0).numpy().reshape(grid_resolution, grid_resolution, grid_resolution).astype(np.float32)
    grid_colors = torch.cat(color_chunks, dim=0).numpy().astype(np.float32)
    near_surface = np.abs(sdf_grid.reshape(-1)) <= float(args.sdf_beta)
    near_points = grid.detach().cpu().numpy()[near_surface]
    near_colors = grid_colors[near_surface]
    points_path = output_dir / "a22_sdf_near_surface_points.ply"
    write_point_ply(points_path, near_points, near_colors)
    output_paths.append(str(points_path))
    npz_path = output_dir / "a22_sdf_grid.npz"
    np.savez_compressed(
        npz_path,
        sdf=sdf_grid,
        bbox_min=bbox_min_np.astype(np.float32),
        bbox_max=bbox_max_np.astype(np.float32),
        view_indices=np.asarray(view_indices, dtype=np.int16),
        eval_view_indices=np.asarray(eval_view_indices, dtype=np.int16),
        level=np.asarray(0.0, dtype=np.float32),
        sdf_beta=np.asarray(float(args.sdf_beta), dtype=np.float32),
    )
    output_paths.append(str(npz_path))

    mesh_status = "not_attempted"
    mesh_path = output_dir / "a22_sdf_zero_level_mesh.ply"
    try:
        from skimage import measure  # type: ignore

        if np.any(sdf_grid <= 0.0) and np.any(sdf_grid >= 0.0):
            verts, faces, _normals, _values = measure.marching_cubes(sdf_grid, level=0.0)
            scale = (bbox_max_np - bbox_min_np) / max(1, grid_resolution - 1)
            verts_world = bbox_min_np[None, :] + verts.astype(np.float32) * scale[None, :]
            mesh_colors = np.full((verts_world.shape[0], 3), [0.64, 0.75, 0.95], dtype=np.float32)
            write_mesh_ply(mesh_path, verts_world, faces.astype(np.int64), mesh_colors)
            output_paths.append(str(mesh_path))
            mesh_status = "extracted"
        else:
            mesh_status = "empty_or_single_sign_sdf"
    except Exception as exc:  # noqa: BLE001
        mesh_status = f"skipped: {exc!r}"

    sdf_values = sdf_grid.reshape(-1)
    sdf_quantiles = {
        "p01": float(np.percentile(sdf_values, 1)) if sdf_values.size else 0.0,
        "p05": float(np.percentile(sdf_values, 5)) if sdf_values.size else 0.0,
        "p50": float(np.percentile(sdf_values, 50)) if sdf_values.size else 0.0,
        "p95": float(np.percentile(sdf_values, 95)) if sdf_values.size else 0.0,
        "p99": float(np.percentile(sdf_values, 99)) if sdf_values.size else 0.0,
        "min": float(sdf_values.min()) if sdf_values.size else 0.0,
        "max": float(sdf_values.max()) if sdf_values.size else 0.0,
    }
    avg_iou = float(np.mean([row["iou"] for row in view_metrics])) if view_metrics else 0.0
    avg_precision = float(np.mean([row["precision"] for row in view_metrics])) if view_metrics else 0.0
    avg_recall = float(np.mean([row["target_recall"] for row in view_metrics])) if view_metrics else 0.0
    avg_rgb = float(np.mean([row["foreground_rgb_residual_mean"] for row in view_metrics])) if view_metrics else 0.0
    eval_avg_iou = float(np.mean([row["iou"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_precision = float(np.mean([row["precision"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_recall = float(np.mean([row["target_recall"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    eval_avg_rgb = (
        float(np.mean([row["foreground_rgb_residual_mean"] for row in eval_view_metrics])) if eval_view_metrics else 0.0
    )
    summary_path = output_dir / "a22_sdf_surface_field_summary.json"
    report_path = output_dir / "a22_sdf_surface_field_summary.md"
    decision = (
        "A2.2 is a research-only SDF surface-field unblocker. It changes A2's free occupancy volume into an "
        "SDF-derived thin-surface representation with eikonal and ray-weight concentration diagnostics. It is not "
        "a teacher/candidate and cannot unblock cloud unless a separate Open3D/strict gate later passes."
    )
    summary = {
        "status": "a22_sdf_surface_field_research_preflight_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "scene_dir": str(scene_dir),
            "mesh_seed": str(args.mesh_seed.resolve()),
            "camera_source": camera_source,
            "views": view_indices,
            "eval_views": eval_view_indices,
            "target_size": height,
            "steps": int(args.steps),
            "ray_batch_size": int(args.ray_batch_size),
            "samples_per_ray": int(args.samples_per_ray),
            "hidden_dim": int(args.hidden_dim),
            "pos_frequencies": int(args.pos_frequencies),
            "bbox_min": [float(v) for v in bbox_min_np],
            "bbox_max": [float(v) for v in bbox_max_np],
            "avg_render_iou": avg_iou,
            "avg_render_precision": avg_precision,
            "avg_render_recall": avg_recall,
            "avg_foreground_rgb_residual_mean": avg_rgb,
            "eval_avg_render_iou": eval_avg_iou,
            "eval_avg_render_precision": eval_avg_precision,
            "eval_avg_render_recall": eval_avg_recall,
            "eval_avg_foreground_rgb_residual_mean": eval_avg_rgb,
            "sdf_grid": grid_resolution,
            "sdf_beta": float(args.sdf_beta),
            "density_scale": float(args.density_scale),
            "render_threshold": float(args.render_threshold),
            "surface_entropy_weight": float(args.surface_entropy_weight),
            "eikonal_weight": float(args.eikonal_weight),
            "sdf_l1_weight": float(args.sdf_l1_weight),
            "sdf_quantiles": sdf_quantiles,
            "near_surface_points": int(near_points.shape[0]),
            "near_surface_fraction": float(near_surface.mean()) if near_surface.size else 0.0,
            "mesh_status": mesh_status,
            "cuda": cuda_info,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "view_metrics": view_metrics,
        "eval_view_metrics": eval_view_metrics,
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
