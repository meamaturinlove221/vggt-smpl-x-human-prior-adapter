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
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    describe_cuda_device,
    load_view_rgb_mask,
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
            "A2 research-only raw RGB/mask/camera neural-field preflight. "
            "This lane trains a tiny occupancy/radiance field with volume rendering to test whether "
            "raw calibrated images provide a dense-surface signal beyond visual-hull shrinking. "
            "It never exports a teacher/candidate and never writes strict passes."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--mesh-seed", type=Path, required=True, help="Visual-hull/template mesh NPZ used only for bbox bounds.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ray-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-ray", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--pos-frequencies", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--color-weight", type=float, default=0.35)
    parser.add_argument("--background-color-weight", type=float, default=0.03)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--occupancy-grid", type=int, default=56)
    parser.add_argument("--occupancy-threshold", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mesh_vertices(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "vertices" in data.files:
            vertices = np.asarray(data["vertices"], dtype=np.float32)
        elif "hybrid_vertices" in data.files:
            vertices = np.asarray(data["hybrid_vertices"], dtype=np.float32)
        else:
            raise KeyError(f"{path} does not contain vertices or hybrid_vertices")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected mesh vertices [N,3] in {path}")
    return vertices


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.full((points.shape[0], 3), 0.7, dtype=np.float32)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors_u8, strict=False):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if colors is None:
        colors = np.full((vertices.shape[0], 3), 0.64, dtype=np.float32)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8, strict=False):
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def mask_metrics(pred_opacity: np.ndarray, target_mask: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = np.asarray(pred_opacity) >= float(threshold)
    target = np.asarray(target_mask).astype(bool)
    inter = int((pred & target).sum())
    union = int((pred | target).sum())
    pred_pixels = int(pred.sum())
    target_pixels = int(target.sum())
    return {
        "pred_pixels": pred_pixels,
        "target_pixels": target_pixels,
        "iou": float(inter / max(union, 1)),
        "target_recall": float(inter / max(target_pixels, 1)),
        "precision": float(inter / max(pred_pixels, 1)),
        "overfill_ratio": float(((pred & ~target).sum()) / max(pred_pixels, 1)),
    }


def make_rays(
    intrinsic: np.ndarray,
    cam_to_world: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    k = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    c2w = torch.as_tensor(cam_to_world, dtype=torch.float32, device=device)
    dirs_cam = torch.stack(
        [
            (xs + 0.5 - k[0, 2]) / k[0, 0].clamp_min(1e-8),
            (ys + 0.5 - k[1, 2]) / k[1, 1].clamp_min(1e-8),
            torch.ones_like(xs),
        ],
        dim=-1,
    )
    dirs_world = dirs_cam.reshape(-1, 3) @ c2w[:3, :3].T
    dirs_world = F.normalize(dirs_world, dim=1, eps=1e-8)
    origins = c2w[:3, 3].view(1, 3).expand_as(dirs_world).contiguous()
    return origins, dirs_world.contiguous()


def ray_aabb_intersection(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_d = 1.0 / torch.where(dirs.abs() < 1e-8, torch.full_like(dirs, 1e-8), dirs)
    t0 = (bbox_min[None, :] - origins) * inv_d
    t1 = (bbox_max[None, :] - origins) * inv_d
    t_min = torch.minimum(t0, t1).amax(dim=1).clamp_min(0.0)
    t_max = torch.maximum(t0, t1).amin(dim=1)
    valid = t_max > (t_min + 1e-5)
    return t_min, t_max, valid


class TinyNeuralField(nn.Module):
    def __init__(self, frequencies: int, hidden_dim: int) -> None:
        super().__init__()
        self.frequencies = int(frequencies)
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

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = [xyz]
        for freq in range(self.frequencies):
            scale = float(2**freq) * np.pi
            feats.append(torch.sin(xyz * scale))
            feats.append(torch.cos(xyz * scale))
        return torch.cat(feats, dim=-1)

    def forward(self, xyz_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(self.encode(xyz_norm))
        sigma = F.softplus(raw[..., 0] - 1.0)
        color = torch.sigmoid(raw[..., 1:4])
        return sigma, color


def render_volume(
    field: TinyNeuralField,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    bbox_center: torch.Tensor,
    bbox_half: torch.Tensor,
    samples_per_ray: int,
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
    pts_norm = ((pts - bbox_center[None, None, :]) / bbox_half[None, None, :].clamp_min(1e-8)).clamp(-1.25, 1.25)
    sigma, color = field(pts_norm.reshape(-1, 3))
    sigma = sigma.reshape(origins.shape[0], sample_count)
    color = color.reshape(origins.shape[0], sample_count, 3)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, deltas[:, -1:]], dim=1).clamp_min(1e-5)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), (1.0 - alpha + 1e-8)], dim=1), dim=1)[:, :-1]
    weights = alpha * trans
    opacity = weights.sum(dim=1).clamp(0.0, 1.0)
    rgb = (weights[..., None] * color).sum(dim=1)
    white_bg = torch.ones_like(rgb)
    rgb = rgb + (1.0 - opacity[:, None]) * white_bg
    depth = (weights * z_vals).sum(dim=1) / opacity.clamp_min(1e-6)
    return {"rgb": rgb, "opacity": opacity, "depth": depth, "weights": weights, "z_vals": z_vals}


def evaluate_view(
    field: TinyNeuralField,
    payload: dict[str, Any],
    bbox_center: torch.Tensor,
    bbox_half: torch.Tensor,
    samples_per_ray: int,
    batch_size: int,
    height: int,
    width: int,
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
                rendered = render_volume(
                    field,
                    payload["origins"][sl][valid],
                    payload["dirs"][sl][valid],
                    payload["near"][sl][valid],
                    payload["far"][sl][valid],
                    bbox_center,
                    bbox_half,
                    samples_per_ray,
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
    metrics = mask_metrics(opacity_np, target_mask)
    visible = target_mask.astype(bool)
    if np.any(visible):
        residual = np.linalg.norm(rgb_np[visible] - payload["rgb_np"][visible].astype(np.float32) / 255.0, axis=1)
        metrics["foreground_rgb_residual_mean"] = float(residual.mean())
        metrics["foreground_rgb_residual_p90"] = float(np.percentile(residual, 90))
    else:
        metrics["foreground_rgb_residual_mean"] = 0.0
        metrics["foreground_rgb_residual_p90"] = 0.0
    return {"rgb": rgb_np, "opacity": opacity_np, "depth": depth_np, "metrics": metrics}


def build_view_payloads(
    scene_dir: Path,
    manifest: dict[str, Any],
    cameras: dict[str, dict[str, np.ndarray]],
    view_indices: list[int],
    target_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    views = manifest["exported_views"]
    payloads: list[dict[str, Any]] = []
    height = width = int(target_size)
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        rgb_np, mask_np = load_view_rgb_mask(view, height)
        intrinsic = align_intrinsics_for_loaded_scene_view(
            np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32),
            view,
            height,
        )
        cam_to_world = np.asarray(cameras[camera_id]["cam_to_world"], dtype=np.float32)
        origins, dirs = make_rays(intrinsic, cam_to_world, height, width, device)
        rgb = torch.as_tensor(rgb_np.astype(np.float32) / 255.0, dtype=torch.float32, device=device).reshape(-1, 3)
        mask = torch.as_tensor(mask_np.astype(np.float32), dtype=torch.float32, device=device).reshape(-1)
        payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "rgb_np": rgb_np,
                "target_mask_np": mask_np,
                "rgb": rgb,
                "mask": mask,
                "origins": origins,
                "dirs": dirs,
            }
        )
    return payloads


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A2 Neural Field Reconstruction Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is research-only. It trains a tiny raw RGB/mask/camera neural field to test for a dense-surface signal; it is not a teacher, not a candidate, and not a cloud unblocker.",
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
            "loss_curve": [],
            "decision": "A2 neural field preflight requires CUDA for volume rendering smoke. No teacher/candidate artifact was created.",
            "outputs": [],
        }
        summary_path = output_dir / "a2_neural_field_summary.json"
        report_path = output_dir / "a2_neural_field_summary.md"
        summary["outputs"] = [str(summary_path), str(report_path)]
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(report_path, summary)
        return 2

    device = torch.device("cuda")
    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
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

    field = TinyNeuralField(int(args.pos_frequencies), int(args.hidden_dim)).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=float(args.lr))
    loss_curve: list[dict[str, Any]] = []
    rng = torch.Generator(device=device)
    rng.manual_seed(20260506)

    for step in range(int(args.steps) + 1):
        field.train()
        payload = payloads[step % len(payloads)]
        valid_indices = torch.nonzero(payload["valid"], as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            raise RuntimeError(f"No rays intersect bbox for view {payload['view_index']}")
        sample_count = min(int(args.ray_batch_size), int(valid_indices.numel()))
        perm = torch.randperm(valid_indices.numel(), generator=rng, device=device)[:sample_count]
        ray_ids = valid_indices[perm]
        rendered = render_volume(
            field,
            payload["origins"][ray_ids],
            payload["dirs"][ray_ids],
            payload["near"][ray_ids],
            payload["far"][ray_ids],
            bbox_center,
            bbox_half,
            int(args.samples_per_ray),
        )
        target_mask = payload["mask"][ray_ids]
        target_rgb = payload["rgb"][ray_ids]
        opacity = rendered["opacity"].clamp(1e-5, 1.0 - 1e-5)
        rgb_pred = rendered["rgb"].clamp(0.0, 1.0)
        mask_loss = F.binary_cross_entropy(opacity, target_mask)
        fg = target_mask > 0.5
        if bool(fg.any()):
            color_loss = torch.sqrt(((rgb_pred[fg] - target_rgb[fg]) ** 2).sum(dim=1) + 1e-4).mean()
        else:
            color_loss = rgb_pred.sum() * 0.0
        bg = target_mask <= 0.5
        if bool(bg.any()):
            bg_color_loss = ((rgb_pred[bg] - 1.0) ** 2).mean()
        else:
            bg_color_loss = rgb_pred.sum() * 0.0
        loss = mask_loss + float(args.color_weight) * color_loss + float(args.background_color_weight) * bg_color_loss
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
                    "train_view_index": int(payload["view_index"]),
                    "train_camera_id": payload["camera_id"],
                    "batch_foreground_fraction": float(target_mask.mean().detach().cpu()),
                    "batch_opacity_mean": float(opacity.mean().detach().cpu()),
                }
            )

    output_paths: list[str] = []
    view_metrics: list[dict[str, Any]] = []
    for payload in payloads:
        result = evaluate_view(
            field,
            payload,
            bbox_center,
            bbox_half,
            int(args.samples_per_ray),
            int(args.ray_batch_size),
            height,
            width,
        )
        prefix = output_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
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
        view_metrics.append(
            {
                "view_index": int(payload["view_index"]),
                "camera_id": payload["camera_id"],
                "valid_ray_count": int(payload["valid_ray_count"]),
                **result["metrics"],
            }
        )

    grid_resolution = int(args.occupancy_grid)
    xs = torch.linspace(float(bbox_min[0]), float(bbox_max[0]), grid_resolution, device=device)
    ys = torch.linspace(float(bbox_min[1]), float(bbox_max[1]), grid_resolution, device=device)
    zs = torch.linspace(float(bbox_min[2]), float(bbox_max[2]), grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
    occ_chunks: list[torch.Tensor] = []
    color_chunks: list[torch.Tensor] = []
    field.eval()
    with torch.no_grad():
        for start in range(0, grid.shape[0], 65536):
            pts = grid[start : start + 65536]
            pts_norm = ((pts - bbox_center[None, :]) / bbox_half[None, :].clamp_min(1e-8)).clamp(-1.25, 1.25)
            sigma, color = field(pts_norm)
            voxel = float((bbox_max - bbox_min).max().detach().cpu()) / max(1, grid_resolution)
            occ = (1.0 - torch.exp(-sigma * voxel)).clamp(0.0, 1.0)
            occ_chunks.append(occ.detach().cpu())
            color_chunks.append(color.detach().cpu())
    occupancy = torch.cat(occ_chunks, dim=0).numpy().reshape(grid_resolution, grid_resolution, grid_resolution).astype(np.float32)
    grid_colors = torch.cat(color_chunks, dim=0).numpy().astype(np.float32)
    occupied = occupancy.reshape(-1) >= float(args.occupancy_threshold)
    occupied_points = grid.detach().cpu().numpy()[occupied]
    occupied_colors = grid_colors[occupied]
    points_path = output_dir / "a2_neural_field_occupied_points.ply"
    write_point_ply(points_path, occupied_points, occupied_colors)
    output_paths.append(str(points_path))
    npz_path = output_dir / "a2_neural_field_occupancy_grid.npz"
    np.savez_compressed(
        npz_path,
        occupancy=occupancy,
        bbox_min=bbox_min_np.astype(np.float32),
        bbox_max=bbox_max_np.astype(np.float32),
        view_indices=np.asarray(view_indices, dtype=np.int16),
        threshold=np.asarray(float(args.occupancy_threshold), dtype=np.float32),
    )
    output_paths.append(str(npz_path))

    mesh_status = "not_attempted"
    mesh_path = output_dir / "a2_neural_field_occupancy_mesh.ply"
    try:
        from skimage import measure  # type: ignore

        if np.any(occupancy >= float(args.occupancy_threshold)) and np.any(occupancy < float(args.occupancy_threshold)):
            verts, faces, _normals, _values = measure.marching_cubes(occupancy, level=float(args.occupancy_threshold))
            scale = (bbox_max_np - bbox_min_np) / max(1, grid_resolution - 1)
            verts_world = bbox_min_np[None, :] + verts.astype(np.float32) * scale[None, :]
            mesh_colors = np.full((verts_world.shape[0], 3), [0.5, 0.72, 0.95], dtype=np.float32)
            write_mesh_ply(mesh_path, verts_world, faces.astype(np.int64), mesh_colors)
            output_paths.append(str(mesh_path))
            mesh_status = "extracted"
        else:
            mesh_status = "empty_or_full_threshold"
    except Exception as exc:  # noqa: BLE001
        mesh_status = f"skipped: {exc!r}"

    avg_iou = float(np.mean([row["iou"] for row in view_metrics])) if view_metrics else 0.0
    avg_precision = float(np.mean([row["precision"] for row in view_metrics])) if view_metrics else 0.0
    avg_recall = float(np.mean([row["target_recall"] for row in view_metrics])) if view_metrics else 0.0
    avg_rgb = float(np.mean([row["foreground_rgb_residual_mean"] for row in view_metrics])) if view_metrics else 0.0
    summary_path = output_dir / "a2_neural_field_summary.json"
    report_path = output_dir / "a2_neural_field_summary.md"
    decision = (
        "A2 is a research-only learned-field unblocker. It uses raw RGB/mask/camera volume rendering rather than "
        "VGGT shell observations or visual-hull shrinkage. The output may justify deeper dense-reconstruction work "
        "only if rendered masks/RGB and the occupancy mesh visibly improve toward a continuous normal human; it is "
        "not a teacher/candidate and cannot unblock cloud."
    )
    summary = {
        "status": "a2_neural_field_research_preflight_complete",
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
            "occupancy_grid": grid_resolution,
            "occupancy_threshold": float(args.occupancy_threshold),
            "occupied_points": int(occupied_points.shape[0]),
            "occupied_fraction": float(occupied.mean()) if occupied.size else 0.0,
            "mesh_status": mesh_status,
            "cuda": cuda_info,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
        "view_metrics": view_metrics,
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
