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

from optimize_raw_surface_nvdiffrast import (  # noqa: E402
    build_vertex_features,
    mask_metrics,
    part_offset_limits,
    unique_edges,
    write_colored_ply,
)
from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    describe_cuda_device,
    import_nvdiffrast,
    load_connected_mesh,
    load_view_rgb_mask,
    normalize_depth,
    parse_view_indices,
    render_nvdiffrast_view,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B1 learned local surface-token backend preflight. This is research-only: "
            "it exports diagnostics and a design scaffold, not a teacher, not a candidate, "
            "and not a cloud unblock signal."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--token-grid", type=int, default=5)
    parser.add_argument("--token-hidden", type=int, default=64)
    parser.add_argument("--photometric-mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-token-views", type=int, default=2)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.55)
    parser.add_argument("--overfill-weight", type=float, default=0.35)
    parser.add_argument("--token-offset-reg-weight", type=float, default=0.08)
    parser.add_argument("--edge-reg-weight", type=float, default=0.05)
    parser.add_argument("--photometric-token-weight", type=float, default=0.06)
    parser.add_argument("--depth-normal-weight", type=float, default=0.03)
    parser.add_argument("--normal-render-weight", type=float, default=0.02)
    parser.add_argument("--z-sign", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quantized_surface_tokens(vertices: np.ndarray, part_ids: np.ndarray, token_grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    token_grid = max(2, int(token_grid))
    unique_keys: list[tuple[int, int, int, int]] = []
    per_part_payload: list[tuple[np.ndarray, list[tuple[int, int, int, int]]]] = []

    for part in sorted(int(v) for v in np.unique(part_ids)):
        idx = np.nonzero(part_ids == part)[0]
        pts = vertices[idx]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        q = np.floor(((pts - lo[None, :]) / span[None, :]) * float(token_grid)).astype(np.int64)
        q = np.clip(q, 0, token_grid - 1)
        keys = [(part, int(row[0]), int(row[1]), int(row[2])) for row in q]
        unique_keys.extend(keys)
        per_part_payload.append((idx, keys))

    ordered_keys = sorted(set(unique_keys))
    key_to_id = {key: i for i, key in enumerate(ordered_keys)}
    token_ids = np.zeros((vertices.shape[0],), dtype=np.int64)
    for idx, keys in per_part_payload:
        token_ids[idx] = np.asarray([key_to_id[key] for key in keys], dtype=np.int64)

    token_part_ids = np.asarray([key[0] for key in ordered_keys], dtype=np.int64)
    meta = {
        "token_grid": int(token_grid),
        "token_count": int(len(ordered_keys)),
        "token_part_ids": token_part_ids,
        "token_part_histogram": {str(int(part)): int((token_part_ids == part).sum()) for part in np.unique(token_part_ids)},
    }
    return token_ids, meta


def aggregate_by_token(values: np.ndarray, token_ids: np.ndarray, token_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    sums = np.zeros((token_count, values.shape[1]), dtype=np.float32)
    counts = np.zeros((token_count, 1), dtype=np.float32)
    np.add.at(sums, token_ids, values)
    np.add.at(counts, token_ids, 1.0)
    return sums / np.maximum(counts, 1.0), counts[:, 0]


def scatter_token_to_vertex(token_values: torch.Tensor, vertex_token_ids: torch.Tensor) -> torch.Tensor:
    return token_values[vertex_token_ids]


def sample_vertex_rgb_mask(
    vertices: torch.Tensor,
    payload: dict[str, Any],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    hom = torch.cat([vertices, ones], dim=1)
    cam = (payload["world_to_cam"] @ hom.T).T[:, :3]
    z = cam[:, 2]
    uvw = (payload["intrinsic"] @ cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-8)
    x = (uv[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    y = (uv[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=1)
    inside = (z > 1e-6) & (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    sample_grid = grid.view(1, -1, 1, 2)
    rgb = F.grid_sample(
        payload["rgb_tensor"],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, :, 0].T
    mask = F.grid_sample(
        payload["mask_tensor"],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, :, 0]
    return rgb, mask, inside


@torch.no_grad()
def build_visibility_aggregates(
    vertices: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    token_ids: np.ndarray,
    token_count: int,
    height: int,
    width: int,
    mask_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    colors: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for payload in view_payloads:
        rgb, mask, inside = sample_vertex_rgb_mask(vertices, payload, height, width)
        weight = inside.to(vertices.dtype) * (mask > float(mask_threshold)).to(vertices.dtype)
        colors.append(rgb)
        weights.append(weight)

    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = weight_stack.sum(dim=0)
    weighted = weight_stack[:, :, None]
    mean = (color_stack * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    variance = (((color_stack - mean[None, :, :]) ** 2) * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    support_norm = (support / max(1, len(view_payloads))).clamp(0.0, 1.0)[:, None]
    vertex_features = torch.cat([mean, variance, support_norm], dim=1).detach().cpu().numpy().astype(np.float32)
    token_features, token_vertex_counts = aggregate_by_token(vertex_features, token_ids, token_count)

    view_support_np = weight_stack.detach().cpu().numpy().astype(np.float32)
    token_view_support = np.zeros((len(view_payloads), token_count), dtype=np.float32)
    for view_pos in range(len(view_payloads)):
        sums = np.zeros((token_count,), dtype=np.float32)
        counts = np.zeros((token_count,), dtype=np.float32)
        np.add.at(sums, token_ids, view_support_np[view_pos])
        np.add.at(counts, token_ids, 1.0)
        token_view_support[view_pos] = sums / np.maximum(counts, 1.0)

    token_support_count = (token_view_support > 0.05).sum(axis=0).astype(np.int64)
    support_hist = {
        str(int(bucket)): int((token_support_count == int(bucket)).sum())
        for bucket in range(len(view_payloads) + 1)
    }
    meta = {
        "image_condition_dim": int(vertex_features.shape[1]),
        "token_image_condition_dim": int(token_features.shape[1]),
        "vertices_with_two_view_support": int((support >= 2).sum().cpu()),
        "mean_vertex_support": float(support[support >= 1].mean().cpu()) if bool((support >= 1).any()) else 0.0,
        "token_support_histogram": support_hist,
        "tokens_with_two_view_support": int((token_support_count >= 2).sum()),
        "token_vertex_count_min": int(token_vertex_counts.min()) if token_vertex_counts.size else 0,
        "token_vertex_count_max": int(token_vertex_counts.max()) if token_vertex_counts.size else 0,
        "token_view_support": token_view_support,
        "token_support_count": token_support_count,
    }
    return token_features, meta


class PartSpecializedSurfaceTokenBackend(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, part_count: int) -> None:
        super().__init__()
        self.visibility_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.offset_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(part_count)
            ]
        )
        self.normal_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(part_count)
            ]
        )
        for head in self.offset_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in self.normal_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.visibility_gate[-1].weight)
        nn.init.constant_(self.visibility_gate[-1].bias, 1.5)

    def forward(
        self,
        token_features: torch.Tensor,
        token_part_ids: torch.Tensor,
        vertex_token_ids: torch.Tensor,
        vertex_limits: torch.Tensor,
        base_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_delta_unit = torch.zeros((token_features.shape[0], 3), dtype=token_features.dtype, device=token_features.device)
        token_normal_residual = torch.zeros_like(token_delta_unit)
        for part_id, head in enumerate(self.offset_heads):
            mask = token_part_ids == int(part_id)
            if bool(mask.any()):
                token_delta_unit[mask] = torch.tanh(head(token_features[mask]))
                token_normal_residual[mask] = 0.35 * torch.tanh(self.normal_heads[part_id](token_features[mask]))
        token_visibility = torch.sigmoid(self.visibility_gate(token_features)).view(-1)
        gated_delta = token_delta_unit * token_visibility[:, None]
        vertex_delta = scatter_token_to_vertex(gated_delta, vertex_token_ids) * vertex_limits
        vertex_normal = F.normalize(base_normals + scatter_token_to_vertex(token_normal_residual, vertex_token_ids), dim=1, eps=1e-6)
        return vertex_delta, token_delta_unit, vertex_normal, token_visibility


def rendered_depth_normal_consistency(depth: torch.Tensor, normal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    visible = mask > 0.5
    if int(visible.sum().detach().cpu()) <= 4:
        return depth.sum() * 0.0
    dzdx = torch.zeros_like(depth)
    dzdy = torch.zeros_like(depth)
    dzdx[:, 1:-1] = 0.5 * (depth[:, 2:] - depth[:, :-2])
    dzdy[1:-1, :] = 0.5 * (depth[2:, :] - depth[:-2, :])
    slope = torch.stack([-dzdx, -dzdy, torch.ones_like(depth)], dim=-1)
    slope = F.normalize(slope, dim=-1, eps=1e-6)
    normal_view = normal
    agreement = (slope * normal_view).sum(dim=-1).abs()
    return (1.0 - agreement[visible]).mean()


def token_photometric_variance(
    vertices: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    token_ids: torch.Tensor,
    token_count: int,
    height: int,
    width: int,
    mask_threshold: float,
    min_views: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    colors: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for payload in view_payloads:
        rgb, mask, inside = sample_vertex_rgb_mask(vertices, payload, height, width)
        weight = inside.to(vertices.dtype) * (mask > float(mask_threshold)).to(vertices.dtype)
        colors.append(rgb)
        weights.append(weight)
    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = weight_stack.sum(dim=0)
    valid_vertex = support >= float(min_views)
    if not bool(valid_vertex.any()):
        zero = vertices.sum() * 0.0
        return zero, {"valid_vertices": 0, "valid_tokens": 0, "mean_support": 0.0}
    weighted = weight_stack[:, :, None]
    mean = (color_stack * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    variance = (((color_stack - mean[None, :, :]) ** 2) * weighted).sum(dim=(0, 2)) / support.clamp_min(1.0)

    token_sum = torch.zeros((token_count,), dtype=vertices.dtype, device=vertices.device)
    token_cnt = torch.zeros((token_count,), dtype=vertices.dtype, device=vertices.device)
    token_sum.scatter_add_(0, token_ids, variance * valid_vertex.to(vertices.dtype))
    token_cnt.scatter_add_(0, token_ids, valid_vertex.to(vertices.dtype))
    valid_token = token_cnt > 0
    token_var = token_sum[valid_token] / token_cnt[valid_token].clamp_min(1.0)
    return token_var.mean(), {
        "valid_vertices": int(valid_vertex.detach().sum().cpu()),
        "valid_tokens": int(valid_token.detach().sum().cpu()),
        "mean_support": float(support[valid_vertex].detach().mean().cpu()),
    }


def save_float_heatmap(path: Path, values: np.ndarray, mask: np.ndarray | None = None) -> None:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    out = np.zeros_like(values, dtype=np.float32)
    if np.any(valid):
        hi = float(np.percentile(values[valid], 98))
        lo = float(np.percentile(values[valid], 2))
        if hi <= lo:
            hi = float(values[valid].max())
            lo = float(values[valid].min())
        if hi > lo:
            out[valid] = np.clip((values[valid] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        else:
            out[valid] = 1.0
    save_image(path, out)


def build_render_diagnostic(
    dr: Any,
    ctx: Any,
    vertices: torch.Tensor,
    faces_t: torch.Tensor,
    normals_t: torch.Tensor,
    token_conf_vertex: torch.Tensor,
    payload: dict[str, Any],
    height: int,
    width: int,
    z_sign: float,
) -> dict[str, torch.Tensor]:
    return render_nvdiffrast_view(
        dr,
        ctx,
        vertices,
        faces_t,
        normals_t,
        token_conf_vertex[:, None].expand(-1, 3),
        payload["world_to_cam"],
        payload["intrinsic"],
        height,
        width,
        z_sign=z_sign,
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Surface Token Backend B1 Research Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a B-line research-only preflight. It is not a teacher, not a candidate, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "teacher/candidate export = blocked",
        "formal VGGT cloud train/infer/export = blocked",
        "```",
        "",
        "## B1 Delta Beyond B0",
        "",
        "- surface tokens remain the optimized representation;",
        "- visibility aggregation is exported as token-by-view support, token confidence, and support maps;",
        "- part-specialized offset and normal heads are both represented;",
        "- rendered mask/depth/normal/photometric diagnostics are first-class outputs;",
        "- all outputs are diagnostics or carrier meshes, not gate payloads.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
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
    for output in summary["outputs"]:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)

    mesh = load_connected_mesh(args.template_payload)
    base_vertices_np = mesh["vertices"].astype(np.float32)
    faces_np = mesh["faces"].astype(np.int32)
    part_ids = mesh["part_ids"].astype(np.int64)
    colors_np = mesh["part_colors"].astype(np.float32)
    base_normals_np = compute_vertex_normals(base_vertices_np, faces_np)
    limits_np = part_offset_limits(part_ids)
    edges_np = unique_edges(faces_np)

    dr, import_error = import_nvdiffrast()
    cuda_info = describe_cuda_device()
    if dr is None:
        summary = {
            "status": "blocked_no_nvdiffrast",
            "summary": {
                "research_only": True,
                "strict_candidate_passes": 0,
                "strict_teacher_passes": 0,
                "formal_cloud": "blocked",
                "nvdiffrast_import_error": import_error,
                "cuda": cuda_info,
            },
            "loss_curve": [],
            "decision": "B1 requires nvdiffrast for rendered visibility diagnostics; no teacher/candidate artifact was created.",
            "outputs": [],
        }
        summary_path = output_dir / "surface_token_b1_summary.json"
        report_path = output_dir / "surface_token_b1_summary.md"
        summary["outputs"] = [str(summary_path), str(report_path)]
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(report_path, summary)
        return 2
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("CUDA is unavailable for B1 nvdiffrast research preflight")

    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    height = width = int(args.target_size)
    base_vertices = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=device).contiguous()
    base_normals = torch.as_tensor(base_normals_np, dtype=torch.float32, device=device).contiguous()
    faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    limits = torch.as_tensor(limits_np, dtype=torch.float32, device=device).view(-1, 1)
    edges = torch.as_tensor(edges_np, dtype=torch.long, device=device).contiguous()

    view_payloads: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, height)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, target_mask_np = load_view_rgb_mask(view, height)
        view_payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "rgb_np": rgb_np,
                "target_mask_np": target_mask_np,
                "rgb_tensor": torch.as_tensor(rgb_np.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous(),
                "mask_tensor": torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device)
                .view(1, 1, height, width)
                .contiguous(),
                "target_mask": torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device),
                "world_to_cam": torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device).contiguous(),
                "intrinsic": torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device).contiguous(),
            }
        )

    token_ids_np, token_meta = quantized_surface_tokens(base_vertices_np, part_ids, int(args.token_grid))
    token_count = int(token_meta["token_count"])
    visibility_token_features_np, visibility_meta = build_visibility_aggregates(
        base_vertices,
        view_payloads,
        token_ids_np,
        token_count,
        height,
        width,
        float(args.photometric_mask_threshold),
    )
    geom_features_np = build_vertex_features(base_vertices_np, base_normals_np, part_ids)
    token_geom_np, token_vertex_counts = aggregate_by_token(geom_features_np, token_ids_np, token_count)
    token_features_np = np.concatenate([token_geom_np, visibility_token_features_np, token_vertex_counts[:, None] / max(1.0, token_vertex_counts.max())], axis=1)

    token_features = torch.as_tensor(token_features_np, dtype=torch.float32, device=device).contiguous()
    token_part_ids = torch.as_tensor(np.asarray(token_meta["token_part_ids"], dtype=np.int64), dtype=torch.long, device=device).contiguous()
    vertex_token_ids = torch.as_tensor(token_ids_np, dtype=torch.long, device=device).contiguous()
    part_count = max(6, int(part_ids.max()) + 1)
    backend = PartSpecializedSurfaceTokenBackend(token_features.shape[1], int(args.token_hidden), part_count).to(device)
    optimizer = torch.optim.Adam(backend.parameters(), lr=float(args.lr))

    loss_curve: list[dict[str, Any]] = []
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta, token_delta_unit, vertex_normals, token_visibility = backend(
            token_features,
            token_part_ids,
            vertex_token_ids,
            limits,
            base_normals,
        )
        vertices = base_vertices + delta
        token_conf_vertex = scatter_token_to_vertex(token_visibility[:, None], vertex_token_ids).view(-1)
        mask_bce_total = torch.zeros((), dtype=torch.float32, device=device)
        recall_total = torch.zeros((), dtype=torch.float32, device=device)
        overfill_total = torch.zeros((), dtype=torch.float32, device=device)
        normal_consistency_total = torch.zeros((), dtype=torch.float32, device=device)
        normal_render_energy_total = torch.zeros((), dtype=torch.float32, device=device)
        metrics_rows: list[dict[str, Any]] = []
        for payload in view_payloads:
            render = build_render_diagnostic(
                dr,
                ctx,
                vertices,
                faces_t,
                vertex_normals,
                token_conf_vertex,
                payload,
                height,
                width,
                float(args.z_sign),
            )
            mask = render["mask"][..., 0].clamp(1e-4, 1.0 - 1e-4)
            target = payload["target_mask"]
            mask_bce_total = mask_bce_total + F.binary_cross_entropy(mask, target)
            recall_total = recall_total + (target * (1.0 - mask)).sum() / target.sum().clamp_min(1.0)
            overfill_total = overfill_total + ((1.0 - target) * mask).sum() / mask.sum().clamp_min(1.0)
            normal_consistency_total = normal_consistency_total + rendered_depth_normal_consistency(
                render["depth"],
                render["normal"],
                render["mask"][..., 0],
            )
            visible = render["mask"][..., 0] > 0.5
            if bool(visible.any()):
                normal_render_energy_total = normal_render_energy_total + (1.0 - render["normal"][visible, 2].abs()).mean()
            if step == 0 or step == int(args.steps):
                metrics_rows.append(
                    {
                        "view_index": payload["view_index"],
                        "camera_id": payload["camera_id"],
                        **mask_metrics(mask.detach().cpu().numpy(), payload["target_mask_np"]),
                    }
                )
        denom = max(1, len(view_payloads))
        mask_bce = mask_bce_total / denom
        recall_loss = recall_total / denom
        overfill_loss = overfill_total / denom
        normal_consistency = normal_consistency_total / denom
        normal_render_energy = normal_render_energy_total / denom
        offset_reg = (token_delta_unit.pow(2).sum(dim=1) * token_visibility.detach()).mean()
        edge_reg = (delta[edges[:, 0]] - delta[edges[:, 1]]).pow(2).mean() / (limits.mean().clamp_min(1e-6) ** 2)
        photo_loss, photo_meta = token_photometric_variance(
            vertices,
            view_payloads,
            vertex_token_ids,
            token_count,
            height,
            width,
            float(args.photometric_mask_threshold),
            int(args.min_token_views),
        )
        loss = (
            float(args.mask_bce_weight) * mask_bce
            + float(args.target_recall_weight) * recall_loss
            + float(args.overfill_weight) * overfill_loss
            + float(args.token_offset_reg_weight) * offset_reg
            + float(args.edge_reg_weight) * edge_reg
            + float(args.photometric_token_weight) * photo_loss
            + float(args.depth_normal_weight) * normal_consistency
            + float(args.normal_render_weight) * normal_render_energy
        )
        if step < int(args.steps):
            loss.backward()
            optimizer.step()
        loss_curve.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "mask_bce": float(mask_bce.detach().cpu()),
                "target_recall_loss": float(recall_loss.detach().cpu()),
                "overfill_loss": float(overfill_loss.detach().cpu()),
                "token_offset_reg": float(offset_reg.detach().cpu()),
                "edge_reg": float(edge_reg.detach().cpu()),
                "photometric_token_variance": float(photo_loss.detach().cpu()),
                "depth_normal_consistency": float(normal_consistency.detach().cpu()),
                "normal_render_energy": float(normal_render_energy.detach().cpu()),
                "token_visibility_mean": float(token_visibility.detach().mean().cpu()),
                "token_visibility_min": float(token_visibility.detach().min().cpu()),
                "token_visibility_max": float(token_visibility.detach().max().cpu()),
                "photometric_meta": photo_meta,
                "metrics": metrics_rows,
            }
        )

    with torch.no_grad():
        delta, token_delta_unit, vertex_normals, token_visibility = backend(
            token_features,
            token_part_ids,
            vertex_token_ids,
            limits,
            base_normals,
        )
        optimized_vertices = base_vertices + delta
        token_conf_vertex = scatter_token_to_vertex(token_visibility[:, None], vertex_token_ids).view(-1)
        optimized_vertices_np = optimized_vertices.detach().cpu().numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)
        vertex_normals_np = vertex_normals.detach().cpu().numpy().astype(np.float32)
        token_delta_np = token_delta_unit.detach().cpu().numpy().astype(np.float32)
        token_visibility_np = token_visibility.detach().cpu().numpy().astype(np.float32)
        token_conf_vertex_np = token_conf_vertex.detach().cpu().numpy().astype(np.float32)

    token_support_count = np.asarray(visibility_meta["token_support_count"], dtype=np.int64)
    token_view_support = np.asarray(visibility_meta["token_view_support"], dtype=np.float32)
    token_table_path = output_dir / "surface_token_b1_token_visibility.json"
    token_rows = [
        {
            "token_id": int(token_id),
            "part_id": int(token_meta["token_part_ids"][token_id]),
            "vertex_count": int(token_vertex_counts[token_id]),
            "support_view_count": int(token_support_count[token_id]),
            "visibility_gate": float(token_visibility_np[token_id]),
            "view_support": [float(v) for v in token_view_support[:, token_id]],
            "delta_unit_norm": float(np.linalg.norm(token_delta_np[token_id])),
        }
        for token_id in range(token_count)
    ]
    token_table_path.write_text(json.dumps(token_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    mesh_path = output_dir / "surface_token_b1_research_mesh.ply"
    carrier_path = output_dir / "surface_token_b1_carrier_mesh.ply"
    normals_path = output_dir / "surface_token_b1_normals.ply"
    token_conf_path = output_dir / "surface_token_b1_token_confidence_mesh.ply"
    write_colored_ply(mesh_path, optimized_vertices_np, faces_np, colors_np)
    write_colored_ply(carrier_path, base_vertices_np, faces_np, colors_np)
    write_colored_ply(normals_path, optimized_vertices_np, faces_np, (vertex_normals_np + 1.0) * 0.5)
    write_colored_ply(token_conf_path, optimized_vertices_np, faces_np, np.repeat(token_conf_vertex_np[:, None], 3, axis=1))

    initial_metrics = loss_curve[0]["metrics"]
    final_metrics = loss_curve[-1]["metrics"]
    avg_initial_iou = float(np.mean([row["iou"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([row["iou"] for row in final_metrics])) if final_metrics else 0.0
    output_paths = [str(mesh_path), str(carrier_path), str(normals_path), str(token_conf_path), str(token_table_path)]
    render_rows: list[dict[str, Any]] = []
    for payload in view_payloads:
        with torch.no_grad():
            render = build_render_diagnostic(
                dr,
                ctx,
                torch.as_tensor(optimized_vertices_np, dtype=torch.float32, device=device).contiguous(),
                faces_t,
                torch.as_tensor(vertex_normals_np, dtype=torch.float32, device=device).contiguous(),
                torch.as_tensor(token_conf_vertex_np, dtype=torch.float32, device=device).contiguous(),
                payload,
                height,
                width,
                float(args.z_sign),
            )
            sampled_rgb, _sampled_mask, _inside = sample_vertex_rgb_mask(
                torch.as_tensor(optimized_vertices_np, dtype=torch.float32, device=device).contiguous(),
                payload,
                height,
                width,
            )
        prefix = output_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
        mask_np = render["mask"][..., 0].detach().cpu().numpy().astype(np.float32)
        depth_np = render["depth"].detach().cpu().numpy().astype(np.float32)
        normal_np = render["normal"].detach().cpu().numpy().astype(np.float32)
        support_np = render["color"][..., 0].detach().cpu().numpy().astype(np.float32)
        rendered_rgb_np = render["color"].detach().cpu().numpy().astype(np.float32)
        target_rgb_np = payload["rgb_np"].astype(np.float32) / 255.0
        photometric_residual_np = np.linalg.norm(rendered_rgb_np - target_rgb_np, axis=2) * (mask_np > 0.5)
        mask_metrics_row = mask_metrics(mask_np, payload["target_mask_np"])
        save_image(prefix.with_name(prefix.name + "_target_mask.png"), payload["target_mask_np"].astype(np.float32))
        save_image(prefix.with_name(prefix.name + "_render_mask.png"), mask_np)
        save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(depth_np, mask_np > 0.5))
        save_image(prefix.with_name(prefix.name + "_normal.png"), (normal_np + 1.0) * 0.5)
        save_float_heatmap(prefix.with_name(prefix.name + "_support.png"), support_np, mask_np > 0.5)
        save_float_heatmap(prefix.with_name(prefix.name + "_photometric_residual.png"), photometric_residual_np, mask_np > 0.5)
        for suffix in ("target_mask", "render_mask", "depth", "normal", "support", "photometric_residual"):
            output_paths.append(str(prefix.with_name(prefix.name + f"_{suffix}.png")))
        visible_residual = photometric_residual_np[mask_np > 0.5]
        render_rows.append(
            {
                "view_index": int(payload["view_index"]),
                "camera_id": payload["camera_id"],
                **mask_metrics_row,
                "visible_pixels": int((mask_np > 0.5).sum()),
                "photometric_residual_mean": float(visible_residual.mean()) if visible_residual.size else 0.0,
                "photometric_residual_p90": float(np.percentile(visible_residual, 90)) if visible_residual.size else 0.0,
                "render_support_mean": float(support_np[mask_np > 0.5].mean()) if np.any(mask_np > 0.5) else 0.0,
                "sampled_vertex_rgb_mean": [float(v) for v in sampled_rgb.detach().mean(dim=0).cpu().numpy()],
            }
        )

    summary_path = output_dir / "surface_token_b1_summary.json"
    report_path = output_dir / "surface_token_b1_summary.md"
    token_support_hist = {
        str(int(bucket)): int((token_support_count == int(bucket)).sum())
        for bucket in range(len(view_payloads) + 1)
    }
    elapsed = float(time.perf_counter() - start_time)
    summary = {
        "status": "surface_token_b1_research_preflight_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "scene_dir": str(scene_dir),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": int(args.target_size),
            "steps": int(args.steps),
            "token_grid": int(args.token_grid),
            "token_hidden": int(args.token_hidden),
            "token_meta": {k: v for k, v in token_meta.items() if k != "token_part_ids"},
            "visibility_meta": {k: v for k, v in visibility_meta.items() if k not in {"token_view_support", "token_support_count"}},
            "token_support_histogram": token_support_hist,
            "tokens_with_min_view_support": int((token_support_count >= int(args.min_token_views)).sum()),
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "max_vertex_delta": float(np.linalg.norm(delta_np, axis=1).max()) if delta_np.size else 0.0,
            "mean_vertex_delta": float(np.linalg.norm(delta_np, axis=1).mean()) if delta_np.size else 0.0,
            "max_token_delta_unit": float(np.linalg.norm(token_delta_np, axis=1).max()) if token_delta_np.size else 0.0,
            "token_visibility_mean": float(token_visibility_np.mean()) if token_visibility_np.size else 0.0,
            "token_visibility_min": float(token_visibility_np.min()) if token_visibility_np.size else 0.0,
            "token_visibility_max": float(token_visibility_np.max()) if token_visibility_np.size else 0.0,
            "cuda": cuda_info,
            "elapsed_seconds": elapsed,
        },
        "render_diagnostics": render_rows,
        "loss_curve": loss_curve,
        "decision": (
            "B1 is a research-only backend preflight beyond image_mlp++: surface tokens are the learned carrier, "
            "visibility is aggregated per token and exported, part-specialized offset/normal heads are exercised, "
            "and rendered mask/depth/normal/photometric diagnostics are written. It still cannot be a teacher or "
            "candidate without a strict local visual pass and real dense supervision."
        ),
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
