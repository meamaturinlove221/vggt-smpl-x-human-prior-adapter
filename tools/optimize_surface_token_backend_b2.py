from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
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


PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline",
    5: "lower_clothing_proxy",
}
FAMILY_NAMES = ("body", "hand", "face", "hair")
FAMILY_TO_ID = {name: idx for idx, name in enumerate(FAMILY_NAMES)}
PART_TO_FAMILY = {
    0: "body",
    1: "hand",
    2: "hand",
    3: "face",
    4: "hair",
    5: "body",
}
FAMILY_COLORS = np.asarray(
    [
        [0.65, 0.70, 0.74],
        [0.20, 0.74, 0.48],
        [1.00, 0.48, 0.30],
        [0.46, 0.36, 0.82],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class FamilyHeadSpec:
    part_ids: tuple[int, ...]
    hidden_layers: int
    delta_limit_scale: float
    normal_residual_scale: float
    confidence_bias: float


FAMILY_HEAD_SPECS = {
    "body": FamilyHeadSpec(part_ids=(0, 5), hidden_layers=1, delta_limit_scale=0.45, normal_residual_scale=0.16, confidence_bias=1.1),
    "hand": FamilyHeadSpec(part_ids=(1, 2), hidden_layers=2, delta_limit_scale=0.95, normal_residual_scale=0.30, confidence_bias=0.9),
    "face": FamilyHeadSpec(part_ids=(3,), hidden_layers=2, delta_limit_scale=0.70, normal_residual_scale=0.26, confidence_bias=1.0),
    "hair": FamilyHeadSpec(part_ids=(4,), hidden_layers=2, delta_limit_scale=1.15, normal_residual_scale=0.36, confidence_bias=0.7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B2 part-specialized surface-token backend research preflight. "
            "This is not B1 hidden/step tuning and not image_mlp++. It builds "
            "face/hair/hand/body token-head diagnostics, visibility-aware RGB/mask/depth/normal "
            "aggregation, and explicit stop reasons. It does not export a teacher, candidate, "
            "strict pass, train job, infer job, or cloud unblock signal."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--token-grid", type=int, default=5)
    parser.add_argument("--token-hidden", type=int, default=64)
    parser.add_argument("--photometric-mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-token-views", type=int, default=2)
    parser.add_argument("--min-family-visible-fraction", type=float, default=0.01)
    parser.add_argument("--min-mask-iou", type=float, default=0.05)
    parser.add_argument("--max-rgb-residual", type=float, default=0.75)
    parser.add_argument("--max-depth-std", type=float, default=10.0)
    parser.add_argument("--max-normal-angular-std-deg", type=float, default=85.0)
    parser.add_argument("--max-vertex-delta", type=float, default=0.020)
    parser.add_argument("--early-stop-window", type=int, default=2)
    parser.add_argument("--min-loss-improvement", type=float, default=1e-5)
    parser.add_argument("--continue-on-visibility-fail", action="store_true")
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.55)
    parser.add_argument("--overfill-weight", type=float, default=0.35)
    parser.add_argument("--token-offset-reg-weight", type=float, default=0.08)
    parser.add_argument("--edge-reg-weight", type=float, default=0.05)
    parser.add_argument("--rgb-residual-weight", type=float, default=0.04)
    parser.add_argument("--depth-normal-weight", type=float, default=0.03)
    parser.add_argument("--normal-render-weight", type=float, default=0.02)
    parser.add_argument("--z-sign", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def family_id_for_part(part_id: int) -> int:
    return FAMILY_TO_ID[PART_TO_FAMILY.get(int(part_id), "body")]


def family_ids_for_parts(part_ids: np.ndarray) -> np.ndarray:
    return np.asarray([family_id_for_part(int(part)) for part in np.asarray(part_ids).reshape(-1)], dtype=np.int64)


def family_histogram(family_ids: np.ndarray) -> dict[str, int]:
    return {
        name: int((np.asarray(family_ids, dtype=np.int64) == int(family_id)).sum())
        for name, family_id in FAMILY_TO_ID.items()
    }


def family_colors_for_part_ids(part_ids: np.ndarray) -> np.ndarray:
    family_ids = family_ids_for_parts(part_ids)
    return FAMILY_COLORS[np.clip(family_ids, 0, FAMILY_COLORS.shape[0] - 1)]


def quantized_surface_tokens(vertices: np.ndarray, part_ids: np.ndarray, token_grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    token_grid = max(2, int(token_grid))
    unique_keys: list[tuple[int, int, int, int]] = []
    per_part_payload: list[tuple[np.ndarray, list[tuple[int, int, int, int]]]] = []

    for part in sorted(int(value) for value in np.unique(part_ids)):
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
    key_to_id = {key: idx for idx, key in enumerate(ordered_keys)}
    token_ids = np.zeros((vertices.shape[0],), dtype=np.int64)
    for idx, keys in per_part_payload:
        token_ids[idx] = np.asarray([key_to_id[key] for key in keys], dtype=np.int64)

    token_part_ids = np.asarray([key[0] for key in ordered_keys], dtype=np.int64)
    token_family_ids = family_ids_for_parts(token_part_ids)
    meta = {
        "token_grid": int(token_grid),
        "token_count": int(len(ordered_keys)),
        "token_keys": [
            {"part_id": int(part), "qx": int(qx), "qy": int(qy), "qz": int(qz)}
            for part, qx, qy, qz in ordered_keys
        ],
        "token_part_ids": token_part_ids,
        "token_family_ids": token_family_ids,
        "token_part_histogram": {
            PART_NAMES.get(int(part), str(int(part))): int((token_part_ids == int(part)).sum())
            for part in np.unique(token_part_ids)
        },
        "token_family_histogram": family_histogram(token_family_ids),
    }
    return token_ids, meta


def aggregate_by_token(values: np.ndarray, token_ids: np.ndarray, token_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    token_ids = np.asarray(token_ids, dtype=np.int64)
    sums = np.zeros((token_count, values.shape[1]), dtype=np.float32)
    counts = np.zeros((token_count, 1), dtype=np.float32)
    np.add.at(sums, token_ids, values)
    np.add.at(counts, token_ids, 1.0)
    return sums / np.maximum(counts, 1.0), counts[:, 0]


def scatter_token_to_vertex(token_values: torch.Tensor, vertex_token_ids: torch.Tensor) -> torch.Tensor:
    return token_values[vertex_token_ids]


def one_hot(ids: np.ndarray, count: int) -> np.ndarray:
    ids_np = np.asarray(ids, dtype=np.int64)
    out = np.zeros((ids_np.shape[0], int(count)), dtype=np.float32)
    valid = (ids_np >= 0) & (ids_np < int(count))
    out[np.nonzero(valid)[0], ids_np[valid]] = 1.0
    return out


def make_view_payloads(
    views: list[dict[str, Any]],
    view_indices: list[int],
    cameras: dict[str, Any],
    target_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    height = width = int(target_size)
    payloads: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, height)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, target_mask_np = load_view_rgb_mask(view, height)
        payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "rgb_np": rgb_np,
                "target_mask_np": target_mask_np.astype(bool),
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
    return payloads


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
def build_projected_rgb_mask_aggregates(
    vertices: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    token_ids: np.ndarray,
    token_family_ids: np.ndarray,
    token_count: int,
    height: int,
    width: int,
    mask_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    colors: list[torch.Tensor] = []
    foreground_weights: list[torch.Tensor] = []
    inside_weights: list[torch.Tensor] = []
    sampled_masks: list[torch.Tensor] = []
    for payload in view_payloads:
        rgb, mask, inside = sample_vertex_rgb_mask(vertices, payload, height, width)
        inside_f = inside.to(vertices.dtype)
        foreground = inside_f * (mask > float(mask_threshold)).to(vertices.dtype)
        colors.append(rgb)
        foreground_weights.append(foreground)
        inside_weights.append(inside_f)
        sampled_masks.append(mask * inside_f)

    color_stack = torch.stack(colors, dim=0)
    foreground_stack = torch.stack(foreground_weights, dim=0)
    inside_stack = torch.stack(inside_weights, dim=0)
    mask_stack = torch.stack(sampled_masks, dim=0)
    support = foreground_stack.sum(dim=0)
    inside_support = inside_stack.sum(dim=0)
    weighted = foreground_stack[:, :, None]
    mean = (color_stack * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    variance = (((color_stack - mean[None, :, :]) ** 2) * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    mask_mean = mask_stack.sum(dim=0) / inside_support.clamp_min(1.0)
    support_norm = (support / max(1, len(view_payloads))).clamp(0.0, 1.0)[:, None]
    inside_norm = (inside_support / max(1, len(view_payloads))).clamp(0.0, 1.0)[:, None]
    vertex_features = torch.cat([mean, variance, support_norm, inside_norm, mask_mean[:, None]], dim=1)
    token_features, token_vertex_counts = aggregate_by_token(vertex_features.cpu().numpy().astype(np.float32), token_ids, token_count)

    foreground_np = foreground_stack.cpu().numpy().astype(np.float32)
    token_view_support = np.zeros((len(view_payloads), token_count), dtype=np.float32)
    for view_pos in range(len(view_payloads)):
        sums = np.zeros((token_count,), dtype=np.float32)
        counts = np.zeros((token_count,), dtype=np.float32)
        np.add.at(sums, token_ids, foreground_np[view_pos])
        np.add.at(counts, token_ids, 1.0)
        token_view_support[view_pos] = sums / np.maximum(counts, 1.0)

    token_support_count = (token_view_support > 0.05).sum(axis=0).astype(np.int64)
    family_rows = []
    for family_name, family_id in FAMILY_TO_ID.items():
        mask = token_family_ids == int(family_id)
        family_rows.append(
            {
                "family": family_name,
                "token_count": int(mask.sum()),
                "projected_tokens_with_two_views": int((token_support_count[mask] >= 2).sum()) if np.any(mask) else 0,
                "projected_visible_token_fraction": float((token_support_count[mask] >= 1).mean()) if np.any(mask) else 0.0,
                "projected_mean_support_views": float(token_support_count[mask].mean()) if np.any(mask) else 0.0,
            }
        )
    meta = {
        "projected_feature_dim": int(token_features.shape[1]),
        "projected_token_view_support": token_view_support,
        "projected_token_support_count": token_support_count,
        "projected_token_support_histogram": {
            str(int(bucket)): int((token_support_count == int(bucket)).sum())
            for bucket in range(len(view_payloads) + 1)
        },
        "projected_tokens_with_two_view_support": int((token_support_count >= 2).sum()),
        "projected_mean_vertex_support": float(support[support >= 1].mean().cpu()) if bool((support >= 1).any()) else 0.0,
        "projected_vertex_count_min": int(token_vertex_counts.min()) if token_vertex_counts.size else 0,
        "projected_vertex_count_max": int(token_vertex_counts.max()) if token_vertex_counts.size else 0,
        "family_projected_visibility": family_rows,
    }
    return token_features.astype(np.float32), meta


def precompute_face_labels(faces: np.ndarray, vertex_token_ids: np.ndarray, vertex_family_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    faces_np = np.asarray(faces, dtype=np.int64)
    token_ids = np.asarray(vertex_token_ids, dtype=np.int64)
    family_ids = np.asarray(vertex_family_ids, dtype=np.int64)
    face_token_ids = np.zeros((faces_np.shape[0],), dtype=np.int64)
    face_family_ids = np.zeros((faces_np.shape[0],), dtype=np.int64)
    for face_idx, tri in enumerate(faces_np):
        tri_tokens = token_ids[tri]
        tri_families = family_ids[tri]
        face_token_ids[face_idx] = int(np.bincount(tri_tokens).argmax())
        face_family_ids[face_idx] = int(np.bincount(tri_families, minlength=len(FAMILY_NAMES)).argmax())
    return face_token_ids, face_family_ids


def safe_bincount(ids: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(np.asarray(ids, dtype=np.int64), weights=np.asarray(weights, dtype=np.float64), minlength=count).astype(np.float32)


def normalize_depth_values(values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values_np) & (values_np > 0)
    out = np.zeros_like(values_np, dtype=np.float32)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(values_np[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values_np[valid].min())
        hi = float(values_np[valid].max())
    if hi <= lo:
        out[valid] = 1.0
    else:
        out[valid] = np.clip((values_np[valid] - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    return out


def save_float_heatmap(path: Path, values: np.ndarray, mask: np.ndarray | None = None) -> None:
    values_np = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values_np)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    out = np.zeros_like(values_np, dtype=np.float32)
    if np.any(valid):
        hi = float(np.percentile(values_np[valid], 98))
        lo = float(np.percentile(values_np[valid], 2))
        if hi <= lo:
            hi = float(values_np[valid].max())
            lo = float(values_np[valid].min())
        if hi > lo:
            out[valid] = np.clip((values_np[valid] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        else:
            out[valid] = 1.0
    save_image(path, out)


@torch.no_grad()
def build_raster_rgb_mask_depth_normal_aggregates(
    dr: Any,
    ctx: Any,
    vertices: torch.Tensor,
    faces_t: torch.Tensor,
    faces_np: np.ndarray,
    normals_t: torch.Tensor,
    render_colors_t: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    face_token_ids: np.ndarray,
    token_family_ids: np.ndarray,
    token_count: int,
    height: int,
    width: int,
    z_sign: float,
    output_dir: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    output_paths: list[str] = []
    view_count = len(view_payloads)
    pixel_counts = np.zeros((view_count, token_count), dtype=np.float32)
    target_counts = np.zeros((view_count, token_count), dtype=np.float32)
    rgb_residual_sums = np.zeros((view_count, token_count), dtype=np.float32)
    depth_sums = np.zeros((view_count, token_count), dtype=np.float32)
    depth_sq_sums = np.zeros((view_count, token_count), dtype=np.float32)
    normal_sums = np.zeros((view_count, token_count, 3), dtype=np.float32)
    view_rows: list[dict[str, Any]] = []

    for view_pos, payload in enumerate(view_payloads):
        render = render_nvdiffrast_view(
            dr,
            ctx,
            vertices,
            faces_t,
            normals_t,
            render_colors_t,
            payload["world_to_cam"],
            payload["intrinsic"],
            height,
            width,
            z_sign=z_sign,
        )
        face_ids = render["visibility"].detach().cpu().numpy().astype(np.int64) - 1
        valid = (face_ids >= 0) & (face_ids < faces_np.shape[0])
        token_image = np.full((height, width), -1, dtype=np.int64)
        if np.any(valid):
            token_image[valid] = face_token_ids[face_ids[valid]]

        mask_np = render["mask"].detach().cpu().numpy().astype(np.float32)
        depth_np = render["depth"].detach().cpu().numpy().astype(np.float32)
        normal_np = render["normal"].detach().cpu().numpy().astype(np.float32)
        render_rgb_np = render["color"].detach().cpu().numpy().astype(np.float32)
        target_rgb_np = payload["rgb_np"].astype(np.float32) / 255.0
        target_mask_np = payload["target_mask_np"].astype(bool)
        rgb_residual_np = np.linalg.norm(render_rgb_np - target_rgb_np, axis=2)

        prefix = output_dir / f"{label}_view_{payload['view_index']:02d}_cam{payload['camera_id']}"
        save_image(prefix.with_name(prefix.name + "_target_rgb.png"), target_rgb_np)
        save_image(prefix.with_name(prefix.name + "_target_mask.png"), target_mask_np.astype(np.float32))
        save_image(prefix.with_name(prefix.name + "_render_mask.png"), mask_np)
        save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(depth_np, mask_np > 0.5))
        save_image(prefix.with_name(prefix.name + "_normal.png"), (normal_np + 1.0) * 0.5)
        save_image(prefix.with_name(prefix.name + "_render_rgb.png"), render_rgb_np)
        save_float_heatmap(prefix.with_name(prefix.name + "_rgb_residual.png"), rgb_residual_np, mask_np > 0.5)
        for suffix in ("target_rgb", "target_mask", "render_mask", "depth", "normal", "render_rgb", "rgb_residual"):
            output_paths.append(str(prefix.with_name(prefix.name + f"_{suffix}.png")))

        valid_tokens = token_image[valid]
        if valid_tokens.size:
            pixel_counts[view_pos] = safe_bincount(valid_tokens, np.ones_like(valid_tokens, dtype=np.float32), token_count)
            target_counts[view_pos] = safe_bincount(valid_tokens, target_mask_np[valid].astype(np.float32), token_count)
            rgb_residual_sums[view_pos] = safe_bincount(valid_tokens, rgb_residual_np[valid].astype(np.float32), token_count)
            depth_sums[view_pos] = safe_bincount(valid_tokens, depth_np[valid].astype(np.float32), token_count)
            depth_sq_sums[view_pos] = safe_bincount(valid_tokens, (depth_np[valid].astype(np.float32) ** 2), token_count)
            for channel in range(3):
                normal_sums[view_pos, :, channel] = safe_bincount(valid_tokens, normal_np[..., channel][valid], token_count)

        visible_residual = rgb_residual_np[mask_np > 0.5]
        visible_depth = depth_np[(mask_np > 0.5) & np.isfinite(depth_np) & (depth_np > 0)]
        visible_normals = normal_np[mask_np > 0.5]
        normal_mean_len = float(np.linalg.norm(visible_normals.mean(axis=0))) if visible_normals.size else 0.0
        view_rows.append(
            {
                "view_index": int(payload["view_index"]),
                "camera_id": payload["camera_id"],
                **mask_metrics(mask_np, target_mask_np),
                "visible_pixels": int((mask_np > 0.5).sum()),
                "rgb_residual_mean": float(visible_residual.mean()) if visible_residual.size else 0.0,
                "rgb_residual_p90": float(np.percentile(visible_residual, 90)) if visible_residual.size else 0.0,
                "depth_mean": float(visible_depth.mean()) if visible_depth.size else 0.0,
                "depth_std": float(visible_depth.std()) if visible_depth.size else 0.0,
                "normal_dispersion": float(max(0.0, 1.0 - normal_mean_len)),
            }
        )

    total_pixels = pixel_counts.sum(axis=0)
    total_targets = target_counts.sum(axis=0)
    total_rgb_residual = rgb_residual_sums.sum(axis=0)
    total_depth = depth_sums.sum(axis=0)
    total_depth_sq = depth_sq_sums.sum(axis=0)
    total_normals = normal_sums.sum(axis=0)
    visible = total_pixels > 0
    visible_view_count = (pixel_counts > 0).sum(axis=0).astype(np.int64)
    target_fraction = np.zeros((token_count,), dtype=np.float32)
    rgb_residual_mean = np.zeros((token_count,), dtype=np.float32)
    depth_mean = np.zeros((token_count,), dtype=np.float32)
    depth_std = np.zeros((token_count,), dtype=np.float32)
    normal_mean = np.zeros((token_count, 3), dtype=np.float32)
    normal_dispersion = np.zeros((token_count,), dtype=np.float32)
    target_fraction[visible] = total_targets[visible] / np.maximum(total_pixels[visible], 1.0)
    rgb_residual_mean[visible] = total_rgb_residual[visible] / np.maximum(total_pixels[visible], 1.0)
    depth_mean[visible] = total_depth[visible] / np.maximum(total_pixels[visible], 1.0)
    depth_var = np.zeros((token_count,), dtype=np.float32)
    depth_var[visible] = total_depth_sq[visible] / np.maximum(total_pixels[visible], 1.0) - depth_mean[visible] ** 2
    depth_std[visible] = np.sqrt(np.maximum(depth_var[visible], 0.0))
    normal_mean[visible] = total_normals[visible] / np.maximum(total_pixels[visible, None], 1.0)
    normal_mean_len = np.linalg.norm(normal_mean, axis=1)
    normal_dispersion[visible] = np.maximum(0.0, 1.0 - np.clip(normal_mean_len[visible], 0.0, 1.0))
    normal_angular_std_deg = np.degrees(np.arccos(np.clip(normal_mean_len, 0.0, 1.0))).astype(np.float32)
    depth_mean_norm = normalize_depth_values(depth_mean)
    pixel_fraction = total_pixels / max(1.0, float(height * width * max(1, view_count)))
    features = np.stack(
        [
            visible_view_count.astype(np.float32) / max(1.0, float(view_count)),
            pixel_fraction.astype(np.float32),
            target_fraction,
            rgb_residual_mean,
            depth_mean_norm,
            depth_std,
            normal_dispersion,
            normal_angular_std_deg / 180.0,
        ],
        axis=1,
    ).astype(np.float32)

    family_rows = []
    for family_name, family_id in FAMILY_TO_ID.items():
        family_mask = token_family_ids == int(family_id)
        visible_family = family_mask & visible
        token_count_family = int(family_mask.sum())
        family_rows.append(
            {
                "family": family_name,
                "token_count": token_count_family,
                "visible_token_count": int(visible_family.sum()),
                "visible_token_fraction": float(visible_family.sum() / max(1, token_count_family)),
                "tokens_with_min_view_support": int((visible_view_count[family_mask] >= 2).sum()) if np.any(family_mask) else 0,
                "pixel_count": float(total_pixels[family_mask].sum()) if np.any(family_mask) else 0.0,
                "target_fraction_mean": float(target_fraction[visible_family].mean()) if np.any(visible_family) else 0.0,
                "rgb_residual_mean": float(rgb_residual_mean[visible_family].mean()) if np.any(visible_family) else 0.0,
                "depth_std_mean": float(depth_std[visible_family].mean()) if np.any(visible_family) else 0.0,
                "normal_angular_std_deg_mean": float(normal_angular_std_deg[visible_family].mean()) if np.any(visible_family) else 0.0,
            }
        )

    meta = {
        "raster_feature_dim": int(features.shape[1]),
        "raster_token_pixel_support": pixel_counts,
        "raster_visible_view_count": visible_view_count,
        "raster_token_target_fraction": target_fraction,
        "raster_token_rgb_residual_mean": rgb_residual_mean,
        "raster_token_depth_mean": depth_mean,
        "raster_token_depth_std": depth_std,
        "raster_token_normal_mean": normal_mean,
        "raster_token_normal_dispersion": normal_dispersion,
        "raster_token_normal_angular_std_deg": normal_angular_std_deg,
        "raster_token_support_histogram": {
            str(int(bucket)): int((visible_view_count == int(bucket)).sum())
            for bucket in range(view_count + 1)
        },
        "family_raster_visibility": family_rows,
        "view_diagnostics": view_rows,
    }
    return features, meta, output_paths


class FamilyTokenHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, spec: FamilyHeadSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, int(spec.hidden_layers) - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.offset = nn.Linear(hidden_dim, 3)
        self.normal = nn.Linear(hidden_dim, 3)
        self.visibility = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.normal.weight)
        nn.init.zeros_(self.normal.bias)
        nn.init.zeros_(self.visibility.weight)
        nn.init.constant_(self.visibility.bias, float(spec.confidence_bias))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.offset(hidden), self.normal(hidden), self.visibility(hidden).view(-1)


class PartSpecializedB2SurfaceTokenBackend(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                family_name: FamilyTokenHead(feature_dim, hidden_dim, FAMILY_HEAD_SPECS[family_name])
                for family_name in FAMILY_NAMES
            }
        )

    def forward(
        self,
        token_features: torch.Tensor,
        token_family_ids: torch.Tensor,
        vertex_token_ids: torch.Tensor,
        vertex_limits: torch.Tensor,
        base_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_delta_unit = torch.zeros((token_features.shape[0], 3), dtype=token_features.dtype, device=token_features.device)
        token_normal_residual = torch.zeros_like(token_delta_unit)
        token_visibility_logits = torch.zeros((token_features.shape[0],), dtype=token_features.dtype, device=token_features.device)
        for family_name, family_id in FAMILY_TO_ID.items():
            mask = token_family_ids == int(family_id)
            if not bool(mask.any()):
                continue
            offset_raw, normal_raw, visibility_logit = self.heads[family_name](token_features[mask])
            spec = FAMILY_HEAD_SPECS[family_name]
            token_delta_unit[mask] = float(spec.delta_limit_scale) * torch.tanh(offset_raw)
            token_normal_residual[mask] = float(spec.normal_residual_scale) * torch.tanh(normal_raw)
            token_visibility_logits[mask] = visibility_logit
        token_visibility = torch.sigmoid(token_visibility_logits)
        gated_delta = token_delta_unit * token_visibility[:, None]
        vertex_delta = scatter_token_to_vertex(gated_delta, vertex_token_ids) * vertex_limits
        vertex_normals = F.normalize(base_normals + scatter_token_to_vertex(token_normal_residual, vertex_token_ids), dim=1, eps=1e-6)
        return vertex_delta, token_delta_unit, vertex_normals, token_visibility, token_visibility_logits


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
    agreement = (slope * normal).sum(dim=-1).abs()
    return (1.0 - agreement[visible]).mean()


def rendered_rgb_residual(render_rgb: torch.Tensor, target_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    visible = mask > 0.5
    if not bool(visible.any()):
        return render_rgb.sum() * 0.0
    target_hwc = target_rgb[0].permute(1, 2, 0)
    return torch.linalg.norm(render_rgb - target_hwc, dim=-1)[visible].mean()


def stopping_condition_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "id": "backend_missing",
            "condition": "nvdiffrast import and CUDA availability are required before raster depth/normal aggregation",
            "action": "stop with blocked_no_nvdiffrast or blocked_no_cuda; keep strict passes at zero",
        },
        {
            "id": "family_visibility_guard",
            "condition": f"each nonempty family must have visible_token_fraction >= {float(args.min_family_visible_fraction):.4f}",
            "action": "stop before optimization unless --continue-on-visibility-fail is set",
        },
        {
            "id": "mask_guard",
            "condition": f"average rendered mask IoU must stay >= {float(args.min_mask_iou):.4f}",
            "action": "stop diagnostics and report the failing view/family metrics",
        },
        {
            "id": "rgb_depth_normal_guard",
            "condition": (
                f"mean RGB residual <= {float(args.max_rgb_residual):.4f}, depth std <= {float(args.max_depth_std):.4f}, "
                f"normal angular std <= {float(args.max_normal_angular_std_deg):.2f} deg"
            ),
            "action": "stop diagnostics; this is not a pass condition",
        },
        {
            "id": "nonfinite_loss",
            "condition": "loss or diagnostics become NaN/Inf",
            "action": "stop immediately and keep only research artifacts",
        },
        {
            "id": "delta_guard",
            "condition": f"max vertex delta must remain <= {float(args.max_vertex_delta):.6f}",
            "action": "stop to avoid hiding template failure with unconstrained residuals",
        },
        {
            "id": "plateau_guard",
            "condition": (
                f"loss improvement over the last {int(args.early_stop_window)} steps is "
                f"< {float(args.min_loss_improvement):.6g}"
            ),
            "action": "stop as no-evidence plateau",
        },
        {
            "id": "budget_guard",
            "condition": f"step reaches --max-steps={int(args.max_steps)}",
            "action": "stop with max_steps_reached, not success",
        },
    ]


def evaluate_family_visibility(meta: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in meta.get("family_raster_visibility", []):
        if int(row.get("token_count", 0)) <= 0:
            continue
        visible_fraction = float(row.get("visible_token_fraction", 0.0))
        if visible_fraction < float(args.min_family_visible_fraction):
            failures.append(
                {
                    "condition_id": "family_visibility_guard",
                    "family": row.get("family"),
                    "visible_token_fraction": visible_fraction,
                    "threshold": float(args.min_family_visible_fraction),
                }
            )
    return failures


def evaluate_global_metric_guards(meta: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    view_rows = meta.get("view_diagnostics", [])
    if view_rows:
        avg_iou = float(np.mean([float(row.get("iou", 0.0)) for row in view_rows]))
        if avg_iou < float(args.min_mask_iou):
            failures.append({"condition_id": "mask_guard", "avg_mask_iou": avg_iou, "threshold": float(args.min_mask_iou)})
        avg_rgb = float(np.mean([float(row.get("rgb_residual_mean", 0.0)) for row in view_rows]))
        if avg_rgb > float(args.max_rgb_residual):
            failures.append({"condition_id": "rgb_depth_normal_guard", "metric": "rgb_residual_mean", "value": avg_rgb, "threshold": float(args.max_rgb_residual)})
        avg_depth_std = float(np.mean([float(row.get("depth_std", 0.0)) for row in view_rows]))
        if avg_depth_std > float(args.max_depth_std):
            failures.append({"condition_id": "rgb_depth_normal_guard", "metric": "depth_std", "value": avg_depth_std, "threshold": float(args.max_depth_std)})
    family_rows = meta.get("family_raster_visibility", [])
    for row in family_rows:
        value = float(row.get("normal_angular_std_deg_mean", 0.0))
        if value > float(args.max_normal_angular_std_deg):
            failures.append(
                {
                    "condition_id": "rgb_depth_normal_guard",
                    "metric": "normal_angular_std_deg_mean",
                    "family": row.get("family"),
                    "value": value,
                    "threshold": float(args.max_normal_angular_std_deg),
                }
            )
    return failures


def backend_parameter_summary(feature_dim: int, hidden_dim: int) -> dict[str, Any]:
    backend = PartSpecializedB2SurfaceTokenBackend(int(feature_dim), int(hidden_dim))
    per_family: dict[str, int] = {}
    for family_name, module in backend.heads.items():
        per_family[family_name] = int(sum(param.numel() for param in module.parameters()))
    return {
        "feature_dim": int(feature_dim),
        "hidden_dim": int(hidden_dim),
        "total_parameters": int(sum(param.numel() for param in backend.parameters())),
        "per_family_parameters": per_family,
    }


def head_contract_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family_name, spec in FAMILY_HEAD_SPECS.items():
        out[family_name] = {
            "part_ids": [int(part) for part in spec.part_ids],
            "part_names": [PART_NAMES.get(int(part), str(int(part))) for part in spec.part_ids],
            "hidden_layers": int(spec.hidden_layers),
            "delta_limit_scale": float(spec.delta_limit_scale),
            "normal_residual_scale": float(spec.normal_residual_scale),
            "confidence_bias": float(spec.confidence_bias),
        }
    return out


def build_token_rows(
    token_meta: dict[str, Any],
    token_vertex_counts: np.ndarray,
    projected_meta: dict[str, Any],
    raster_meta: dict[str, Any] | None,
    token_visibility: np.ndarray | None = None,
    token_delta_unit: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    token_part_ids = np.asarray(token_meta["token_part_ids"], dtype=np.int64)
    token_family_ids = np.asarray(token_meta["token_family_ids"], dtype=np.int64)
    projected_support = np.asarray(projected_meta["projected_token_support_count"], dtype=np.int64)
    raster_visible = (
        np.asarray(raster_meta["raster_visible_view_count"], dtype=np.int64)
        if raster_meta is not None and "raster_visible_view_count" in raster_meta
        else np.zeros_like(projected_support)
    )
    raster_target = (
        np.asarray(raster_meta["raster_token_target_fraction"], dtype=np.float32)
        if raster_meta is not None and "raster_token_target_fraction" in raster_meta
        else np.zeros_like(projected_support, dtype=np.float32)
    )
    raster_rgb = (
        np.asarray(raster_meta["raster_token_rgb_residual_mean"], dtype=np.float32)
        if raster_meta is not None and "raster_token_rgb_residual_mean" in raster_meta
        else np.zeros_like(projected_support, dtype=np.float32)
    )
    raster_depth_std = (
        np.asarray(raster_meta["raster_token_depth_std"], dtype=np.float32)
        if raster_meta is not None and "raster_token_depth_std" in raster_meta
        else np.zeros_like(projected_support, dtype=np.float32)
    )
    raster_normal_deg = (
        np.asarray(raster_meta["raster_token_normal_angular_std_deg"], dtype=np.float32)
        if raster_meta is not None and "raster_token_normal_angular_std_deg" in raster_meta
        else np.zeros_like(projected_support, dtype=np.float32)
    )
    rows: list[dict[str, Any]] = []
    token_count = int(token_meta["token_count"])
    for token_id in range(token_count):
        family_id = int(token_family_ids[token_id])
        row = {
            "token_id": int(token_id),
            "part_id": int(token_part_ids[token_id]),
            "part_name": PART_NAMES.get(int(token_part_ids[token_id]), str(int(token_part_ids[token_id]))),
            "family": FAMILY_NAMES[family_id],
            "vertex_count": int(token_vertex_counts[token_id]),
            "projected_support_views": int(projected_support[token_id]),
            "raster_visible_views": int(raster_visible[token_id]),
            "raster_target_fraction": float(raster_target[token_id]),
            "raster_rgb_residual_mean": float(raster_rgb[token_id]),
            "raster_depth_std": float(raster_depth_std[token_id]),
            "raster_normal_angular_std_deg": float(raster_normal_deg[token_id]),
            "visibility_gate": float(token_visibility[token_id]) if token_visibility is not None else None,
            "delta_unit_norm": float(np.linalg.norm(token_delta_unit[token_id])) if token_delta_unit is not None else None,
        }
        rows.append(row)
    return rows


def write_token_table(json_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(json_path, rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Surface Token Backend B2 Research Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is B-line research-only. It is not a teacher, not a candidate, not a strict pass, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "teacher/candidate export = blocked",
        "formal VGGT train/infer/export = untouched",
        "```",
        "",
        "## B2 Contract",
        "",
        "```json",
        json.dumps(summary["contract"], indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Stopping Conditions",
        "",
        "```json",
        json.dumps(summary["stopping_conditions"], indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Stop Result",
        "",
        "```json",
        json.dumps(summary.get("stop_result", {}), indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Render Diagnostics",
        "",
        "```json",
        json.dumps(summary.get("render_diagnostics", []), indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Loss Curve",
        "",
        "```json",
        json.dumps(summary.get("loss_curve", []), indent=2, ensure_ascii=False, default=json_default),
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
    for item in summary.get("outputs", []):
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_summary(
    *,
    output_dir: Path,
    status: str,
    args: argparse.Namespace,
    scene_dir: Path,
    template_payload: Path,
    camera_source: str,
    view_indices: list[int],
    token_meta: dict[str, Any],
    token_vertex_counts: np.ndarray,
    projected_meta: dict[str, Any],
    cuda_info: dict[str, Any],
    nvdiffrast_import_error: str | None,
    output_paths: list[str],
    start_time: float,
) -> int:
    token_table_json = output_dir / "surface_token_b2_token_diagnostics.json"
    token_table_csv = output_dir / "surface_token_b2_token_diagnostics.csv"
    rows = build_token_rows(token_meta, token_vertex_counts, projected_meta, None)
    write_token_table(token_table_json, token_table_csv, rows)
    output_paths.extend([str(token_table_json), str(token_table_csv)])
    summary_path = output_dir / "surface_token_b2_summary.json"
    report_path = output_dir / "surface_token_b2_summary.md"
    elapsed = float(time.perf_counter() - start_time)
    summary = {
        "status": status,
        "contract": {
            "research_only": True,
            "not_b1_hidden_step_tuning": True,
            "not_image_mlp_plus_plus": True,
            "part_specialized_families": list(FAMILY_NAMES),
            "head_contract": head_contract_summary(),
            "backend_parameters_projected_only": backend_parameter_summary(
                int(build_vertex_features(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32), np.zeros((1,), dtype=np.int64)).shape[1])
                + int(projected_meta["projected_feature_dim"])
                + len(FAMILY_NAMES)
                + 1,
                int(args.token_hidden),
            ),
        },
        "stopping_conditions": stopping_condition_specs(args),
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_train_infer_export": "untouched",
            "scene_dir": str(scene_dir),
            "template_payload": str(template_payload),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": int(args.target_size),
            "token_grid": int(args.token_grid),
            "token_meta": {k: v for k, v in token_meta.items() if k not in {"token_part_ids", "token_family_ids", "token_keys"}},
            "projected_meta": {
                k: v
                for k, v in projected_meta.items()
                if k not in {"projected_token_view_support", "projected_token_support_count"}
            },
            "cuda": cuda_info,
            "nvdiffrast_import_error": nvdiffrast_import_error,
            "elapsed_seconds": elapsed,
        },
        "stop_result": {
            "stopped": True,
            "reason": status,
            "details": "B2 cannot aggregate raster depth/normal diagnostics without nvdiffrast CUDA.",
        },
        "render_diagnostics": [],
        "loss_curve": [],
        "decision": (
            "B2 projected RGB/mask token and head-contract diagnostics were written, but raster depth/normal "
            "aggregation is blocked in this environment. No teacher, candidate, strict pass, train, infer, "
            "export, or cloud artifact was created."
        ),
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    write_json(summary_path, summary)
    write_report(report_path, summary)
    return 2


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    scene_dir = args.scene_dir.resolve()
    template_payload = args.template_payload.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)

    mesh = load_connected_mesh(template_payload)
    base_vertices_np = mesh["vertices"].astype(np.float32)
    faces_np = mesh["faces"].astype(np.int32)
    part_ids = mesh["part_ids"].astype(np.int64)
    part_colors_np = mesh["part_colors"].astype(np.float32)
    family_ids_np = family_ids_for_parts(part_ids)
    family_colors_np = family_colors_for_part_ids(part_ids)
    base_normals_np = compute_vertex_normals(base_vertices_np, faces_np)
    limits_np = part_offset_limits(part_ids)
    edges_np = unique_edges(faces_np)
    output_paths: list[str] = []

    token_ids_np, token_meta = quantized_surface_tokens(base_vertices_np, part_ids, int(args.token_grid))
    token_count = int(token_meta["token_count"])
    token_family_ids_np = np.asarray(token_meta["token_family_ids"], dtype=np.int64)
    face_token_ids_np, _face_family_ids_np = precompute_face_labels(faces_np, token_ids_np, family_ids_np)

    cpu_device = torch.device("cpu")
    cpu_payloads = make_view_payloads(views, view_indices, cameras, int(args.target_size), cpu_device)
    base_vertices_cpu = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=cpu_device).contiguous()
    projected_token_features_np, projected_meta = build_projected_rgb_mask_aggregates(
        base_vertices_cpu,
        cpu_payloads,
        token_ids_np,
        token_family_ids_np,
        token_count,
        int(args.target_size),
        int(args.target_size),
        float(args.photometric_mask_threshold),
    )
    geom_features_np = build_vertex_features(base_vertices_np, base_normals_np, part_ids)
    token_geom_np, token_vertex_counts = aggregate_by_token(geom_features_np, token_ids_np, token_count)
    token_vertex_count_feature = (token_vertex_counts[:, None] / max(1.0, float(token_vertex_counts.max()))).astype(np.float32)
    token_family_one_hot_np = one_hot(token_family_ids_np, len(FAMILY_NAMES))

    carrier_path = output_dir / "surface_token_b2_carrier_mesh.ply"
    family_path = output_dir / "surface_token_b2_family_carrier_mesh.ply"
    write_colored_ply(carrier_path, base_vertices_np, faces_np, part_colors_np)
    write_colored_ply(family_path, base_vertices_np, faces_np, family_colors_np)
    output_paths.extend([str(carrier_path), str(family_path)])

    dr, import_error = import_nvdiffrast()
    cuda_info = describe_cuda_device()
    if dr is None:
        return write_blocked_summary(
            output_dir=output_dir,
            status="blocked_no_nvdiffrast",
            args=args,
            scene_dir=scene_dir,
            template_payload=template_payload,
            camera_source=camera_source,
            view_indices=view_indices,
            token_meta=token_meta,
            token_vertex_counts=token_vertex_counts,
            projected_meta=projected_meta,
            cuda_info=cuda_info,
            nvdiffrast_import_error=import_error,
            output_paths=output_paths,
            start_time=start_time,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return write_blocked_summary(
            output_dir=output_dir,
            status="blocked_no_cuda",
            args=args,
            scene_dir=scene_dir,
            template_payload=template_payload,
            camera_source=camera_source,
            view_indices=view_indices,
            token_meta=token_meta,
            token_vertex_counts=token_vertex_counts,
            projected_meta=projected_meta,
            cuda_info=cuda_info,
            nvdiffrast_import_error=import_error,
            output_paths=output_paths,
            start_time=start_time,
        )

    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    height = width = int(args.target_size)
    view_payloads = make_view_payloads(views, view_indices, cameras, height, device)
    base_vertices = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=device).contiguous()
    base_normals = torch.as_tensor(base_normals_np, dtype=torch.float32, device=device).contiguous()
    faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    limits = torch.as_tensor(limits_np, dtype=torch.float32, device=device).view(-1, 1)
    edges = torch.as_tensor(edges_np, dtype=torch.long, device=device).contiguous()
    vertex_token_ids = torch.as_tensor(token_ids_np, dtype=torch.long, device=device).contiguous()
    token_family_ids = torch.as_tensor(token_family_ids_np, dtype=torch.long, device=device).contiguous()
    token_mean_rgb = torch.as_tensor(projected_token_features_np[:, :3], dtype=torch.float32, device=device).contiguous()
    vertex_mean_rgb = scatter_token_to_vertex(token_mean_rgb, vertex_token_ids).contiguous()

    raster_features_np, initial_raster_meta, raster_output_paths = build_raster_rgb_mask_depth_normal_aggregates(
        dr,
        ctx,
        base_vertices,
        faces_t,
        faces_np,
        base_normals,
        vertex_mean_rgb,
        view_payloads,
        face_token_ids_np,
        token_family_ids_np,
        token_count,
        height,
        width,
        float(args.z_sign),
        output_dir,
        "initial",
    )
    output_paths.extend(raster_output_paths)

    token_features_np = np.concatenate(
        [
            token_geom_np,
            projected_token_features_np,
            raster_features_np,
            token_family_one_hot_np,
            token_vertex_count_feature,
        ],
        axis=1,
    ).astype(np.float32)
    token_features = torch.as_tensor(token_features_np, dtype=torch.float32, device=device).contiguous()
    backend = PartSpecializedB2SurfaceTokenBackend(token_features.shape[1], int(args.token_hidden)).to(device)
    optimizer = torch.optim.Adam(backend.parameters(), lr=float(args.lr))

    stop_failures = evaluate_family_visibility(initial_raster_meta, args) + evaluate_global_metric_guards(initial_raster_meta, args)
    stop_result: dict[str, Any] = {"stopped": False, "reason": None, "details": [], "initial_failures": stop_failures}
    loss_curve: list[dict[str, Any]] = []
    optimized_vertices = base_vertices
    optimized_normals = base_normals
    token_visibility_np: np.ndarray | None = None
    token_delta_np: np.ndarray | None = None

    should_skip_optimization = bool(args.diagnostics_only)
    if should_skip_optimization:
        stop_result.update({"stopped": True, "reason": "diagnostics_only", "details": ["--diagnostics-only was set"]})
    elif stop_failures and not bool(args.continue_on_visibility_fail):
        should_skip_optimization = True
        stop_result.update(
            {
                "stopped": True,
                "reason": "initial_guard_failed",
                "details": stop_failures,
            }
        )

    if not should_skip_optimization:
        for step in range(int(args.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            delta, token_delta_unit, vertex_normals, token_visibility, _token_visibility_logits = backend(
                token_features,
                token_family_ids,
                vertex_token_ids,
                limits,
                base_normals,
            )
            vertices = base_vertices + delta
            token_conf_vertex = scatter_token_to_vertex(token_visibility[:, None], vertex_token_ids).view(-1)
            render_rgb = vertex_mean_rgb * token_conf_vertex[:, None] + family_colors_for_part_ids(part_ids).astype(np.float32)
            render_rgb_t = torch.as_tensor(render_rgb, dtype=torch.float32, device=device).contiguous()
            render_rgb_t = render_rgb_t.clamp(0.0, 1.0)

            mask_bce_total = torch.zeros((), dtype=torch.float32, device=device)
            recall_total = torch.zeros((), dtype=torch.float32, device=device)
            overfill_total = torch.zeros((), dtype=torch.float32, device=device)
            rgb_total = torch.zeros((), dtype=torch.float32, device=device)
            depth_normal_total = torch.zeros((), dtype=torch.float32, device=device)
            normal_energy_total = torch.zeros((), dtype=torch.float32, device=device)
            metrics_rows: list[dict[str, Any]] = []
            for payload in view_payloads:
                render = render_nvdiffrast_view(
                    dr,
                    ctx,
                    vertices,
                    faces_t,
                    vertex_normals,
                    render_rgb_t,
                    payload["world_to_cam"],
                    payload["intrinsic"],
                    height,
                    width,
                    z_sign=float(args.z_sign),
                )
                mask = render["mask"].clamp(1e-4, 1.0 - 1e-4)
                target = payload["target_mask"]
                mask_bce_total = mask_bce_total + F.binary_cross_entropy(mask, target)
                recall_total = recall_total + (target * (1.0 - mask)).sum() / target.sum().clamp_min(1.0)
                overfill_total = overfill_total + ((1.0 - target) * mask).sum() / mask.sum().clamp_min(1.0)
                rgb_total = rgb_total + rendered_rgb_residual(render["color"], payload["rgb_tensor"], render["mask"])
                depth_normal_total = depth_normal_total + rendered_depth_normal_consistency(render["depth"], render["normal"], render["mask"])
                visible = render["mask"] > 0.5
                if bool(visible.any()):
                    normal_energy_total = normal_energy_total + (1.0 - render["normal"][..., 2][visible].abs()).mean()
                if step == 0 or step == int(args.max_steps):
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
            rgb_loss = rgb_total / denom
            depth_normal_loss = depth_normal_total / denom
            normal_energy = normal_energy_total / denom
            offset_reg = (token_delta_unit.pow(2).sum(dim=1) * token_visibility.detach()).mean()
            edge_reg = (delta[edges[:, 0]] - delta[edges[:, 1]]).pow(2).mean() / (limits.mean().clamp_min(1e-6) ** 2)
            loss = (
                float(args.mask_bce_weight) * mask_bce
                + float(args.target_recall_weight) * recall_loss
                + float(args.overfill_weight) * overfill_loss
                + float(args.rgb_residual_weight) * rgb_loss
                + float(args.depth_normal_weight) * depth_normal_loss
                + float(args.normal_render_weight) * normal_energy
                + float(args.token_offset_reg_weight) * offset_reg
                + float(args.edge_reg_weight) * edge_reg
            )

            max_delta = float(torch.linalg.norm(delta, dim=1).detach().max().cpu()) if delta.numel() else 0.0
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "mask_bce": float(mask_bce.detach().cpu()),
                "target_recall_loss": float(recall_loss.detach().cpu()),
                "overfill_loss": float(overfill_loss.detach().cpu()),
                "rgb_residual": float(rgb_loss.detach().cpu()),
                "depth_normal_consistency": float(depth_normal_loss.detach().cpu()),
                "normal_render_energy": float(normal_energy.detach().cpu()),
                "token_offset_reg": float(offset_reg.detach().cpu()),
                "edge_reg": float(edge_reg.detach().cpu()),
                "max_vertex_delta": max_delta,
                "token_visibility_mean": float(token_visibility.detach().mean().cpu()),
                "token_visibility_min": float(token_visibility.detach().min().cpu()),
                "token_visibility_max": float(token_visibility.detach().max().cpu()),
                "metrics": metrics_rows,
            }
            loss_curve.append(row)

            if not math.isfinite(float(row["loss"])):
                stop_result.update({"stopped": True, "reason": "nonfinite_loss", "details": [row]})
                break
            if max_delta > float(args.max_vertex_delta):
                stop_result.update({"stopped": True, "reason": "delta_guard", "details": [row]})
                break
            if step >= int(args.early_stop_window):
                prev = loss_curve[-1 - int(args.early_stop_window)]
                improvement = float(prev["loss"]) - float(row["loss"])
                if improvement < float(args.min_loss_improvement):
                    stop_result.update(
                        {
                            "stopped": True,
                            "reason": "plateau_guard",
                            "details": [{"step": int(step), "window_improvement": improvement}],
                        }
                    )
                    break
            if step < int(args.max_steps):
                loss.backward()
                optimizer.step()
            else:
                stop_result.update({"stopped": True, "reason": "max_steps_reached", "details": [row]})

        with torch.no_grad():
            delta, token_delta_unit, optimized_normals, token_visibility, _ = backend(
                token_features,
                token_family_ids,
                vertex_token_ids,
                limits,
                base_normals,
            )
            optimized_vertices = base_vertices + delta
            token_visibility_np = token_visibility.detach().cpu().numpy().astype(np.float32)
            token_delta_np = token_delta_unit.detach().cpu().numpy().astype(np.float32)

    if token_visibility_np is None or token_delta_np is None:
        with torch.no_grad():
            delta, token_delta_unit, optimized_normals, token_visibility, _ = backend(
                token_features,
                token_family_ids,
                vertex_token_ids,
                limits,
                base_normals,
            )
            optimized_vertices = base_vertices + delta
            token_visibility_np = token_visibility.detach().cpu().numpy().astype(np.float32)
            token_delta_np = token_delta_unit.detach().cpu().numpy().astype(np.float32)

    optimized_vertices_np = optimized_vertices.detach().cpu().numpy().astype(np.float32)
    optimized_normals_np = optimized_normals.detach().cpu().numpy().astype(np.float32)
    token_conf_vertex_np = token_visibility_np[token_ids_np].astype(np.float32)
    final_rgb_np = projected_token_features_np[token_ids_np, :3].astype(np.float32)
    optimized_mesh_path = output_dir / "surface_token_b2_research_mesh.ply"
    normals_path = output_dir / "surface_token_b2_normals.ply"
    confidence_path = output_dir / "surface_token_b2_visibility_gate_mesh.ply"
    rgb_path = output_dir / "surface_token_b2_projected_rgb_mesh.ply"
    write_colored_ply(optimized_mesh_path, optimized_vertices_np, faces_np, part_colors_np)
    write_colored_ply(normals_path, optimized_vertices_np, faces_np, (optimized_normals_np + 1.0) * 0.5)
    write_colored_ply(confidence_path, optimized_vertices_np, faces_np, np.repeat(token_conf_vertex_np[:, None], 3, axis=1))
    write_colored_ply(rgb_path, optimized_vertices_np, faces_np, final_rgb_np)
    output_paths.extend([str(optimized_mesh_path), str(normals_path), str(confidence_path), str(rgb_path)])

    final_vertex_rgb = torch.as_tensor(final_rgb_np, dtype=torch.float32, device=device).contiguous()
    final_raster_features_np, final_raster_meta, final_output_paths = build_raster_rgb_mask_depth_normal_aggregates(
        dr,
        ctx,
        optimized_vertices,
        faces_t,
        faces_np,
        optimized_normals,
        final_vertex_rgb,
        view_payloads,
        face_token_ids_np,
        token_family_ids_np,
        token_count,
        height,
        width,
        float(args.z_sign),
        output_dir,
        "final",
    )
    _ = final_raster_features_np
    output_paths.extend(final_output_paths)
    final_failures = evaluate_family_visibility(final_raster_meta, args) + evaluate_global_metric_guards(final_raster_meta, args)
    if final_failures and stop_result.get("reason") in {None, "max_steps_reached"}:
        stop_result.update({"stopped": True, "reason": "final_guard_failed", "details": final_failures})

    token_table_json = output_dir / "surface_token_b2_token_diagnostics.json"
    token_table_csv = output_dir / "surface_token_b2_token_diagnostics.csv"
    token_rows = build_token_rows(
        token_meta,
        token_vertex_counts,
        projected_meta,
        final_raster_meta,
        token_visibility=token_visibility_np,
        token_delta_unit=token_delta_np,
    )
    write_token_table(token_table_json, token_table_csv, token_rows)
    output_paths.extend([str(token_table_json), str(token_table_csv)])

    initial_metrics = loss_curve[0]["metrics"] if loss_curve else initial_raster_meta.get("view_diagnostics", [])
    final_metrics = loss_curve[-1]["metrics"] if loss_curve and loss_curve[-1]["metrics"] else final_raster_meta.get("view_diagnostics", [])
    avg_initial_iou = float(np.mean([float(row.get("iou", 0.0)) for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([float(row.get("iou", 0.0)) for row in final_metrics])) if final_metrics else 0.0
    elapsed = float(time.perf_counter() - start_time)
    summary_path = output_dir / "surface_token_b2_summary.json"
    report_path = output_dir / "surface_token_b2_summary.md"
    summary = {
        "status": "surface_token_b2_research_preflight_complete",
        "contract": {
            "research_only": True,
            "not_b1_hidden_step_tuning": True,
            "not_image_mlp_plus_plus": True,
            "part_specialized_families": list(FAMILY_NAMES),
            "head_contract": head_contract_summary(),
            "backend_parameters": backend_parameter_summary(int(token_features.shape[1]), int(args.token_hidden)),
            "diagnostic_channels": ["projected_rgb", "projected_mask", "raster_mask", "raster_depth", "raster_normal", "raster_rgb_residual"],
        },
        "stopping_conditions": stopping_condition_specs(args),
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_train_infer_export": "untouched",
            "scene_dir": str(scene_dir),
            "template_payload": str(template_payload),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": int(args.target_size),
            "max_steps": int(args.max_steps),
            "diagnostics_only": bool(args.diagnostics_only),
            "token_grid": int(args.token_grid),
            "token_hidden": int(args.token_hidden),
            "token_meta": {k: v for k, v in token_meta.items() if k not in {"token_part_ids", "token_family_ids", "token_keys"}},
            "projected_meta": {
                k: v
                for k, v in projected_meta.items()
                if k not in {"projected_token_view_support", "projected_token_support_count"}
            },
            "initial_raster_meta": {
                k: v
                for k, v in initial_raster_meta.items()
                if not str(k).startswith("raster_token_")
            },
            "final_raster_meta": {
                k: v
                for k, v in final_raster_meta.items()
                if not str(k).startswith("raster_token_")
            },
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "max_vertex_delta": float(np.linalg.norm(optimized_vertices_np - base_vertices_np, axis=1).max()),
            "mean_vertex_delta": float(np.linalg.norm(optimized_vertices_np - base_vertices_np, axis=1).mean()),
            "token_visibility_mean": float(token_visibility_np.mean()),
            "token_visibility_min": float(token_visibility_np.min()),
            "token_visibility_max": float(token_visibility_np.max()),
            "cuda": cuda_info,
            "elapsed_seconds": elapsed,
        },
        "stop_result": stop_result,
        "render_diagnostics": final_raster_meta.get("view_diagnostics", []),
        "loss_curve": loss_curve,
        "decision": (
            "B2 is a research-only backend scaffold with explicit face/hair/hand/body token heads and "
            "visibility-aware RGB/mask/depth/normal diagnostics. Its stop reason is reported separately. "
            "No output here is a teacher, candidate, strict pass, training run, inference run, export, or cloud unblock signal."
        ),
        "outputs": output_paths + [str(summary_path), str(report_path)],
    }
    write_json(summary_path, summary)
    write_report(report_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
