from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rasterize_shared_surfel_predictions import (  # noqa: E402
    ROI_NAMES,
    build_roi_masks,
    camera_to_world,
    channel_index,
    collect_observations,
    json_ready,
    load_npz,
    parse_sources,
    quantize_canonical,
    recompute_normals,
    unproject_pixel_depth,
    world_to_camera,
)
from render_open3d_pointcloud import _save_open3d_renders, load_rgb_stack  # noqa: E402


NEIGHBOR_OFFSETS_6 = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "r47 diagnostic: build an optimized visibility-aware shared human surface from VGGT observations. "
            "SMPL-X canonical maps are used only for correspondence/body-part keys, never as face/hair geometry truth."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--canonical-bin-size", type=float, default=0.012)
    parser.add_argument("--min-surfel-observations", type=int, default=4)
    parser.add_argument("--min-surfel-views", type=int, default=2)
    parser.add_argument("--critical-min-views", type=int, default=3)
    parser.add_argument("--max-surfel-spread", type=float, default=0.055)
    parser.add_argument("--smooth-lambda", type=float, default=0.35)
    parser.add_argument("--critical-smooth-scale", type=float, default=0.35)
    parser.add_argument("--smooth-iterations", type=int, default=18)
    parser.add_argument("--max-raster-distance", type=float, default=0.070)
    parser.add_argument("--neighbor-fill-radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--visible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-dilate", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mode_int(values: np.ndarray) -> int:
    if values.size == 0:
        return -1
    vals, counts = np.unique(values.astype(np.int32), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def body_part_ids(
    *,
    priors: dict[str, np.ndarray],
    channel_names: list[str],
    support_base: np.ndarray,
    canonical_q: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    body_channels = [
        idx for idx, name in enumerate(channel_names) if str(name).startswith("smplx_body_part_emb_")
    ]
    if not body_channels:
        return {}
    body = np.asarray(priors["prior_maps"][:, body_channels], dtype=np.float32).transpose(0, 2, 3, 1)
    body_id = np.argmax(body, axis=-1).astype(np.int16)
    flat_keys = canonical_q.reshape(-1, 3)
    flat_part = body_id.reshape(-1)
    selected = np.flatnonzero(support_base.reshape(-1))
    if selected.size == 0:
        return {}
    keys = flat_keys[selected]
    parts = flat_part[selected]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    out: dict[tuple[int, int, int], int] = {}
    for idx, key in enumerate(unique):
        out[tuple(int(v) for v in key.tolist())] = mode_int(parts[inverse == idx])
    return out


def robust_group_center(points: np.ndarray, conf: np.ndarray) -> np.ndarray:
    if points.shape[0] == 1:
        return points[0].astype(np.float32)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median[None, :], axis=1)
    cutoff = np.percentile(distances, 75.0)
    keep = distances <= max(float(cutoff), 1e-6)
    if int(keep.sum()) < 2:
        return median.astype(np.float32)
    weights = np.maximum(conf[keep].astype(np.float32), 1e-6)
    return np.average(points[keep].astype(np.float32), axis=0, weights=weights).astype(np.float32)


def aggregate_surface(
    observations: dict[str, np.ndarray],
    *,
    key_to_body_part: dict[tuple[int, int, int], int],
    min_observations: int,
    min_views: int,
    critical_min_views: int,
    max_spread: float,
) -> tuple[dict[tuple[int, int, int], int], dict[str, np.ndarray], dict[str, Any]]:
    keys = observations["keys"]
    points = observations["points"]
    conf = observations["conf"]
    views = observations["views"]
    roi_flags = observations["roi_flags"]
    sources = observations["sources"]
    rejected = {"too_few_observations": 0, "too_few_views": 0, "spread": 0, "critical_low_views": 0}
    if keys.shape[0] == 0:
        empty = {
            "canonical_key": np.zeros((0, 3), dtype=np.int32),
            "position_data": np.zeros((0, 3), dtype=np.float32),
            "position_optimized": np.zeros((0, 3), dtype=np.float32),
            "confidence": np.zeros((0,), dtype=np.float32),
            "view_support": np.zeros((0,), dtype=np.int16),
            "observation_count": np.zeros((0,), dtype=np.int32),
            "spread_p75": np.zeros((0,), dtype=np.float32),
            "source_mask": np.zeros((0,), dtype=np.uint8),
            "roi_counts": np.zeros((0, len(ROI_NAMES)), dtype=np.int32),
            "body_part_id": np.zeros((0,), dtype=np.int16),
            "raster_eligible": np.zeros((0,), dtype=bool),
        }
        return {}, empty, {"accepted_surfels": 0, "rejected": rejected}

    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], order.size]

    key_to_index: dict[tuple[int, int, int], int] = {}
    out_keys: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    confidences: list[float] = []
    view_counts: list[int] = []
    obs_counts: list[int] = []
    spreads: list[float] = []
    source_masks: list[int] = []
    roi_counts: list[np.ndarray] = []
    body_parts: list[int] = []
    eligible: list[bool] = []

    for start, end in zip(starts, ends):
        group = order[start:end]
        if group.size < int(min_observations):
            rejected["too_few_observations"] += 1
            continue
        group_views = np.unique(views[group])
        if group_views.size < int(min_views):
            rejected["too_few_views"] += 1
            continue
        group_points = points[group].astype(np.float32)
        center = robust_group_center(group_points, conf[group])
        distances = np.linalg.norm(group_points - center[None, :], axis=1)
        spread = float(np.percentile(distances, 75.0)) if distances.size else float("inf")
        if not np.isfinite(spread) or spread > float(max_spread):
            rejected["spread"] += 1
            continue
        roi_count = roi_flags[group].astype(np.int32).sum(axis=0)
        critical = bool(roi_count[ROI_NAMES.index("face")] > 0 or roi_count[ROI_NAMES.index("hands")] > 0)
        surfel_eligible = bool(group_views.size >= (int(critical_min_views) if critical else int(min_views)))
        if critical and not surfel_eligible:
            rejected["critical_low_views"] += 1
        key = unique[int(sorted_inverse[start])].astype(np.int32)
        index = len(centers)
        key_tuple = tuple(int(v) for v in key.tolist())
        key_to_index[key_tuple] = index
        out_keys.append(key)
        centers.append(center)
        weights = np.maximum(conf[group].astype(np.float32), 1e-6)
        confidences.append(float(np.average(conf[group].astype(np.float32), weights=weights)))
        view_counts.append(int(group_views.size))
        obs_counts.append(int(group.size))
        spreads.append(spread)
        source_masks.append(int(np.bitwise_or.reduce(sources[group])))
        roi_counts.append(roi_count)
        body_parts.append(int(key_to_body_part.get(key_tuple, -1)))
        eligible.append(surfel_eligible)

    arrays = {
        "canonical_key": np.stack(out_keys, axis=0) if out_keys else np.zeros((0, 3), dtype=np.int32),
        "position_data": np.stack(centers, axis=0) if centers else np.zeros((0, 3), dtype=np.float32),
        "position_optimized": np.stack(centers, axis=0) if centers else np.zeros((0, 3), dtype=np.float32),
        "confidence": np.asarray(confidences, dtype=np.float32),
        "view_support": np.asarray(view_counts, dtype=np.int16),
        "observation_count": np.asarray(obs_counts, dtype=np.int32),
        "spread_p75": np.asarray(spreads, dtype=np.float32),
        "source_mask": np.asarray(source_masks, dtype=np.uint8),
        "roi_counts": np.stack(roi_counts, axis=0) if roi_counts else np.zeros((0, len(ROI_NAMES)), dtype=np.int32),
        "body_part_id": np.asarray(body_parts, dtype=np.int16),
        "raster_eligible": np.asarray(eligible, dtype=bool),
    }
    roi_surfel_counts = {
        roi: int((arrays["roi_counts"][:, idx] > 0).sum()) if arrays["roi_counts"].size else 0
        for idx, roi in enumerate(ROI_NAMES)
    }
    summary = {
        "unique_canonical_bins": int(unique.shape[0]),
        "accepted_surfels": int(arrays["position_data"].shape[0]),
        "raster_eligible_surfels": int(arrays["raster_eligible"].sum()),
        "rejected": rejected,
        "roi_surfel_counts": roi_surfel_counts,
        "observation_count_percentiles": [float(v) for v in np.percentile(obs_counts, [0, 25, 50, 75, 95, 100])]
        if obs_counts
        else [],
        "view_support_percentiles": [float(v) for v in np.percentile(view_counts, [0, 25, 50, 75, 95, 100])]
        if view_counts
        else [],
        "spread_p75_percentiles": [float(v) for v in np.percentile(spreads, [0, 25, 50, 75, 95, 100])]
        if spreads
        else [],
    }
    return key_to_index, arrays, summary


def build_adjacency(keys: np.ndarray, key_to_index: dict[tuple[int, int, int], int]) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(keys.shape[0])]
    for idx, key in enumerate(keys):
        key_tuple = tuple(int(v) for v in key.tolist())
        neighbors: list[int] = []
        for offset in NEIGHBOR_OFFSETS_6:
            neighbor_key = (key_tuple[0] + offset[0], key_tuple[1] + offset[1], key_tuple[2] + offset[2])
            neighbor_idx = key_to_index.get(neighbor_key)
            if neighbor_idx is not None:
                neighbors.append(int(neighbor_idx))
        out[idx] = neighbors
    return out


def optimize_positions(
    surfels: dict[str, np.ndarray],
    key_to_index: dict[tuple[int, int, int], int],
    *,
    smooth_lambda: float,
    critical_smooth_scale: float,
    iterations: int,
) -> dict[str, Any]:
    centers = np.asarray(surfels["position_data"], dtype=np.float32)
    if centers.shape[0] == 0 or int(iterations) <= 0 or float(smooth_lambda) <= 0:
        surfels["position_optimized"] = centers.copy()
        return {"enabled": False, "iterations": int(iterations), "mean_shift": 0.0, "p95_shift": 0.0}
    keys = np.asarray(surfels["canonical_key"], dtype=np.int32)
    adjacency = build_adjacency(keys, key_to_index)
    confidence = np.maximum(np.asarray(surfels["confidence"], dtype=np.float32), 1e-6)
    view_support = np.maximum(np.asarray(surfels["view_support"], dtype=np.float32), 1.0)
    obs_count = np.maximum(np.asarray(surfels["observation_count"], dtype=np.float32), 1.0)
    roi_counts = np.asarray(surfels["roi_counts"], dtype=np.int32)
    critical = (roi_counts[:, ROI_NAMES.index("face")] > 0) | (roi_counts[:, ROI_NAMES.index("hands")] > 0)
    data_weight = np.sqrt(confidence) * np.sqrt(view_support) * np.log1p(obs_count)
    data_weight = data_weight / max(float(np.median(data_weight)), 1e-6)
    data_weight = np.clip(data_weight, 0.25, 8.0).astype(np.float32)

    x = centers.copy()
    for _ in range(int(iterations)):
        new_x = x.copy()
        for idx, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            neigh = x[np.asarray(neighbors, dtype=np.int32)]
            lam = float(smooth_lambda) * (float(critical_smooth_scale) if bool(critical[idx]) else 1.0)
            deg = float(len(neighbors))
            numerator = data_weight[idx] * centers[idx] + lam * deg * np.mean(neigh, axis=0)
            denominator = data_weight[idx] + lam * deg
            new_x[idx] = numerator / max(denominator, 1e-6)
        x = new_x.astype(np.float32)
    shift = np.linalg.norm(x - centers, axis=1)
    surfels["position_optimized"] = x.astype(np.float32)
    return {
        "enabled": True,
        "iterations": int(iterations),
        "smooth_lambda": float(smooth_lambda),
        "critical_smooth_scale": float(critical_smooth_scale),
        "surfel_count": int(centers.shape[0]),
        "adjacency_edges_directed": int(sum(len(n) for n in adjacency)),
        "isolated_surfels": int(sum(1 for n in adjacency if not n)),
        "mean_shift": float(np.mean(shift)) if shift.size else 0.0,
        "p50_shift": float(np.percentile(shift, 50)) if shift.size else 0.0,
        "p95_shift": float(np.percentile(shift, 95)) if shift.size else 0.0,
        "max_shift": float(np.max(shift)) if shift.size else 0.0,
    }


def neighbor_lookup(
    key: tuple[int, int, int],
    key_to_index: dict[tuple[int, int, int], int],
    radius: int,
) -> int | None:
    exact = key_to_index.get(key)
    if exact is not None or int(radius) <= 0:
        return exact
    best_idx: int | None = None
    best_dist = 10**9
    r = int(radius)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = abs(dx) + abs(dy) + abs(dz)
                if dist == 0 or dist > r or dist >= best_dist:
                    continue
                idx = key_to_index.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if idx is not None:
                    best_idx = int(idx)
                    best_dist = dist
    return best_idx


def rasterize_optimized_surface(
    *,
    predictions: dict[str, np.ndarray],
    canonical_q: np.ndarray,
    support_base: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    key_to_index: dict[tuple[int, int, int], int],
    surfels: dict[str, np.ndarray],
    max_distance: float,
    neighbor_fill_radius: int,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    depth = np.asarray(predictions["depth"], dtype=np.float32)
    extrinsics = np.asarray(predictions["extrinsic"], dtype=np.float32)
    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float32)
    fused_world = world_points.copy()
    fused_depth = depth.copy()
    changed = np.zeros(world_points.shape[:3], dtype=bool)
    support_map = np.zeros(world_points.shape[:3], dtype=np.uint8)
    obs_count_map = np.zeros(world_points.shape[:3], dtype=np.uint16)
    distance_map = np.full(world_points.shape[:3], np.nan, dtype=np.float32)
    reason_counts = {
        "no_surfel": 0,
        "ineligible_low_support": 0,
        "distance_rejected": 0,
        "applied_exact": 0,
        "applied_neighbor": 0,
    }

    positions = np.asarray(surfels["position_optimized"], dtype=np.float32)
    view_support = np.asarray(surfels["view_support"], dtype=np.int16)
    obs_count = np.asarray(surfels["observation_count"], dtype=np.int32)
    eligible = np.asarray(surfels["raster_eligible"], dtype=bool)
    view_count, height, width = world_points.shape[:3]
    flat_q = canonical_q.reshape(-1, 3)
    flat_support = support_base.reshape(-1)
    flat_current = world_points.reshape(-1, 3)
    flat_fused = fused_world.reshape(-1, 3)
    flat_depth = fused_depth.reshape(-1)
    flat_changed = changed.reshape(-1)
    flat_support_map = support_map.reshape(-1)
    flat_obs_map = obs_count_map.reshape(-1)
    flat_distance = distance_map.reshape(-1)
    flat_view = np.broadcast_to(np.arange(view_count, dtype=np.int32)[:, None, None], (view_count, height, width)).reshape(-1)

    for flat_idx in np.flatnonzero(flat_support):
        key = tuple(int(v) for v in flat_q[flat_idx].tolist())
        exact_idx = key_to_index.get(key)
        surfel_idx = exact_idx if exact_idx is not None else neighbor_lookup(key, key_to_index, int(neighbor_fill_radius))
        if surfel_idx is None:
            reason_counts["no_surfel"] += 1
            continue
        if not bool(eligible[surfel_idx]):
            reason_counts["ineligible_low_support"] += 1
            continue
        target = positions[surfel_idx]
        current = flat_current[flat_idx]
        distance = float(np.linalg.norm(current - target))
        flat_support_map[flat_idx] = np.uint8(min(int(view_support[surfel_idx]), 255))
        flat_obs_map[flat_idx] = np.uint16(min(int(obs_count[surfel_idx]), 65535))
        flat_distance[flat_idx] = distance
        if not np.isfinite(distance) or distance > float(max_distance):
            reason_counts["distance_rejected"] += 1
            continue
        view_idx = int(flat_view[flat_idx])
        target_cam = world_to_camera(target, extrinsics[view_idx])
        current_depth = float(flat_depth[flat_idx])
        target_depth = float(target_cam[2])
        if not np.isfinite(target_depth) or target_depth <= 1e-6 or not np.isfinite(current_depth):
            reason_counts["distance_rejected"] += 1
            continue
        fused_depth_value = current_depth + float(alpha) * (target_depth - current_depth)
        if not np.isfinite(fused_depth_value) or fused_depth_value <= 1e-6:
            reason_counts["distance_rejected"] += 1
            continue
        flat_fused[flat_idx] = unproject_pixel_depth(
            view_idx=view_idx,
            flat_idx=int(flat_idx),
            depth_value=fused_depth_value,
            height=height,
            width=width,
            intrinsic=intrinsics[view_idx],
            extrinsic=extrinsics[view_idx],
        )
        flat_depth[flat_idx] = fused_depth_value
        flat_changed[flat_idx] = True
        if exact_idx is None:
            reason_counts["applied_neighbor"] += 1
        else:
            reason_counts["applied_exact"] += 1

    roi_summary: dict[str, Any] = {}
    for roi in ROI_NAMES:
        mask = support_base & roi_masks[roi]
        changed_roi = changed & roi_masks[roi]
        support_values = support_map[mask & (support_map > 0)]
        obs_values = obs_count_map[mask & (obs_count_map > 0)]
        dist_values = distance_map[mask & np.isfinite(distance_map)]
        roi_summary[roi] = {
            "apply_candidates": int(mask.sum()),
            "changed_pixels": int(changed_roi.sum()),
            "changed_fraction": float(changed_roi.sum() / max(int(mask.sum()), 1)),
            "view_support_percentiles": [float(v) for v in np.percentile(support_values, [0, 25, 50, 75, 95, 100])]
            if support_values.size
            else [],
            "observation_count_percentiles": [float(v) for v in np.percentile(obs_values, [0, 25, 50, 75, 95, 100])]
            if obs_values.size
            else [],
            "raster_distance_percentiles": [float(v) for v in np.percentile(dist_values, [0, 25, 50, 75, 95, 100])]
            if dist_values.size
            else [],
        }

    return {
        "world_points": fused_world,
        "depth": fused_depth,
        "r47_changed_mask": changed.astype(np.uint8),
        "r47_view_support": support_map,
        "r47_observation_count": obs_count_map,
        "r47_raster_distance": distance_map,
    }, {
        "apply_candidates": int(flat_support.sum()),
        "changed_pixels_total": int(changed.sum()),
        "changed_pixels_per_view": [int(changed[idx].sum()) for idx in range(changed.shape[0])],
        "reason_counts": reason_counts,
        "neighbor_fill_radius": int(neighbor_fill_radius),
        "alpha": float(alpha),
        "roi_rasterization": roi_summary,
    }


def write_support_heatmaps(output_dir: Path, support: np.ndarray, changed: np.ndarray) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for view_idx in range(support.shape[0]):
        support_img = np.asarray(support[view_idx], dtype=np.float32)
        support_img = support_img / max(float(support_img.max()), 1.0)
        changed_img = np.asarray(changed[view_idx], dtype=np.float32)
        rgb = np.zeros((*support_img.shape, 3), dtype=np.uint8)
        rgb[..., 1] = np.clip(support_img * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 0] = np.clip(changed_img * 255.0, 0, 255).astype(np.uint8)
        path = output_dir / f"view{view_idx:02d}_support_green_changed_red.png"
        Image.fromarray(rgb).save(path)
        saved.append(str(path.resolve()))
    return saved


def write_surface_debug(
    *,
    output_dir: Path,
    surfels: dict[str, np.ndarray],
    point_size: float,
) -> dict[str, Any]:
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(surfels["position_optimized"], dtype=np.float32)
    if points.shape[0] == 0:
        return {"status": "empty", "point_count": 0}
    view_support = np.asarray(surfels["view_support"], dtype=np.float32)
    support_norm = view_support / max(float(view_support.max()), 1.0)
    colors = np.zeros_like(points, dtype=np.float32)
    colors[:, 0] = 1.0 - support_norm
    colors[:, 1] = support_norm
    colors[:, 2] = 0.25
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    ply_path = output_dir / "optimized_shared_surface_view_support.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)
    screenshots = _save_open3d_renders(
        points=points,
        colors=np.clip(colors * 255.0, 0, 255).astype(np.uint8),
        output_dir=output_dir / "open3d",
        roi="r47_shared_surface",
        width=1400,
        height=1100,
        point_size=float(point_size),
        interactive=False,
    )
    roi_counts = np.asarray(surfels["roi_counts"], dtype=np.int32)
    roi_plys: dict[str, str] = {}
    for roi in ("face", "head", "hands"):
        idx = ROI_NAMES.index(roi)
        mask = roi_counts[:, idx] > 0 if roi_counts.size else np.zeros((points.shape[0],), dtype=bool)
        if not np.any(mask):
            continue
        pcd_roi = o3d.geometry.PointCloud()
        pcd_roi.points = o3d.utility.Vector3dVector(points[mask].astype(np.float64))
        pcd_roi.colors = o3d.utility.Vector3dVector(colors[mask].astype(np.float64))
        roi_path = output_dir / f"optimized_shared_surface_{roi}.ply"
        o3d.io.write_point_cloud(str(roi_path), pcd_roi, write_ascii=False, compressed=False)
        roi_plys[roi] = str(roi_path.resolve())
    return {
        "status": "rendered",
        "point_count": int(points.shape[0]),
        "ply": str(ply_path.resolve()),
        "screenshots": screenshots,
        "roi_plys": roi_plys,
    }


def failure_analysis(surface_summary: dict[str, Any], opt_summary: dict[str, Any], raster_summary: dict[str, Any]) -> dict[str, Any]:
    roi_surfel_counts = surface_summary.get("roi_surfel_counts", {})
    roi_raster = raster_summary.get("roi_rasterization", {})
    changed_face = float(roi_raster.get("face", {}).get("changed_fraction", 0.0))
    changed_hands = float(roi_raster.get("hands", {}).get("changed_fraction", 0.0))
    view_support = surface_summary.get("view_support_percentiles", [])
    median_view_support = float(view_support[2]) if len(view_support) >= 3 else 0.0
    risks = {
        "accepted_surface_tiny": int(surface_summary.get("accepted_surfels", 0)) < 2500,
        "face_surface_sparse": int(roi_surfel_counts.get("face", 0) or 0) < 2500,
        "hands_surface_sparse": int(roi_surfel_counts.get("hands", 0) or 0) < 800,
        "median_view_support_low": median_view_support < 3.0,
        "face_raster_change_low": changed_face < 0.25,
        "hands_raster_change_low": changed_hands < 0.20,
        "optimization_shift_tiny": float(opt_summary.get("p95_shift", 0.0)) < 1e-4,
    }
    return {
        "truthful_status": "diagnostic_only_not_a_pass",
        "risk_flags": risks,
        "interpretation": [
            "r47 optimizes a shared surface graph, but it still uses only VGGT-observed geometry.",
            "SMPL-X canonical/body-part maps are correspondence and topology hints, not face/hair/clothing truth.",
            "Low face/hands support or shell-like Open3D output means this route must be frozen, not tuned by thresholds.",
        ],
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_sources(args.sources)
    predictions = load_npz(args.predictions)
    priors = load_npz(args.scene_dir / "prior_maps.npz")
    channel_names = [str(value) for value in priors["prior_channels"].tolist()]
    canonical_indices = [
        channel_index(channel_names, "smplx_canonical_x"),
        channel_index(channel_names, "smplx_canonical_y"),
        channel_index(channel_names, "smplx_canonical_z"),
    ]
    visible_index = channel_names.index("smplx_visible_mask") if "smplx_visible_mask" in channel_names else None

    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    view_count, height, width, _ = world_points.shape
    rgb = load_rgb_stack(args.scene_dir / "images", target_size=height)
    roi_masks = build_roi_masks(args.scene_dir, height, rgb)
    canonical = np.asarray(priors["prior_maps"][:, canonical_indices], dtype=np.float32).transpose(0, 2, 3, 1)
    prior_mask = np.asarray(priors["prior_mask"], dtype=bool)
    visible = np.ones_like(prior_mask, dtype=bool)
    if bool(args.visible_only) and visible_index is not None:
        visible = np.asarray(priors["prior_maps"][:, visible_index], dtype=np.float32) > 0.5
    support_base = roi_masks["full"] & prior_mask & visible
    canonical_q = quantize_canonical(canonical, float(args.canonical_bin_size))
    key_to_body_part = body_part_ids(
        priors=priors,
        channel_names=channel_names,
        support_base=support_base,
        canonical_q=canonical_q,
    )

    observations, observation_summary = collect_observations(
        sources=sources,
        predictions=predictions,
        canonical_q=canonical_q,
        support_base=support_base,
        roi_masks=roi_masks,
        conf_percentile=float(args.conf_percentile),
    )
    key_to_index, surfels, surface_summary = aggregate_surface(
        observations,
        key_to_body_part=key_to_body_part,
        min_observations=int(args.min_surfel_observations),
        min_views=int(args.min_surfel_views),
        critical_min_views=int(args.critical_min_views),
        max_spread=float(args.max_surfel_spread),
    )
    opt_summary = optimize_positions(
        surfels,
        key_to_index,
        smooth_lambda=float(args.smooth_lambda),
        critical_smooth_scale=float(args.critical_smooth_scale),
        iterations=int(args.smooth_iterations),
    )
    rasterized, raster_summary = rasterize_optimized_surface(
        predictions=predictions,
        canonical_q=canonical_q,
        support_base=support_base,
        roi_masks=roi_masks,
        key_to_index=key_to_index,
        surfels=surfels,
        max_distance=float(args.max_raster_distance),
        neighbor_fill_radius=int(args.neighbor_fill_radius),
        alpha=float(args.alpha),
    )

    out = dict(predictions)
    out["world_points"] = rasterized["world_points"].astype(predictions["world_points"].dtype, copy=False)
    out["depth"] = rasterized["depth"].astype(predictions["depth"].dtype, copy=False)
    out["r47_changed_mask"] = rasterized["r47_changed_mask"]
    out["r47_view_support"] = rasterized["r47_view_support"]
    out["r47_observation_count"] = rasterized["r47_observation_count"]
    out["r47_raster_distance"] = rasterized["r47_raster_distance"]
    normal_summary = {"enabled": False}
    if "normal" in predictions and "normal_conf" in predictions:
        normal, normal_conf, normal_summary = recompute_normals(
            predictions,
            out["world_points"],
            rasterized["r47_changed_mask"].astype(bool),
            dilate=int(args.normal_dilate),
        )
        out["normal"] = normal.astype(predictions["normal"].dtype, copy=False)
        out["normal_conf"] = normal_conf.astype(predictions["normal_conf"].dtype, copy=False)

    np.savez_compressed(args.output_dir / "predictions.npz", **out)
    np.savez_compressed(args.output_dir / "r47_optimized_surfels.npz", **surfels)
    heatmaps = write_support_heatmaps(
        args.output_dir / "support_heatmaps",
        rasterized["r47_view_support"],
        rasterized["r47_changed_mask"],
    )
    surface_debug = write_surface_debug(output_dir=args.output_dir / "surface_debug", surfels=surfels, point_size=float(args.point_size))
    analysis = failure_analysis(surface_summary, opt_summary, raster_summary)
    summary = {
        "task": "r47_visibility_aware_surface_backend",
        "truthful_status": "local_diagnostic_not_pass_not_cloud",
        "predictions": str(args.predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_predictions": str((args.output_dir / "predictions.npz").resolve()),
        "output_surfels": str((args.output_dir / "r47_optimized_surfels.npz").resolve()),
        "sources": sources,
        "parameters": {
            "conf_percentile": float(args.conf_percentile),
            "canonical_bin_size": float(args.canonical_bin_size),
            "min_surfel_observations": int(args.min_surfel_observations),
            "min_surfel_views": int(args.min_surfel_views),
            "critical_min_views": int(args.critical_min_views),
            "max_surfel_spread": float(args.max_surfel_spread),
            "smooth_lambda": float(args.smooth_lambda),
            "critical_smooth_scale": float(args.critical_smooth_scale),
            "smooth_iterations": int(args.smooth_iterations),
            "max_raster_distance": float(args.max_raster_distance),
            "neighbor_fill_radius": int(args.neighbor_fill_radius),
            "alpha": float(args.alpha),
            "visible_only": bool(args.visible_only),
        },
        "support_counts": {
            "full_mask": int(roi_masks["full"].sum()),
            "prior_mask": int(prior_mask.sum()),
            "visible": int(visible.sum()),
            "support_base": int(support_base.sum()),
            "roi_masks": {roi: int(mask.sum()) for roi, mask in roi_masks.items()},
        },
        "observation_summary": observation_summary,
        "surface_summary": surface_summary,
        "optimization_summary": opt_summary,
        "rasterization_summary": raster_summary,
        "normal_recompute": normal_summary,
        "surface_debug": surface_debug,
        "support_heatmaps": heatmaps,
        "failure_analysis": analysis,
        "notes": [
            "This is not r43 with only a bin/support/raster parameter change: r47 adds a graph-regularized optimized surface and critical ROI support gating.",
            "It still cannot claim mentor pass without the full strict candidate gate and explicit Open3D visual review.",
            "No hard teacher, 60-view, Kinect, LHM, Sapiens depth, or SMPL-X face/hair geometry truth is used.",
        ],
    }
    (args.output_dir / "r47_visibility_surface_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# r47 Visibility-Aware Shared Surface Diagnostic",
        "",
        f"- output predictions: `{(args.output_dir / 'predictions.npz').resolve()}`",
        f"- accepted surfels: `{surface_summary.get('accepted_surfels', 0)}`",
        f"- raster-eligible surfels: `{surface_summary.get('raster_eligible_surfels', 0)}`",
        f"- optimization p95 shift: `{opt_summary.get('p95_shift', 0.0)}`",
        f"- changed pixels: `{raster_summary.get('changed_pixels_total', 0)}`",
        f"- face changed fraction: `{raster_summary.get('roi_rasterization', {}).get('face', {}).get('changed_fraction', 0.0):.4f}`",
        f"- hands changed fraction: `{raster_summary.get('roi_rasterization', {}).get('hands', {}).get('changed_fraction', 0.0):.4f}`",
        "",
        "## Truthful Scope",
        "",
        "This is a local diagnostic for a graph-regularized shared surface backend. It does not unblock cloud and does not claim mentor pass.",
        "",
        "## Risk Flags",
        "",
    ]
    for key, value in analysis["risk_flags"].items():
        report_lines.append(f"- `{key}`: `{value}`")
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
