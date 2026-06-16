from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from render_open3d_pointcloud import (  # noqa: E402
    load_2d_roi_mask_stack,
    load_mask_stack,
    load_rgb_stack,
    unproject_depth_map_to_point_map_numpy,
)
from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


ROI_NAMES = ("full", "head", "face", "hands")
SOURCE_BITS = {"world_points": 1, "depth_unprojection": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and rasterize a shared human surfel representation from VGGT "
            "observations. SMPL-X canonical channels are used only as "
            "cross-view correspondence keys; surfel positions always come from "
            "VGGT world/depth observations."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--canonical-bin-size", type=float, default=0.014)
    parser.add_argument("--min-surfel-observations", type=int, default=4)
    parser.add_argument("--min-surfel-views", type=int, default=2)
    parser.add_argument("--max-surfel-spread", type=float, default=0.060)
    parser.add_argument("--max-raster-distance", type=float, default=0.080)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--visible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-dilate", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def channel_index(channel_names: list[str], name: str) -> int:
    try:
        return channel_names.index(name)
    except ValueError as exc:
        raise KeyError(f"Missing prior channel {name!r}; available={channel_names}") from exc


def parse_sources(spec: str) -> list[str]:
    sources = [item.strip() for item in str(spec).split(",") if item.strip()]
    valid = {"world_points", "depth_unprojection"}
    bad = [item for item in sources if item not in valid]
    if bad:
        raise ValueError(f"Unknown source(s) {bad}; valid={sorted(valid)}")
    if not sources:
        raise ValueError("At least one source is required")
    return sources


def quantize_canonical(canonical: np.ndarray, bin_size: float) -> np.ndarray:
    safe_bin = max(float(bin_size), 1e-6)
    return np.floor(canonical.astype(np.float32) / safe_bin).astype(np.int32)


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return np.einsum("...j,ij->...i", points_world.astype(np.float32), rotation) + translation


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[:, dy : dy + out.shape[1], dx : dx + out.shape[2]]
        out = grown
    return out


def build_roi_masks(scene_dir: Path, target_size: int, rgb: np.ndarray) -> dict[str, np.ndarray]:
    masks = load_mask_stack(scene_dir / "masks", target_size=target_size).astype(bool)
    return {
        "full": masks,
        "head": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="head", rgb_stack=rgb).astype(bool),
        "face": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="face", rgb_stack=rgb).astype(bool),
        "hands": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="hands", rgb_stack=rgb).astype(bool),
    }


def percentile_thresholds(conf: np.ndarray, support: np.ndarray, percentile: float) -> np.ndarray:
    thresholds = np.zeros((conf.shape[0],), dtype=np.float32)
    for view_idx in range(conf.shape[0]):
        values = conf[view_idx][support[view_idx] & np.isfinite(conf[view_idx]) & (conf[view_idx] > 0.0)]
        thresholds[view_idx] = float(np.percentile(values, float(percentile))) if values.size else np.inf
    return thresholds


def source_points_and_conf(predictions: dict[str, np.ndarray], source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "world_points":
        return (
            np.asarray(predictions["world_points"], dtype=np.float32),
            np.asarray(predictions["world_points_conf"], dtype=np.float32),
        )
    if source == "depth_unprojection":
        return (
            unproject_depth_map_to_point_map_numpy(
                np.asarray(predictions["depth"], dtype=np.float32),
                np.asarray(predictions["extrinsic"], dtype=np.float32),
                np.asarray(predictions["intrinsic"], dtype=np.float32),
            ),
            np.asarray(predictions["depth_conf"], dtype=np.float32),
        )
    raise ValueError(source)


def collect_observations(
    *,
    sources: list[str],
    predictions: dict[str, np.ndarray],
    canonical_q: np.ndarray,
    support_base: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    conf_percentile: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    view_count, height, width = support_base.shape
    flat_views = np.broadcast_to(np.arange(view_count, dtype=np.int16)[:, None, None], (view_count, height, width)).reshape(-1)
    flat_q_all = canonical_q.reshape(-1, 3)
    flat_roi = {roi: mask.reshape(-1) for roi, mask in roi_masks.items()}

    obs_keys: list[np.ndarray] = []
    obs_points: list[np.ndarray] = []
    obs_colors_conf: list[np.ndarray] = []
    obs_views: list[np.ndarray] = []
    obs_sources: list[np.ndarray] = []
    obs_roi_flags: list[np.ndarray] = []
    thresholds: dict[str, list[float | None]] = {}
    source_counts: dict[str, Any] = {}

    for source in sources:
        points, conf = source_points_and_conf(predictions, source)
        finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf > 0.0)
        source_support = support_base & finite
        threshold = percentile_thresholds(conf, source_support, float(conf_percentile))
        high_conf = source_support & (conf >= threshold[:, None, None])
        selected = np.flatnonzero(high_conf.reshape(-1))
        thresholds[source] = [float(v) if np.isfinite(v) else None for v in threshold.tolist()]
        source_counts[source] = {
            "support_pixels": int(source_support.sum()),
            "selected_observations": int(selected.size),
            "roi_selected": {
                roi: int((high_conf & roi_masks[roi]).sum())
                for roi in ROI_NAMES
            },
        }
        if selected.size == 0:
            continue
        flags = np.stack([flat_roi[roi][selected] for roi in ROI_NAMES], axis=1).astype(np.uint8)
        obs_keys.append(flat_q_all[selected])
        obs_points.append(points.reshape(-1, 3)[selected].astype(np.float32))
        obs_colors_conf.append(conf.reshape(-1)[selected].astype(np.float32))
        obs_views.append(flat_views[selected])
        obs_sources.append(np.full((selected.size,), SOURCE_BITS[source], dtype=np.uint8))
        obs_roi_flags.append(flags)

    if not obs_keys:
        empty = {
            "keys": np.zeros((0, 3), dtype=np.int32),
            "points": np.zeros((0, 3), dtype=np.float32),
            "conf": np.zeros((0,), dtype=np.float32),
            "views": np.zeros((0,), dtype=np.int16),
            "sources": np.zeros((0,), dtype=np.uint8),
            "roi_flags": np.zeros((0, len(ROI_NAMES)), dtype=np.uint8),
        }
    else:
        empty = {
            "keys": np.concatenate(obs_keys, axis=0),
            "points": np.concatenate(obs_points, axis=0),
            "conf": np.concatenate(obs_colors_conf, axis=0),
            "views": np.concatenate(obs_views, axis=0),
            "sources": np.concatenate(obs_sources, axis=0),
            "roi_flags": np.concatenate(obs_roi_flags, axis=0),
        }
    summary = {
        "sources": sources,
        "thresholds": thresholds,
        "source_counts": source_counts,
        "total_observations": int(empty["points"].shape[0]),
    }
    return empty, summary


def aggregate_surfels(
    observations: dict[str, np.ndarray],
    *,
    min_observations: int,
    min_views: int,
    max_spread: float,
) -> tuple[dict[tuple[int, int, int], int], dict[str, np.ndarray], dict[str, Any]]:
    keys = observations["keys"]
    points = observations["points"]
    conf = observations["conf"]
    views = observations["views"]
    sources = observations["sources"]
    roi_flags = observations["roi_flags"]
    rejected = {"too_few_observations": 0, "too_few_views": 0, "spread": 0}
    if keys.shape[0] == 0:
        empty_arrays = {
            "canonical_key": np.zeros((0, 3), dtype=np.int32),
            "position": np.zeros((0, 3), dtype=np.float32),
            "confidence": np.zeros((0,), dtype=np.float32),
            "view_support": np.zeros((0,), dtype=np.int16),
            "observation_count": np.zeros((0,), dtype=np.int32),
            "source_mask": np.zeros((0,), dtype=np.uint8),
            "spread_p75": np.zeros((0,), dtype=np.float32),
            "roi_counts": np.zeros((0, len(ROI_NAMES)), dtype=np.int32),
        }
        return {}, empty_arrays, {"accepted_surfels": 0, "rejected": rejected}

    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_indices = order
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], sorted_indices.size]

    key_to_index: dict[tuple[int, int, int], int] = {}
    surfel_keys: list[np.ndarray] = []
    surfel_positions: list[np.ndarray] = []
    surfel_conf: list[float] = []
    surfel_views: list[int] = []
    surfel_obs: list[int] = []
    surfel_sources: list[int] = []
    surfel_spread: list[float] = []
    surfel_roi_counts: list[np.ndarray] = []

    counts: list[int] = []
    view_counts: list[int] = []
    spreads: list[float] = []
    source_masks: list[int] = []

    for start, end in zip(starts, ends):
        group_indices = sorted_indices[start:end]
        if group_indices.size < int(min_observations):
            rejected["too_few_observations"] += 1
            continue
        group_views = np.unique(views[group_indices])
        if group_views.size < int(min_views):
            rejected["too_few_views"] += 1
            continue
        group_points = points[group_indices].astype(np.float32)
        center = np.median(group_points, axis=0)
        distances = np.linalg.norm(group_points - center[None, :], axis=1)
        spread = float(np.percentile(distances, 75)) if distances.size else float("inf")
        if not np.isfinite(spread) or spread > float(max_spread):
            rejected["spread"] += 1
            continue
        weights = np.maximum(conf[group_indices].astype(np.float32), 1e-6)
        confidence = float(np.average(conf[group_indices].astype(np.float32), weights=weights))
        source_mask = int(np.bitwise_or.reduce(sources[group_indices])) if group_indices.size else 0
        roi_count = roi_flags[group_indices].astype(np.int32).sum(axis=0)
        surfel_idx = len(surfel_positions)
        key = unique[int(sorted_inverse[start])]
        key_to_index[tuple(int(v) for v in key.tolist())] = surfel_idx
        surfel_keys.append(key.astype(np.int32))
        surfel_positions.append(center.astype(np.float32))
        surfel_conf.append(confidence)
        surfel_views.append(int(group_views.size))
        surfel_obs.append(int(group_indices.size))
        surfel_sources.append(source_mask)
        surfel_spread.append(spread)
        surfel_roi_counts.append(roi_count.astype(np.int32))
        counts.append(int(group_indices.size))
        view_counts.append(int(group_views.size))
        spreads.append(spread)
        source_masks.append(source_mask)

    arrays = {
        "canonical_key": np.stack(surfel_keys, axis=0) if surfel_keys else np.zeros((0, 3), dtype=np.int32),
        "position": np.stack(surfel_positions, axis=0) if surfel_positions else np.zeros((0, 3), dtype=np.float32),
        "confidence": np.asarray(surfel_conf, dtype=np.float32),
        "view_support": np.asarray(surfel_views, dtype=np.int16),
        "observation_count": np.asarray(surfel_obs, dtype=np.int32),
        "source_mask": np.asarray(surfel_sources, dtype=np.uint8),
        "spread_p75": np.asarray(surfel_spread, dtype=np.float32),
        "roi_counts": np.stack(surfel_roi_counts, axis=0) if surfel_roi_counts else np.zeros((0, len(ROI_NAMES)), dtype=np.int32),
    }
    roi_surfel_counts = {
        roi: int((arrays["roi_counts"][:, idx] > 0).sum()) if arrays["roi_counts"].size else 0
        for idx, roi in enumerate(ROI_NAMES)
    }
    both_source = sum(1 for value in source_masks if value == (SOURCE_BITS["world_points"] | SOURCE_BITS["depth_unprojection"]))
    summary = {
        "unique_canonical_bins": int(unique.shape[0]),
        "accepted_surfels": int(arrays["position"].shape[0]),
        "rejected": rejected,
        "roi_surfel_counts": roi_surfel_counts,
        "both_source_surfels": int(both_source),
        "observation_count_percentiles": [float(v) for v in np.percentile(counts, [0, 25, 50, 75, 95, 100])] if counts else [],
        "view_support_percentiles": [float(v) for v in np.percentile(view_counts, [0, 25, 50, 75, 95, 100])] if view_counts else [],
        "spread_p75_percentiles": [float(v) for v in np.percentile(spreads, [0, 25, 50, 75, 95, 100])] if spreads else [],
    }
    return key_to_index, arrays, summary


def rasterize_surfels(
    *,
    predictions: dict[str, np.ndarray],
    canonical_q: np.ndarray,
    support_base: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    key_to_index: dict[tuple[int, int, int], int],
    surfels: dict[str, np.ndarray],
    max_distance: float,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    depth = np.asarray(predictions["depth"], dtype=np.float32)
    extrinsics = np.asarray(predictions["extrinsic"], dtype=np.float32)
    fused_world = world_points.copy()
    fused_depth = depth.copy()
    changed = np.zeros(world_points.shape[:3], dtype=bool)
    support_map = np.zeros(world_points.shape[:3], dtype=np.uint8)
    obs_count_map = np.zeros(world_points.shape[:3], dtype=np.uint16)
    spread_map = np.zeros(world_points.shape[:3], dtype=np.float32)
    distance_map = np.full(world_points.shape[:3], np.nan, dtype=np.float32)
    reason_counts = {"no_surfel": 0, "distance_rejected": 0, "applied": 0}

    flat_q = canonical_q.reshape(-1, 3)
    flat_support = support_base.reshape(-1)
    flat_current = world_points.reshape(-1, 3)
    flat_fused = fused_world.reshape(-1, 3)
    flat_changed = changed.reshape(-1)
    flat_support_map = support_map.reshape(-1)
    flat_obs_map = obs_count_map.reshape(-1)
    flat_spread_map = spread_map.reshape(-1)
    flat_distance_map = distance_map.reshape(-1)
    selected = np.flatnonzero(flat_support)
    positions = surfels["position"]
    view_support = surfels["view_support"]
    obs_count = surfels["observation_count"]
    spread = surfels["spread_p75"]

    for flat_idx in selected:
        key = tuple(int(v) for v in flat_q[flat_idx].tolist())
        surfel_idx = key_to_index.get(key)
        if surfel_idx is None:
            reason_counts["no_surfel"] += 1
            continue
        target = positions[surfel_idx]
        current = flat_current[flat_idx]
        distance = float(np.linalg.norm(current - target))
        flat_support_map[flat_idx] = np.uint8(min(int(view_support[surfel_idx]), 255))
        flat_obs_map[flat_idx] = np.uint16(min(int(obs_count[surfel_idx]), 65535))
        flat_spread_map[flat_idx] = float(spread[surfel_idx])
        flat_distance_map[flat_idx] = distance
        if not np.isfinite(distance) or distance > float(max_distance):
            reason_counts["distance_rejected"] += 1
            continue
        flat_fused[flat_idx] = (current + float(alpha) * (target - current)).astype(np.float32)
        flat_changed[flat_idx] = True
        reason_counts["applied"] += 1

    for view_idx in range(fused_world.shape[0]):
        use = changed[view_idx]
        if not use.any():
            continue
        cam = world_to_camera(fused_world[view_idx], extrinsics[view_idx])
        positive = use & np.isfinite(cam[..., 2]) & (cam[..., 2] > 1e-6)
        fused_depth[view_idx, positive, 0] = cam[..., 2][positive]

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

    outputs = {
        "world_points": fused_world,
        "depth": fused_depth,
        "shared_surfel_changed_mask": changed.astype(np.uint8),
        "shared_surfel_view_support": support_map,
        "shared_surfel_observation_count": obs_count_map,
        "shared_surfel_spread_p75": spread_map,
        "shared_surfel_raster_distance": distance_map,
    }
    summary = {
        "apply_candidates": int(selected.size),
        "changed_pixels_total": int(changed.sum()),
        "changed_pixels_per_view": [int(changed[idx].sum()) for idx in range(changed.shape[0])],
        "reason_counts": reason_counts,
        "roi_rasterization": roi_summary,
    }
    return outputs, summary


def recompute_normals(
    base: dict[str, np.ndarray],
    world_points: np.ndarray,
    changed_mask: np.ndarray,
    *,
    dilate: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsics = np.asarray(base["extrinsic"], dtype=np.float32)
    use_mask = dilate_mask(changed_mask, int(dilate))
    per_view: dict[str, Any] = {}
    for view_idx in range(world_points.shape[0]):
        cam = world_to_camera(world_points[view_idx], extrinsics[view_idx])
        finite = np.isfinite(cam).all(axis=-1)
        normal_map, valid = point_map_to_normal_numpy(cam, finite)
        use = use_mask[view_idx] & valid
        mean_dot = None
        flipped = False
        if use.any():
            dot = np.sum(normal[view_idx][use] * normal_map[use], axis=-1)
            mean_dot = float(np.nanmean(dot)) if dot.size else 0.0
            if mean_dot < 0.0:
                normal_map = -normal_map
                flipped = True
            normal[view_idx][use] = normal_map[use]
            # No confidence boost: support maps carry evidence, confidence cannot hide low support.
        per_view[str(view_idx)] = {
            "seed_pixels": int(changed_mask[view_idx].sum()),
            "candidate_pixels_after_dilate": int(use_mask[view_idx].sum()),
            "normal_replaced_pixels": int(use.sum()),
            "mean_dot_before_optional_flip": mean_dot,
            "flipped_to_match_previous_convention": bool(flipped),
        }
    return normal, normal_conf, {
        "enabled": True,
        "normal_replaced_pixels_total": int(sum(row["normal_replaced_pixels"] for row in per_view.values())),
        "per_view": per_view,
    }


def build_failure_analysis(
    *,
    surfel_summary: dict[str, Any],
    raster_summary: dict[str, Any],
) -> dict[str, Any]:
    roi_surfel_counts = surfel_summary.get("roi_surfel_counts", {})
    roi_raster = raster_summary.get("roi_rasterization", {})
    view_support_p = surfel_summary.get("view_support_percentiles", [])
    median_view_support = float(view_support_p[2]) if len(view_support_p) >= 3 else 0.0
    face_change = float(roi_raster.get("face", {}).get("changed_fraction", 0.0))
    head_change = float(roi_raster.get("head", {}).get("changed_fraction", 0.0))
    hands_change = float(roi_raster.get("hands", {}).get("changed_fraction", 0.0))
    accepted = int(surfel_summary.get("accepted_surfels", 0))
    face_bins = int(roi_surfel_counts.get("face", 0) or 0)
    hands_bins = int(roi_surfel_counts.get("hands", 0) or 0)
    risks = {
        "surfel_set_empty_or_tiny": bool(accepted < 1000),
        "face_surfel_support_sparse": bool(face_bins < 2500),
        "hand_surfel_support_sparse": bool(hands_bins < 800),
        "median_view_support_only_two": bool(median_view_support <= 2.0),
        "face_rasterization_low_change": bool(face_change < 0.20),
        "head_rasterization_low_change": bool(head_change < 0.20),
        "hands_rasterization_low_change": bool(hands_change < 0.20),
    }
    return {
        "truthful_status": "diagnostic_only_not_a_pass",
        "risk_flags": risks,
        "interpretation": [
            "This backend can only consolidate VGGT-observed geometry; it cannot invent unobserved facial relief.",
            "A strict pass still requires same-protocol face/head improvement plus full-body/hands Open3D visual review.",
            "If face/head Open3D remains shell-like, freeze this r43 diagnostic instead of tuning thresholds.",
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
    if canonical.shape[:3] != (view_count, height, width):
        raise ValueError(f"canonical shape {canonical.shape} does not match predictions {(view_count, height, width)}")
    prior_mask = np.asarray(priors["prior_mask"], dtype=bool)
    if prior_mask.shape != (view_count, height, width):
        raise ValueError(f"prior_mask shape {prior_mask.shape} does not match predictions {(view_count, height, width)}")
    visible = np.ones_like(prior_mask, dtype=bool)
    if bool(args.visible_only) and visible_index is not None:
        visible = np.asarray(priors["prior_maps"][:, visible_index], dtype=np.float32) > 0.5
    support_base = roi_masks["full"] & prior_mask & visible
    canonical_q = quantize_canonical(canonical, float(args.canonical_bin_size))

    observations, obs_summary = collect_observations(
        sources=sources,
        predictions=predictions,
        canonical_q=canonical_q,
        support_base=support_base,
        roi_masks=roi_masks,
        conf_percentile=float(args.conf_percentile),
    )
    key_to_index, surfels, surfel_summary = aggregate_surfels(
        observations,
        min_observations=int(args.min_surfel_observations),
        min_views=int(args.min_surfel_views),
        max_spread=float(args.max_surfel_spread),
    )
    rasterized, raster_summary = rasterize_surfels(
        predictions=predictions,
        canonical_q=canonical_q,
        support_base=support_base,
        roi_masks=roi_masks,
        key_to_index=key_to_index,
        surfels=surfels,
        max_distance=float(args.max_raster_distance),
        alpha=float(args.alpha),
    )

    out = dict(predictions)
    out["world_points"] = rasterized["world_points"].astype(predictions["world_points"].dtype, copy=False)
    out["depth"] = rasterized["depth"].astype(predictions["depth"].dtype, copy=False)
    out["shared_surfel_changed_mask"] = rasterized["shared_surfel_changed_mask"]
    out["shared_surfel_view_support"] = rasterized["shared_surfel_view_support"]
    out["shared_surfel_observation_count"] = rasterized["shared_surfel_observation_count"]
    out["shared_surfel_spread_p75"] = rasterized["shared_surfel_spread_p75"]
    out["shared_surfel_raster_distance"] = rasterized["shared_surfel_raster_distance"]
    normal_summary = {"enabled": False}
    if "normal" in predictions and "normal_conf" in predictions:
        normal, normal_conf, normal_summary = recompute_normals(
            predictions,
            out["world_points"],
            rasterized["shared_surfel_changed_mask"].astype(bool),
            dilate=int(args.normal_dilate),
        )
        out["normal"] = normal.astype(predictions["normal"].dtype, copy=False)
        out["normal_conf"] = normal_conf.astype(predictions["normal_conf"].dtype, copy=False)

    output_path = args.output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)
    np.savez_compressed(args.output_dir / "shared_surfels.npz", **surfels)

    failure_analysis = build_failure_analysis(surfel_summary=surfel_summary, raster_summary=raster_summary)
    summary = {
        "task": "rasterize_shared_surfel_predictions",
        "truthful_status": "r43_local_shared_surface_diagnostic_not_pass_not_cloud",
        "predictions": str(args.predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_predictions": str(output_path.resolve()),
        "output_surfels": str((args.output_dir / "shared_surfels.npz").resolve()),
        "sources": sources,
        "conf_percentile": float(args.conf_percentile),
        "canonical_bin_size": float(args.canonical_bin_size),
        "min_surfel_observations": int(args.min_surfel_observations),
        "min_surfel_views": int(args.min_surfel_views),
        "max_surfel_spread": float(args.max_surfel_spread),
        "max_raster_distance": float(args.max_raster_distance),
        "alpha": float(args.alpha),
        "visible_only": bool(args.visible_only),
        "support_counts": {
            "full_mask": int(roi_masks["full"].sum()),
            "prior_mask": int(prior_mask.sum()),
            "visible": int(visible.sum()),
            "support_base": int(support_base.sum()),
            "roi_masks": {roi: int(mask.sum()) for roi, mask in roi_masks.items()},
        },
        "observation_summary": obs_summary,
        "surfel_summary": surfel_summary,
        "rasterization_summary": raster_summary,
        "normal_recompute": normal_summary,
        "failure_analysis": failure_analysis,
        "notes": [
            "SMPL-X canonical maps are used only as correspondence/bin keys.",
            "No hard teacher, 60-view, Kinect, or SMPL-X face geometry is used.",
            "Confidence is not boosted; support maps are saved so low-support regions cannot pass invisibly.",
            "This diagnostic must be evaluated by the full strict candidate gate and Open3D visual review.",
        ],
    }
    summary_path = args.output_dir / "shared_surfel_rasterization_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_lines = [
        "# r43 Shared Human Surfel Diagnostic",
        "",
        f"- Output predictions: `{output_path.resolve()}`",
        f"- Output surfels: `{(args.output_dir / 'shared_surfels.npz').resolve()}`",
        f"- Accepted surfels: `{surfel_summary.get('accepted_surfels', 0)}`",
        f"- Changed pixels: `{raster_summary.get('changed_pixels_total', 0)}`",
        f"- Face changed fraction: `{raster_summary.get('roi_rasterization', {}).get('face', {}).get('changed_fraction', 0.0):.4f}`",
        f"- Head changed fraction: `{raster_summary.get('roi_rasterization', {}).get('head', {}).get('changed_fraction', 0.0):.4f}`",
        f"- Hands changed fraction: `{raster_summary.get('roi_rasterization', {}).get('hands', {}).get('changed_fraction', 0.0):.4f}`",
        "",
        "## Truthful Scope",
        "",
        "This is a local representation diagnostic. It does not claim mentor pass and does not unblock cloud. The strict candidate gate must decide whether the shared-surface representation actually improves head/face while preserving full body and hands.",
        "",
        "## Risk Flags",
        "",
    ]
    for key, value in failure_analysis["risk_flags"].items():
        report_lines.append(f"- `{key}`: `{value}`")
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
