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
    _save_open3d_camera_renders,
    _save_open3d_renders,
    load_2d_roi_mask_stack,
    load_mask_stack,
    load_rgb_stack,
    unproject_depth_map_to_point_map_numpy,
)


ROI_ORDER = ("full", "head", "face", "hands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shared human point cloud by aggregating VGGT observations "
            "through SMPL-X canonical bins. SMPL-X is used only as a "
            "correspondence/index; output geometry always comes from VGGT "
            "world/depth observations."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--point-source", choices=("world_points", "depth_unprojection"), default="world_points")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--canonical-bin-size", type=float, default=0.014)
    parser.add_argument("--min-bin-count", type=int, default=2)
    parser.add_argument("--min-bin-views", type=int, default=2)
    parser.add_argument("--max-bin-spread", type=float, default=0.060)
    parser.add_argument("--visible-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-points-per-roi", type=int, default=260000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--camera-view-indices", default="0,3")
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


def parse_camera_indices(spec: str, count: int) -> list[int]:
    out: list[int] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= count:
            raise IndexError(f"camera view index {index} outside [0,{count})")
        out.append(index)
    return sorted(set(out))


def quantize_canonical(canonical: np.ndarray, bin_size: float) -> np.ndarray:
    safe_bin = max(float(bin_size), 1e-6)
    return np.floor(canonical.astype(np.float32) / safe_bin).astype(np.int32)


def percentile_thresholds(conf: np.ndarray, support: np.ndarray, percentile: float) -> np.ndarray:
    thresholds = np.zeros((conf.shape[0],), dtype=np.float32)
    for view_idx in range(conf.shape[0]):
        values = conf[view_idx][support[view_idx] & np.isfinite(conf[view_idx]) & (conf[view_idx] > 0)]
        thresholds[view_idx] = float(np.percentile(values, float(percentile))) if values.size else np.inf
    return thresholds


def build_roi_masks(scene_dir: Path, target_size: int, rgb: np.ndarray) -> dict[str, np.ndarray]:
    masks = load_mask_stack(scene_dir / "masks", target_size=target_size).astype(bool)
    return {
        "full": masks,
        "head": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="head", rgb_stack=rgb).astype(bool),
        "face": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="face", rgb_stack=rgb).astype(bool),
        "hands": load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi="hands", rgb_stack=rgb).astype(bool),
    }


def aggregate_bins(
    canonical_q: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    build_mask: np.ndarray,
    *,
    min_bin_count: int,
    min_bin_views: int,
    max_bin_spread: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flat_q = canonical_q.reshape(-1, 3)
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    flat_mask = build_mask.reshape(-1)
    view_ids = np.broadcast_to(np.arange(points.shape[0], dtype=np.int32)[:, None, None], points.shape[:3]).reshape(-1)
    selected = np.flatnonzero(flat_mask)
    rejected = {"too_few_count": 0, "too_few_views": 0, "spread": 0}
    if selected.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {
            "selected_observations": 0,
            "unique_bins": 0,
            "accepted_bins": 0,
            "rejected": rejected,
        }

    keys = flat_q[selected]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_selected = selected[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], sorted_selected.size]

    out_points: list[np.ndarray] = []
    out_colors: list[np.ndarray] = []
    spreads: list[float] = []
    counts: list[int] = []
    view_counts: list[int] = []

    for start, end in zip(starts, ends):
        group_indices = sorted_selected[start:end]
        if group_indices.size < int(min_bin_count):
            rejected["too_few_count"] += 1
            continue
        group_views = np.unique(view_ids[group_indices])
        if group_views.size < int(min_bin_views):
            rejected["too_few_views"] += 1
            continue
        group_points = flat_points[group_indices].astype(np.float32)
        center = np.median(group_points, axis=0)
        distances = np.linalg.norm(group_points - center[None, :], axis=1)
        spread = float(np.percentile(distances, 75)) if distances.size else float("inf")
        if not np.isfinite(spread) or spread > float(max_bin_spread):
            rejected["spread"] += 1
            continue
        weights = np.maximum(flat_conf[group_indices].astype(np.float32), 1e-6)
        color = np.average(flat_colors[group_indices].astype(np.float32), axis=0, weights=weights)
        out_points.append(center.astype(np.float32))
        out_colors.append(np.clip(color, 0, 255).astype(np.uint8))
        spreads.append(spread)
        counts.append(int(group_indices.size))
        view_counts.append(int(group_views.size))

    points_out = np.stack(out_points, axis=0) if out_points else np.zeros((0, 3), dtype=np.float32)
    colors_out = np.stack(out_colors, axis=0) if out_colors else np.zeros((0, 3), dtype=np.uint8)
    summary = {
        "selected_observations": int(selected.size),
        "unique_bins": int(unique.shape[0]),
        "accepted_bins": int(points_out.shape[0]),
        "rejected": rejected,
        "count_percentiles": [float(v) for v in np.percentile(counts, [0, 25, 50, 75, 95, 100])] if counts else [],
        "view_count_percentiles": [float(v) for v in np.percentile(view_counts, [0, 25, 50, 75, 95, 100])] if view_counts else [],
        "spread_percentiles": [float(v) for v in np.percentile(spreads, [0, 25, 50, 75, 95, 100])] if spreads else [],
    }
    return points_out, colors_out, summary


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def render_roi(
    *,
    roi: str,
    points: np.ndarray,
    colors: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    point_size: float,
    camera_indices: list[int],
) -> dict[str, Any]:
    roi_dir = output_dir / roi
    roi_dir.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        return {"roi": roi, "points": 0, "screenshots": [], "status": "empty"}
    if points.shape[0] > 260000:
        rng = np.random.default_rng(0)
        keep = rng.choice(np.arange(points.shape[0]), size=260000, replace=False)
        points = points[keep]
        colors = colors[keep]
    ply_path = roi_dir / f"shared_{roi}.ply"
    write_ply(ply_path, points, colors)
    screenshots = _save_open3d_renders(
        points=points,
        colors=colors,
        output_dir=roi_dir,
        roi=roi,
        width=1400,
        height=1100,
        point_size=float(point_size),
        interactive=False,
    )
    screenshots.extend(
        _save_open3d_camera_renders(
            points=points,
            colors=colors,
            extrinsic=np.asarray(predictions["extrinsic"], dtype=np.float32),
            intrinsic=np.asarray(predictions["intrinsic"], dtype=np.float32),
            output_dir=roi_dir,
            camera_indices=camera_indices,
            point_size=float(point_size),
            render_size=700,
        )
    )
    extent = np.ptp(points, axis=0) if points.shape[0] else np.zeros(3, dtype=np.float32)
    return {
        "roi": roi,
        "points": int(points.shape[0]),
        "ply": str(ply_path.resolve()),
        "screenshots": screenshots,
        "extent": [float(v) for v in extent.tolist()],
        "status": "rendered",
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_npz(args.predictions)
    priors = load_npz(args.scene_dir / "prior_maps.npz")
    channel_names = [str(value) for value in priors["prior_channels"].tolist()]
    canonical_indices = [
        channel_index(channel_names, "smplx_canonical_x"),
        channel_index(channel_names, "smplx_canonical_y"),
        channel_index(channel_names, "smplx_canonical_z"),
    ]
    visible_index = channel_names.index("smplx_visible_mask") if "smplx_visible_mask" in channel_names else None

    if args.point_source == "world_points":
        points = np.asarray(predictions["world_points"], dtype=np.float32)
        conf = np.asarray(predictions["world_points_conf"], dtype=np.float32)
    else:
        points = unproject_depth_map_to_point_map_numpy(
            np.asarray(predictions["depth"], dtype=np.float32),
            np.asarray(predictions["extrinsic"], dtype=np.float32),
            np.asarray(predictions["intrinsic"], dtype=np.float32),
        )
        conf = np.asarray(predictions["depth_conf"], dtype=np.float32)
    views, height, width, _ = points.shape
    rgb = load_rgb_stack(args.scene_dir / "images", target_size=height)
    roi_masks = build_roi_masks(args.scene_dir, height, rgb)
    canonical = np.asarray(priors["prior_maps"][:, canonical_indices], dtype=np.float32).transpose(0, 2, 3, 1)
    if canonical.shape[:3] != points.shape[:3]:
        raise ValueError(f"canonical shape {canonical.shape} does not match points {points.shape}")
    prior_mask = np.asarray(priors["prior_mask"], dtype=bool)
    visible = np.ones(prior_mask.shape, dtype=bool)
    if bool(args.visible_only) and visible_index is not None:
        visible = np.asarray(priors["prior_maps"][:, visible_index], dtype=np.float32) > 0.5

    finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf > 0.0)
    support_base = finite & prior_mask & visible
    thresholds = percentile_thresholds(conf, support_base, float(args.conf_percentile))
    high_conf = support_base & (conf >= thresholds[:, None, None])
    canonical_q = quantize_canonical(canonical, float(args.canonical_bin_size))
    camera_indices = parse_camera_indices(str(args.camera_view_indices), int(views))

    roi_summaries: dict[str, Any] = {}
    render_summaries: dict[str, Any] = {}
    for roi in ROI_ORDER:
        roi_support = high_conf & roi_masks[roi]
        shared_points, shared_colors, summary = aggregate_bins(
            canonical_q,
            points,
            rgb,
            conf,
            roi_support,
            min_bin_count=int(args.min_bin_count),
            min_bin_views=int(args.min_bin_views),
            max_bin_spread=float(args.max_bin_spread),
        )
        roi_summaries[roi] = summary
        render_summaries[roi] = render_roi(
            roi=roi,
            points=shared_points,
            colors=shared_colors,
            predictions=predictions,
            output_dir=args.output_dir,
            point_size=float(args.point_size),
            camera_indices=camera_indices,
        )

    payload = {
        "task": "build_vggt_canonical_shared_human_points",
        "truthful_status": "diagnostic_probe_not_candidate_not_teacher_pass",
        "predictions": str(args.predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "point_source": str(args.point_source),
        "conf_percentile": float(args.conf_percentile),
        "conf_thresholds": [float(v) if np.isfinite(v) else None for v in thresholds.tolist()],
        "canonical_bin_size": float(args.canonical_bin_size),
        "min_bin_count": int(args.min_bin_count),
        "min_bin_views": int(args.min_bin_views),
        "max_bin_spread": float(args.max_bin_spread),
        "visible_only": bool(args.visible_only),
        "support_counts": {
            "finite_prior_visible": int(support_base.sum()),
            "high_conf": int(high_conf.sum()),
            "roi_masks": {roi: int(mask.sum()) for roi, mask in roi_masks.items()},
        },
        "roi_summaries": roi_summaries,
        "render_summaries": render_summaries,
        "notes": [
            "SMPL-X canonical bins are only used as multi-view correspondence keys.",
            "Output points are robust medians of VGGT observations, not SMPL-X vertices.",
            "This probe cannot claim pass; it must be visually reviewed as a complete human point representation.",
        ],
    }
    summary_path = args.output_dir / "shared_human_points_summary.json"
    summary_path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
