from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from vggt.utils.normal_refiner import shoulder_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a VGGT predictions.npz by projecting an externally reconstructed "
            "PLY point cloud into the VGGT views with a nearest-depth z-buffer. This "
            "is a local diagnostic/teacher probe; it does not claim mentor pass."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--external-pointcloud", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--patch-roi",
        choices=("full", "head", "face", "head_face", "shoulder", "all"),
        default="head_face",
    )
    parser.add_argument(
        "--align-roi",
        choices=("full", "head", "face", "head_face", "shoulder", "all"),
        default="head_face",
    )
    parser.add_argument("--views", default="all", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--align-mode", choices=("none", "umeyama_icp"), default="umeyama_icp")
    parser.add_argument("--anchor-conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-anchor-points", type=int, default=140000)
    parser.add_argument("--max-source-points", type=int, default=900000)
    parser.add_argument("--align-sample-points", type=int, default=45000)
    parser.add_argument("--align-iterations", type=int, default=14)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.08)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--statistical-nb-neighbors", type=int, default=0)
    parser.add_argument("--statistical-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius-outlier-radius", type=float, default=0.0)
    parser.add_argument("--radius-outlier-min-nb", type=int, default=8)
    parser.add_argument("--robust-percentile", type=float, default=0.0)
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--splat-radius", type=int, default=0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument(
        "--max-base-distance",
        type=float,
        default=0.12,
        help="Reject projected external hits farther than this from the base VGGT point. <=0 disables.",
    )
    parser.add_argument(
        "--max-depth-residual",
        type=float,
        default=0.16,
        help="Reject projected external hits whose camera z differs too much from base depth. <=0 disables.",
    )
    parser.add_argument("--surface-alpha", type=float, default=1.0)
    parser.add_argument("--confidence-boost", type=float, default=180.0)
    parser.add_argument("--depth-confidence-boost", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--write-debug-ply", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required; run with the g3splat Python environment.") from exc
    return o3d


def parse_views(spec: str, view_count: int) -> list[int]:
    if str(spec).strip().lower() == "all":
        return list(range(view_count))
    out: list[int] = []
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= view_count:
            raise ValueError(f"view index {idx} outside [0, {view_count})")
        out.append(idx)
    if not out:
        raise ValueError("--views selected no views")
    return sorted(set(out))


def box_mask(box: tuple[int, int, int, int] | None, shape: tuple[int, int], support: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if box is None:
        return out
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(shape[1], x0))
    x1 = max(0, min(shape[1], x1))
    y0 = max(0, min(shape[0], y0))
    y1 = max(0, min(shape[0], y1))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = support[y0:y1, x0:x1]
    return out


def roi_mask(mask: np.ndarray, roi: str) -> np.ndarray:
    support = np.asarray(mask, dtype=bool)
    if roi in {"full", "all"}:
        return support
    rois = build_roi_masks(support)
    if roi == "head":
        return rois["head"]
    if roi == "face":
        return rois["face"]
    if roi == "head_face":
        return rois["head"] | rois["face"]
    if roi == "shoulder":
        return box_mask(shoulder_box_from_mask(support), support.shape, support)
    raise ValueError(f"unsupported ROI: {roi}")


def load_roi_masks(scene_dir: Path, view_count: int, height: int, width: int, roi: str, dilate: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        mask = roi_mask(scene.mask.astype(bool), roi)
        if int(dilate) > 0:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=int(dilate)).astype(bool)
        masks.append(mask.astype(bool))
    return np.stack(masks, axis=0)


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return points_world.astype(np.float32) @ rotation.T + translation[None, :]


def _transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (float(scale) * points.astype(np.float64) @ rotation.T + translation[None, :]).astype(np.float32)


def _estimate_similarity(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if len(source) < 4 or len(target) < 4 or len(source) != len(target):
        raise ValueError("Need matching source/target correspondences for similarity estimation.")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean[None]
    target_centered = target - target_mean[None]
    covariance = target_centered.T @ source_centered / float(len(source))
    left_singular, singular_values, right_singular_t = np.linalg.svd(covariance)
    handedness = np.ones(3, dtype=np.float64)
    if np.linalg.det(left_singular @ right_singular_t) < 0:
        handedness[-1] = -1.0
    rotation = left_singular @ np.diag(handedness) @ right_singular_t
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular_values * handedness) / max(source_variance, 1e-12))
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def nearest_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    o3d = _load_open3d()
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    max_distance_sq = float(max_distance) * float(max_distance)
    source_corr: list[np.ndarray] = []
    target_corr: list[np.ndarray] = []
    for point in source_points.astype(np.float64):
        _, indices, distances = tree.search_knn_vector_3d(point, 1)
        if indices and distances[0] <= max_distance_sq:
            source_corr.append(point)
            target_corr.append(target_points[int(indices[0])])
    if not source_corr:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(source_corr, dtype=np.float32), np.asarray(target_corr, dtype=np.float32)


def load_external_points(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    o3d = _load_open3d()
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in external point cloud: {path}")
    pcd.remove_non_finite_points()
    if float(args.voxel_size) > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))
    if int(args.statistical_nb_neighbors) > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(args.statistical_nb_neighbors),
            std_ratio=float(args.statistical_std_ratio),
        )
    if float(args.radius_outlier_radius) > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=int(args.radius_outlier_min_nb),
            radius=float(args.radius_outlier_radius),
        )
    points = np.asarray(pcd.points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    if 0.0 < float(args.robust_percentile) < 50.0 and finite.any():
        q = float(args.robust_percentile)
        lo, hi = np.percentile(points[finite], [q, 100.0 - q], axis=0)
        finite &= np.all((points >= lo) & (points <= hi), axis=1)
    points = points[finite]
    if len(points) > int(args.max_source_points) > 0:
        rng = np.random.default_rng(int(args.seed))
        keep = rng.choice(len(points), size=int(args.max_source_points), replace=False)
        points = points[keep]
    if len(points) < 256:
        raise RuntimeError(f"Too few usable external points after filtering: {len(points)}")
    return points.astype(np.float32), {
        "input_path": str(path.resolve()),
        "points_after_filter": int(len(points)),
        "voxel_size": float(args.voxel_size),
        "statistical_nb_neighbors": int(args.statistical_nb_neighbors),
        "radius_outlier_radius": float(args.radius_outlier_radius),
        "robust_percentile": float(args.robust_percentile),
    }


def select_anchor_points(
    predictions: dict[str, np.ndarray],
    masks: np.ndarray,
    conf_percentile: float,
    max_points: int,
    seed: int,
) -> np.ndarray:
    points = np.asarray(predictions["world_points"], dtype=np.float32)
    conf = np.asarray(predictions.get("world_points_conf", predictions.get("depth_conf")), dtype=np.float32)
    selected = []
    for view_idx in range(points.shape[0]):
        valid = masks[view_idx] & np.isfinite(points[view_idx]).all(axis=-1) & np.isfinite(conf[view_idx])
        if not valid.any():
            continue
        threshold = float(np.percentile(conf[view_idx][valid], float(conf_percentile)))
        valid &= conf[view_idx] >= threshold
        if valid.any():
            selected.append(points[view_idx][valid])
    if not selected:
        raise RuntimeError("No anchor points selected for alignment.")
    anchor = np.concatenate(selected, axis=0).astype(np.float32)
    if len(anchor) > int(max_points) > 0:
        rng = np.random.default_rng(int(seed))
        anchor = anchor[rng.choice(len(anchor), size=int(max_points), replace=False)]
    return anchor


def align_points_to_anchor(points: np.ndarray, anchor: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if str(args.align_mode) == "none":
        return points, {"align_mode": "none"}
    rng = np.random.default_rng(int(args.seed))
    source_sample = points
    target_sample = anchor
    if len(source_sample) > int(args.align_sample_points):
        source_sample = source_sample[rng.choice(len(source_sample), size=int(args.align_sample_points), replace=False)]
    if len(target_sample) > int(args.align_sample_points):
        target_sample = target_sample[rng.choice(len(target_sample), size=int(args.align_sample_points), replace=False)]

    source_center = source_sample.mean(axis=0)
    target_center = target_sample.mean(axis=0)
    source_radius = float(np.sqrt(np.mean(np.sum((source_sample - source_center[None]) ** 2, axis=1))))
    target_radius = float(np.sqrt(np.mean(np.sum((target_sample - target_center[None]) ** 2, axis=1))))
    scale = target_radius / max(source_radius, 1e-8)
    rotation = np.eye(3, dtype=np.float64)
    translation = target_center.astype(np.float64) - scale * source_center.astype(np.float64)

    history: list[dict[str, Any]] = []
    for iteration in range(int(args.align_iterations)):
        transformed = _transform_points(source_sample, scale, rotation, translation)
        source_corr, target_corr = nearest_correspondences(
            transformed,
            target_sample,
            max_distance=float(args.max_correspondence_distance),
        )
        if len(source_corr) < 64:
            history.append({"iteration": iteration, "correspondences": int(len(source_corr)), "stopped": "too_few"})
            break
        delta_scale, delta_rotation, delta_translation = _estimate_similarity(source_corr, target_corr)
        rotation = delta_rotation @ rotation
        translation = delta_scale * (translation @ delta_rotation.T) + delta_translation
        scale = float(delta_scale * scale)
        residual = np.linalg.norm(
            _transform_points(source_corr, delta_scale, delta_rotation, delta_translation) - target_corr,
            axis=1,
        )
        history.append(
            {
                "iteration": iteration,
                "correspondences": int(len(source_corr)),
                "median_residual": float(np.median(residual)),
                "mean_residual": float(np.mean(residual)),
                "scale": float(scale),
            }
        )
    aligned = _transform_points(points, scale, rotation, translation)
    return aligned.astype(np.float32), {
        "align_mode": "umeyama_icp",
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "history": history,
    }


def project_zbuffer_one_view(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    support: np.ndarray,
    base_points: np.ndarray,
    base_depth: np.ndarray,
    *,
    min_depth: float,
    max_base_distance: float,
    max_depth_residual: float,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = support.shape
    points_cam = world_to_camera(points_world, extrinsic)
    z = points_cam[:, 2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * points_cam[:, 0] / z + cx
        v = fy * points_cam[:, 1] / z + cy
    xi = np.rint(u).astype(np.int32)
    yi = np.rint(v).astype(np.int32)
    valid = (
        np.isfinite(points_cam).all(axis=1)
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(z)
        & (z > float(min_depth))
        & (xi >= 0)
        & (xi < width)
        & (yi >= 0)
        & (yi < height)
    )
    input_inside = int(valid.sum())
    if not valid.any():
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=bool),
            {"projected_inside": 0, "support_hits": 0, "accepted_hits": 0},
        )

    candidates: list[tuple[int, int, int]] = []
    radius = max(0, int(splat_radius))
    valid_indices = np.nonzero(valid)[0]
    for point_idx in valid_indices.tolist():
        px = int(xi[point_idx])
        py = int(yi[point_idx])
        for dy in range(-radius, radius + 1):
            yy = py + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = px + dx
                if xx < 0 or xx >= width:
                    continue
                if not support[yy, xx]:
                    continue
                candidates.append((point_idx, yy, xx))
    if not candidates:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=bool),
            {"projected_inside": input_inside, "support_hits": 0, "accepted_hits": 0},
        )

    idx = np.asarray([row[0] for row in candidates], dtype=np.int64)
    yy = np.asarray([row[1] for row in candidates], dtype=np.int64)
    xx = np.asarray([row[2] for row in candidates], dtype=np.int64)
    candidate_z = z[idx].astype(np.float32)
    candidate_world = points_world[idx].astype(np.float32)
    keep = np.ones(idx.shape[0], dtype=bool)
    if float(max_depth_residual) > 0:
        base_z = base_depth[yy, xx].astype(np.float32)
        keep &= np.isfinite(base_z) & (np.abs(candidate_z - base_z) <= float(max_depth_residual))
    if float(max_base_distance) > 0:
        base = base_points[yy, xx].astype(np.float32)
        dist = np.linalg.norm(candidate_world - base, axis=1)
        keep &= np.isfinite(base).all(axis=1) & np.isfinite(dist) & (dist <= float(max_base_distance))
    if not keep.any():
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width), dtype=bool),
            {"projected_inside": input_inside, "support_hits": int(len(candidates)), "accepted_hits": 0},
        )
    idx = idx[keep]
    yy = yy[keep]
    xx = xx[keep]
    candidate_z = candidate_z[keep]
    pixel_index = yy.astype(np.int64) * width + xx.astype(np.int64)
    order = np.lexsort((candidate_z, pixel_index))
    sorted_pixels = pixel_index[order]
    keep_sorted = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    chosen_idx = idx[order][keep_sorted]
    chosen_pixels = sorted_pixels[keep_sorted]
    out_y = (chosen_pixels // width).astype(np.int64)
    out_x = (chosen_pixels % width).astype(np.int64)

    world_map = np.zeros((height, width, 3), dtype=np.float32)
    depth_map = np.zeros((height, width), dtype=np.float32)
    hit_mask = np.zeros((height, width), dtype=bool)
    world_map[out_y, out_x] = points_world[chosen_idx].astype(np.float32)
    depth_map[out_y, out_x] = z[chosen_idx].astype(np.float32)
    hit_mask[out_y, out_x] = True
    distances = np.linalg.norm(world_map[hit_mask] - base_points[hit_mask], axis=1) if hit_mask.any() else np.array([])
    return world_map, depth_map, hit_mask, {
        "projected_inside": input_inside,
        "support_hits": int(len(candidates)),
        "accepted_hits": int(keep.sum()),
        "zbuffer_pixels": int(hit_mask.sum()),
        "base_distance_percentiles": [float(v) for v in np.percentile(distances, [0, 50, 90, 99, 100])] if distances.size else [],
    }


def write_debug_cloud(path: Path, points: np.ndarray) -> None:
    o3d = _load_open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "predictions.npz"
    summary_path = output_dir / "external_pointcloud_projection_summary.json"

    with np.load(args.base_predictions, allow_pickle=False) as payload:
        base = {key: np.array(payload[key]) for key in payload.files}
    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)
    view_count, height, width = world_points.shape[:3]

    patch_views = parse_views(str(args.views), view_count)
    patch_masks = load_roi_masks(Path(args.scene_dir), view_count, height, width, str(args.patch_roi), int(args.mask_dilate))
    align_masks = load_roi_masks(Path(args.scene_dir), view_count, height, width, str(args.align_roi), 0)
    external_points, source_summary = load_external_points(Path(args.external_pointcloud), args)
    anchor_points = select_anchor_points(
        base,
        align_masks,
        conf_percentile=float(args.anchor_conf_percentile),
        max_points=int(args.max_anchor_points),
        seed=int(args.seed),
    )
    aligned_points, alignment_summary = align_points_to_anchor(external_points, anchor_points, args)

    if bool(args.write_debug_ply):
        debug_count = min(len(aligned_points), 250000)
        rng = np.random.default_rng(int(args.seed))
        debug_points = aligned_points
        if len(debug_points) > debug_count:
            debug_points = debug_points[rng.choice(len(debug_points), size=debug_count, replace=False)]
        write_debug_cloud(output_dir / "external_pointcloud_aligned_debug.ply", debug_points)

    alpha = float(np.clip(float(args.surface_alpha), 0.0, 1.0))
    per_view: dict[str, Any] = {}
    patch_total = 0
    for view_idx in patch_views:
        world_map, depth_map, hit_mask, view_summary = project_zbuffer_one_view(
            aligned_points,
            extrinsic[view_idx],
            intrinsic[view_idx],
            patch_masks[view_idx],
            world_points[view_idx],
            depth[view_idx, ..., 0],
            min_depth=float(args.min_depth),
            max_base_distance=float(args.max_base_distance),
            max_depth_residual=float(args.max_depth_residual),
            splat_radius=int(args.splat_radius),
        )
        if hit_mask.any():
            blended_world = (1.0 - alpha) * world_points[view_idx][hit_mask] + alpha * world_map[hit_mask]
            blended_depth = (1.0 - alpha) * depth[view_idx, ..., 0][hit_mask] + alpha * depth_map[hit_mask]
            world_points[view_idx][hit_mask] = blended_world.astype(np.float32)
            depth[view_idx, ..., 0][hit_mask] = blended_depth.astype(np.float32)
            world_conf[view_idx][hit_mask] = np.maximum(world_conf[view_idx][hit_mask], float(args.confidence_boost))
            depth_conf[view_idx][hit_mask] = np.maximum(depth_conf[view_idx][hit_mask], float(args.depth_confidence_boost))
            patch_total += int(hit_mask.sum())
        view_summary["patched_pixels"] = int(hit_mask.sum())
        view_summary["roi_pixels"] = int(patch_masks[view_idx].sum())
        per_view[str(view_idx)] = view_summary

    base["world_points"] = world_points.astype(np.float32)
    base["world_points_conf"] = world_conf.astype(np.float32)
    base["depth"] = depth.astype(np.float32)
    base["depth_conf"] = depth_conf.astype(np.float32)
    np.savez_compressed(output_npz, **base)

    summary = {
        "base_predictions": str(Path(args.base_predictions).resolve()),
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "external_pointcloud": str(Path(args.external_pointcloud).resolve()),
        "output_npz": str(output_npz),
        "patch_roi": str(args.patch_roi),
        "align_roi": str(args.align_roi),
        "patch_views": patch_views,
        "surface_alpha": float(alpha),
        "max_base_distance": float(args.max_base_distance),
        "max_depth_residual": float(args.max_depth_residual),
        "splat_radius": int(args.splat_radius),
        "source": source_summary,
        "anchor_points": int(len(anchor_points)),
        "alignment": alignment_summary,
        "patched_pixels_total": int(patch_total),
        "patched_pixels_per_view": per_view,
        "truthful_status": "local_external_pointcloud_projection_probe_not_mentor_final",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if patch_total > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
