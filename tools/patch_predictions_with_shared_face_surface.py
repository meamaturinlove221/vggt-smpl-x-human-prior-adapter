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
from patch_predictions_with_image_face_relief import (  # noqa: E402
    camera_grid_from_depth,
    camera_to_world,
    relief_for_skin,
    skin_mask_for_view,
    update_predicted_normals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one image-guided face relief surface in a source view, then "
            "project that same 3D surface into selected views. This is a local "
            "geometry-teacher probe: confidence is preserved by default so fixed "
            "threshold checks cannot pass through calibration alone."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--source-view", type=int, default=3)
    parser.add_argument("--target-views", default="2,3", help="Comma-separated views to receive the shared surface.")
    parser.add_argument("--nose-mode", choices=("center", "left", "right"), default="center")
    parser.add_argument("--face-bulge", type=float, default=0.010)
    parser.add_argument("--nose-bulge", type=float, default=0.018)
    parser.add_argument("--eye-depression", type=float, default=0.003)
    parser.add_argument("--mouth-depression", type=float, default=0.002)
    parser.add_argument("--max-face-y-frac", type=float, default=0.88)
    parser.add_argument("--skin-dilate", type=int, default=1)
    parser.add_argument("--skin-erode", type=int, default=0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--min-source-pixels", type=int, default=512)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def parse_views(spec: str, view_count: int) -> list[int]:
    views: list[int] = []
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        view_idx = int(item)
        if view_idx < 0 or view_idx >= view_count:
            raise ValueError(f"view index {view_idx} outside [0, {view_count})")
        views.append(view_idx)
    if not views:
        raise ValueError("--target-views must select at least one view")
    return sorted(set(views))


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    return points_world @ rotation.T + translation[None, :]


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    u = fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + cy
    return u, v, z


def crop_lower_face_roi(roi: np.ndarray, max_face_y_frac: float) -> np.ndarray:
    out = np.asarray(roi, dtype=bool).copy()
    ys = np.nonzero(out)[0]
    if ys.size:
        y_limit = int(ys.min() + float(max_face_y_frac) * max(1, ys.max() - ys.min() + 1))
        out &= np.arange(out.shape[0])[:, None] <= y_limit
    return out


def save_debug(path: Path, rgb: np.ndarray, support: np.ndarray, selected: np.ndarray) -> None:
    overlay = rgb.astype(np.float32).copy()
    overlay[support] = overlay[support] * 0.75 + np.array([40, 80, 255], dtype=np.float32) * 0.25
    overlay[selected] = overlay[selected] * 0.35 + np.array([255, 60, 40], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def splat_surface_to_view(
    *,
    view_idx: int,
    surface_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    support: np.ndarray,
    radius: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_cam = world_to_camera(surface_world, extrinsic)
    u, v, z = project_points(points_cam, intrinsic)
    finite = np.isfinite(points_cam).all(axis=1) & np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0.05)
    if not np.any(finite):
        return np.zeros((height, width), dtype=bool), np.zeros((height, width), dtype=np.float32), np.zeros((height, width, 3), dtype=np.float32)

    kept_z = np.full((height, width), np.inf, dtype=np.float32)
    kept_points = np.zeros((height, width, 3), dtype=np.float32)
    selected = np.zeros((height, width), dtype=bool)
    radius = max(0, int(radius))

    valid_indices = np.nonzero(finite)[0]
    for point_idx in valid_indices:
        px = int(round(float(u[point_idx])))
        py = int(round(float(v[point_idx])))
        depth_value = float(z[point_idx])
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
                if depth_value < kept_z[yy, xx]:
                    kept_z[yy, xx] = depth_value
                    kept_points[yy, xx] = surface_world[point_idx]
                    selected[yy, xx] = True

    kept_z[~selected] = 0.0
    return selected, kept_z, kept_points


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_predictions)
    scene_dir = Path(args.scene_dir)
    output_npz = Path(args.output_npz)
    output_summary = Path(args.output_summary) if args.output_summary else output_npz.with_suffix(".json")

    base = np.load(base_path, allow_pickle=False)
    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    source_view = int(args.source_view)
    if source_view < 0 or source_view >= view_count:
        raise ValueError(f"--source-view {source_view} outside [0, {view_count})")
    target_views = parse_views(args.target_views, view_count)

    source_scene = load_scene_view(scene_dir, source_view, (height, width))
    source_roi = crop_lower_face_roi(build_roi_masks(source_scene.mask.astype(bool))["face"], float(args.max_face_y_frac))
    source_skin = skin_mask_for_view(
        source_scene.rgb,
        source_scene.mask.astype(bool),
        source_roi,
        skin_dilate=int(args.skin_dilate),
        skin_erode=int(args.skin_erode),
    )
    source_depth = depth[source_view, ..., 0] if depth.ndim == 4 and depth.shape[-1] == 1 else depth[source_view]
    source_valid = source_skin & np.isfinite(source_depth) & (source_depth > 0.05)
    if int(source_valid.sum()) < int(args.min_source_pixels):
        raise RuntimeError(f"Source view selected only {int(source_valid.sum())} pixels")

    relief = relief_for_skin(
        source_scene.rgb,
        source_valid,
        nose_mode=str(args.nose_mode),
        face_bulge=float(args.face_bulge),
        nose_bulge=float(args.nose_bulge),
        eye_depression=float(args.eye_depression),
        mouth_depression=float(args.mouth_depression),
    )
    source_new_depth = np.maximum(source_depth + relief, 0.05)
    source_cam_grid = camera_grid_from_depth(source_new_depth, intrinsic[source_view])
    source_world_grid = camera_to_world(source_cam_grid, extrinsic[source_view])
    surface_world = source_world_grid[source_valid].astype(np.float32)

    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    patch_stats: dict[str, Any] = {}
    for view_idx in target_views:
        view_scene = load_scene_view(scene_dir, view_idx, (height, width))
        view_face = crop_lower_face_roi(build_roi_masks(view_scene.mask.astype(bool))["face"], float(args.max_face_y_frac))
        # Keep the projection in the semantic face region. The skin support keeps
        # profile views from splatting onto hair or shirt when the generic face
        # surface falls outside visible skin.
        view_skin = skin_mask_for_view(
            view_scene.rgb,
            view_scene.mask.astype(bool),
            view_face,
            skin_dilate=max(0, int(args.skin_dilate)),
            skin_erode=int(args.skin_erode),
        )
        support = view_face & (view_skin | (view_idx == source_view))
        if view_idx == source_view:
            support = source_valid
        selected, z_map, point_map = splat_surface_to_view(
            view_idx=view_idx,
            surface_world=surface_world,
            extrinsic=extrinsic[view_idx],
            intrinsic=intrinsic[view_idx],
            support=support,
            radius=int(args.splat_radius),
            height=height,
            width=width,
        )
        if view_idx == source_view:
            selected = source_valid
            z_map = source_new_depth.astype(np.float32)
            point_map = source_world_grid.astype(np.float32)

        world_points[view_idx][selected] = point_map[selected]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth[view_idx, ..., 0][selected] = z_map[selected]
        else:
            depth[view_idx][selected] = z_map[selected]
        patch_mask[view_idx] = selected

        patch_stats[str(view_idx)] = {
            "support_pixels": int(support.sum()),
            "patched_pixels": int(selected.sum()),
        }
        if bool(args.write_debug):
            save_debug(output_npz.parent / f"debug_projected_view_{view_idx:02d}.png", view_scene.rgb, support, selected)

    normal, normal_update_count = update_predicted_normals(normal, world_points, extrinsic, patch_mask)

    out: dict[str, Any] = {key: np.asarray(base[key]) for key in base.files}
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    summary = {
        "base_predictions": str(base_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_npz": str(output_npz.resolve()),
        "source_view": source_view,
        "target_views": target_views,
        "source_pixels": int(source_valid.sum()),
        "surface_points": int(surface_world.shape[0]),
        "face_bulge": float(args.face_bulge),
        "nose_bulge": float(args.nose_bulge),
        "eye_depression": float(args.eye_depression),
        "mouth_depression": float(args.mouth_depression),
        "splat_radius": int(args.splat_radius),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "confidence_preserved": True,
        "patch_stats": patch_stats,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
