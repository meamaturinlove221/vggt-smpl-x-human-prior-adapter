from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from vggt.utils.normal_refiner import point_map_to_normal_numpy, points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a local candidate with a 4K4D annotated face-keypoint surface. "
            "This uses dataset 3D keypoints to add view-local face relief without "
            "turning the whole head into an SMPL-X template."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--annotations-smc", default="", help="Defaults to scene_manifest annotations_smc when present.")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--view-indices", default="", help="Comma-separated view indices. Empty means all views.")
    parser.add_argument("--keypoint-start", type=int, default=55)
    parser.add_argument("--keypoint-end", type=int, default=125)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.05)
    parser.add_argument("--min-landmarks", type=int, default=18)
    parser.add_argument("--sample-radius", type=int, default=3)
    parser.add_argument("--roi-dilate", type=int, default=5)
    parser.add_argument("--surface-alpha", type=float, default=0.85)
    parser.add_argument("--relief-scale", type=float, default=1.0)
    parser.add_argument("--max-depth-delta", type=float, default=0.085)
    parser.add_argument("--conf-boost", type=float, default=0.0)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def _resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    # Some manifests were written through a non-UTF8 console and contain mojibake
    # for "G:\数据集". Search the mounted drive instead of trusting that string.
    name = path.name
    for root_name in ("G:\\", "D:\\"):
        root = Path(root_name)
        if not root.exists():
            continue
        for child in root.iterdir():
            candidate = child / "datasets" / "data_used_in_4K4D" / "annotations" / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path_text)


def _load_manifest(scene_dir: Path) -> dict[str, Any]:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _parse_view_indices(spec: str, view_count: int) -> list[int]:
    if not spec.strip():
        return list(range(view_count))
    out: list[int] = []
    for piece in spec.split(","):
        item = piece.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= view_count:
            raise ValueError(f"view index {idx} outside [0, {view_count})")
        out.append(idx)
    return sorted(set(out))


def _camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    return (points_cam - translation[None, None, :]) @ rotation


def _camera_grid_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xx - cx) * depth / fx
    y = (yy - cy) * depth / fy
    return np.stack((x, y, depth), axis=-1).astype(np.float32)


def _normalize_vectors(values: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1)
    valid = np.isfinite(vectors).all(axis=-1) & (norms > eps)
    out = np.zeros_like(vectors, dtype=np.float32)
    out[valid] = vectors[valid] / norms[valid, None]
    return out, valid


def _update_normals(normal: np.ndarray, world_points: np.ndarray, extrinsic: np.ndarray, patch_mask: np.ndarray) -> tuple[np.ndarray, int]:
    out = normal.copy()
    update_count = 0
    for view_idx in range(world_points.shape[0]):
        cam = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
        finite = np.isfinite(cam).all(axis=-1)
        point_normal, surface_valid = point_map_to_normal_numpy(cam, finite)
        point_normal, vector_valid = _normalize_vectors(point_normal)
        valid = patch_mask[view_idx] & surface_valid & vector_valid
        out[view_idx][valid] = -point_normal[valid]
        update_count += int(valid.sum())
    out, _ = _normalize_vectors(out)
    return out.astype(np.float32), update_count


def _scene_xy_from_raw(
    raw_xy: np.ndarray,
    *,
    view_meta: dict[str, Any],
    output_width: int,
    output_height: int,
) -> np.ndarray:
    raw_h, raw_w = view_meta.get("source_image_size", view_meta.get("original_source_image_size", [output_height, output_width]))
    preprocess = dict(view_meta.get("preprocess_meta", {}))
    aligned_size = preprocess.get("source_aligned_size") or preprocess.get("aligned_source_size") or view_meta.get("image_size")
    aligned_w, aligned_h = float(aligned_size[0]), float(aligned_size[1])
    bbox = preprocess.get("crop_bbox_xyxy")
    if bbox is None:
        scale_x = output_width / max(float(raw_w), 1.0)
        scale_y = output_height / max(float(raw_h), 1.0)
        return raw_xy * np.array([scale_x, scale_y], dtype=np.float32)[None]
    x0, y0, x1, y1 = [float(value) for value in bbox]
    crop_size = max(x1 - x0, y1 - y0, 1.0)
    full = raw_xy * np.array([aligned_w / max(float(raw_w), 1.0), aligned_h / max(float(raw_h), 1.0)], dtype=np.float32)[None]
    return (full - np.array([x0, y0], dtype=np.float32)[None]) * np.array(
        [output_width / crop_size, output_height / crop_size], dtype=np.float32
    )[None]


def _sample_depth(depth: np.ndarray, xy: np.ndarray, radius: int) -> np.ndarray:
    height, width = depth.shape
    samples = []
    for x_float, y_float in xy:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        samples.append(float(np.median(valid)) if valid.size else np.nan)
    return np.asarray(samples, dtype=np.float32)


def _fit_keypoint_depth(real_z: np.ndarray, base_z: np.ndarray, relief_scale: float, max_delta: float) -> np.ndarray:
    valid = np.isfinite(real_z) & np.isfinite(base_z) & (base_z > 0.05)
    if int(valid.sum()) < 8:
        return np.full_like(real_z, np.nan, dtype=np.float32)
    rz = real_z[valid].astype(np.float64)
    bz = base_z[valid].astype(np.float64)
    rz_med = float(np.median(rz))
    bz_med = float(np.median(bz))
    rz_center = rz - rz_med
    bz_center = bz - bz_med
    denom = float(np.sum(rz_center * rz_center))
    slope = float(np.sum(rz_center * bz_center) / denom) if denom > 1e-10 else 0.0
    if not np.isfinite(slope) or abs(slope) < 1e-5:
        slope = float(np.std(bz) / max(np.std(rz), 1e-6))
    slope = float(np.clip(slope, -2.5, 2.5))
    target = bz_med + float(relief_scale) * slope * (real_z.astype(np.float64) - rz_med)
    # Keep the existing VGGT surface as the coarse anchor; only inject the
    # annotated face relief locally.
    target = np.where(np.isfinite(base_z), base_z.astype(np.float64) + np.clip(target - base_z, -max_delta, max_delta), target)
    return target.astype(np.float32)


def _interpolate_surface_depth(
    xy: np.ndarray,
    z: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_points = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy_valid = xy[valid_points].astype(np.float64)
    z_valid = z[valid_points].astype(np.float64)
    if xy_valid.shape[0] < 3:
        return np.zeros(support.shape, dtype=np.float32), np.zeros(support.shape, dtype=bool)
    tri = Delaunay(xy_valid)
    ys, xs = np.nonzero(support)
    query = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    simplex = tri.find_simplex(query)
    inside = simplex >= 0
    if not inside.any():
        return np.zeros(support.shape, dtype=np.float32), np.zeros(support.shape, dtype=bool)
    query_inside = query[inside]
    simplex_inside = simplex[inside]
    transform = tri.transform[simplex_inside]
    delta = query_inside - transform[:, 2]
    bary = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
    bary = np.c_[bary, 1.0 - bary.sum(axis=1)]
    vertices = tri.simplices[simplex_inside]
    depth_values = np.sum(z_valid[vertices] * bary, axis=1)
    out = np.zeros(support.shape, dtype=np.float32)
    mask = np.zeros(support.shape, dtype=bool)
    out_ys = ys[inside]
    out_xs = xs[inside]
    finite = np.isfinite(depth_values)
    out[out_ys[finite], out_xs[finite]] = depth_values[finite].astype(np.float32)
    mask[out_ys[finite], out_xs[finite]] = True
    return out, mask


def _save_debug(path: Path, rgb: np.ndarray, roi: np.ndarray, patch: np.ndarray, xy: np.ndarray) -> None:
    overlay = rgb.astype(np.float32).copy()
    overlay[roi] = overlay[roi] * 0.72 + np.array([30, 120, 255], dtype=np.float32) * 0.28
    overlay[patch] = overlay[patch] * 0.45 + np.array([0, 230, 80], dtype=np.float32) * 0.55
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for x, y in xy:
        if np.isfinite(x + y):
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_predictions)
    scene_dir = Path(args.scene_dir)
    output_npz = Path(args.output_npz)
    output_summary = Path(args.output_summary) if args.output_summary else output_npz.with_suffix(".json")
    manifest = _load_manifest(scene_dir)
    annotations_path = _resolve_existing_path(args.annotations_smc or str(manifest.get("annotations_smc", "")))

    with np.load(base_path, allow_pickle=False) as payload:
        base = {key: np.asarray(payload[key]) for key in payload.files}
    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    view_indices = _parse_view_indices(str(args.view_indices), view_count)
    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    per_view: dict[str, Any] = {}
    kernel = np.ones((3, 3), dtype=np.uint8)

    with h5py.File(annotations_path, "r") as annot:
        keypoints_3d = np.asarray(annot["Keypoints_3D/keypoints3d"][int(args.frame)], dtype=np.float32)
        start = int(args.keypoint_start)
        end = min(int(args.keypoint_end), keypoints_3d.shape[0])
        face_indices = np.arange(start, end, dtype=np.int64)
        face_points_world = keypoints_3d[face_indices, :3]
        face_conf = keypoints_3d[face_indices, 3]

        for view_idx in view_indices:
            view_meta = manifest["exported_views"][view_idx]
            camera_id = str(view_meta["camera_id"])
            K = np.asarray(annot[f"Camera_Parameter/{camera_id}/K"], dtype=np.float64)
            RT = np.asarray(annot[f"Camera_Parameter/{camera_id}/RT"], dtype=np.float64)
            real_cam = (np.linalg.inv(RT) @ np.c_[face_points_world, np.ones(len(face_points_world))].T).T[:, :3]
            raw_uv_h = (K @ real_cam.T).T
            raw_uv = raw_uv_h[:, :2] / np.clip(raw_uv_h[:, 2:3], 1e-8, None)
            scene_xy = _scene_xy_from_raw(raw_uv.astype(np.float32), view_meta=view_meta, output_width=width, output_height=height)

            scene = load_scene_view(scene_dir, view_idx, (height, width))
            roi = build_roi_masks(scene.mask.astype(bool))["face"]
            if int(args.roi_dilate) > 0:
                roi_support = cv2.dilate(roi.astype(np.uint8), kernel, iterations=int(args.roi_dilate)).astype(bool)
                roi_support &= scene.mask.astype(bool)
            else:
                roi_support = roi.copy()
            in_frame = (
                (scene_xy[:, 0] >= 0)
                & (scene_xy[:, 0] < width)
                & (scene_xy[:, 1] >= 0)
                & (scene_xy[:, 1] < height)
                & (real_cam[:, 2] > 0.05)
                & (face_conf >= float(args.min_keypoint_conf))
            )
            xy_int = np.round(scene_xy).astype(np.int32)
            in_support = np.zeros(len(scene_xy), dtype=bool)
            good = in_frame.copy()
            good_indices = np.nonzero(good)[0]
            if good_indices.size:
                xs = np.clip(xy_int[good_indices, 0], 0, width - 1)
                ys = np.clip(xy_int[good_indices, 1], 0, height - 1)
                in_support[good_indices] = roi_support[ys, xs]
            selected = in_frame & in_support

            current_depth = depth[view_idx, ..., 0] if depth.ndim == 4 and depth.shape[-1] == 1 else depth[view_idx]
            sampled_base = _sample_depth(current_depth, scene_xy[selected], radius=int(args.sample_radius))
            target_landmark_depth = _fit_keypoint_depth(
                real_cam[selected, 2],
                sampled_base,
                relief_scale=float(args.relief_scale),
                max_delta=float(args.max_depth_delta),
            )
            valid_target = np.isfinite(target_landmark_depth)
            if int(valid_target.sum()) < int(args.min_landmarks):
                per_view[str(view_idx)] = {
                    "camera_id": camera_id,
                    "selected_landmarks": int(selected.sum()),
                    "valid_landmarks": int(valid_target.sum()),
                    "patch_pixels": 0,
                    "skipped": "too_few_landmarks",
                }
                continue
            surface_depth, surface_mask = _interpolate_surface_depth(
                scene_xy[selected][valid_target],
                target_landmark_depth[valid_target],
                roi_support,
            )
            finite_surface = surface_mask & np.isfinite(current_depth) & (current_depth > 0.05) & (surface_depth > 0.05)
            delta = np.clip(surface_depth - current_depth, -float(args.max_depth_delta), float(args.max_depth_delta))
            new_depth = current_depth.copy()
            new_depth[finite_surface] = (
                (1.0 - float(args.surface_alpha)) * current_depth[finite_surface]
                + float(args.surface_alpha) * (current_depth[finite_surface] + delta[finite_surface])
            )

            cam_grid = _camera_grid_from_depth(new_depth, intrinsic[view_idx])
            new_world = _camera_to_world(cam_grid, extrinsic[view_idx])
            world_points[view_idx][finite_surface] = new_world[finite_surface]
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth[view_idx, ..., 0][finite_surface] = new_depth[finite_surface]
            else:
                depth[view_idx][finite_surface] = new_depth[finite_surface]
            patch_mask[view_idx] = finite_surface
            if float(args.conf_boost) > 0:
                for key in ("world_points_conf", "depth_conf", "point_conf", "normal_conf"):
                    if key in base:
                        arr = np.asarray(base[key]).copy()
                        arr[view_idx][finite_surface] = np.maximum(arr[view_idx][finite_surface], float(args.conf_boost))
                        base[key] = arr
            per_view[str(view_idx)] = {
                "camera_id": camera_id,
                "selected_landmarks": int(selected.sum()),
                "valid_landmarks": int(valid_target.sum()),
                "patch_pixels": int(finite_surface.sum()),
                "depth_delta_percentiles": [float(v) for v in np.percentile(delta[finite_surface], [0, 25, 50, 75, 100])]
                if finite_surface.any()
                else [],
            }
            if bool(args.write_debug):
                _save_debug(output_npz.parent / f"debug_view_{view_idx:02d}.png", scene.rgb, roi_support, finite_surface, scene_xy[selected])

    normal, normal_update_count = _update_normals(normal, world_points, extrinsic, patch_mask)

    out: dict[str, Any] = {key: value for key, value in base.items()}
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    summary = {
        "base_predictions": str(base_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "annotations_smc": str(annotations_path.resolve()),
        "output_npz": str(output_npz.resolve()),
        "frame": int(args.frame),
        "keypoint_range": [int(args.keypoint_start), int(args.keypoint_end)],
        "view_indices": view_indices,
        "surface_alpha": float(args.surface_alpha),
        "relief_scale": float(args.relief_scale),
        "max_depth_delta": float(args.max_depth_delta),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "confidence_boost": float(args.conf_boost),
        "truthful_status": "local_4k4d_face_keypoint_surface_probe_not_final_pass",
        "per_view": per_view,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
