from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

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
            "Replace only the face ROI with a conservative, smoothed depth/point "
            "surface. This is a diagnostic to test whether the Open3D face failure "
            "is mostly a multi-view shell/low-continuity issue."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-summary", default="", type=Path)
    parser.add_argument("--view-indices", default="")
    parser.add_argument("--roi-dilate", type=int, default=2)
    parser.add_argument("--smooth-ksize", type=int, default=17)
    parser.add_argument("--smooth-sigma", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-depth-delta", type=float, default=0.055)
    parser.add_argument("--confidence-boost", type=float, default=185.0)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_view_indices(spec: str, view_count: int) -> np.ndarray:
    selected = np.zeros((view_count,), dtype=bool)
    if not str(spec).strip():
        selected[:] = True
        return selected
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= view_count:
            raise ValueError(f"view index {idx} outside [0, {view_count})")
        selected[idx] = True
    if not selected.any():
        raise ValueError("--view-indices selected nothing")
    return selected


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def camera_grid_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xx - cx) * depth / fx
    y = (yy - cy) * depth / fy
    return np.stack((x, y, depth), axis=-1).astype(np.float32)


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    return (points_cam - translation[None, None, :]) @ rotation


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (norms > eps)
    out = np.zeros_like(values, dtype=np.float32)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def smooth_depth_in_roi(depth: np.ndarray, roi: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    roi = np.asarray(roi, dtype=bool)
    valid = roi & np.isfinite(depth) & (depth > 0.05)
    if not valid.any():
        return depth.copy()
    median = float(np.median(depth[valid]))
    filled = depth.astype(np.float32).copy()
    filled[~valid] = median
    k = max(3, int(ksize) | 1)
    weight = valid.astype(np.float32)
    numer = cv2.GaussianBlur(filled * weight, (k, k), float(sigma))
    denom = cv2.GaussianBlur(weight, (k, k), float(sigma))
    smooth = numer / np.maximum(denom, 1e-6)
    out = depth.copy()
    out[roi] = smooth[roi]
    return out


def recompute_normals(normal: np.ndarray, world_points: np.ndarray, extrinsic: np.ndarray, patch_mask: np.ndarray) -> tuple[np.ndarray, int]:
    out = normal.copy()
    replaced = 0
    for view_idx in range(world_points.shape[0]):
        cam = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
        finite = np.isfinite(cam).all(axis=-1)
        point_normal, surface_valid = point_map_to_normal_numpy(cam, finite)
        point_normal, vector_valid = normalize_vectors(point_normal)
        use = np.asarray(patch_mask[view_idx], dtype=bool) & surface_valid & vector_valid
        out[view_idx][use] = -point_normal[use]
        replaced += int(use.sum())
    out, _ = normalize_vectors(out)
    return out.astype(np.float32), replaced


def write_debug(path: Path, rgb: np.ndarray, roi: np.ndarray, patch: np.ndarray) -> None:
    arr = rgb.astype(np.float32).copy()
    arr[roi] = arr[roi] * 0.65 + np.asarray([30, 120, 255], dtype=np.float32) * 0.35
    arr[patch] = arr[patch] * 0.35 + np.asarray([20, 235, 90], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth2 = depth[..., 0].copy() if depth.ndim == 4 else depth.copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    selected_views = parse_view_indices(args.view_indices, view_count)
    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    per_view: list[dict[str, Any]] = []
    kernel = np.ones((3, 3), dtype=np.uint8)

    for view_idx in range(view_count):
        if not bool(selected_views[view_idx]):
            per_view.append({"view_index": int(view_idx), "skipped": True, "reason": "view_not_selected"})
            continue
        scene = load_scene_view(args.scene_dir, view_idx, (height, width))
        roi = build_roi_masks(scene.mask.astype(bool))["face"]
        if int(args.roi_dilate) > 0:
            roi = cv2.dilate(roi.astype(np.uint8), kernel, iterations=int(args.roi_dilate)).astype(bool)
            roi &= scene.mask.astype(bool)
        current_depth = depth2[view_idx]
        smooth = smooth_depth_in_roi(current_depth, roi, int(args.smooth_ksize), float(args.smooth_sigma))
        delta = np.clip(smooth - current_depth, -float(args.max_depth_delta), float(args.max_depth_delta))
        new_depth = current_depth.copy()
        finite = roi & np.isfinite(current_depth) & (current_depth > 0.05)
        new_depth[finite] = current_depth[finite] + float(args.alpha) * delta[finite]
        cam_grid = camera_grid_from_depth(new_depth, intrinsic[view_idx])
        new_world = camera_to_world(cam_grid, extrinsic[view_idx])
        world_points[view_idx][finite] = new_world[finite]
        depth2[view_idx][finite] = new_depth[finite]
        world_conf[view_idx][finite] = np.maximum(world_conf[view_idx][finite], float(args.confidence_boost))
        depth_conf[view_idx][finite] = np.maximum(depth_conf[view_idx][finite], float(args.confidence_boost))
        normal_conf[view_idx][finite] = np.maximum(normal_conf[view_idx][finite], 1.0)
        patch_mask[view_idx] = finite
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": str(json.loads((args.scene_dir / "scene_manifest.json").read_text(encoding="utf-8"))["exported_views"][view_idx].get("camera_id")),
                "roi_pixels": int(roi.sum()),
                "patch_pixels": int(finite.sum()),
                "delta_percentiles": [float(v) for v in np.percentile(delta[finite], [0, 25, 50, 75, 100])] if finite.any() else [],
            }
        )
        if bool(args.write_debug):
            write_debug(args.output_npz.parent / "overlays" / f"view_{view_idx:02d}_face_smooth_fill.png", scene.rgb, roi, finite)

    normal, normal_replaced = recompute_normals(normal, world_points, extrinsic, patch_mask)
    out = dict(base)
    out["world_points"] = world_points.astype(np.float32)
    out["world_points_conf"] = world_conf.astype(np.float32)
    out["depth"] = depth2[..., None].astype(np.float32) if depth.ndim == 4 else depth2.astype(np.float32)
    out["depth_conf"] = depth_conf.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    out["normal_conf"] = normal_conf.astype(np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_npz": str(args.output_npz.resolve()),
        "roi_dilate": int(args.roi_dilate),
        "smooth_ksize": int(args.smooth_ksize),
        "smooth_sigma": float(args.smooth_sigma),
        "alpha": float(args.alpha),
        "max_depth_delta": float(args.max_depth_delta),
        "confidence_boost": float(args.confidence_boost),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_replaced": int(normal_replaced),
        "per_view": per_view,
        "truthful_status": "local_face_roi_depth_smooth_fill_diagnostic_not_final_pass",
    }
    output_summary = args.output_summary if str(args.output_summary) else args.output_npz.with_suffix(".json")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
