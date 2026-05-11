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
            "Create a local image-guided face relief candidate by editing depth/world points "
            "inside skin-colored face ROI pixels. Confidence is preserved so fixed-threshold "
            "audits cannot be faked by this tool."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--view-indices", required=True, help="Comma-separated views to patch.")
    parser.add_argument(
        "--nose-modes",
        default="",
        help="Optional comma-separated view:mode entries, where mode is center,left,right. Example: 2:left,3:center",
    )
    parser.add_argument("--face-bulge", type=float, default=0.018, help="Base convex face relief in camera-depth units.")
    parser.add_argument("--nose-bulge", type=float, default=0.025, help="Extra nose relief in camera-depth units.")
    parser.add_argument("--eye-depression", type=float, default=0.006, help="Small dark-feature recession.")
    parser.add_argument("--mouth-depression", type=float, default=0.004, help="Small lower dark-feature recession.")
    parser.add_argument("--max-face-y-frac", type=float, default=0.9, help="Drop lower neck pixels within the face box.")
    parser.add_argument("--skin-dilate", type=int, default=1)
    parser.add_argument("--skin-erode", type=int, default=0)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def parse_view_indices(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for piece in spec.split(","):
        item = piece.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= view_count:
            raise ValueError(f"view index {idx} outside [0, {view_count})")
        out.append(idx)
    if not out:
        raise ValueError("--view-indices must select at least one view")
    return sorted(set(out))


def parse_nose_modes(spec: str) -> dict[int, str]:
    modes: dict[int, str] = {}
    if not spec.strip():
        return modes
    for piece in spec.split(","):
        item = piece.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        mode = right.strip().lower()
        if mode not in {"center", "left", "right"}:
            raise ValueError(f"Unsupported nose mode: {mode}")
        modes[int(left.strip())] = mode
    return modes


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (norms > eps)
    out = np.zeros_like(values, dtype=np.float32)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    return (points_cam - translation[None, None, :]) @ rotation


def camera_grid_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
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


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    return labels == best


def skin_mask_for_view(rgb: np.ndarray, support: np.ndarray, roi: np.ndarray, skin_dilate: int, skin_erode: int) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin = (cr > 130) & (cr < 180) & (cb > 70) & (cb < 135) & (y > 40)
    skin &= support & roi
    kernel = np.ones((3, 3), dtype=np.uint8)
    if skin_dilate > 0:
        skin = cv2.dilate(skin.astype(np.uint8), kernel, iterations=int(skin_dilate)).astype(bool)
    if skin_erode > 0:
        skin = cv2.erode(skin.astype(np.uint8), kernel, iterations=int(skin_erode)).astype(bool)
    skin &= support & roi
    if int(skin.sum()) > 0:
        skin = largest_component(skin)
    return skin


def relief_for_skin(
    rgb: np.ndarray,
    skin: np.ndarray,
    nose_mode: str,
    face_bulge: float,
    nose_bulge: float,
    eye_depression: float,
    mouth_depression: float,
) -> np.ndarray:
    relief = np.zeros(skin.shape, dtype=np.float32)
    ys, xs = np.nonzero(skin)
    if ys.size < 64:
        return relief
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.50 * y0 + 0.50 * y1
    sx = max(0.5 * (x1 - x0), 1.0)
    sy = max(0.5 * (y1 - y0), 1.0)
    yy, xx = np.meshgrid(
        np.arange(skin.shape[0], dtype=np.float32),
        np.arange(skin.shape[1], dtype=np.float32),
        indexing="ij",
    )
    nx = (xx - cx) / sx
    ny = (yy - cy) / sy
    face = np.exp(-0.5 * ((nx / 0.95) ** 2 + (ny / 1.05) ** 2))
    if nose_mode == "left":
        nose_x = -0.48
    elif nose_mode == "right":
        nose_x = 0.48
    else:
        nose_x = 0.08
    nose_y = -0.05
    nose = np.exp(-0.5 * (((nx - nose_x) / 0.23) ** 2 + ((ny - nose_y) / 0.34) ** 2))

    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dark = skin & (gray < np.percentile(gray[skin], 42.0))
    eye_band = dark & (ny > -0.45) & (ny < 0.10)
    mouth_band = dark & (ny > 0.18) & (ny < 0.65)

    # Smaller depth means closer to the camera. The face/nose terms create local
    # relief; dark eye/mouth bands are allowed to recede slightly.
    relief[skin] -= float(face_bulge) * face[skin]
    relief[skin] -= float(nose_bulge) * nose[skin]
    relief[eye_band] += float(eye_depression)
    relief[mouth_band] += float(mouth_depression)
    return relief


def update_predicted_normals(
    normal: np.ndarray,
    world_points: np.ndarray,
    extrinsic: np.ndarray,
    patch_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    out = normal.copy()
    update_count = 0
    for view_idx in range(world_points.shape[0]):
        cam = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
        finite = np.isfinite(cam).all(axis=-1)
        point_normal, surface_valid = point_map_to_normal_numpy(cam, finite)
        point_normal, vector_valid = normalize_vectors(point_normal)
        valid = patch_mask[view_idx] & surface_valid & vector_valid
        out[view_idx][valid] = -point_normal[valid]
        update_count += int(valid.sum())
    out, _ = normalize_vectors(out)
    return out.astype(np.float32), update_count


def save_debug_overlay(path: Path, rgb: np.ndarray, roi: np.ndarray, skin: np.ndarray, relief: np.ndarray) -> None:
    overlay = rgb.astype(np.float32).copy()
    overlay[roi] = overlay[roi] * 0.75 + np.array([40, 80, 255], dtype=np.float32) * 0.25
    overlay[skin] = overlay[skin] * 0.45 + np.array([255, 180, 70], dtype=np.float32) * 0.55
    strong = relief < np.percentile(relief[skin], 20.0) if np.any(skin) else np.zeros_like(skin)
    overlay[strong] = overlay[strong] * 0.35 + np.array([255, 40, 40], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


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
    view_indices = parse_view_indices(str(args.view_indices), view_count)
    nose_modes = parse_nose_modes(str(args.nose_modes))
    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    relief_stats: dict[str, Any] = {}

    for view_idx in view_indices:
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        roi = build_roi_masks(scene.mask.astype(bool))["face"]
        ys = np.nonzero(roi)[0]
        if ys.size:
            y_limit = int(ys.min() + float(args.max_face_y_frac) * max(1, ys.max() - ys.min() + 1))
            roi = roi & (np.arange(height)[:, None] <= y_limit)
        skin = skin_mask_for_view(
            scene.rgb,
            scene.mask.astype(bool),
            roi,
            skin_dilate=int(args.skin_dilate),
            skin_erode=int(args.skin_erode),
        )
        relief = relief_for_skin(
            scene.rgb,
            skin,
            nose_mode=nose_modes.get(view_idx, "center"),
            face_bulge=float(args.face_bulge),
            nose_bulge=float(args.nose_bulge),
            eye_depression=float(args.eye_depression),
            mouth_depression=float(args.mouth_depression),
        )
        current_depth = (
            depth[view_idx, ..., 0].copy()
            if depth.ndim == 4 and depth.shape[-1] == 1
            else depth[view_idx].copy()
        )
        valid = skin & np.isfinite(current_depth) & (current_depth > 0.05)
        new_depth = np.maximum(current_depth + relief, 0.05)
        cam_points = camera_grid_from_depth(new_depth, intrinsic[view_idx])
        new_world = camera_to_world(cam_points, extrinsic[view_idx])
        world_points[view_idx][valid] = new_world[valid]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth[view_idx, ..., 0][valid] = new_depth[valid]
        else:
            depth[view_idx][valid] = new_depth[valid]
        patch_mask[view_idx] = valid
        selected_relief = relief[valid]
        relief_stats[str(view_idx)] = {
            "roi_pixels": int(roi.sum()),
            "skin_pixels": int(skin.sum()),
            "patch_pixels": int(valid.sum()),
            "nose_mode": nose_modes.get(view_idx, "center"),
            "relief_min": float(selected_relief.min()) if selected_relief.size else 0.0,
            "relief_mean": float(selected_relief.mean()) if selected_relief.size else 0.0,
            "relief_max": float(selected_relief.max()) if selected_relief.size else 0.0,
        }
        if bool(args.write_debug):
            save_debug_overlay(output_npz.parent / f"debug_view_{view_idx:02d}.png", scene.rgb, roi, valid, relief)

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
        "view_indices": view_indices,
        "nose_modes": {str(k): v for k, v in sorted(nose_modes.items())},
        "face_bulge": float(args.face_bulge),
        "nose_bulge": float(args.nose_bulge),
        "eye_depression": float(args.eye_depression),
        "mouth_depression": float(args.mouth_depression),
        "max_face_y_frac": float(args.max_face_y_frac),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": normal_update_count,
        "confidence_preserved": True,
        "relief_stats": relief_stats,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
