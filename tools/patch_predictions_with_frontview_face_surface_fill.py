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
from patch_predictions_with_image_face_relief import skin_mask_for_view, update_predicted_normals  # noqa: E402
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill a front/side visible face ROI with a bounded 3D surface patch. "
            "This is a diagnostic for whether the Open3D face hole is caused by "
            "missing visible face surface points rather than confidence alone."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-summary", default="", type=Path)
    parser.add_argument("--view-indices", default="3")
    parser.add_argument("--direction-view", type=int, default=3)
    parser.add_argument("--face-bulge", type=float, default=0.010)
    parser.add_argument("--nose-bulge", type=float, default=0.020)
    parser.add_argument("--eye-depression", type=float, default=0.004)
    parser.add_argument("--mouth-depression", type=float, default=0.003)
    parser.add_argument("--max-face-y-frac", type=float, default=0.88)
    parser.add_argument("--skin-dilate", type=int, default=1)
    parser.add_argument("--skin-erode", type=int, default=0)
    parser.add_argument("--conf-boost", type=float, default=220.0)
    parser.add_argument("--normal-conf-boost", type=float, default=1.0)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


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


def parse_views(spec: str, view_count: int) -> list[int]:
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
        raise ValueError("--view-indices selected nothing")
    return sorted(set(out))


def normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return value / norm


def face_direction(extrinsic: np.ndarray, view_idx: int) -> np.ndarray:
    rotation = extrinsic[view_idx, :3, :3].astype(np.float32)
    forward = rotation.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normalize(-forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def face_roi(mask: np.ndarray, max_face_y_frac: float) -> np.ndarray:
    roi = build_roi_masks(mask.astype(bool))["face"]
    ys = np.nonzero(roi)[0]
    if ys.size:
        y_limit = int(ys.min() + float(max_face_y_frac) * max(1, ys.max() - ys.min() + 1))
        roi &= np.arange(roi.shape[0])[:, None] <= y_limit
    return roi


def dark_feature_masks(rgb: np.ndarray, skin: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if not np.any(skin):
        empty = np.zeros_like(skin, dtype=bool)
        return empty, empty
    threshold = float(np.percentile(gray[skin], 42.0))
    dark = skin & (gray < threshold)
    eyes = dark & (ny > -0.48) & (ny < 0.08)
    mouth = dark & (ny > 0.18) & (ny < 0.68)
    return eyes, mouth


def save_debug(path: Path, rgb: np.ndarray, roi: np.ndarray, skin: np.ndarray, patch: np.ndarray) -> None:
    arr = rgb.astype(np.float32).copy()
    arr[roi] = arr[roi] * 0.75 + np.asarray([40, 80, 255], dtype=np.float32) * 0.25
    arr[skin] = arr[skin] * 0.50 + np.asarray([255, 180, 60], dtype=np.float32) * 0.50
    arr[patch] = arr[patch] * 0.35 + np.asarray([255, 40, 40], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def surface_patch_points(
    world_points: np.ndarray,
    patch: np.ndarray,
    rgb: np.ndarray,
    direction: np.ndarray,
    face_bulge: float,
    nose_bulge: float,
    eye_depression: float,
    mouth_depression: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    yy, xx = np.nonzero(patch)
    selected = world_points[patch]
    if selected.shape[0] < 64:
        return selected, {"reason": "too_few_patch_pixels", "patch_pixels": int(selected.shape[0])}

    up = normalize(np.asarray([0.0, -1.0, 0.0], dtype=np.float32), np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
    horizontal = normalize(np.cross(up, direction), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    normal = normalize(np.cross(horizontal, up), direction)
    if float(np.dot(normal, direction)) < 0.0:
        normal = -normal

    center = np.median(selected, axis=0).astype(np.float32)
    rel = selected - center[None, :]
    h_existing = rel @ horizontal
    v_existing = rel @ up
    n_existing = rel @ normal
    h_lo, h_hi = np.percentile(h_existing, [3.0, 97.0])
    v_lo, v_hi = np.percentile(v_existing, [3.0, 97.0])
    n_center = float(np.median(n_existing))

    x0, x1 = float(xx.min()), float(xx.max())
    y0, y1 = float(yy.min()), float(yy.max())
    sx = max(0.5 * (x1 - x0), 1.0)
    sy = max(0.5 * (y1 - y0), 1.0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    nx = (xx.astype(np.float32) - cx) / sx
    ny = (yy.astype(np.float32) - cy) / sy

    h = np.interp(xx.astype(np.float32), [x0, x1], [h_lo, h_hi]).astype(np.float32)
    v = np.interp(yy.astype(np.float32), [y0, y1], [v_lo, v_hi]).astype(np.float32)
    face = np.exp(-0.5 * ((nx / 0.95) ** 2 + (ny / 1.05) ** 2)).astype(np.float32)
    nose = np.exp(-0.5 * (((nx - 0.05) / 0.24) ** 2 + ((ny + 0.04) / 0.34) ** 2)).astype(np.float32)

    full_nx = np.zeros(patch.shape, dtype=np.float32)
    full_ny = np.zeros(patch.shape, dtype=np.float32)
    full_nx[yy, xx] = nx
    full_ny[yy, xx] = ny
    eyes, mouth = dark_feature_masks(rgb, patch, full_nx, full_ny)
    eye_term = eyes[yy, xx].astype(np.float32)
    mouth_term = mouth[yy, xx].astype(np.float32)
    relief = (
        float(face_bulge) * face
        + float(nose_bulge) * nose
        - float(eye_depression) * eye_term
        - float(mouth_depression) * mouth_term
    ).astype(np.float32)
    n = n_center + relief
    filled = center[None, :] + h[:, None] * horizontal[None, :] + v[:, None] * up[None, :] + n[:, None] * normal[None, :]
    stats = {
        "patch_pixels": int(selected.shape[0]),
        "h_range": [float(h_lo), float(h_hi)],
        "v_range": [float(v_lo), float(v_hi)],
        "n_center": float(n_center),
        "relief_mean": float(np.mean(relief)),
        "relief_p90": float(np.percentile(relief, 90.0)),
        "eye_pixels": int(eyes.sum()),
        "mouth_pixels": int(mouth.sum()),
        "direction": direction.tolist(),
        "normal": normal.tolist(),
    }
    return filled.astype(np.float32), stats


def main() -> int:
    args = parse_args()
    with np.load(args.base_predictions, allow_pickle=False) as payload:
        base = {key: np.asarray(payload[key]) for key in payload.files}

    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth2 = depth[..., 0].copy() if depth.ndim == 4 else depth.copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    views = parse_views(str(args.view_indices), view_count)
    direction = face_direction(extrinsic, int(args.direction_view))
    patch_mask = np.zeros(world_points.shape[:3], dtype=bool)
    per_view: dict[str, Any] = {}

    for view_idx in views:
        scene = load_scene_view(args.scene_dir, view_idx, (height, width))
        roi = face_roi(scene.mask.astype(bool), float(args.max_face_y_frac))
        skin = skin_mask_for_view(
            scene.rgb,
            scene.mask.astype(bool),
            roi,
            skin_dilate=int(args.skin_dilate),
            skin_erode=int(args.skin_erode),
        )
        patch = skin & np.isfinite(world_points[view_idx]).all(axis=-1)
        filled, stats = surface_patch_points(
            world_points[view_idx],
            patch,
            scene.rgb,
            direction,
            float(args.face_bulge),
            float(args.nose_bulge),
            float(args.eye_depression),
            float(args.mouth_depression),
        )
        if filled.shape[0] and "reason" not in stats:
            yy, xx = np.nonzero(patch)
            world_points[view_idx, yy, xx] = filled
            cam_points = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
            depth2[view_idx, yy, xx] = np.maximum(1e-4, cam_points[yy, xx, 2])
            world_conf[view_idx, yy, xx] = np.maximum(world_conf[view_idx, yy, xx], float(args.conf_boost))
            depth_conf[view_idx, yy, xx] = np.maximum(depth_conf[view_idx, yy, xx], float(args.conf_boost))
            normal_conf[view_idx, yy, xx] = np.maximum(normal_conf[view_idx, yy, xx], float(args.normal_conf_boost))
            patch_mask[view_idx] = patch
        if bool(args.write_debug):
            save_debug(args.output_npz.parent / f"debug_view_{view_idx:02d}.png", scene.rgb, roi, skin, patch)
        per_view[str(view_idx)] = {
            **stats,
            "roi_pixels": int(roi.sum()),
            "skin_pixels": int(skin.sum()),
        }

    normal, normal_update_count = update_predicted_normals(normal, world_points, extrinsic, patch_mask)
    out: dict[str, Any] = dict(base)
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth2[..., None].astype(np.float32) if depth.ndim == 4 else depth2.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    out["world_points_conf"] = world_conf.astype(np.float32)
    out["depth_conf"] = depth_conf.astype(np.float32)
    out["normal_conf"] = normal_conf.astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_npz": str(args.output_npz.resolve()),
        "view_indices": views,
        "direction_view": int(args.direction_view),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "conf_boost": float(args.conf_boost),
        "face_bulge": float(args.face_bulge),
        "nose_bulge": float(args.nose_bulge),
        "eye_depression": float(args.eye_depression),
        "mouth_depression": float(args.mouth_depression),
        "per_view": per_view,
        "truthful_status": "local_frontview_face_surface_fill_diagnostic_not_final_pass",
    }
    output_summary = args.output_summary if str(args.output_summary) and args.output_summary != Path(".") else args.output_npz.with_suffix(".json")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
