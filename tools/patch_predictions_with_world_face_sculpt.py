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
    skin_mask_for_view,
    update_predicted_normals,
)
from vggt.utils.normal_refiner import points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a small, confidence-preserving face sculpt in world space. "
            "The displacement direction is shared across views and projected "
            "back to depth, so this probes geometry coupling rather than per-view "
            "depth hallucination."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--view-indices", default="2,3")
    parser.add_argument("--direction-view", type=int, default=3)
    parser.add_argument("--face-bulge", type=float, default=0.006)
    parser.add_argument("--nose-bulge", type=float, default=0.012)
    parser.add_argument("--eye-depression", type=float, default=0.002)
    parser.add_argument("--mouth-depression", type=float, default=0.0015)
    parser.add_argument("--max-face-y-frac", type=float, default=0.88)
    parser.add_argument("--skin-dilate", type=int, default=1)
    parser.add_argument("--skin-erode", type=int, default=0)
    parser.add_argument("--clamp-face-box", action="store_true")
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def parse_views(spec: str, view_count: int) -> list[int]:
    result: list[int] = []
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value >= view_count:
            raise ValueError(f"view index {value} outside [0, {view_count})")
        result.append(value)
    if not result:
        raise ValueError("--view-indices must select at least one view")
    return sorted(set(result))


def face_roi(mask: np.ndarray, max_face_y_frac: float) -> np.ndarray:
    roi = build_roi_masks(mask.astype(bool))["face"]
    ys = np.nonzero(roi)[0]
    if ys.size:
        y_limit = int(ys.min() + float(max_face_y_frac) * max(1, ys.max() - ys.min() + 1))
        roi &= np.arange(roi.shape[0])[:, None] <= y_limit
    return roi


def shared_face_direction(extrinsic: np.ndarray, view_idx: int) -> np.ndarray:
    rotation = extrinsic[view_idx, :3, :3].astype(np.float32)
    forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    toward_camera = -forward
    direction = np.array([toward_camera[0], 0.0, toward_camera[2]], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        direction /= norm
    return direction


def dark_feature_masks(rgb: np.ndarray, skin: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if not np.any(skin):
        empty = np.zeros_like(skin, dtype=bool)
        return empty, empty
    threshold = float(np.percentile(gray[skin], 42.0))
    dark = skin & (gray < threshold)
    eye_band = dark & (ny > -0.50) & (ny < 0.08)
    mouth_band = dark & (ny > 0.16) & (ny < 0.70)
    return eye_band, mouth_band


def sculpt_weights(rgb: np.ndarray, skin: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.zeros(skin.shape, dtype=np.float32)
    ys, xs = np.nonzero(skin)
    if ys.size < 64:
        return weights, {"skin_pixels": int(ys.size), "reason": "too_few_skin_pixels"}
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
    face = np.exp(-0.5 * ((nx / 0.92) ** 2 + (ny / 1.00) ** 2))
    nose = np.exp(-0.5 * (((nx - 0.06) / 0.23) ** 2 + ((ny + 0.04) / 0.32) ** 2))
    eye_band, mouth_band = dark_feature_masks(rgb, skin, nx, ny)
    weights[skin] = face[skin]
    stats = {
        "skin_pixels": int(ys.size),
        "eye_pixels": int(eye_band.sum()),
        "mouth_pixels": int(mouth_band.sum()),
        "nose_weight_mean": float(nose[skin].mean()),
    }
    return np.stack([face, nose, eye_band.astype(np.float32), mouth_band.astype(np.float32)], axis=-1), stats


def save_debug(path: Path, rgb: np.ndarray, roi: np.ndarray, skin: np.ndarray, patch: np.ndarray) -> None:
    overlay = rgb.astype(np.float32).copy()
    overlay[roi] = overlay[roi] * 0.72 + np.array([40, 80, 255], dtype=np.float32) * 0.28
    overlay[skin] = overlay[skin] * 0.52 + np.array([255, 180, 70], dtype=np.float32) * 0.48
    overlay[patch] = overlay[patch] * 0.35 + np.array([255, 40, 40], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def hard_face_box(points: np.ndarray, conf: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    flat = points.reshape(-1, 3)
    score = conf.reshape(-1)
    support = masks.reshape(-1)
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(score) & (score > 0.0) & support
    threshold = float(np.percentile(score[valid], 40.0))
    kept = flat[valid & (score >= threshold)]
    height_like = -kept[:, 1]
    head_cut = float(np.percentile(height_like, 74.0))
    head_mask = height_like >= head_cut
    head_points = kept[head_mask]
    x_lo, x_hi = np.percentile(head_points[:, 0], [20.0, 80.0])
    z_lo, z_hi = np.percentile(head_points[:, 2], [15.0, 85.0])
    head_height = -head_points[:, 1]
    y_hi = -float(np.percentile(head_height, 25.0))
    y_lo = float(np.percentile(kept[:, 1], 1.0))
    return {
        "x_lo": float(x_lo),
        "x_hi": float(x_hi),
        "y_lo": float(y_lo),
        "y_hi": float(y_hi),
        "z_lo": float(z_lo),
        "z_hi": float(z_hi),
    }


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
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32)

    view_count, height, width, _ = world_points.shape
    view_indices = parse_views(args.view_indices, view_count)
    direction = shared_face_direction(extrinsic, int(args.direction_view))

    scene_masks: list[np.ndarray] = []
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        scene_masks.append(scene.mask.astype(bool))
    masks = np.stack(scene_masks, axis=0)
    box = hard_face_box(world_points, world_conf, masks)

    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    stats: dict[str, Any] = {}
    for view_idx in view_indices:
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        roi = face_roi(scene.mask.astype(bool), float(args.max_face_y_frac))
        skin = skin_mask_for_view(
            scene.rgb,
            scene.mask.astype(bool),
            roi,
            skin_dilate=int(args.skin_dilate),
            skin_erode=int(args.skin_erode),
        )
        weights, weight_stats = sculpt_weights(scene.rgb, skin)
        if weights.ndim != 3:
            stats[str(view_idx)] = weight_stats
            continue
        face_w = weights[..., 0]
        nose_w = weights[..., 1]
        eye_w = weights[..., 2]
        mouth_w = weights[..., 3]
        patch = skin & np.isfinite(world_points[view_idx]).all(axis=-1)
        displacement_amount = (
            float(args.face_bulge) * face_w
            + float(args.nose_bulge) * nose_w
            - float(args.eye_depression) * eye_w
            - float(args.mouth_depression) * mouth_w
        ).astype(np.float32)
        updated = world_points[view_idx].copy()
        updated[patch] = updated[patch] + displacement_amount[patch, None] * direction[None, :]
        if bool(args.clamp_face_box):
            # Keep sculpted pixels inside the same hard face box used by the
            # local gate, with a tiny margin to avoid artificial boundary spikes.
            updated[..., 0][patch] = np.clip(updated[..., 0][patch], box["x_lo"] + 1e-4, box["x_hi"] - 1e-4)
            updated[..., 1][patch] = np.clip(updated[..., 1][patch], box["y_lo"], box["y_hi"] - 1e-4)
            updated[..., 2][patch] = np.clip(updated[..., 2][patch], box["z_lo"] + 1e-4, box["z_hi"] - 1e-4)
        world_points[view_idx][patch] = updated[patch]
        cam_points = points_world_to_camera(updated, extrinsic[view_idx])
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth[view_idx, ..., 0][patch] = cam_points[..., 2][patch]
        else:
            depth[view_idx][patch] = cam_points[..., 2][patch]
        patch_mask[view_idx] = patch
        selected = displacement_amount[patch]
        stats[str(view_idx)] = {
            **weight_stats,
            "roi_pixels": int(roi.sum()),
            "patch_pixels": int(patch.sum()),
            "displacement_min": float(selected.min()) if selected.size else 0.0,
            "displacement_mean": float(selected.mean()) if selected.size else 0.0,
            "displacement_max": float(selected.max()) if selected.size else 0.0,
        }
        if bool(args.write_debug):
            save_debug(output_npz.parent / f"debug_view_{view_idx:02d}.png", scene.rgb, roi, skin, patch)

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
        "direction_view": int(args.direction_view),
        "world_direction": direction.tolist(),
        "face_bulge": float(args.face_bulge),
        "nose_bulge": float(args.nose_bulge),
        "eye_depression": float(args.eye_depression),
        "mouth_depression": float(args.mouth_depression),
        "clamp_face_box": bool(args.clamp_face_box),
        "hard_face_box": box,
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "confidence_preserved": True,
        "per_view": stats,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
