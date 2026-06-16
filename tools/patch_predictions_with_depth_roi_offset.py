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
    skin_mask_for_view,
    update_predicted_normals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a small depth offset inside 2D face ROI pixels and rebuild the "
            "corresponding world_points from the same depth map. Confidence is "
            "preserved, so fixed-threshold checks remain meaningful."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--view-indices", default="2,3,4,5")
    parser.add_argument("--offset", type=float, default=0.025)
    parser.add_argument("--roi", choices=("face", "head"), default="face")
    parser.add_argument("--support", choices=("roi", "skin"), default="roi")
    parser.add_argument("--max-face-y-frac", type=float, default=0.88)
    parser.add_argument("--skin-dilate", type=int, default=1)
    parser.add_argument("--skin-erode", type=int, default=0)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def parse_views(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for piece in str(spec).split(","):
        item = piece.strip()
        if not item:
            continue
        view_idx = int(item)
        if view_idx < 0 or view_idx >= view_count:
            raise ValueError(f"view index {view_idx} outside [0, {view_count})")
        out.append(view_idx)
    if not out:
        raise ValueError("--view-indices must select at least one view")
    return sorted(set(out))


def select_roi(mask: np.ndarray, roi_name: str, max_face_y_frac: float) -> np.ndarray:
    masks = build_roi_masks(mask.astype(bool))
    roi = masks[roi_name].copy()
    if roi_name == "face":
        ys = np.nonzero(roi)[0]
        if ys.size:
            y_limit = int(ys.min() + float(max_face_y_frac) * max(1, ys.max() - ys.min() + 1))
            roi &= np.arange(roi.shape[0])[:, None] <= y_limit
    return roi


def save_debug(path: Path, rgb: np.ndarray, roi: np.ndarray, support: np.ndarray) -> None:
    overlay = rgb.astype(np.float32).copy()
    overlay[roi] = overlay[roi] * 0.72 + np.array([40, 80, 255], dtype=np.float32) * 0.28
    overlay[support] = overlay[support] * 0.40 + np.array([255, 60, 40], dtype=np.float32) * 0.60
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


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
    view_indices = parse_views(str(args.view_indices), view_count)
    patch_mask = np.zeros((view_count, height, width), dtype=bool)
    per_view: dict[str, Any] = {}

    for view_idx in view_indices:
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        roi = select_roi(scene.mask.astype(bool), str(args.roi), float(args.max_face_y_frac))
        if str(args.support) == "skin":
            support = skin_mask_for_view(
                scene.rgb,
                scene.mask.astype(bool),
                roi,
                skin_dilate=int(args.skin_dilate),
                skin_erode=int(args.skin_erode),
            )
        else:
            support = roi
        current_depth = depth[view_idx, ..., 0] if depth.ndim == 4 and depth.shape[-1] == 1 else depth[view_idx]
        valid = support & np.isfinite(current_depth) & (current_depth > 0.05)
        new_depth = current_depth.copy()
        new_depth[valid] = np.maximum(new_depth[valid] + float(args.offset), 0.05)
        cam_points = camera_grid_from_depth(new_depth, intrinsic[view_idx])
        new_world = camera_to_world(cam_points, extrinsic[view_idx])
        world_points[view_idx][valid] = new_world[valid]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth[view_idx, ..., 0][valid] = new_depth[valid]
        else:
            depth[view_idx][valid] = new_depth[valid]
        patch_mask[view_idx] = valid
        per_view[str(view_idx)] = {
            "roi_pixels": int(roi.sum()),
            "support_pixels": int(support.sum()),
            "patch_pixels": int(valid.sum()),
            "offset": float(args.offset),
        }
        if bool(args.write_debug):
            save_debug(output_npz.parent / f"debug_view_{view_idx:02d}.png", scene.rgb, roi, valid)

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
        "roi": str(args.roi),
        "support": str(args.support),
        "offset": float(args.offset),
        "max_face_y_frac": float(args.max_face_y_frac),
        "patch_pixels_total": int(patch_mask.sum()),
        "normal_update_count": int(normal_update_count),
        "confidence_preserved": True,
        "per_view": per_view,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
