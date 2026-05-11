from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from render_open3d_pointcloud import mask_to_2d_roi  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch only low-confidence body/hand pixels from an already aligned "
            "teacher_targets.npz. Head, face, and hairline are always protected."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--teacher-targets", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--max-distance", type=float, default=0.10)
    parser.add_argument("--confidence-boost", type=float, default=180.0)
    parser.add_argument("--repair-hands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-body", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-fill-pixels", type=int, default=16)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
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


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3].astype(np.float32)
    translation = extrinsic[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def protect_head_face_hair(mask: np.ndarray) -> np.ndarray:
    protected = np.zeros_like(mask, dtype=bool)
    head = head_box_from_mask(mask)
    face = face_box_from_mask(mask)
    for box, pad in ((head, 10), (face, 6)):
        if box is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(mask.shape[1], x1 + pad)
        y1 = min(mask.shape[0], y1 + pad)
        protected[y0:y1, x0:x1] = True
    if head is not None:
        x0, y0, x1, y1 = [int(v) for v in head]
        h = max(1, y1 - y0)
        protected[
            max(0, y0 - 6) : min(mask.shape[0], y0 + int(round(0.58 * h))),
            max(0, x0 - 10) : min(mask.shape[1], x1 + 10),
        ] = True
    return protected & mask


def save_overlay(path: Path, rgb: np.ndarray, fill: np.ndarray, protect: np.ndarray, text: str) -> None:
    arr = rgb.astype(np.float32).copy()
    arr[protect] = arr[protect] * 0.60 + np.asarray([255, 230, 0], dtype=np.float32) * 0.40
    arr[fill] = arr[fill] * 0.42 + np.asarray([40, 110, 255], dtype=np.float32) * 0.58
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), text, fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    teacher = load_npz(args.teacher_targets)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    world = np.asarray(base["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth2 = depth[..., 0].copy() if depth.ndim == 4 else depth.copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)

    teacher_world = np.asarray(teacher["world_points"], dtype=np.float32)
    teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
    teacher_normals = np.asarray(teacher["teacher_normals"], dtype=np.float32) if "teacher_normals" in teacher else None
    if teacher_world.shape != world.shape:
        raise ValueError(f"teacher/world shape mismatch: {teacher_world.shape} vs {world.shape}")
    if teacher_mask.shape != world.shape[:3]:
        raise ValueError(f"teacher_mask shape mismatch: {teacher_mask.shape} vs {world.shape[:3]}")

    view_count, height, width, _ = world.shape
    filled = np.zeros(world.shape[:3], dtype=bool)
    per_view: list[dict[str, Any]] = []
    for view_idx in range(view_count):
        scene = load_scene_view(args.scene_dir, view_idx, (height, width))
        mask = scene.mask.astype(bool)
        finite_base = np.isfinite(world[view_idx]).all(axis=-1)
        finite_teacher = np.isfinite(teacher_world[view_idx]).all(axis=-1)
        support = mask & finite_base & np.isfinite(world_conf[view_idx]) & (world_conf[view_idx] > 0.0)
        threshold = float(np.percentile(world_conf[view_idx][support], float(args.conf_percentile))) if support.any() else 0.0
        low = support & (world_conf[view_idx] < threshold)
        protected = protect_head_face_hair(mask)
        hand_roi = mask_to_2d_roi(mask, "hands", rgb=scene.rgb)
        body_roi = mask & ~protected
        target_roi = np.zeros_like(mask, dtype=bool)
        if bool(args.repair_body):
            target_roi |= body_roi
        if bool(args.repair_hands):
            target_roi |= hand_roi
        distance = np.linalg.norm(teacher_world[view_idx] - world[view_idx], axis=-1)
        candidate = (
            low
            & target_roi
            & ~protected
            & teacher_mask[view_idx]
            & finite_teacher
            & np.isfinite(distance)
            & (distance <= float(args.max_distance))
        )
        if int(candidate.sum()) < int(args.min_fill_pixels):
            save_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_teacher_lowconf_bodyhands.png",
                scene.rgb,
                candidate,
                protected,
                f"skip fill={int(candidate.sum())}",
            )
            per_view.append(
                {
                    "view_index": int(view_idx),
                    "fill_pixels": int(candidate.sum()),
                    "skipped": True,
                    "conf_threshold": threshold,
                    "teacher_pixels": int((teacher_mask[view_idx] & target_roi & ~protected).sum()),
                }
            )
            continue

        updated = world[view_idx][candidate] + float(args.alpha) * (
            teacher_world[view_idx][candidate] - world[view_idx][candidate]
        )
        world[view_idx][candidate] = updated
        filled[view_idx] = candidate
        world_conf[view_idx][candidate] = np.maximum(world_conf[view_idx][candidate], float(args.confidence_boost))
        depth_conf[view_idx][candidate] = np.maximum(depth_conf[view_idx][candidate], float(args.confidence_boost))
        if teacher_normals is not None:
            normal[view_idx][candidate] = teacher_normals[view_idx][candidate]
            normal_conf[view_idx][candidate] = np.maximum(normal_conf[view_idx][candidate], 1.0)

        save_overlay(
            output_dir / "overlays" / f"view_{view_idx:02d}_teacher_lowconf_bodyhands.png",
            scene.rgb,
            candidate,
            protected,
            f"fill={int(candidate.sum())}",
        )
        dist_values = distance[candidate]
        per_view.append(
            {
                "view_index": int(view_idx),
                "fill_pixels": int(candidate.sum()),
                "skipped": False,
                "conf_threshold": threshold,
                "teacher_pixels": int((teacher_mask[view_idx] & target_roi & ~protected).sum()),
                "hand_pixels": int((candidate & hand_roi).sum()),
                "body_pixels": int((candidate & body_roi).sum()),
                "distance_percentiles": [float(v) for v in np.percentile(dist_values, [0, 25, 50, 75, 90, 99])],
            }
        )

    cam = world_to_camera(world, extrinsic)
    valid_depth = filled & np.isfinite(cam[..., 2]) & (cam[..., 2] > 1e-6)
    depth2[valid_depth] = cam[..., 2][valid_depth]

    out = dict(base)
    out["world_points"] = world.astype(base["world_points"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth"] = depth2[..., None].astype(base["depth"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["normal"] = normal.astype(base["normal"].dtype, copy=False)
    out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)
    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)

    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "teacher_targets": str(args.teacher_targets.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_predictions": str(output_path),
        "alpha": float(args.alpha),
        "max_distance": float(args.max_distance),
        "confidence_boost": float(args.confidence_boost),
        "filled_pixels": int(filled.sum()),
        "per_view": per_view,
        "truthful_status": "local_lowconf_bodyhands_teacher_patch_diagnostic_not_final_pass",
    }
    (output_dir / "teacher_lowconf_bodyhands_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
