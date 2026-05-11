from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import head_box_from_mask, point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a local diagnostic predictions.npz with MediaPipe FaceMesh depth relief "
            "on views where a face is detected. This is image-evidence relief, not a "
            "training result."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--face-landmarker-task", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--relief-scale", type=float, default=0.35)
    parser.add_argument("--max-relief", type=float, default=0.075)
    parser.add_argument("--head-crop-pad", type=int, default=30)
    parser.add_argument("--min-detection-confidence", type=float, default=0.05)
    parser.add_argument("--confidence-boost", type=float, default=120.0)
    parser.add_argument("--only-views", default="", help="Optional comma-separated view indices.")
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


def parse_view_indices(spec: str, view_count: int) -> set[int] | None:
    if not str(spec).strip():
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        if not part.strip():
            continue
        idx = int(part)
        if idx < 0 or idx >= view_count:
            raise IndexError(f"view index {idx} out of range [0,{view_count})")
        out.add(idx)
    return out


def closed_form_inverse_se3(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    out = np.tile(np.eye(4, dtype=np.float32), (extrinsic.shape[0], 1, 1))
    out[:, :3, :3] = rotation_t
    out[:, :3, 3] = -np.einsum("vij,vj->vi", rotation_t, translation)
    return out


def depth_to_world(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    views, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cam = np.zeros((views, height, width, 3), dtype=np.float32)
    for view_idx in range(views):
        z = depth[view_idx].astype(np.float32)
        fx = float(intrinsic[view_idx, 0, 0])
        fy = float(intrinsic[view_idx, 1, 1])
        cx = float(intrinsic[view_idx, 0, 2])
        cy = float(intrinsic[view_idx, 1, 2])
        cam[view_idx, ..., 0] = (xx - cx) / max(fx, 1e-6) * z
        cam[view_idx, ..., 1] = (yy - cy) / max(fy, 1e-6) * z
        cam[view_idx, ..., 2] = z
    cam_to_world = closed_form_inverse_se3(extrinsic.astype(np.float32))
    rotation = cam_to_world[:, :3, :3]
    translation = cam_to_world[:, :3, 3]
    return np.einsum("vij,vhwj->vhwi", rotation, cam) + translation[:, None, None, :]


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3].astype(np.float32)
    translation = extrinsic[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def detect_face_relief(
    detector: vision.FaceLandmarker,
    image: Image.Image,
    mask: np.ndarray,
    *,
    pad: int,
    relief_scale: float,
    max_relief: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    head_box = head_box_from_mask(mask)
    height, width = mask.shape
    if head_box is None:
        return np.zeros(mask.shape, dtype=np.float32), np.zeros(mask.shape, dtype=bool), {"detected": False, "reason": "no_head_box"}
    x0, y0, x1, y1 = head_box
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(width, x1 + int(pad))
    y1 = min(height, y1 + int(pad))
    if x1 <= x0 + 16 or y1 <= y0 + 16:
        return np.zeros(mask.shape, dtype=np.float32), np.zeros(mask.shape, dtype=bool), {"detected": False, "reason": "tiny_head_box"}

    crop = image.crop((x0, y0, x1, y1)).resize((512, 512), Image.Resampling.BICUBIC)
    crop_arr = np.asarray(crop.convert("RGB"))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_arr))
    if not result.face_landmarks:
        return np.zeros(mask.shape, dtype=np.float32), np.zeros(mask.shape, dtype=bool), {
            "detected": False,
            "head_box": [int(x0), int(y0), int(x1), int(y1)],
            "reason": "no_facemesh",
        }

    landmarks = result.face_landmarks[0]
    points = []
    z_values = []
    for lm in landmarks:
        x = x0 + float(lm.x) * (x1 - x0)
        y = y0 + float(lm.y) * (y1 - y0)
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y))
            z_values.append(float(lm.z))
    if len(points) < 64:
        return np.zeros(mask.shape, dtype=np.float32), np.zeros(mask.shape, dtype=bool), {"detected": False, "reason": "too_few_landmarks"}
    pts = np.asarray(points, dtype=np.float32)
    z = np.asarray(z_values, dtype=np.float32)
    hull = cv2.convexHull(np.rint(pts).astype(np.int32))
    face_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 1)
    face_mask = face_mask.astype(bool) & mask.astype(bool)
    yy, xx = np.nonzero(face_mask)
    if yy.size < 64:
        return np.zeros(mask.shape, dtype=np.float32), np.zeros(mask.shape, dtype=bool), {"detected": False, "reason": "tiny_face_mask"}
    z_centered = z - float(np.median(z))
    interp = griddata(pts, z_centered, (xx.astype(np.float32), yy.astype(np.float32)), method="linear")
    nearest = griddata(pts, z_centered, (xx.astype(np.float32), yy.astype(np.float32)), method="nearest")
    values = np.where(np.isfinite(interp), interp, nearest).astype(np.float32)
    offset = np.zeros(mask.shape, dtype=np.float32)
    offset[yy, xx] = np.clip(values * float(relief_scale), -float(max_relief), float(max_relief))
    offset = gaussian_filter(offset, sigma=1.25)
    offset[~face_mask] = 0.0
    return offset, face_mask, {
        "detected": True,
        "head_box": [int(x0), int(y0), int(x1), int(y1)],
        "landmarks": int(len(points)),
        "face_mask_pixels": int(face_mask.sum()),
        "raw_z_percentiles": [float(v) for v in np.percentile(z, [0, 5, 25, 50, 75, 95, 100])],
        "offset_percentiles": [float(v) for v in np.percentile(offset[face_mask], [0, 5, 25, 50, 75, 95, 100])],
    }


def save_overlay(path: Path, image: Image.Image, face_mask: np.ndarray, offset: np.ndarray, meta: dict[str, Any]) -> None:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    positive = face_mask & (offset > 0)
    negative = face_mask & (offset < 0)
    arr[positive] = arr[positive] * 0.55 + np.asarray([40, 80, 255], dtype=np.float32) * 0.45
    arr[negative] = arr[negative] * 0.55 + np.asarray([255, 80, 40], dtype=np.float32) * 0.45
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    if "head_box" in meta:
        draw.rectangle(tuple(meta["head_box"]), outline=(255, 255, 0), width=2)
    draw.text((8, 8), f"detected={meta.get('detected')} pixels={int(face_mask.sum())}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    if depth.ndim == 4:
        depth2 = depth[..., 0].copy()
    else:
        depth2 = depth.copy()
    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    intrinsic = np.asarray(base["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    view_count, height, width = depth2.shape
    selected = parse_view_indices(str(args.only_views), view_count)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.face_landmarker_task)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=float(args.min_detection_confidence),
        min_face_presence_confidence=float(args.min_detection_confidence),
    )
    patched_mask = np.zeros(depth2.shape, dtype=bool)
    per_view: list[dict[str, Any]] = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for view_idx in range(view_count):
            if selected is not None and view_idx not in selected:
                per_view.append({"view_index": int(view_idx), "skipped": True})
                continue
            image_path = scene_dir / "images" / f"{view_idx:02d}.png"
            mask_path = scene_dir / "masks" / f"{view_idx:02d}.png"
            image = Image.open(image_path).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = Image.open(mask_path).convert("L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
            mask_arr = np.asarray(mask) > 127
            offset, face_mask, meta = detect_face_relief(
                detector,
                image,
                mask_arr,
                pad=int(args.head_crop_pad),
                relief_scale=float(args.relief_scale),
                max_relief=float(args.max_relief),
            )
            if meta.get("detected"):
                depth2[view_idx][face_mask] = np.maximum(1e-4, depth2[view_idx][face_mask] + offset[face_mask])
                patched_mask[view_idx] = face_mask
                depth_conf[view_idx][face_mask] = np.maximum(depth_conf[view_idx][face_mask], float(args.confidence_boost))
                world_conf[view_idx][face_mask] = np.maximum(world_conf[view_idx][face_mask], float(args.confidence_boost))
                normal_conf[view_idx][face_mask] = np.maximum(normal_conf[view_idx][face_mask], 1.0)
            save_overlay(output_dir / "overlays" / f"view_{view_idx:02d}_facemesh_relief_overlay.png", image, face_mask, offset, meta)
            per_view.append({"view_index": int(view_idx), **meta})

    new_world = depth_to_world(depth2, intrinsic, extrinsic)
    world_points[patched_mask] = new_world[patched_mask]
    cam_points = world_to_camera(world_points, extrinsic)
    for view_idx in range(view_count):
        if patched_mask[view_idx].any():
            nmap, valid = point_map_to_normal_numpy(cam_points[view_idx], patched_mask[view_idx])
            use = valid & patched_mask[view_idx]
            # Match the project's predicted-normal convention used by prior
            # patches: derived camera normals are usually opposite to predicted
            # normals, so choose the sign closer to the old prediction.
            old = normal[view_idx]
            dot = np.sum(old[use] * nmap[use], axis=-1)
            if dot.size and float(np.nanmean(dot)) < 0.0:
                nmap = -nmap
            normal[view_idx][use] = nmap[use]

    out = dict(base)
    out["depth"] = depth2[..., None].astype(base["depth"].dtype, copy=False)
    out["world_points"] = world_points.astype(base["world_points"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["normal"] = normal.astype(base["normal"].dtype, copy=False)
    out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)
    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)
    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "scene_dir": str(scene_dir),
        "output_predictions": str(output_path),
        "relief_scale": float(args.relief_scale),
        "max_relief": float(args.max_relief),
        "confidence_boost": float(args.confidence_boost),
        "patched_pixels": int(patched_mask.sum()),
        "per_view": per_view,
        "truthful_status": "local_mediapipe_face_relief_diagnostic_not_training_result",
    }
    (output_dir / "mediapipe_face_relief_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
