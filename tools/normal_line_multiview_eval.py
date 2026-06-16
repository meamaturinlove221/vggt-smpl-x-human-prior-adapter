from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import (  # noqa: E402
    face_box_from_mask,
    head_box_from_mask,
    normal_to_rgb,
    point_map_to_normal_numpy,
    points_world_to_camera,
    preprocess_mask_image,
    preprocess_rgb_image,
)


WORLD_POINT_KEYS = ("world_points", "points_world", "pred_world_points", "point_map", "points3d", "points_3d")
DEPTH_KEYS = ("depth", "depths", "depth_map", "depth_maps", "pred_depth")
NORMAL_KEYS = ("normal", "normals", "pred_normal", "pred_normals", "refined_normal", "teacher_normal")
WORLD_CONF_KEYS = ("world_points_conf", "point_conf", "points_conf", "depth_conf")
DEPTH_CONF_KEYS = ("depth_conf", "depths_conf", "world_points_conf")
NORMAL_CONF_KEYS = ("normal_conf", "normals_conf", "pred_normal_conf", "refined_normal_conf", "teacher_normal_conf")
INTRINSIC_KEYS = ("intrinsic", "intrinsics", "camera_intrinsics", "K")
EXTRINSIC_KEYS = ("extrinsic", "extrinsics", "camera_extrinsics", "world_to_camera")
ROI_ORDER = ("full", "head", "face")
COMPARISON_ORDER = ("pred_vs_depth", "pred_vs_point", "depth_vs_point")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EntrySpec:
    name: str
    predictions_npz: Path
    scene_dir: Path


@dataclass
class PredictionBundle:
    keys: dict[str, str | None]
    world_points: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    normal_valid: np.ndarray
    world_points_conf: np.ndarray
    depth_conf: np.ndarray
    normal_conf: np.ndarray
    intrinsic: np.ndarray | None
    extrinsic: np.ndarray | None
    normal_format: str


@dataclass
class SceneView:
    rgb: np.ndarray
    mask: np.ndarray
    image_path: Path
    mask_path: Path
    scene_view_count: int
    manifest_path: Path | None


@dataclass
class EntryArtifacts:
    summary: dict[str, Any]
    rgb: np.ndarray
    pred_normal_rgb: np.ndarray
    depth_normal_rgb: np.ndarray
    point_normal_rgb: np.ndarray
    pred_depth_error_rgb: np.ndarray
    pred_point_error_rgb: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether predicted normals agree with depth-derived and point-derived normals "
            "for one target view across multiple multiview predictions.npz entries."
        )
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Entry in name:predictions.npz:scene_dir form. Windows absolute paths are supported "
            "when predictions path ends with .npz, e.g. four:D:\\case\\predictions.npz:D:\\case\\scene."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/Markdown/PNG outputs")
    parser.add_argument("--target-view", type=int, default=0, help="View index to evaluate and visualize")
    parser.add_argument(
        "--normal-format",
        choices=("auto", "vector", "rgb01", "rgb255"),
        default="auto",
        help="How to decode the predictions normal array before vector normalization",
    )
    parser.add_argument("--overview-tile-size", type=int, default=220)
    parser.add_argument("--max-error-deg", type=float, default=45.0)
    return parser.parse_args()


def parse_entry_spec(text: str) -> EntrySpec:
    name_end = text.find(":")
    if name_end <= 0:
        raise ValueError(f"Invalid --entry '{text}'. Expected name:predictions.npz:scene_dir")

    name = text[:name_end].strip()
    remainder = text[name_end + 1 :]
    marker = ".npz:"
    marker_index = remainder.lower().rfind(marker)
    if marker_index >= 0:
        predictions_text = remainder[: marker_index + 4]
        scene_text = remainder[marker_index + len(marker) :]
    else:
        pieces = remainder.split(":")
        if len(pieces) != 2:
            raise ValueError(
                f"Invalid --entry '{text}'. Use name:predictions.npz:scene_dir; "
                "for Windows absolute paths, keep the .npz extension on predictions."
            )
        predictions_text, scene_text = pieces

    if not name or not predictions_text or not scene_text:
        raise ValueError(f"Invalid --entry '{text}'. Empty name, predictions path, or scene_dir.")

    predictions_npz = Path(predictions_text).expanduser()
    scene_dir = Path(scene_text).expanduser()
    if not predictions_npz.is_absolute():
        predictions_npz = Path.cwd() / predictions_npz
    if not scene_dir.is_absolute():
        scene_dir = Path.cwd() / scene_dir
    return EntrySpec(name=name, predictions_npz=predictions_npz.resolve(), scene_dir=scene_dir.resolve())


def find_key(payload: np.lib.npyio.NpzFile, aliases: tuple[str, ...], label: str, required: bool) -> str | None:
    available = set(payload.files)
    for key in aliases:
        if key in available:
            return key
    if required:
        raise KeyError(f"Missing required {label}; tried keys: {', '.join(aliases)}")
    return None


def as_vector_stack(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] == 3:
        return values[None].astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[0] == 3:
        return np.transpose(values, (1, 2, 0))[None].astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[-1] == 3:
        return values.astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[1] == 3:
        return np.transpose(values, (0, 2, 3, 1)).astype(np.float32, copy=False)
    raise ValueError(f"{label} must have shape [V,H,W,3], [V,3,H,W], [H,W,3], or [3,H,W]; got {values.shape}")


def as_scalar_stack(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        return values[None].astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0][None].astype(np.float32, copy=False)
    if values.ndim == 3:
        return values.astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[-1] == 1:
        return values[..., 0].astype(np.float32, copy=False)
    if values.ndim == 4 and values.shape[1] == 1:
        return values[:, 0].astype(np.float32, copy=False)
    raise ValueError(f"{label} must have scalar map shape [V,H,W], [V,H,W,1], [V,1,H,W], or [H,W]; got {values.shape}")


def ensure_stack_shape(array: np.ndarray, expected_views: int, target_hw: tuple[int, int], label: str) -> np.ndarray:
    if array.shape[0] == 1 and expected_views > 1:
        array = np.repeat(array, expected_views, axis=0)
    if array.shape[0] != expected_views:
        raise ValueError(f"{label} view count {array.shape[0]} does not match normal view count {expected_views}")
    if tuple(array.shape[1:3]) != target_hw:
        raise ValueError(f"{label} resolution {array.shape[1:3]} does not match normal resolution {target_hw}")
    return array


def as_optional_conf_stack(
    array: np.ndarray | None,
    expected_views: int,
    target_hw: tuple[int, int],
    label: str,
) -> np.ndarray:
    if array is None:
        return np.ones((expected_views, target_hw[0], target_hw[1]), dtype=np.float32)
    conf = as_scalar_stack(array, label)
    return ensure_stack_shape(conf, expected_views, target_hw, label).astype(np.float32, copy=False)


def as_intrinsic_stack(array: np.ndarray | None, expected_views: int) -> np.ndarray | None:
    if array is None:
        return None
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"intrinsic must have shape [V,3,3], [V,4,4], [3,3], or [4,4]; got {values.shape}")
    if values.shape[-2:] == (4, 4):
        values = values[:, :3, :3]
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsic must end with 3x3 or 4x4 matrices; got {values.shape}")
    if values.shape[0] == 1 and expected_views > 1:
        values = np.repeat(values, expected_views, axis=0)
    if values.shape[0] != expected_views:
        raise ValueError(f"intrinsic view count {values.shape[0]} does not match normal view count {expected_views}")
    return values.astype(np.float32, copy=False)


def as_extrinsic_stack(array: np.ndarray | None, expected_views: int) -> np.ndarray | None:
    if array is None:
        return None
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"extrinsic must have shape [V,3,4], [V,4,4], [3,4], or [4,4]; got {values.shape}")
    if values.shape[-2:] == (4, 4):
        values = values[:, :3, :4]
    if values.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic must end with 3x4 or 4x4 matrices; got {values.shape}")
    if values.shape[0] == 1 and expected_views > 1:
        values = np.repeat(values, expected_views, axis=0)
    if values.shape[0] != expected_views:
        raise ValueError(f"extrinsic view count {values.shape[0]} does not match normal view count {expected_views}")
    return values.astype(np.float32, copy=False)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    finite = np.isfinite(values).all(axis=-1)
    safe_values = np.where(finite[..., None], values, 0.0)
    norms = np.linalg.norm(safe_values, axis=-1)
    valid = finite & (norms > eps)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[valid] = values[valid] / norms[valid, None]
    return normalized, valid


def decode_normal_stack(raw_normal: np.ndarray, normal_format: str) -> tuple[np.ndarray, np.ndarray, str]:
    values = as_vector_stack(raw_normal, "normal")
    chosen_format = normal_format
    if normal_format == "auto":
        finite_values = values[np.isfinite(values)]
        max_value = float(np.max(finite_values)) if finite_values.size else 0.0
        min_value = float(np.min(finite_values)) if finite_values.size else 0.0
        chosen_format = "rgb255" if max_value > 2.0 or min_value < -2.0 else "vector"
    if chosen_format == "rgb255":
        values = values / 127.5 - 1.0
    elif chosen_format == "rgb01":
        values = values * 2.0 - 1.0
    elif chosen_format != "vector":
        raise ValueError(f"Unsupported normal format: {chosen_format}")
    return (*normalize_vectors(values), chosen_format)


def load_prediction_bundle(path: Path, normal_format: str) -> PredictionBundle:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        world_points_key = find_key(payload, WORLD_POINT_KEYS, "world_points", required=True)
        depth_key = find_key(payload, DEPTH_KEYS, "depth", required=True)
        normal_key = find_key(payload, NORMAL_KEYS, "normal", required=True)
        world_conf_key = find_key(payload, WORLD_CONF_KEYS, "world_points_conf", required=False)
        depth_conf_key = find_key(payload, DEPTH_CONF_KEYS, "depth_conf", required=False)
        normal_conf_key = find_key(payload, NORMAL_CONF_KEYS, "normal_conf", required=False)
        intrinsic_key = find_key(payload, INTRINSIC_KEYS, "intrinsic", required=False)
        extrinsic_key = find_key(payload, EXTRINSIC_KEYS, "extrinsic", required=False)

        world_points = as_vector_stack(payload[world_points_key], world_points_key)
        depth = as_scalar_stack(payload[depth_key], depth_key)
        normal, normal_valid, chosen_normal_format = decode_normal_stack(payload[normal_key], normal_format)
        expected_views = normal.shape[0]
        target_hw = tuple(int(value) for value in normal.shape[1:3])
        world_points = ensure_stack_shape(world_points, expected_views, target_hw, world_points_key)
        depth = ensure_stack_shape(depth, expected_views, target_hw, depth_key)
        world_points_conf = as_optional_conf_stack(
            payload[world_conf_key] if world_conf_key else None,
            expected_views,
            target_hw,
            world_conf_key or "world_points_conf",
        )
        depth_conf = as_optional_conf_stack(
            payload[depth_conf_key] if depth_conf_key else None,
            expected_views,
            target_hw,
            depth_conf_key or "depth_conf",
        )
        normal_conf = as_optional_conf_stack(
            payload[normal_conf_key] if normal_conf_key else None,
            expected_views,
            target_hw,
            normal_conf_key or "normal_conf",
        )
        intrinsic = as_intrinsic_stack(payload[intrinsic_key] if intrinsic_key else None, expected_views)
        extrinsic = as_extrinsic_stack(payload[extrinsic_key] if extrinsic_key else None, expected_views)

    return PredictionBundle(
        keys={
            "world_points": world_points_key,
            "depth": depth_key,
            "normal": normal_key,
            "world_points_conf": world_conf_key,
            "depth_conf": depth_conf_key,
            "normal_conf": normal_conf_key,
            "intrinsic": intrinsic_key,
            "extrinsic": extrinsic_key,
        },
        world_points=world_points.astype(np.float32, copy=False),
        depth=depth.astype(np.float32, copy=False),
        normal=normal.astype(np.float32, copy=False),
        normal_valid=normal_valid.astype(bool, copy=False),
        world_points_conf=world_points_conf.astype(np.float32, copy=False),
        depth_conf=depth_conf.astype(np.float32, copy=False),
        normal_conf=normal_conf.astype(np.float32, copy=False),
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        normal_format=chosen_normal_format,
    )


def read_scene_manifest(scene_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for manifest_name in ("scene_manifest.json", "scene_manifest"):
        manifest_path = scene_dir / manifest_name
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path
    return None, None


def resolve_scene_path(scene_dir: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    scene_relative = scene_dir / path
    if scene_relative.exists():
        return scene_relative.resolve()
    return path.resolve()


def sorted_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def scene_view_paths(scene_dir: Path) -> tuple[list[Path], list[Path], Path | None]:
    manifest, manifest_path = read_scene_manifest(scene_dir)
    exported_views = manifest.get("exported_views", []) if manifest else []
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    if exported_views:
        for view_record in exported_views:
            image_raw = view_record.get("image_path") or view_record.get("rgb_path") or view_record.get("image")
            mask_raw = view_record.get("mask_path") or view_record.get("mask")
            if image_raw:
                image_paths.append(resolve_scene_path(scene_dir, image_raw))
            if mask_raw:
                mask_paths.append(resolve_scene_path(scene_dir, mask_raw))
    if not image_paths:
        image_paths = sorted_image_files(scene_dir / "images")
    if not mask_paths:
        mask_paths = sorted_image_files(scene_dir / "masks")
    return image_paths, mask_paths, manifest_path


def letterbox_image_to_shape(
    path: Path,
    target_hw: tuple[int, int],
    mode: str,
    resampling: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> np.ndarray:
    image = Image.open(path).convert(mode)
    target_height, target_width = target_hw
    source_width, source_height = image.size
    scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1))
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    image = image.resize((new_width, new_height), resampling)
    canvas = Image.new(mode, (target_width, target_height), fill)
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    canvas.paste(image, (pad_left, pad_top))
    return np.asarray(canvas)


def load_rgb_for_shape(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    if target_hw[0] == target_hw[1]:
        return preprocess_rgb_image(path, target_hw[0]).astype(np.uint8)
    return letterbox_image_to_shape(path, target_hw, "RGB", Image.Resampling.BICUBIC, (255, 255, 255)).astype(np.uint8)


def load_mask_for_shape(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    if target_hw[0] == target_hw[1]:
        return preprocess_mask_image(path, target_hw[0]).astype(bool)
    mask = letterbox_image_to_shape(path, target_hw, "L", Image.Resampling.NEAREST, 0)
    return (mask > 127).astype(bool)


def load_scene_view(scene_dir: Path, target_view: int, target_hw: tuple[int, int]) -> SceneView:
    image_paths, mask_paths, manifest_path = scene_view_paths(scene_dir)
    if not image_paths:
        raise FileNotFoundError(f"No scene images found in manifest or {scene_dir / 'images'}")
    if not mask_paths:
        raise FileNotFoundError(f"No scene masks found in manifest or {scene_dir / 'masks'}")
    if target_view >= len(image_paths):
        raise IndexError(f"target view {target_view} is outside scene image count {len(image_paths)}")
    if target_view >= len(mask_paths):
        raise IndexError(f"target view {target_view} is outside scene mask count {len(mask_paths)}")
    image_path = image_paths[target_view]
    mask_path = mask_paths[target_view]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    return SceneView(
        rgb=load_rgb_for_shape(image_path, target_hw),
        mask=load_mask_for_shape(mask_path, target_hw),
        image_path=image_path,
        mask_path=mask_path,
        scene_view_count=len(image_paths),
        manifest_path=manifest_path,
    )


def normalize_target_view(target_view: int, view_count: int) -> int:
    resolved_view = target_view
    if resolved_view < 0:
        resolved_view = view_count + resolved_view
    if resolved_view < 0 or resolved_view >= view_count:
        raise IndexError(f"target view {target_view} is outside prediction view count {view_count}")
    return resolved_view


def box_mask(box: tuple[int, int, int, int] | None, shape: tuple[int, int], support_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if box is None:
        return mask
    x0, y0, x1, y1 = [int(value) for value in box]
    mask[y0:y1, x0:x1] = True
    return mask & np.asarray(support_mask, dtype=bool)


def build_roi_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    support_mask = np.asarray(mask, dtype=bool)
    return {
        "full": support_mask,
        "head": box_mask(head_box_from_mask(support_mask), support_mask.shape, support_mask),
        "face": box_mask(face_box_from_mask(support_mask), support_mask.shape, support_mask),
    }


def conf_valid_mask(conf: np.ndarray) -> np.ndarray:
    values = np.asarray(conf, dtype=np.float32)
    return np.isfinite(values) & (values > 0.0)


def synthetic_intrinsic(height: int, width: int) -> np.ndarray:
    focal = float(max(height, width))
    return np.array(
        [
            [focal, 0.0, 0.5 * (width - 1)],
            [0.0, focal, 0.5 * (height - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def depth_to_camera_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth, dtype=np.float32)
    height, width = depth_map.shape
    pixel_y, pixel_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    focal_x = max(abs(float(intrinsic[0, 0])), 1e-6)
    focal_y = max(abs(float(intrinsic[1, 1])), 1e-6)
    center_x = float(intrinsic[0, 2])
    center_y = float(intrinsic[1, 2])
    camera_x = (pixel_x - center_x) * depth_map / focal_x
    camera_y = (pixel_y - center_y) * depth_map / focal_y
    return np.stack((camera_x, camera_y, depth_map), axis=-1).astype(np.float32)


def unoriented_angle_map(
    first_normal: np.ndarray,
    first_valid: np.ndarray,
    second_normal: np.ndarray,
    second_valid: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.asarray(roi_mask, dtype=bool)
        & np.asarray(first_valid, dtype=bool)
        & np.asarray(second_valid, dtype=bool)
        & np.isfinite(first_normal).all(axis=-1)
        & np.isfinite(second_normal).all(axis=-1)
    )
    dot = np.zeros(valid.shape, dtype=np.float32)
    angle = np.full(valid.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        dot_values = np.abs(np.sum(first_normal[valid] * second_normal[valid], axis=-1))
        dot_values = np.clip(dot_values, 0.0, 1.0)
        dot[valid] = dot_values.astype(np.float32)
        angle[valid] = np.degrees(np.arccos(dot_values)).astype(np.float32)
    return angle, dot, valid


def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def compute_metrics(
    first_normal: np.ndarray,
    first_valid: np.ndarray,
    second_normal: np.ndarray,
    second_valid: np.ndarray,
    roi_mask: np.ndarray,
) -> dict[str, Any]:
    roi = np.asarray(roi_mask, dtype=bool)
    angle, dot, valid = unoriented_angle_map(first_normal, first_valid, second_normal, second_valid, roi)
    valid_angles = angle[valid]
    valid_cosines = dot[valid]
    roi_pixels = int(roi.sum())
    valid_pixels = int(valid.sum())
    metrics: dict[str, Any] = {
        "roi_pixels": roi_pixels,
        "valid_pixels": valid_pixels,
        "valid_ratio": float(valid_pixels / roi_pixels) if roi_pixels else None,
        "cos_mean": None,
        "cos_median": None,
        "cos_p10": None,
        "angle_mean_deg": None,
        "angle_median_deg": None,
        "angle_p90_deg": None,
        "angle_p95_deg": None,
        "bad10_frac": None,
        "bad20_frac": None,
        "bad30_frac": None,
    }
    if valid_pixels == 0:
        return metrics
    metrics.update(
        {
            "cos_mean": float(np.mean(valid_cosines)),
            "cos_median": float(np.median(valid_cosines)),
            "cos_p10": percentile_or_none(valid_cosines, 10.0),
            "angle_mean_deg": float(np.mean(valid_angles)),
            "angle_median_deg": float(np.median(valid_angles)),
            "angle_p90_deg": percentile_or_none(valid_angles, 90.0),
            "angle_p95_deg": percentile_or_none(valid_angles, 95.0),
            "bad10_frac": float(np.mean(valid_angles > 10.0)),
            "bad20_frac": float(np.mean(valid_angles > 20.0)),
            "bad30_frac": float(np.mean(valid_angles > 30.0)),
        }
    )
    return metrics


def angle_error_to_rgb(angle: np.ndarray, valid: np.ndarray, max_error_deg: float) -> np.ndarray:
    safe_max_error = max(float(max_error_deg), 1e-6)
    normalized = np.zeros(angle.shape, dtype=np.float32)
    normalized[valid] = np.clip(angle[valid] / safe_max_error, 0.0, 1.0)
    red = (255.0 * normalized).astype(np.uint8)
    green = (255.0 * (1.0 - normalized)).astype(np.uint8)
    blue = np.zeros_like(red, dtype=np.uint8)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid] = 255
    return rgb


def evaluate_entry(spec: EntrySpec, args: argparse.Namespace) -> EntryArtifacts:
    bundle = load_prediction_bundle(spec.predictions_npz, args.normal_format)
    view_count = int(bundle.normal.shape[0])
    target_view = normalize_target_view(args.target_view, view_count)
    target_hw = tuple(int(value) for value in bundle.normal.shape[1:3])
    scene = load_scene_view(spec.scene_dir, target_view, target_hw)
    warnings: list[str] = []
    if scene.scene_view_count != view_count:
        warnings.append(f"scene view count {scene.scene_view_count} differs from prediction view count {view_count}")

    support_mask = scene.mask.astype(bool)
    pred_normal = bundle.normal[target_view]
    pred_valid = bundle.normal_valid[target_view] & support_mask & conf_valid_mask(bundle.normal_conf[target_view])

    depth_view = bundle.depth[target_view]
    if bundle.intrinsic is None:
        depth_intrinsic = synthetic_intrinsic(target_hw[0], target_hw[1])
        depth_intrinsic_source = "synthetic_centered"
        warnings.append("intrinsic missing; depth normals use a synthetic centered pinhole intrinsic")
    else:
        depth_intrinsic = bundle.intrinsic[target_view]
        depth_intrinsic_source = bundle.keys["intrinsic"] or "intrinsic"
    depth_input_valid = (
        support_mask
        & conf_valid_mask(bundle.depth_conf[target_view])
        & np.isfinite(depth_view)
        & (depth_view > 0.0)
    )
    depth_points = depth_to_camera_points(depth_view, depth_intrinsic)
    depth_normal, depth_surface_valid = point_map_to_normal_numpy(depth_points, depth_input_valid)
    depth_normal, depth_vector_valid = normalize_vectors(depth_normal)
    depth_valid = depth_surface_valid & depth_vector_valid & depth_input_valid

    world_points_view = bundle.world_points[target_view]
    if bundle.extrinsic is None:
        camera_points = world_points_view.astype(np.float32, copy=False)
        point_camera_source = "world_points_assumed_camera"
        warnings.append("extrinsic missing; point normals assume world_points are already camera-space")
    else:
        camera_points = points_world_to_camera(world_points_view, bundle.extrinsic[target_view])
        point_camera_source = bundle.keys["extrinsic"] or "extrinsic"
    point_input_valid = (
        support_mask
        & conf_valid_mask(bundle.world_points_conf[target_view])
        & np.isfinite(camera_points).all(axis=-1)
    )
    point_normal, point_surface_valid = point_map_to_normal_numpy(camera_points, point_input_valid)
    point_normal, point_vector_valid = normalize_vectors(point_normal)
    point_valid = point_surface_valid & point_vector_valid & point_input_valid

    roi_masks = build_roi_masks(support_mask)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for roi_name in ROI_ORDER:
        roi_mask = roi_masks[roi_name]
        metrics[roi_name] = {
            "pred_vs_depth": compute_metrics(pred_normal, pred_valid, depth_normal, depth_valid, roi_mask),
            "pred_vs_point": compute_metrics(pred_normal, pred_valid, point_normal, point_valid, roi_mask),
            "depth_vs_point": compute_metrics(depth_normal, depth_valid, point_normal, point_valid, roi_mask),
        }

    full_roi = roi_masks["full"]
    pred_depth_angle, _, pred_depth_valid = unoriented_angle_map(pred_normal, pred_valid, depth_normal, depth_valid, full_roi)
    pred_point_angle, _, pred_point_valid = unoriented_angle_map(pred_normal, pred_valid, point_normal, point_valid, full_roi)

    pred_normal_rgb = normal_to_rgb(pred_normal, pred_valid & full_roi)
    depth_normal_rgb = normal_to_rgb(depth_normal, depth_valid & full_roi)
    point_normal_rgb = normal_to_rgb(point_normal, point_valid & full_roi)
    pred_depth_error_rgb = angle_error_to_rgb(pred_depth_angle, pred_depth_valid, args.max_error_deg)
    pred_point_error_rgb = angle_error_to_rgb(pred_point_angle, pred_point_valid, args.max_error_deg)

    summary = {
        "name": spec.name,
        "predictions_npz": str(spec.predictions_npz),
        "scene_dir": str(spec.scene_dir),
        "view_count": view_count,
        "target_view": target_view,
        "height": int(target_hw[0]),
        "width": int(target_hw[1]),
        "scene_view_count": int(scene.scene_view_count),
        "image_path": str(scene.image_path),
        "mask_path": str(scene.mask_path),
        "scene_manifest_path": str(scene.manifest_path) if scene.manifest_path else None,
        "keys": bundle.keys,
        "normal_format": bundle.normal_format,
        "depth_intrinsic_source": depth_intrinsic_source,
        "point_camera_source": point_camera_source,
        "roi_pixels": {roi_name: int(roi_masks[roi_name].sum()) for roi_name in ROI_ORDER},
        "valid_pixels": {
            "pred_normal": int(pred_valid.sum()),
            "depth_normal": int(depth_valid.sum()),
            "point_normal": int(point_valid.sum()),
        },
        "metrics": metrics,
        "warnings": warnings,
    }
    return EntryArtifacts(
        summary=summary,
        rgb=scene.rgb,
        pred_normal_rgb=pred_normal_rgb,
        depth_normal_rgb=depth_normal_rgb,
        point_normal_rgb=point_normal_rgb,
        pred_depth_error_rgb=pred_depth_error_rgb,
        pred_point_error_rgb=pred_point_error_rgb,
    )


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def iter_metric_rows(entry_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_by_roi = entry_summary["metrics"]
    for roi_name in ROI_ORDER:
        for comparison in COMPARISON_ORDER:
            metrics = metrics_by_roi[roi_name][comparison]
            row = {
                "entry": entry_summary["name"],
                "view_count": entry_summary["view_count"],
                "target_view": entry_summary["target_view"],
                "roi": roi_name,
                "comparison": comparison,
                **metrics,
            }
            rows.append(row)
    return rows


def write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    headers = [
        "entry",
        "view_count",
        "target_view",
        "roi",
        "comparison",
        "roi_pixels",
        "valid_pixels",
        "valid_ratio",
        "cos_mean",
        "cos_median",
        "cos_p10",
        "angle_mean_deg",
        "angle_median_deg",
        "angle_p90_deg",
        "angle_p95_deg",
        "bad10_frac",
        "bad20_frac",
        "bad30_frac",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for entry in entries:
            for row in iter_metric_rows(entry):
                writer.writerow({header: format_value(row.get(header), digits=6) for header in headers})


def write_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Multiview normal consistency summary",
        "",
        "| Entry | Views | ROI | Comparison | Valid | Mean cos | Mean angle deg | P90 angle deg | Bad >20 deg |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        for row in iter_metric_rows(entry):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["entry"]),
                        str(row["view_count"]),
                        str(row["roi"]),
                        str(row["comparison"]),
                        str(row["valid_pixels"]),
                        format_value(row["cos_mean"]),
                        format_value(row["angle_mean_deg"]),
                        format_value(row["angle_p90_deg"]),
                        format_value(row["bad20_frac"]),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def image_to_tile(image: np.ndarray, tile_size: int) -> Image.Image:
    values = np.asarray(image)
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    tile_image = Image.fromarray(values).convert("RGB")
    tile_image.thumbnail((tile_size, tile_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
    pad_left = (tile_size - tile_image.size[0]) // 2
    pad_top = (tile_size - tile_image.size[1]) // 2
    canvas.paste(tile_image, (pad_left, pad_top))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def make_overview(path: Path, artifacts: list[EntryArtifacts], tile_size: int) -> None:
    columns = [
        ("rgb", "RGB"),
        ("pred_normal_rgb", "predicted normal"),
        ("depth_normal_rgb", "depth-normal"),
        ("point_normal_rgb", "point-normal"),
        ("pred_depth_error_rgb", "pred-vs-depth error"),
        ("pred_point_error_rgb", "pred-vs-point error"),
    ]
    header_height = 28
    caption_height = 32
    row_height = caption_height + tile_size
    width = len(columns) * tile_size
    height = header_height + len(artifacts) * row_height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for column_index, (_, title) in enumerate(columns):
        draw_text(draw, (column_index * tile_size + 6, 8), title, (0, 0, 0))
    for row_index, artifact in enumerate(artifacts):
        top = header_height + row_index * row_height
        row_label = f"{artifact.summary['name']} | {artifact.summary['view_count']} views | v{artifact.summary['target_view']}"
        for column_index, (field_name, _) in enumerate(columns):
            left = column_index * tile_size
            caption = row_label if column_index == 0 else ""
            draw.rectangle((left, top, left + tile_size, top + caption_height), fill=(245, 245, 245))
            if caption:
                draw_text(draw, (left + 6, top + 8), caption[: max(16, tile_size // 6)], (0, 0, 0))
            tile = image_to_tile(getattr(artifact, field_name), tile_size)
            canvas.paste(tile, (left, top + caption_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [parse_entry_spec(entry_text) for entry_text in args.entry]

    artifacts: list[EntryArtifacts] = []
    for spec in specs:
        artifacts.append(evaluate_entry(spec, args))

    json_path = output_dir / "multiview_normal_consistency_summary.json"
    csv_path = output_dir / "multiview_normal_consistency_summary.csv"
    markdown_path = output_dir / "multiview_normal_consistency_summary.md"
    overview_path = output_dir / "multiview_normal_consistency_overview.png"

    entries = [artifact.summary for artifact in artifacts]
    make_overview(overview_path, artifacts, int(args.overview_tile_size))
    write_csv(csv_path, entries)
    write_markdown(markdown_path, entries)

    summary = {
        "target_view": int(args.target_view),
        "normal_format_arg": args.normal_format,
        "max_error_deg": float(args.max_error_deg),
        "entries": entries,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "overview_png": str(overview_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {overview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
