from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, preprocess_mask_image, preprocess_rgb_image  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse VGGT depth maps into a TSDF point cloud for a geometry-only multi-view depth gate."
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-key", choices=("depth_conf", "world_points_conf"), default="depth_conf")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--voxel-length", type=float, default=0.004)
    parser.add_argument("--sdf-trunc", type=float, default=0.02)
    parser.add_argument("--depth-trunc", type=float, default=8.0)
    parser.add_argument("--target-view", type=int, default=0)
    parser.add_argument("--max-integrate-views", type=int, default=-1)
    parser.add_argument("--no-human-mask", action="store_true")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_manifest(scene_dir: Path) -> dict[str, Any] | None:
    for name in ("scene_manifest.json", "scene_manifest"):
        path = scene_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _resolve_scene_path(scene_dir: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = scene_dir / path
    if candidate.exists():
        return candidate
    return path


def _sorted_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _scene_paths(scene_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest = _read_manifest(scene_dir)
    images: list[Path] = []
    masks: list[Path] = []
    if manifest:
        for record in manifest.get("exported_views", []):
            image_raw = record.get("image_path") or record.get("rgb_path") or record.get("image")
            mask_raw = record.get("mask_path") or record.get("mask")
            if image_raw:
                images.append(_resolve_scene_path(scene_dir, image_raw))
            if mask_raw:
                masks.append(_resolve_scene_path(scene_dir, mask_raw))
    if not images:
        images = _sorted_files(scene_dir / "images")
    if not masks:
        masks = _sorted_files(scene_dir / "masks")
    if not images:
        raise FileNotFoundError(f"No images found in {scene_dir}")
    if not masks:
        raise FileNotFoundError(f"No masks found in {scene_dir}")
    return images, masks


def _load_scene(scene_dir: Path, height: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_paths, mask_paths = _scene_paths(scene_dir)
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}")
    images = [preprocess_rgb_image(path, height) for path in image_paths]
    masks = [preprocess_mask_image(path, height) for path in mask_paths]
    return np.stack(images), np.stack(masks).astype(bool), [path.name for path in image_paths]


def _box_mask(mask: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    if box is None:
        return out
    x0, y0, x1, y1 = [int(v) for v in box]
    out[y0:y1, x0:x1] = True
    return out & mask


def _roi_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "full": mask.astype(bool),
        "head": _box_mask(mask, head_box_from_mask(mask)),
        "face": _box_mask(mask, face_box_from_mask(mask)),
    }


def _project(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = extrinsic[:, :3].astype(np.float32)
    translation = extrinsic[:, 3].astype(np.float32)
    cam = points @ rotation.T + translation
    z = cam[:, 2]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u = fx * cam[:, 0] / np.clip(z, 1e-6, None) + cx
    v = fy * cam[:, 1] / np.clip(z, 1e-6, None) + cy
    return u, v, z


def _save_camera_projection(
    points: np.ndarray,
    colors: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    u, v, z = _project(points, intrinsic, extrinsic)
    valid = (z > 0.05) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    order = np.argsort(z[valid])[::-1]
    uu = np.round(u[valid][order]).astype(np.int32)
    vv = np.round(v[valid][order]).astype(np.int32)
    cc = colors[valid][order]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[vv, uu] = np.clip(cc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(output_path)


def _save_ortho(points: np.ndarray, colors: np.ndarray, output_path: Path, axes: tuple[int, int], title: str, width: int, height: int) -> None:
    xy = points[:, axes]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    uv = (xy - lo) / span
    u = np.clip((uv[:, 0] * (width - 1)).round().astype(np.int32), 0, width - 1)
    v = np.clip(((1.0 - uv[:, 1]) * (height - 1)).round().astype(np.int32), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[v, u] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 360, 28), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0))
    image.save(output_path)


def main() -> int:
    args = parse_args()
    import open3d as o3d

    pred_path = Path(args.predictions_npz)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(pred_path, allow_pickle=False) as payload:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        if depth.ndim == 4:
            depth = depth[..., 0]
        intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
        conf = np.asarray(payload[args.conf_key], dtype=np.float32)
    views, height, width = depth.shape
    images, masks, names = _load_scene(scene_dir, height)
    if images.shape[0] != views:
        raise ValueError(f"scene views {images.shape[0]} != predictions views {views}")
    finite_conf = conf[np.isfinite(conf)]
    threshold = float(args.conf_threshold) if args.conf_threshold is not None else float(np.percentile(finite_conf, float(args.conf_percentile)))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_length),
        sdf_trunc=float(args.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrate_count = views if int(args.max_integrate_views) <= 0 else min(views, int(args.max_integrate_views))
    records = []
    for view_idx in range(integrate_count):
        valid = np.isfinite(depth[view_idx]) & (depth[view_idx] > 0.05) & (depth[view_idx] < float(args.depth_trunc))
        valid &= np.isfinite(conf[view_idx]) & (conf[view_idx] >= threshold)
        if not bool(args.no_human_mask):
            valid &= masks[view_idx]
        filtered_depth = np.where(valid, depth[view_idx], 0.0).astype(np.float32)
        color = o3d.geometry.Image(images[view_idx].astype(np.uint8))
        depth_image = o3d.geometry.Image(filtered_depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_image,
            depth_scale=1.0,
            depth_trunc=float(args.depth_trunc),
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[view_idx, 0, 0]),
            float(intrinsics[view_idx, 1, 1]),
            float(intrinsics[view_idx, 0, 2]),
            float(intrinsics[view_idx, 1, 2]),
        )
        extrinsic4 = np.eye(4, dtype=np.float64)
        extrinsic4[:3, :4] = extrinsics[view_idx].astype(np.float64)
        volume.integrate(rgbd, intrinsic, extrinsic4)
        records.append({"view": int(view_idx), "image": names[view_idx], "integrated_pixels": int(valid.sum())})
    pcd = volume.extract_point_cloud()
    if len(pcd.points) == 0:
        raise RuntimeError("TSDF produced zero points")
    pcd.estimate_normals()
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    o3d.io.write_point_cloud(str(output_dir / "tsdf_pointcloud.ply"), pcd, write_ascii=False, compressed=False)
    target_view = int(args.target_view)
    if target_view < 0:
        target_view = views + target_view
    roi_counts = {}
    projected = {}
    u, v, z = _project(points, intrinsics[target_view], extrinsics[target_view])
    in_image = (z > 0.05) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    uu = np.clip(np.round(u).astype(np.int32), 0, width - 1)
    vv = np.clip(np.round(v).astype(np.int32), 0, height - 1)
    for roi_name, roi_mask in _roi_masks(masks[target_view]).items():
        inside = in_image & roi_mask[vv, uu]
        roi_counts[roi_name] = int(inside.sum())
        projected[roi_name] = inside
    _save_camera_projection(points, colors, intrinsics[target_view], extrinsics[target_view], output_dir / "target_camera_projection.png", width, height)
    _save_ortho(points, colors, output_dir / "front_ortho.png", (0, 1), "TSDF front ortho", int(args.width), int(args.height))
    _save_ortho(points, colors, output_dir / "side_ortho.png", (2, 1), "TSDF side ortho", int(args.width), int(args.height))
    summary = {
        "predictions_npz": str(pred_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "conf_key": args.conf_key,
        "conf_threshold": threshold,
        "conf_percentile": float(args.conf_percentile),
        "voxel_length": float(args.voxel_length),
        "sdf_trunc": float(args.sdf_trunc),
        "depth_trunc": float(args.depth_trunc),
        "integrated_views": integrate_count,
        "integrated_records": records,
        "points_after_tsdf": int(len(points)),
        "target_view": target_view,
        "target_roi_projected_counts": roi_counts,
        "truthful_gate": "TSDF is a postprocess candidate; it passes only if face/head/full visuals form cleaner geometry without new holes or shell artifacts.",
    }
    (output_dir / "tsdf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
