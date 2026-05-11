from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


POINT_SOURCES = ("world_points", "depth_unprojection")


def preprocess_rgb(path: Path, target_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def load_rgb_stack(image_dir: Path, target_size: int) -> np.ndarray:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    images = [preprocess_rgb(path, target_size=target_size) for path in image_paths]
    return np.stack(images, axis=0)


def preprocess_mask(mask_path: Path, target_size: int) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def load_mask_stack(mask_dir: Path, target_size: int) -> np.ndarray:
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    masks = [preprocess_mask(path, target_size=target_size) for path in mask_paths]
    return np.stack(masks, axis=0)


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map_numpy(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    world_points = []
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)

        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]

        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))

    return np.stack(world_points, axis=0)


def resolve_point_source(data: np.lib.npyio.NpzFile, point_source: str) -> tuple[np.ndarray, np.ndarray]:
    if point_source == "world_points":
        return data["world_points"], data["world_points_conf"]
    if point_source == "depth_unprojection":
        world_points = unproject_depth_map_to_point_map_numpy(data["depth"], data["extrinsic"], data["intrinsic"])
        return world_points, data["depth_conf"]
    raise ValueError(f"Unsupported point source: {point_source}")


def build_filtered_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    colors: np.ndarray,
    masks: np.ndarray | None,
    max_points: int,
    conf_percentile: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    rgb = colors.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
    if masks is not None:
        valid &= masks.reshape(-1) > 0

    if not np.any(valid):
        raise RuntimeError("No valid points after filtering.")

    conf_valid = conf[valid]
    conf_threshold = float(np.percentile(conf_valid, conf_percentile))
    keep = valid & (conf >= conf_threshold)
    if not np.any(keep):
        keep = valid

    kept_indices = np.flatnonzero(keep)
    if len(kept_indices) > max_points:
        kept_indices = rng.choice(kept_indices, size=max_points, replace=False)

    kept_points = points[kept_indices]
    kept_rgb = rgb[kept_indices]
    summary = {
        "valid_points_before_conf": int(valid.sum()),
        "conf_threshold": conf_threshold,
        "points_after_conf": int(keep.sum()),
        "points_written": int(len(kept_indices)),
    }
    return kept_points, kept_rgb, summary


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Open3D is required for render_open3d_pointcloud.py. "
            "Install it with `pip install open3d` in the active environment."
        ) from exc
    return o3d


def _camera_presets(center: np.ndarray, radius: float) -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "front",
            {
                "front": [0.0, 0.0, -1.0],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": 0.55,
            },
        ),
        (
            "side",
            {
                "front": [1.0, 0.0, 0.0],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": 0.55,
            },
        ),
        (
            "top",
            {
                "front": [0.0, -1.0, 0.0],
                "lookat": center.tolist(),
                "up": [0.0, 0.0, -1.0],
                "zoom": 0.55,
            },
        ),
        (
            "iso",
            {
                "front": [0.65, -0.25, -0.72],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": 0.52,
            },
        ),
        (
            "head_close",
            {
                "front": [0.08, -0.02, -0.996],
                "lookat": (center + np.array([0.0, -0.10 * radius, 0.10 * radius], dtype=np.float32)).tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": 0.78,
            },
        ),
        (
            "face_close",
            {
                "front": [0.15, -0.05, -0.99],
                "lookat": (center + np.array([0.0, -0.18 * radius, 0.18 * radius], dtype=np.float32)).tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": 0.95,
            },
        ),
    ]


def _save_open3d_renders(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    width: int,
    height: int,
    point_size: float,
    interactive: bool,
) -> list[str]:
    o3d = _load_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0.0, 1.0).astype(np.float64))

    bounds = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float32)
    extent = np.asarray(bounds.get_extent(), dtype=np.float32)
    radius = float(np.linalg.norm(extent) + 1e-6)

    if interactive:  # pragma: no cover - requires GUI
        o3d.visualization.draw_geometries([pcd], window_name="VGGT Human Point Cloud", width=width, height=height)
        return []

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VGGT Open3D Render", width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    render_option.point_size = float(point_size)
    render_option.light_on = True

    ctr = vis.get_view_control()
    saved = []
    for name, preset in _camera_presets(center=center, radius=radius):
        ctr.set_front(preset["front"])
        ctr.set_lookat(preset["lookat"])
        ctr.set_up(preset["up"])
        ctr.set_zoom(float(preset["zoom"]))
        vis.poll_events()
        vis.update_renderer()
        output_path = output_dir / f"{name}.png"
        vis.capture_screen_image(str(output_path), do_render=True)
        saved.append(str(output_path))

    vis.destroy_window()
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VGGT point clouds with Open3D for clearer human-region visualization.")
    parser.add_argument("--predictions-npz", required=True, help="Path to predictions.npz")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing images/ and optionally masks/")
    parser.add_argument("--output-dir", required=True, help="Output directory for PLY and Open3D screenshots")
    parser.add_argument(
        "--point-source",
        choices=POINT_SOURCES,
        default="world_points",
        help="3D point source: precomputed world_points or depth+camera unprojection.",
    )
    parser.add_argument("--max-points", type=int, default=300000, help="Maximum points to keep after filtering")
    parser.add_argument("--conf-percentile", type=float, default=40.0, help="Keep points at or above this confidence percentile")
    parser.add_argument("--human-only", action="store_true", help="Filter the point cloud using scene masks if available")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument("--width", type=int, default=1600, help="Render width")
    parser.add_argument("--height", type=int, default=1200, help="Render height")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive Open3D viewer instead of offscreen screenshots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(args.predictions_npz, allow_pickle=False)
    world_points, world_points_conf = resolve_point_source(predictions, args.point_source)

    scene_dir = Path(args.scene_dir).resolve()
    target_size = int(world_points.shape[1])
    colors = load_rgb_stack(scene_dir / "images", target_size=target_size)
    masks = None
    if args.human_only and (scene_dir / "masks").is_dir():
        masks = load_mask_stack(scene_dir / "masks", target_size=target_size)

    rng = np.random.default_rng(args.seed)
    points, rgb, summary = build_filtered_cloud(
        world_points=world_points,
        world_points_conf=world_points_conf,
        colors=colors,
        masks=masks,
        max_points=int(args.max_points),
        conf_percentile=float(args.conf_percentile),
        rng=rng,
    )

    o3d = _load_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32) / 255.0).astype(np.float64))
    ply_path = output_dir / "pointcloud_open3d.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)

    screenshots = _save_open3d_renders(
        points=points,
        colors=rgb,
        output_dir=output_dir,
        width=int(args.width),
        height=int(args.height),
        point_size=float(args.point_size),
        interactive=bool(args.interactive),
    )

    payload = {
        "predictions_npz": str(Path(args.predictions_npz).resolve()),
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "point_source": args.point_source,
        "human_only": bool(args.human_only),
        "summary": summary,
        "ply_path": str(ply_path),
        "screenshots": screenshots,
    }
    (output_dir / "open3d_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
