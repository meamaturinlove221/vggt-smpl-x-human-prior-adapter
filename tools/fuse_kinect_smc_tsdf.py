from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse a 4K4D Kinect SMC into a TSDF mesh in the dataset real-world coordinate system. "
            "This is an asset-quality diagnostic, not a VGGT teacher pass."
        )
    )
    parser.add_argument("--kinect-smc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--kinect-cameras", nargs="*", default=["all"])
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--voxel-length", type=float, default=0.005)
    parser.add_argument("--sdf-trunc", type=float, default=0.025)
    parser.add_argument("--mask-erode", type=int, default=1)
    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-iterations", type=int, default=6)
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-width", type=int, default=1200)
    parser.add_argument("--render-height", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
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


def numeric_key(text: str) -> int:
    return int(str(text))


def select_cameras(handle: h5py.File, requested: list[str]) -> list[str]:
    available = sorted([str(k) for k in handle["Kinect"].keys() if str(k).isdigit()], key=numeric_key)
    requested_l = [str(x).strip().lower() for x in requested if str(x).strip()]
    if not requested_l or "all" in requested_l:
        return available
    selected = [str(int(x)) for x in requested_l]
    missing = [x for x in selected if x not in available]
    if missing:
        raise KeyError(f"missing Kinect cameras {missing}; available={available}")
    return selected


def maybe_undistort(depth_mm: np.ndarray, mask: np.ndarray, k_mat: np.ndarray, d_vec: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return depth_mm, mask
    try:
        import cv2
    except Exception:
        return depth_mm, mask
    height, width = depth_mm.shape
    map_x, map_y = cv2.initUndistortRectifyMap(
        k_mat.astype(np.float64),
        np.asarray(d_vec, dtype=np.float64).reshape(-1),
        None,
        k_mat.astype(np.float64),
        (int(width), int(height)),
        cv2.CV_32FC1,
    )
    depth_u = cv2.remap(depth_mm, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_u = cv2.remap(mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return depth_u, mask_u


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    try:
        import cv2
    except Exception:
        return mask.astype(bool)
    kernel = np.ones((int(pixels) * 2 + 1, int(pixels) * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def remove_small_mesh_components(mesh):
    clusters, counts, _areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if clusters.size == 0 or counts.size == 0:
        return mesh, {"component_count": 0, "kept_triangles": int(len(mesh.triangles)), "removed_triangles": 0}
    keep_cluster = int(np.argmax(counts))
    remove = clusters != keep_cluster
    mesh.remove_triangles_by_mask(remove.tolist())
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "component_count": int(counts.size),
        "kept_cluster": keep_cluster,
        "kept_triangles": int((~remove).sum()),
        "removed_triangles": int(remove.sum()),
    }


def render_solid(mesh, output_dir: Path, *, width: int, height: int) -> list[str]:
    import open3d as o3d

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return []
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.72, 0.72])
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    radius = float(np.linalg.norm(extent) + 1e-6)
    presets = [
        ("front", [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
        ("back", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ("side", [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ("left_side", [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ("top", [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ("iso", [0.65, -0.35, -0.68], [0.0, -1.0, 0.0]),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Kinect TSDF Asset Review", width=int(width), height=int(height), visible=False)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    opt.light_on = True
    ctr = vis.get_view_control()
    saved: list[str] = []
    for name, front, up in presets:
        ctr.set_front(front)
        ctr.set_lookat(center)
        ctr.set_up(up)
        ctr.set_zoom(0.88 if name != "top" else 0.75)
        vis.poll_events()
        vis.update_renderer()
        out = output_dir / f"{name}.png"
        vis.capture_screen_image(str(out), do_render=True)
        saved.append(str(out))
    vis.destroy_window()
    return saved


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; re-run with --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    with h5py.File(args.kinect_smc, "r") as handle:
        selected = select_cameras(handle, list(args.kinect_cameras))
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(args.voxel_length),
            sdf_trunc=float(args.sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        per_camera = []
        for camera_id in sorted(selected, key=numeric_key):
            calib_id = f"{int(camera_id):02d}"
            depth_raw = handle[f"Kinect/{int(camera_id)}/depth/{int(args.frame)}"][()]
            mask_raw = handle[f"Kinect/{int(camera_id)}/mask/{int(args.frame)}"][()] > 0
            intrinsic = handle[f"Calibration/Kinect/{calib_id}/K"][()].astype(np.float64)
            distortion = handle[f"Calibration/Kinect/{calib_id}/D"][()].astype(np.float64)
            cam_to_world = handle[f"Calibration/Kinect/{calib_id}/RT"][()].astype(np.float64)
            depth_u, mask_u = maybe_undistort(depth_raw, mask_raw, intrinsic, distortion, bool(args.undistort))
            mask_u = erode_mask(mask_u, int(args.mask_erode))
            depth_m = depth_u.astype(np.float32) / float(args.depth_scale)
            valid = mask_u & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
            depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
            height, width = depth_m.shape
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
                o3d.geometry.Image(np.ascontiguousarray(depth_m)),
                depth_scale=1.0,
                depth_trunc=float(args.max_depth_m),
                convert_rgb_to_intensity=False,
            )
            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                int(width),
                int(height),
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
            )
            volume.integrate(rgbd, o3d_intr, np.linalg.inv(cam_to_world))
            vals = depth_m[valid]
            per_camera.append(
                {
                    "kinect_camera": str(camera_id),
                    "valid_pixels": int(valid.sum()),
                    "valid_ratio": float(valid.sum() / max(valid.size, 1)),
                    "depth_percentiles_m": [float(v) for v in np.percentile(vals, [5, 50, 95])] if vals.size else [],
                }
            )

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    component_summary: dict[str, Any] = {}
    if bool(args.keep_largest_component):
        mesh, component_summary = remove_small_mesh_components(mesh)
    if int(args.smooth_iterations) > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))
    mesh.compute_vertex_normals()
    mesh_path = output_dir / "kinect_tsdf_real_world.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    screenshots = render_solid(mesh, output_dir / "solid_mesh", width=int(args.render_width), height=int(args.render_height))

    summary = {
        "task": "kinect_smc_tsdf_asset_quality",
        "truthful_status": "asset_quality_diagnostic_not_teacher_pass",
        "kinect_smc": str(args.kinect_smc.resolve()),
        "frame": int(args.frame),
        "selected_kinect_cameras": selected,
        "parameters": {
            "voxel_length": float(args.voxel_length),
            "sdf_trunc": float(args.sdf_trunc),
            "mask_erode": int(args.mask_erode),
            "undistort": bool(args.undistort),
            "smooth_iterations": int(args.smooth_iterations),
        },
        "per_camera": per_camera,
        "mesh": {
            "path": str(mesh_path),
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
            "extent": np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64),
            "component_filter": component_summary,
        },
        "screenshots": screenshots,
        "notes": [
            "This only checks whether the Kinect asset itself forms a clean surface.",
            "It is not aligned to the target VGGT world and cannot be used for training unless later pose/deformation and strict teacher gates pass.",
        ],
    }
    (output_dir / "kinect_smc_tsdf_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
