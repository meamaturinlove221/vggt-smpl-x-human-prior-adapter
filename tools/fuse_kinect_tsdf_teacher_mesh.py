from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_kinect_depth_teacher_targets import (  # noqa: E402
    apply_similarity,
    build_camera_alignment_correspondences,
    json_ready,
    load_json,
    load_rgb_camera_params,
    resolve_kinect_smc,
    robust_transform,
    select_kinect_cameras,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse real 4K4D Kinect depth/masks with TSDF and export an aligned VGGT-world "
            "teacher mesh. This is a teacher-source diagnostic only; it never trains or "
            "claims mentor-final quality by itself."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kinect-smc", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--kinect-cameras", nargs="*", default=["all"])
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--voxel-length", type=float, default=0.006)
    parser.add_argument("--sdf-trunc", type=float, default=0.025)
    parser.add_argument("--mask-erode", type=int, default=2)
    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-axis-scale", type=float, default=0.005)
    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _numeric_key(value: str) -> int:
    return int(str(value))


def _as_o3d_image(array: np.ndarray):
    import open3d as o3d

    return o3d.geometry.Image(np.ascontiguousarray(array))


def _maybe_undistort(depth_mm: np.ndarray, mask: np.ndarray, k_mat: np.ndarray, d_vec: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
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


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    try:
        import cv2
    except Exception:
        return mask.astype(bool)
    kernel_size = int(max(1, pixels * 2 + 1))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _remove_small_mesh_components(mesh):
    import numpy as np

    clusters, counts, _areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if clusters.size == 0 or counts.size == 0:
        return mesh, {"component_count": 0, "kept_triangles": int(len(mesh.triangles)), "removed_triangles": 0}
    keep_cluster = int(np.argmax(counts))
    remove = clusters != keep_cluster
    removed = int(remove.sum())
    mesh.remove_triangles_by_mask(remove.tolist())
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "component_count": int(counts.size),
        "kept_cluster": keep_cluster,
        "kept_triangles": int((~remove).sum()),
        "removed_triangles": removed,
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    scene_dir = args.scene_dir.resolve()
    scene_manifest = load_json(scene_dir / "scene_manifest.json")
    frame = int(args.frame if args.frame is not None else scene_manifest.get("frame_id", 0))
    kinect_smc = resolve_kinect_smc(scene_manifest, args.kinect_smc)
    rgb_params = load_rgb_camera_params(scene_manifest)
    with np.load(args.base_predictions, allow_pickle=False) as pred:
        predicted_extrinsic = np.asarray(pred["extrinsic"], dtype=np.float64)

    with h5py.File(kinect_smc, "r") as handle:
        selected_cameras = select_kinect_cameras(handle, list(args.kinect_cameras))
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(args.voxel_length),
            sdf_trunc=float(args.sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        per_camera: list[dict[str, Any]] = []
        for camera_id in sorted(selected_cameras, key=_numeric_key):
            calib_id = f"{int(camera_id):02d}"
            depth_raw = handle[f"Kinect/{int(camera_id)}/depth/{int(frame)}"][()]
            mask_raw = handle[f"Kinect/{int(camera_id)}/mask/{int(frame)}"][()] > 0
            intrinsic = handle[f"Calibration/Kinect/{calib_id}/K"][()].astype(np.float64)
            distortion = handle[f"Calibration/Kinect/{calib_id}/D"][()].astype(np.float64)
            cam_to_world = handle[f"Calibration/Kinect/{calib_id}/RT"][()].astype(np.float64)
            depth_u, mask_u = _maybe_undistort(depth_raw, mask_raw, intrinsic, distortion, bool(args.undistort))
            mask_u = _erode_mask(mask_u, int(args.mask_erode))
            depth_m = depth_u.astype(np.float32) / float(args.depth_scale)
            valid = mask_u & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
            depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
            height, width = depth_m.shape
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(width),
                int(height),
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                _as_o3d_image(np.zeros((height, width, 3), dtype=np.uint8)),
                _as_o3d_image(depth_m),
                depth_scale=1.0,
                depth_trunc=float(args.max_depth_m),
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, o3d_intrinsic, np.linalg.inv(cam_to_world))
            values = depth_m[valid]
            per_camera.append(
                {
                    "kinect_camera": str(camera_id),
                    "valid_pixels": int(valid.sum()),
                    "valid_ratio": float(valid.sum() / max(valid.size, 1)),
                    "depth_percentiles_m": [float(v) for v in np.percentile(values, [5, 50, 95])] if values.size else [],
                }
            )

    real_mesh = volume.extract_triangle_mesh()
    real_mesh.remove_degenerate_triangles()
    real_mesh.remove_duplicated_triangles()
    real_mesh.remove_duplicated_vertices()
    real_mesh.remove_non_manifold_edges()
    component_summary: dict[str, Any] = {}
    if bool(args.keep_largest_component):
        real_mesh, component_summary = _remove_small_mesh_components(real_mesh)
    if int(args.smooth_iterations) > 0 and len(real_mesh.triangles) > 0:
        real_mesh = real_mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))
    real_mesh.compute_vertex_normals()

    real_mesh_path = output_dir / "kinect_tsdf_real_world.ply"
    o3d.io.write_triangle_mesh(str(real_mesh_path), real_mesh)

    target_camera_id = normalize_camera_id(scene_manifest["exported_views"][0]["camera_id"])
    target_w2c_real = np.asarray(rgb_params[target_camera_id]["world_to_cam"], dtype=np.float64)
    source_corr, target_corr, alignment_summary = build_camera_alignment_correspondences(
        scene_manifest,
        rgb_params,
        predicted_extrinsic,
        include_axes=True,
        axis_scale=float(args.camera_axis_scale),
    )
    transform_summary, transform_matrix = robust_transform(
        source_corr,
        target_corr,
        mode="similarity",
        max_correspondences=0,
        seed=20260502,
    )
    vertices_world = np.asarray(real_mesh.vertices, dtype=np.float64)
    vertices_target_cam = vertices_world @ target_w2c_real[:3, :3].T + target_w2c_real[:3, 3]
    vertices_vggt = apply_similarity(
        vertices_target_cam.reshape(1, -1, 3),
        float(transform_summary["scale"]),
        np.asarray(transform_summary["rotation"], dtype=np.float64),
        np.asarray(transform_summary["translation"], dtype=np.float64),
    ).reshape(-1, 3)
    vggt_mesh = o3d.geometry.TriangleMesh(real_mesh)
    vggt_mesh.vertices = o3d.utility.Vector3dVector(vertices_vggt.astype(np.float64))
    vggt_mesh.compute_vertex_normals()
    vggt_mesh_path = output_dir / "kinect_tsdf_vggt_world.ply"
    o3d.io.write_triangle_mesh(str(vggt_mesh_path), vggt_mesh)

    summary = {
        "task": "kinect_tsdf_teacher_mesh",
        "truthful_status": "teacher_source_diagnostic_not_pass",
        "scene_dir": str(scene_dir),
        "base_predictions": str(args.base_predictions.resolve()),
        "kinect_smc": str(kinect_smc),
        "frame": int(frame),
        "selected_kinect_cameras": selected_cameras,
        "parameters": {
            "voxel_length": float(args.voxel_length),
            "sdf_trunc": float(args.sdf_trunc),
            "mask_erode": int(args.mask_erode),
            "undistort": bool(args.undistort),
            "camera_axis_scale": float(args.camera_axis_scale),
            "smooth_iterations": int(args.smooth_iterations),
        },
        "per_camera": per_camera,
        "real_mesh": {
            "path": str(real_mesh_path),
            "vertices": int(len(real_mesh.vertices)),
            "triangles": int(len(real_mesh.triangles)),
            "component_filter": component_summary,
        },
        "vggt_mesh": {
            "path": str(vggt_mesh_path),
            "vertices": int(len(vggt_mesh.vertices)),
            "triangles": int(len(vggt_mesh.triangles)),
        },
        "alignment": alignment_summary,
        "target_alignment": transform_summary,
        "transform_matrix_real_targetcam_to_vggt_world": transform_matrix,
        "notes": [
            "TSDF mesh must still pass audit_headface_teacher_surface plus explicit Open3D visual review.",
            "This script avoids point-aligning to the current VGGT shell; it uses calibrated RGB camera centers/axes.",
            "Do not train or upload to cloud unless the strict teacher gate and later candidate gate pass.",
        ],
    }
    _write_summary(output_dir / "kinect_tsdf_teacher_mesh_summary.json", summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
