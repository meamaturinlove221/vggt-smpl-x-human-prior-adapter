from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only continuous-topology teacher construction. It aligns a base "
            "mesh to a teacher point cloud and moves only nearby vertices toward denoised "
            "teacher evidence, then smooths the existing topology. This does not create "
            "new face/hair truth and must pass strict teacher gate before any training."
        )
    )
    parser.add_argument("--base-mesh", required=True)
    parser.add_argument("--target-pointcloud", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="deformed_mesh.ply")
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--target-stat-nb-neighbors", type=int, default=32)
    parser.add_argument("--target-stat-std-ratio", type=float, default=1.8)
    parser.add_argument("--icp-threshold", type=float, default=0.08)
    parser.add_argument("--icp-max-iter", type=int, default=80)
    parser.add_argument("--deform-max-distance", type=float, default=0.055)
    parser.add_argument("--deform-strength", type=float, default=0.65)
    parser.add_argument("--bbox-padding", type=float, default=0.035)
    parser.add_argument("--smooth-iterations", type=int, default=12)
    parser.add_argument("--preserve-unobserved", action="store_true")
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


def _bbox_mask(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.all((points >= bbox_min[None, :]) & (points <= bbox_max[None, :]), axis=1)


def _copy_mesh(mesh):
    import open3d as o3d

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64).copy())
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles, dtype=np.int32).copy())
    if np.asarray(mesh.vertex_colors).shape[0] == np.asarray(mesh.vertices).shape[0]:
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors, dtype=np.float64).copy())
    out.compute_vertex_normals()
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(args.output_name)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    import open3d as o3d

    base_path = Path(args.base_mesh)
    target_path = Path(args.target_pointcloud)
    base_mesh = o3d.io.read_triangle_mesh(str(base_path))
    if len(base_mesh.vertices) == 0 or len(base_mesh.triangles) == 0:
        raise RuntimeError(f"Bad base mesh: {base_path}")
    base_mesh.compute_vertex_normals()
    target_pcd = o3d.io.read_point_cloud(str(target_path))
    if len(target_pcd.points) == 0:
        raise RuntimeError(f"Bad target point cloud: {target_path}")
    if float(args.voxel_size) > 0:
        target_pcd = target_pcd.voxel_down_sample(float(args.voxel_size))
    if int(args.target_stat_nb_neighbors) > 0 and len(target_pcd.points) > int(args.target_stat_nb_neighbors):
        target_pcd, keep = target_pcd.remove_statistical_outlier(
            nb_neighbors=int(args.target_stat_nb_neighbors),
            std_ratio=float(args.target_stat_std_ratio),
        )

    target_points = np.asarray(target_pcd.points, dtype=np.float64)
    bbox_min = target_points.min(axis=0) - float(args.bbox_padding)
    bbox_max = target_points.max(axis=0) + float(args.bbox_padding)
    base_vertices = np.asarray(base_mesh.vertices, dtype=np.float64)
    observed_vertex_mask = _bbox_mask(base_vertices, bbox_min, bbox_max)
    if not bool(observed_vertex_mask.any()):
        raise RuntimeError("No base mesh vertices overlap the teacher pointcloud bbox.")

    base_crop = o3d.geometry.PointCloud()
    base_crop.points = o3d.utility.Vector3dVector(base_vertices[observed_vertex_mask])
    base_crop.estimate_normals()
    target_pcd.estimate_normals()
    icp = o3d.pipelines.registration.registration_icp(
        base_crop,
        target_pcd,
        float(args.icp_threshold),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(args.icp_max_iter)),
    )

    aligned_mesh = _copy_mesh(base_mesh)
    aligned_mesh.transform(icp.transformation)
    aligned_vertices = np.asarray(aligned_mesh.vertices, dtype=np.float64)
    observed_vertex_mask = _bbox_mask(aligned_vertices, bbox_min, bbox_max)

    tree = o3d.geometry.KDTreeFlann(target_pcd)
    deformed_vertices = aligned_vertices.copy()
    moved = np.zeros(aligned_vertices.shape[0], dtype=bool)
    distances = np.full(aligned_vertices.shape[0], np.inf, dtype=np.float64)
    for vertex_idx in np.nonzero(observed_vertex_mask)[0].tolist():
        point = aligned_vertices[vertex_idx]
        count, indices, sq_dists = tree.search_knn_vector_3d(point, 1)
        if count <= 0:
            continue
        distance = float(np.sqrt(sq_dists[0]))
        distances[vertex_idx] = distance
        if distance > float(args.deform_max_distance):
            continue
        nearest = target_points[int(indices[0])]
        falloff = float(np.exp(-((distance / max(float(args.deform_max_distance), 1e-6)) ** 2)))
        strength = float(args.deform_strength) * falloff
        deformed_vertices[vertex_idx] = (1.0 - strength) * point + strength * nearest
        moved[vertex_idx] = True

    if bool(args.preserve_unobserved):
        deformed_vertices[~observed_vertex_mask] = aligned_vertices[~observed_vertex_mask]

    deformed_mesh = _copy_mesh(aligned_mesh)
    deformed_mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)
    deformed_mesh.compute_vertex_normals()
    if int(args.smooth_iterations) > 0:
        # Smooth only after the bounded pull to suppress noisy Kinect spikes while keeping topology continuous.
        deformed_mesh = deformed_mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))
        deformed_mesh.compute_vertex_normals()
    deformed_mesh.remove_degenerate_triangles()
    deformed_mesh.remove_duplicated_triangles()
    deformed_mesh.remove_duplicated_vertices()
    deformed_mesh.remove_non_manifold_edges()
    deformed_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), deformed_mesh, write_ascii=False, compressed=False)

    summary = {
        "task": "deform_mesh_to_teacher_points",
        "truthful_status": "diagnostic_candidate_requires_strict_teacher_gate",
        "base_mesh": str(base_path.resolve()),
        "target_pointcloud": str(target_path.resolve()),
        "output_mesh": str(output_path.resolve()),
        "target_points_after_filter": int(len(target_pcd.points)),
        "base_vertices": int(base_vertices.shape[0]),
        "base_triangles": int(np.asarray(base_mesh.triangles).shape[0]),
        "observed_vertices_after_icp": int(observed_vertex_mask.sum()),
        "moved_vertices": int(moved.sum()),
        "distance_percentiles_observed": np.percentile(
            distances[np.isfinite(distances)], [0, 25, 50, 75, 90, 95, 99, 100]
        )
        if np.isfinite(distances).any()
        else [],
        "icp": {
            "fitness": float(icp.fitness),
            "inlier_rmse": float(icp.inlier_rmse),
            "threshold": float(args.icp_threshold),
            "transformation": np.asarray(icp.transformation, dtype=np.float64),
        },
        "parameters": {
            "voxel_size": float(args.voxel_size),
            "target_stat_nb_neighbors": int(args.target_stat_nb_neighbors),
            "target_stat_std_ratio": float(args.target_stat_std_ratio),
            "deform_max_distance": float(args.deform_max_distance),
            "deform_strength": float(args.deform_strength),
            "bbox_padding": float(args.bbox_padding),
            "smooth_iterations": int(args.smooth_iterations),
        },
        "warning": (
            "This keeps a SMPL-X-like topology and may still be template-biased or over-smoothed. "
            "It is not a valid teacher unless strict numeric gate and explicit Open3D visual review pass."
        ),
    }
    (output_dir / "deform_mesh_to_teacher_points_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
