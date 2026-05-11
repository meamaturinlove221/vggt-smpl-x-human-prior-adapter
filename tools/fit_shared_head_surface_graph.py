from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_headface_teacher_surface import _roi_mask, load_scene_image, load_scene_mask  # noqa: E402
from build_mesh_raycast_training_case import _world_to_cam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one shared head/face surface mesh from already aligned, depth-compatible "
            "teacher target NPZ files. This is a teacher-construction diagnostic only: "
            "the exported mesh must pass audit_headface_teacher_surface.py before any training."
        )
    )
    parser.add_argument("--teacher-npz", action="append", required=True, help="Dense teacher target NPZ. May repeat.")
    parser.add_argument(
        "--teacher-name",
        action="append",
        default=[],
        help="Optional source label for each --teacher-npz. May repeat.",
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-kinds", default="face_core,head_face,hairline")
    parser.add_argument("--target-views", default="all")
    parser.add_argument("--depth-tolerance", type=float, default=0.06)
    parser.add_argument("--max-points-per-source-view-roi", type=int, default=18000)
    parser.add_argument("--max-total-points", type=int, default=450000)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--stat-nb-neighbors", type=int, default=32)
    parser.add_argument("--stat-std-ratio", type=float, default=2.0)
    parser.add_argument("--normal-knn", type=int, default=48)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.06)
    parser.add_argument("--max-vertex-distance", type=float, default=0.045)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_views(spec: str, count: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(count))
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= count:
            raise IndexError(f"view index {idx} outside [0,{count})")
        out.append(idx)
    return sorted(set(out))


def parse_rois(spec: str) -> list[str]:
    rois = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not rois:
        raise ValueError("empty --roi-kinds")
    return rois


def load_teacher(path: Path, extrinsic: np.ndarray) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if "world_points" not in payload.files:
        raise KeyError(f"{path} does not contain world_points")
    world = np.asarray(payload["world_points"], dtype=np.float32)
    if "teacher_mask" in payload.files:
        mask = np.asarray(payload["teacher_mask"], dtype=bool)
    elif "roi_mask" in payload.files:
        mask = np.asarray(payload["roi_mask"], dtype=bool)
    else:
        mask = np.isfinite(world).all(axis=-1)
    if "depths" in payload.files:
        depth = np.asarray(payload["depths"], dtype=np.float32)
    elif "depth" in payload.files:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
    else:
        depths = []
        for view_idx in range(world.shape[0]):
            cam = _world_to_cam(world[view_idx].reshape(-1, 3), extrinsic[view_idx])
            depths.append(cam[:, 2].reshape(world.shape[1:3]))
        depth = np.stack(depths, axis=0).astype(np.float32)
    finite = np.isfinite(world).all(axis=-1) & np.isfinite(depth) & (depth > 0.05)
    return {"world": world, "depth": depth, "mask": mask & finite}


def subsample(points: np.ndarray, colors: np.ndarray, *, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= int(max_points):
        return points, colors
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[keep], colors[keep]


def collect_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    predictions_path = Path(args.predictions_npz)
    scene_dir = Path(args.scene_dir)
    with np.load(predictions_path, allow_pickle=False) as payload:
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float32)
        depth_anchor = np.asarray(payload["depth"], dtype=np.float32)
        if depth_anchor.ndim == 4 and depth_anchor.shape[-1] == 1:
            depth_anchor = depth_anchor[..., 0]
    views, height, width = depth_anchor.shape
    view_indices = parse_views(str(args.target_views), views)
    roi_kinds = parse_rois(str(args.roi_kinds))
    teacher_paths = [Path(path) for path in args.teacher_npz]
    names = list(args.teacher_name)
    while len(names) < len(teacher_paths):
        names.append(teacher_paths[len(names)].parent.name)

    rgb_cache: dict[int, np.ndarray] = {}
    mask_cache: dict[int, np.ndarray] = {}
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for source_idx, (path, name) in enumerate(zip(teacher_paths, names)):
        teacher = load_teacher(path, extrinsic)
        source_points = 0
        for view_idx in view_indices:
            if view_idx >= teacher["world"].shape[0]:
                continue
            if view_idx not in rgb_cache:
                rgb_cache[view_idx] = load_scene_image(scene_dir, view_index=view_idx, target_size=height)
                mask_cache[view_idx] = load_scene_mask(scene_dir, view_index=view_idx, target_size=height)
            rgb = rgb_cache[view_idx]
            human_mask = mask_cache[view_idx]
            for roi_kind in roi_kinds:
                roi = _roi_mask(human_mask, roi_kind)
                valid = (
                    roi
                    & teacher["mask"][view_idx]
                    & np.isfinite(teacher["depth"][view_idx])
                    & (np.abs(teacher["depth"][view_idx] - depth_anchor[view_idx]) <= float(args.depth_tolerance))
                )
                yy, xx = np.nonzero(valid)
                pts = teacher["world"][view_idx, yy, xx]
                cols = rgb[yy, xx]
                pts, cols = subsample(
                    pts.astype(np.float32),
                    cols.astype(np.uint8),
                    max_points=int(args.max_points_per_source_view_roi),
                    seed=20260502 + source_idx * 100 + view_idx * 10 + len(roi_kind),
                )
                if pts.shape[0]:
                    all_points.append(pts)
                    all_colors.append(cols)
                    source_points += int(pts.shape[0])
                records.append(
                    {
                        "source": name,
                        "path": str(path),
                        "view_index": int(view_idx),
                        "roi_kind": roi_kind,
                        "selected_points": int(pts.shape[0]),
                    }
                )
        records.append({"source": name, "path": str(path), "selected_points_total": int(source_points)})
    if not all_points:
        raise RuntimeError("No depth-compatible teacher points selected.")
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    points, colors = subsample(points, colors, max_points=int(args.max_total_points), seed=20260502)
    return points, colors, {
        "predictions_npz": str(predictions_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "teacher_sources": [str(path.resolve()) for path in teacher_paths],
        "teacher_names": names,
        "target_views": view_indices,
        "roi_kinds": roi_kinds,
        "depth_tolerance": float(args.depth_tolerance),
        "records": records,
        "selected_points_after_global_sample": int(points.shape[0]),
    }


def fit_mesh(points: np.ndarray, colors: np.ndarray, args: argparse.Namespace):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).astype(np.float64))
    raw_count = len(pcd.points)
    if float(args.voxel_size) > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))
    voxel_count = len(pcd.points)
    if voxel_count == 0:
        raise RuntimeError("Voxel downsample removed all points.")
    if int(args.stat_nb_neighbors) > 0:
        pcd, keep = pcd.remove_statistical_outlier(
            nb_neighbors=int(args.stat_nb_neighbors),
            std_ratio=float(args.stat_std_ratio),
        )
        stat_count = len(pcd.points)
    else:
        keep = []
        stat_count = voxel_count
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(args.normal_knn)))
    try:
        pcd.orient_normals_consistent_tangent_plane(int(args.normal_knn))
    except RuntimeError:
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(args.poisson_depth),
        n_threads=0,
    )
    densities_np = np.asarray(densities)
    if densities_np.size and float(args.poisson_density_quantile) > 0:
        keep_density = densities_np >= np.quantile(densities_np, float(args.poisson_density_quantile))
        mesh.remove_vertices_by_mask(~keep_density)

    pts_np = np.asarray(pcd.points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=pts_np.min(axis=0) - float(args.bbox_padding),
        max_bound=pts_np.max(axis=0) + float(args.bbox_padding),
    )
    mesh = mesh.crop(bbox)

    if len(mesh.vertices) and float(args.max_vertex_distance) > 0:
        tree = o3d.geometry.KDTreeFlann(pcd)
        remove = []
        max_dist2 = float(args.max_vertex_distance) ** 2
        for vertex in np.asarray(mesh.vertices):
            _, _, dist2 = tree.search_knn_vector_3d(vertex, 1)
            remove.append(not dist2 or dist2[0] > max_dist2)
        mesh.remove_vertices_by_mask(np.asarray(remove, dtype=bool))

    if len(mesh.triangles):
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(triangle_clusters)
        counts = np.asarray(cluster_n_triangles)
        if counts.size:
            keep_label = int(np.argmax(counts))
            mesh.remove_triangles_by_mask(labels != keep_label)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if int(args.smooth_iterations) > 0 and len(mesh.triangles):
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(args.smooth_iterations))
    mesh.compute_vertex_normals()
    return pcd, mesh, {
        "raw_points": int(raw_count),
        "voxel_points": int(voxel_count),
        "statistical_outlier_points": int(stat_count),
        "statistical_keep_count": int(len(keep)) if len(keep) else None,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "poisson_depth": int(args.poisson_depth),
        "voxel_size": float(args.voxel_size),
        "max_vertex_distance": float(args.max_vertex_distance),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors, collection_summary = collect_points(args)
    pcd, mesh, fit_summary = fit_mesh(points, colors, args)
    import open3d as o3d

    pcd_path = output_dir / "selected_depth_compatible_points.ply"
    mesh_path = output_dir / "surface_original6v_world.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)
    summary = {
        "task": "fit_shared_head_surface_graph",
        "truthful_status": "teacher_candidate_requires_strict_gate",
        "collection": collection_summary,
        "fit": fit_summary,
        "outputs": {
            "selected_pointcloud": str(pcd_path.resolve()),
            "surface_mesh": str(mesh_path.resolve()),
        },
        "next_required_gate": (
            "Run tools/audit_headface_teacher_surface.py --teacher-mesh on surface_original6v_world.ply. "
            "Do not train unless that strict numeric gate and explicit Open3D visual review both pass."
        ),
    }
    (output_dir / "shared_surface_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
