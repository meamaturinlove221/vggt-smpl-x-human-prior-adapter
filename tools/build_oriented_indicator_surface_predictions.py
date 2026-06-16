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

from build_mesh_raycast_training_case import _rays_for_pixels  # noqa: E402
from render_open3d_pointcloud import (  # noqa: E402
    load_mask_stack,
    unproject_depth_map_to_point_map_numpy,
)
from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic HART-like oriented indicator surface probe. It builds a "
            "continuous implicit surface from VGGT oriented observations, then "
            "raycasts that surface back to predictions.npz. It is not a teacher, "
            "not a camera replacement, not training, and not a mentor pass."
        )
    )
    parser.add_argument("--predictions-npz", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources", default="world_points,depth_unprojection")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--grid-resolution", type=int, default=96)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--bbox-percentile-low", type=float, default=0.5)
    parser.add_argument("--bbox-percentile-high", type=float, default=99.5)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--trunc-ratio", type=float, default=0.045)
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-hit-depth", type=float, default=0.05)
    parser.add_argument("--normal-source", choices=("derived", "predicted", "mixed"), default="mixed")
    parser.add_argument("--raycast-human-mask-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def parse_sources(spec: str) -> list[str]:
    allowed = {"world_points", "depth_unprojection"}
    out = [item.strip() for item in str(spec).split(",") if item.strip()]
    bad = sorted(set(out) - allowed)
    if bad:
        raise ValueError(f"unknown source(s): {bad}; allowed={sorted(allowed)}")
    if not out:
        raise ValueError("at least one source is required")
    return out


def closed_form_inverse_se3(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=extrinsic.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    return np.einsum("...j,ij->...i", points_world.astype(np.float32), rotation) + translation


def camera_normals_to_world(normals_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    return np.einsum("...j,ji->...i", normals_cam.astype(np.float32), rotation)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(vectors.astype(np.float32), axis=-1, keepdims=True)
    valid = np.isfinite(norm[..., 0]) & (norm[..., 0] > eps)
    out = np.zeros_like(vectors, dtype=np.float32)
    out[valid] = vectors[valid] / norm[valid]
    return out, valid


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, :3, 3], dtype=np.float32)
    return np.einsum("vi,vij->vj", -translation, rotation)


def percentile_thresholds(conf: np.ndarray, support: np.ndarray, percentile: float) -> np.ndarray:
    thresholds = np.full((conf.shape[0],), np.inf, dtype=np.float32)
    for view_idx in range(conf.shape[0]):
        values = conf[view_idx][support[view_idx] & np.isfinite(conf[view_idx]) & (conf[view_idx] > 0.0)]
        if values.size:
            thresholds[view_idx] = float(np.percentile(values, float(percentile)))
    return thresholds


def source_points_and_conf(predictions: dict[str, np.ndarray], source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "world_points":
        return (
            np.asarray(predictions["world_points"], dtype=np.float32),
            np.asarray(predictions["world_points_conf"], dtype=np.float32),
        )
    if source == "depth_unprojection":
        return (
            unproject_depth_map_to_point_map_numpy(
                np.asarray(predictions["depth"], dtype=np.float32),
                np.asarray(predictions["extrinsic"], dtype=np.float32),
                np.asarray(predictions["intrinsic"], dtype=np.float32),
            ),
            np.asarray(predictions["depth_conf"], dtype=np.float32),
        )
    raise ValueError(source)


def derived_world_normals(points_world: np.ndarray, extrinsic: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(points_world, dtype=np.float32)
    valid_out = np.zeros(points_world.shape[:3], dtype=bool)
    for view_idx in range(points_world.shape[0]):
        cam_points = world_to_camera(points_world[view_idx], extrinsic[view_idx])
        normals_cam, valid = point_map_to_normal_numpy(cam_points.astype(np.float32), valid_mask[view_idx].astype(bool))
        normals_world = camera_normals_to_world(normals_cam, extrinsic[view_idx])
        normals_world, normal_valid = normalize_vectors(normals_world)
        out[view_idx] = normals_world
        valid_out[view_idx] = valid & normal_valid
    return out, valid_out


def predicted_world_normals(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(predictions.get("normal"), dtype=np.float32)
    if normal.ndim != 4 or normal.shape[-1] != 3:
        raise ValueError("predictions.npz has no usable normal field")
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    out = np.zeros_like(normal, dtype=np.float32)
    valid = np.isfinite(normal).all(axis=-1)
    for view_idx in range(normal.shape[0]):
        world = camera_normals_to_world(normal[view_idx], extrinsic[view_idx])
        world, ok = normalize_vectors(world)
        out[view_idx] = world
        valid[view_idx] &= ok
    return out, valid


def orient_normals_view_facing(points: np.ndarray, normals: np.ndarray, view_ids: np.ndarray, centers: np.ndarray) -> np.ndarray:
    to_camera = centers[view_ids] - points
    to_camera, ok = normalize_vectors(to_camera)
    oriented = normals.copy()
    dot = np.einsum("ij,ij->i", oriented, to_camera)
    flip = ok & np.isfinite(dot) & (dot < 0.0)
    oriented[flip] *= -1.0
    oriented, _ = normalize_vectors(oriented)
    return oriented


def collect_observations(
    *,
    predictions: dict[str, np.ndarray],
    masks: np.ndarray,
    sources: list[str],
    conf_percentile: float,
    normal_source: str,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    centers = camera_centers(extrinsic)
    pred_normals = pred_valid = None
    if normal_source in {"predicted", "mixed"}:
        pred_normals, pred_valid = predicted_world_normals(predictions)
    point_batches: list[np.ndarray] = []
    normal_batches: list[np.ndarray] = []
    view_batches: list[np.ndarray] = []
    summary: dict[str, Any] = {"sources": {}, "normal_source": str(normal_source)}
    view_template = np.broadcast_to(
        np.arange(masks.shape[0], dtype=np.int16)[:, None, None],
        masks.shape,
    )
    for source in sources:
        points, conf = source_points_and_conf(predictions, source)
        finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf > 0.0) & masks.astype(bool)
        thresholds = percentile_thresholds(conf, finite, float(conf_percentile))
        keep = finite & (conf >= thresholds[:, None, None])
        derived_normals = derived_valid = None
        if normal_source in {"derived", "mixed"}:
            derived_normals, derived_valid = derived_world_normals(points, extrinsic, finite)
        if normal_source == "derived":
            normals = derived_normals
            normal_valid = derived_valid
        elif normal_source == "predicted":
            assert pred_normals is not None and pred_valid is not None
            normals = pred_normals
            normal_valid = pred_valid
        else:
            assert derived_normals is not None and derived_valid is not None
            assert pred_normals is not None and pred_valid is not None
            normals = derived_normals.copy()
            normal_valid = derived_valid.copy()
            fallback = (~normal_valid) & pred_valid
            normals[fallback] = pred_normals[fallback]
            normal_valid[fallback] = True
        keep &= normal_valid
        selected_points = points[keep].astype(np.float32)
        selected_normals = normals[keep].astype(np.float32)
        selected_views = view_template[keep].astype(np.int16)
        if selected_points.size:
            selected_normals = orient_normals_view_facing(selected_points, selected_normals, selected_views, centers)
            point_batches.append(selected_points)
            normal_batches.append(selected_normals)
            view_batches.append(selected_views)
        summary["sources"][source] = {
            "thresholds": [float(v) if np.isfinite(v) else None for v in thresholds.tolist()],
            "finite_mask_points": int(finite.sum()),
            "selected_points": int(selected_points.shape[0]),
        }
    if not point_batches:
        raise RuntimeError("no oriented observations selected")
    points_all = np.concatenate(point_batches, axis=0)
    normals_all = np.concatenate(normal_batches, axis=0)
    views_all = np.concatenate(view_batches, axis=0)
    if points_all.shape[0] > int(max_points):
        rng = np.random.default_rng(20260504)
        keep = rng.choice(points_all.shape[0], size=int(max_points), replace=False)
        points_all = points_all[keep]
        normals_all = normals_all[keep]
        views_all = views_all[keep]
        summary["downsampled_to"] = int(max_points)
    summary["selected_total"] = int(points_all.shape[0])
    summary["view_counts"] = {
        str(int(view_idx)): int((views_all == view_idx).sum())
        for view_idx in range(masks.shape[0])
    }
    return points_all, normals_all, views_all, summary


def build_oriented_mesh(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    output_dir: Path,
    grid_resolution: int,
    bbox_low: float,
    bbox_high: float,
    bbox_pad_ratio: float,
    trunc_ratio: float,
    smooth_sigma: float,
    keep_largest_component: bool,
) -> tuple[Path, dict[str, Any]]:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from skimage import measure
    import open3d as o3d

    lo = np.percentile(points, float(bbox_low), axis=0).astype(np.float32)
    hi = np.percentile(points, float(bbox_high), axis=0).astype(np.float32)
    center = 0.5 * (lo + hi)
    extent = float(np.max(np.maximum(hi - lo, 1e-4)))
    extent *= 1.0 + 2.0 * float(bbox_pad_ratio)
    lo_cube = center - 0.5 * extent
    hi_cube = center + 0.5 * extent
    res = int(max(32, grid_resolution))
    xs = np.linspace(float(lo_cube[0]), float(hi_cube[0]), res, dtype=np.float32)
    ys = np.linspace(float(lo_cube[1]), float(hi_cube[1]), res, dtype=np.float32)
    zs = np.linspace(float(lo_cube[2]), float(hi_cube[2]), res, dtype=np.float32)
    voxel = float(extent / max(res - 1, 1))
    trunc = float(max(voxel * 2.0, extent * float(trunc_ratio)))
    tree = cKDTree(points.astype(np.float32))
    field = np.empty((res, res, res), dtype=np.float32)
    chunk = 220000
    flat_total = res * res * res
    grid_summary: dict[str, Any] = {
        "grid_resolution": res,
        "bbox_low": lo_cube,
        "bbox_high": hi_cube,
        "voxel": voxel,
        "trunc": trunc,
    }
    for start in range(0, flat_total, chunk):
        end = min(flat_total, start + chunk)
        idx = np.arange(start, end, dtype=np.int64)
        ix = idx // (res * res)
        rem = idx - ix * res * res
        iy = rem // res
        iz = rem - iy * res
        coords = np.column_stack((xs[ix], ys[iy], zs[iz])).astype(np.float32)
        dist, nearest = tree.query(coords, k=1, workers=-1)
        nearest_points = points[nearest]
        nearest_normals = normals[nearest]
        signed = np.einsum("ij,ij->i", coords - nearest_points, nearest_normals).astype(np.float32)
        signed = np.where(dist <= trunc, signed, trunc).astype(np.float32)
        field.reshape(-1)[start:end] = np.clip(signed, -trunc, trunc)
    if float(smooth_sigma) > 0.0:
        field = gaussian_filter(field, sigma=float(smooth_sigma)).astype(np.float32)
    if not (float(np.nanmin(field)) < 0.0 < float(np.nanmax(field))):
        # Fallback diagnostic: an unsigned iso-surface around points. This is
        # recorded because it is weaker than a signed oriented field.
        unsigned = np.empty_like(field)
        for start in range(0, flat_total, chunk):
            end = min(flat_total, start + chunk)
            idx = np.arange(start, end, dtype=np.int64)
            ix = idx // (res * res)
            rem = idx - ix * res * res
            iy = rem // res
            iz = rem - iy * res
            coords = np.column_stack((xs[ix], ys[iy], zs[iz])).astype(np.float32)
            dist, _nearest = tree.query(coords, k=1, workers=-1)
            unsigned.reshape(-1)[start:end] = (dist - 1.75 * voxel).astype(np.float32)
        field = unsigned
        grid_summary["fallback_unsigned_distance"] = True
    else:
        grid_summary["fallback_unsigned_distance"] = False
    verts, faces, _normals, _values = measure.marching_cubes(
        field,
        level=0.0,
        spacing=(voxel, voxel, voxel),
    )
    verts = verts.astype(np.float32) + lo_cube[None, :].astype(np.float32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    component_summary: dict[str, Any] = {}
    if bool(keep_largest_component) and len(mesh.triangles) > 0:
        clusters, counts, _areas = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        if counts.size:
            keep = int(np.argmax(counts))
            remove = clusters != keep
            mesh.remove_triangles_by_mask(remove.tolist())
            mesh.remove_unreferenced_vertices()
            component_summary = {
                "component_count": int(counts.size),
                "kept_cluster": keep,
                "kept_triangles": int((~remove).sum()),
                "removed_triangles": int(remove.sum()),
            }
    mesh.compute_vertex_normals()
    mesh_path = output_dir / "oriented_indicator_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)
    grid_summary.update(
        {
            "field_min": float(np.nanmin(field)),
            "field_max": float(np.nanmax(field)),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_triangles": int(len(mesh.triangles)),
            "component_filter": component_summary,
        }
    )
    return mesh_path, grid_summary


def raycast_mesh_to_predictions(
    *,
    mesh_path: Path,
    predictions: dict[str, np.ndarray],
    masks: np.ndarray,
    human_mask_only: bool,
    min_hit_depth: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"mesh has no triangles: {mesh_path}")
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    out = {key: np.asarray(value).copy() for key, value in predictions.items()}
    world_points = np.asarray(out["world_points"], dtype=np.float32)
    depth = np.asarray(out["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth_2d = depth[..., 0]
        depth_has_channel = True
    else:
        depth_2d = depth
        depth_has_channel = False
    extrinsic = np.asarray(out["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(out["intrinsic"], dtype=np.float32)
    views, height, width = world_points.shape[:3]
    yy, xx = np.mgrid[0:height, 0:width]
    records = []
    normal = np.zeros_like(world_points, dtype=np.float32)
    normal_valid_all = np.zeros((views, height, width), dtype=bool)
    for view_idx in range(views):
        support = np.asarray(masks[view_idx], dtype=bool) if bool(human_mask_only) else np.ones((height, width), dtype=bool)
        xs = xx[support].astype(np.float32)
        ys = yy[support].astype(np.float32)
        rays = _rays_for_pixels(xs, ys, intrinsic[view_idx], extrinsic[view_idx])
        answers = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = answers["t_hit"].numpy()
        valid = np.isfinite(t_hit)
        origins = rays[:, :3]
        dirs = rays[:, 3:]
        hit_world = origins + dirs * np.where(valid, t_hit, 0.0)[:, None]
        hit_cam = world_to_camera(hit_world.astype(np.float32), extrinsic[view_idx])
        hit_depth = hit_cam[:, 2]
        valid &= np.isfinite(hit_depth) & (hit_depth > float(min_hit_depth))
        flat_y = ys.astype(np.int64)
        flat_x = xs.astype(np.int64)
        if valid.any():
            world_points[view_idx, flat_y[valid], flat_x[valid]] = hit_world[valid].astype(np.float32)
            depth_2d[view_idx, flat_y[valid], flat_x[valid]] = hit_depth[valid].astype(np.float32)
        cam_map = world_to_camera(world_points[view_idx], extrinsic[view_idx])
        n_cam, n_valid = point_map_to_normal_numpy(cam_map.astype(np.float32), support.astype(bool))
        normal[view_idx] = n_cam.astype(np.float32)
        normal_valid_all[view_idx] = n_valid
        records.append(
            {
                "view_index": int(view_idx),
                "ray_pixels": int(support.sum()),
                "hit_pixels": int(valid.sum()),
                "hit_ratio": float(valid.sum() / max(int(support.sum()), 1)),
            }
        )
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth_2d[..., None].astype(np.float32) if depth_has_channel else depth_2d.astype(np.float32)
    if "normal" in out:
        out["normal"] = normal.astype(np.float32)
    if "normal_conf" in out:
        normal_conf = np.asarray(out["normal_conf"], dtype=np.float32).copy()
        normal_conf[normal_valid_all] = np.maximum(normal_conf[normal_valid_all], np.nanmedian(normal_conf[np.isfinite(normal_conf)]))
        out["normal_conf"] = normal_conf.astype(np.float32)
    return out, {"per_view": records}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_npz(args.predictions_npz)
    height = int(np.asarray(predictions["world_points"]).shape[1])
    masks = load_mask_stack(args.scene_dir / "masks", target_size=height).astype(bool)
    sources = parse_sources(str(args.sources))
    points, normals, views, collect_summary = collect_observations(
        predictions=predictions,
        masks=masks,
        sources=sources,
        conf_percentile=float(args.conf_percentile),
        normal_source=str(args.normal_source),
        max_points=int(args.max_points),
    )
    mesh_path, mesh_summary = build_oriented_mesh(
        points=points,
        normals=normals,
        output_dir=output_dir,
        grid_resolution=int(args.grid_resolution),
        bbox_low=float(args.bbox_percentile_low),
        bbox_high=float(args.bbox_percentile_high),
        bbox_pad_ratio=float(args.bbox_pad_ratio),
        trunc_ratio=float(args.trunc_ratio),
        smooth_sigma=float(args.smooth_sigma),
        keep_largest_component=bool(args.keep_largest_component),
    )
    patched, raycast_summary = raycast_mesh_to_predictions(
        mesh_path=mesh_path,
        predictions=predictions,
        masks=masks,
        human_mask_only=bool(args.raycast_human_mask_only),
        min_hit_depth=float(args.min_hit_depth),
    )
    output_npz = output_dir / "predictions.npz"
    np.savez_compressed(output_npz, **patched)
    summary = {
        "task": "oriented_indicator_surface_predictions",
        "truthful_status": "diagnostic_only_not_teacher_not_training_not_mentor_pass",
        "predictions_npz": str(args.predictions_npz.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_dir": str(output_dir),
        "output_npz": str(output_npz),
        "mesh_path": str(mesh_path),
        "parameters": {
            "sources": sources,
            "conf_percentile": float(args.conf_percentile),
            "grid_resolution": int(args.grid_resolution),
            "max_points": int(args.max_points),
            "normal_source": str(args.normal_source),
            "raycast_human_mask_only": bool(args.raycast_human_mask_only),
        },
        "collection": collect_summary,
        "mesh": mesh_summary,
        "raycast": raycast_summary,
        "notes": [
            "This is the minimal local HART-like continuous-field diagnostic available without DPSR/indicator-grid code.",
            "It must pass the same full candidate gate before any mentor success or cloud work.",
            "Failure means the current VGGT observations do not contain enough surface signal for this unlearned field backend.",
        ],
    }
    (output_dir / "oriented_indicator_surface_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
