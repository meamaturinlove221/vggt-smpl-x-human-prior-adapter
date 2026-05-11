from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v15_common import (
    DEFAULT_2DGS_30K_PLY,
    DEFAULT_2DGS_SCENE,
    LOCAL_ROOT,
    REPORTS,
    derive_2dgs_world_normals,
    json_ready,
    load_binary_ply_vertices,
    load_colmap_cameras,
    normal_angle_metrics,
    normalize_vectors,
    project_points,
    safe_v15_output_dir,
    scalar_stats,
    sigmoid,
    utc_now,
    write_json,
    write_report,
)


def rasterize_normals(
    vertices: np.ndarray,
    props: list[str],
    cameras: list[dict[str, Any]],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    world_normals, normal_meta = derive_2dgs_world_normals(vertices, props)
    if max_points > 0 and len(xyz) > max_points:
        strength = np.ones(len(xyz), dtype=np.float32)
        if "opacity" in props:
            strength = sigmoid(vertices["opacity"].astype(np.float32))
        order = np.argsort(strength)[::-1][:max_points]
        xyz = xyz[order]
        world_normals = world_normals[order]
        selected_count = int(order.size)
    else:
        selected_count = int(len(xyz))

    if "opacity" in props:
        weights = sigmoid(vertices["opacity"].astype(np.float32))
        if len(weights) != len(world_normals):
            weights = np.ones(len(world_normals), dtype=np.float32)
    else:
        weights = np.ones(len(world_normals), dtype=np.float32)
    if len(weights) != len(xyz):
        weights = weights[np.argsort(weights)[::-1][:selected_count]]

    height = int(cameras[0]["height"]) if cameras else 0
    width = int(cameras[0]["width"]) if cameras else 0
    normal_maps = np.zeros((len(cameras), height, width, 3), dtype=np.float32)
    world_normal_maps = np.zeros_like(normal_maps)
    depth_maps = np.full((len(cameras), height, width), np.nan, dtype=np.float32)
    visibility = np.zeros((len(cameras), height, width), dtype=bool)
    per_view: list[dict[str, Any]] = []

    for cam_idx, camera in enumerate(cameras):
        uv, depth, _ = project_points(xyz, camera)
        valid = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 1e-5)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < int(camera["width"]))
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < int(camera["height"]))
        )
        if not np.any(valid):
            per_view.append({"name": camera["name"], "projected_points": 0, "valid_pixels": 0})
            continue
        xy = np.rint(uv[valid]).astype(np.int32)
        z = depth[valid].astype(np.float32)
        wn = world_normals[valid]
        cn = wn @ np.asarray(camera["rotation_w2c"], dtype=np.float32).T
        cn, cn_valid = normalize_vectors(cn)
        wn, wn_valid = normalize_vectors(wn)
        xy = xy[cn_valid & wn_valid]
        z = z[cn_valid & wn_valid]
        cn = cn[cn_valid & wn_valid]
        wn = wn[cn_valid & wn_valid]
        order = np.argsort(z)[::-1]
        for idx in order:
            x, y = int(xy[idx, 0]), int(xy[idx, 1])
            old = depth_maps[cam_idx, y, x]
            if not np.isfinite(old) or z[idx] < old:
                depth_maps[cam_idx, y, x] = z[idx]
                normal_maps[cam_idx, y, x] = cn[idx]
                world_normal_maps[cam_idx, y, x] = wn[idx]
                visibility[cam_idx, y, x] = True
        nlen = np.linalg.norm(normal_maps[cam_idx], axis=-1)
        valid_pix = visibility[cam_idx] & (nlen > 0.5)
        per_view.append(
            {
                "index": int(cam_idx),
                "name": camera["name"],
                "camera_id": camera.get("camera_id"),
                "projected_points": int(valid.sum()),
                "valid_pixels": int(valid_pix.sum()),
                "valid_ratio": float(valid_pix.sum() / max(height * width, 1)),
                "depth": scalar_stats(depth_maps[cam_idx][visibility[cam_idx]]),
                "camera_normal_length": scalar_stats(nlen[valid_pix]),
            }
        )
    return normal_maps, world_normal_maps, depth_maps, visibility, per_view, normal_meta


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 true 2DGS normal rasterizer from raw Gaussian orientation fields.")
    parser.add_argument("--point-cloud", type=Path, default=DEFAULT_2DGS_30K_PLY)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_2DGS_SCENE)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V15_GS_2DGS_true_normal_rasterizer")
    parser.add_argument("--max-points", type=int, default=180000)
    args = parser.parse_args()

    out = safe_v15_output_dir(args.output_dir)
    vertices, props = load_binary_ply_vertices(args.point_cloud)
    cameras = load_colmap_cameras(args.scene_dir)
    normal_maps, world_normal_maps, depth_maps, visibility, per_view, normal_meta = rasterize_normals(
        vertices, props, cameras, max_points=args.max_points
    )
    view_names = np.asarray([cam["name"] for cam in cameras])
    normal_path = out / "v15_2dgs_true_camera_normals_6view.npz"
    world_path = out / "v15_2dgs_true_world_normals_6view.npz"
    depth_path = out / "v15_2dgs_true_depth_6view.npz"
    visibility_path = out / "v15_2dgs_true_visibility_6view.npz"
    np.savez_compressed(normal_path, view_names=view_names, normal=normal_maps, visibility=visibility, research_only=True)
    np.savez_compressed(world_path, view_names=view_names, normal=world_normal_maps, visibility=visibility, research_only=True)
    np.savez_compressed(depth_path, view_names=view_names, depth=depth_maps, visibility=visibility, research_only=True)
    np.savez_compressed(visibility_path, view_names=view_names, visibility=visibility, research_only=True)

    valid_pixels = int(visibility.sum())
    total_pixels = int(visibility.size)
    norm_len = np.linalg.norm(normal_maps, axis=-1)
    metrics = {
        "vertex_count": int(len(vertices)),
        "camera_count": len(cameras),
        "normal_valid_pixels": valid_pixels,
        "normal_valid_ratio": float(valid_pixels / max(total_pixels, 1)),
        "normal_nonzero_pixels": int((norm_len > 0.5).sum()),
        "nx_ny_nz_nonzero_count": normal_meta.get("nx_ny_nz_nonzero_count"),
        "normal_source": normal_meta.get("normal_source"),
    }
    gates = {
        "raw_point_cloud_exists": args.point_cloud.is_file(),
        "colmap_camera_contract_exists": len(cameras) == 6,
        "raster_has_nonzero_normals": metrics["normal_nonzero_pixels"] > 0,
        "nx_ny_nz_fields_are_real": int(normal_meta.get("nx_ny_nz_nonzero_count", 0) or 0) > 0,
        "uses_orientation_fallback": normal_meta.get("normal_source") == "rot_quaternion_local_z",
    }
    blockers = []
    if not gates["nx_ny_nz_fields_are_real"]:
        blockers.append("Raw 2DGS PLY nx/ny/nz fields exist but are all zero; normals were derived from rot_0..rot_3 local-z orientation.")
    if not gates["raster_has_nonzero_normals"]:
        blockers.append("No nonzero 2DGS normal pixels were rasterized from the local artifact.")
    blockers.append("Rasterized orientation normals are research evidence only; this is not a differentiable 2DGS/SuGaR surface export or strict teacher.")

    summary = {
        "task": "v15_2dgs_true_normal_rasterizer",
        "created_utc": utc_now(),
        "status": "v15_2dgs_orientation_normals_ready_research_only" if gates["raster_has_nonzero_normals"] else "v15_2dgs_true_normal_raster_blocked",
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "inputs": {"point_cloud": str(args.point_cloud.resolve()), "scene_dir": str(args.scene_dir.resolve())},
        "metrics": metrics,
        "gates": gates,
        "normal_meta": normal_meta,
        "per_view": per_view,
        "outputs": {
            "camera_normals": str(normal_path.resolve()),
            "world_normals": str(world_path.resolve()),
            "depth": str(depth_path.resolve()),
            "visibility": str(visibility_path.resolve()),
        },
        "decision": "A bounded 6-view nonzero normal raster can be audited from 2DGS quaternion orientation, but raw nx/ny/nz remains zero and no strict teacher export is allowed.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v15_2dgs_true_normal_rasterizer.json", summary)
    write_report(REPORTS / "20260508_v15_2dgs_true_normal_rasterizer.md", "V15 2DGS True Normal Rasterizer", summary)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": metrics, "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
