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

from bridge_teacher_targets_between_vggt_worlds import (  # noqa: E402
    camera_center_and_rotation,
    camera_ids,
    estimate_similarity,
    load_manifest,
    median_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic helper: transform a mesh/pointcloud from one VGGT "
            "prediction world to another using shared camera IDs. This only "
            "bridges coordinate frames; the output must still pass the strict "
            "teacher gate before any training."
        )
    )
    parser.add_argument("--source-surface", required=True, help="Input mesh or pointcloud PLY/OBJ in source VGGT world.")
    parser.add_argument("--source-predictions-npz", required=True)
    parser.add_argument("--source-scene-dir", required=True)
    parser.add_argument("--target-predictions-npz", required=True)
    parser.add_argument("--target-scene-dir", required=True)
    parser.add_argument("--output-surface", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--axis-scale", type=float, default=0.01)
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_extrinsics(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        key = "extrinsic" if "extrinsic" in payload.files else "extrinsics"
        return np.asarray(payload[key], dtype=np.float64)


def _similarity_from_shared_cameras(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = load_manifest(Path(args.source_scene_dir))
    target_manifest = load_manifest(Path(args.target_scene_dir))
    source_ids = camera_ids(source_manifest)
    target_ids = camera_ids(target_manifest)
    source_index = {cam_id: idx for idx, cam_id in enumerate(source_ids)}
    source_extrinsic = _load_extrinsics(Path(args.source_predictions_npz))
    target_extrinsic = _load_extrinsics(Path(args.target_predictions_npz))

    source_corr: list[np.ndarray] = []
    target_corr: list[np.ndarray] = []
    per_camera: list[dict[str, Any]] = []
    shared_ids = [cam_id for cam_id in target_ids if cam_id in source_index]
    for target_view_idx, cam_id in enumerate(target_ids):
        if cam_id not in source_index:
            continue
        source_view_idx = source_index[cam_id]
        src_center, src_rot = camera_center_and_rotation(source_extrinsic[source_view_idx])
        tgt_center, tgt_rot = camera_center_and_rotation(target_extrinsic[target_view_idx])
        source_corr.append(src_center)
        target_corr.append(tgt_center)
        per_camera.append(
            {
                "camera_id": cam_id,
                "source_view_index": int(source_view_idx),
                "target_view_index": int(target_view_idx),
            }
        )

    if not source_corr:
        raise SystemExit("No shared camera IDs found between source and target scenes.")

    source_centers = np.stack(source_corr, axis=0)
    target_centers = np.stack(target_corr, axis=0)
    source_axis_len = float(args.axis_scale) * median_baseline(source_centers)
    target_axis_len = float(args.axis_scale) * median_baseline(target_centers)

    if not args.no_axes:
        for item in per_camera:
            src_idx = int(item["source_view_index"])
            tgt_idx = int(item["target_view_index"])
            src_center, src_rot = camera_center_and_rotation(source_extrinsic[src_idx])
            tgt_center, tgt_rot = camera_center_and_rotation(target_extrinsic[tgt_idx])
            for axis_idx in range(3):
                source_corr.append(src_center + source_axis_len * src_rot[:, axis_idx])
                target_corr.append(tgt_center + target_axis_len * tgt_rot[:, axis_idx])

    source_arr = np.stack(source_corr, axis=0)
    target_arr = np.stack(target_corr, axis=0)
    scale, rotation, translation = estimate_similarity(source_arr, target_arr)
    mapped = scale * (source_arr @ rotation.T) + translation[None, :]
    residual = np.linalg.norm(mapped - target_arr, axis=1)
    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "shared_camera_ids": shared_ids,
        "per_camera": per_camera,
        "used_axes": not bool(args.no_axes),
        "axis_scale": float(args.axis_scale),
        "source_axis_len": source_axis_len,
        "target_axis_len": target_axis_len,
        "scale": float(scale),
        "rotation": rotation.astype(np.float64),
        "translation": translation.astype(np.float64),
        "residual_percentiles": np.percentile(residual, [0, 25, 50, 75, 90, 95, 100]).tolist(),
        "correspondence_count": int(source_arr.shape[0]),
    }


def _transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (scale * (points @ rotation.T) + translation[None, :]).astype(np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> int:
    args = parse_args()
    output_surface = Path(args.output_surface)
    if output_surface.exists() and not args.overwrite:
        raise FileExistsError(output_surface)
    output_surface.parent.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    sim = _similarity_from_shared_cameras(args)
    scale = float(sim["scale"])
    rotation = np.asarray(sim["rotation"], dtype=np.float64)
    translation = np.asarray(sim["translation"], dtype=np.float64)
    source_surface = Path(args.source_surface)

    mesh = o3d.io.read_triangle_mesh(str(source_surface))
    source_kind = "mesh" if len(mesh.vertices) > 0 and len(mesh.triangles) > 0 else "pointcloud"
    if source_kind == "mesh":
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.vertices = o3d.utility.Vector3dVector(_transform_points(vertices, scale, rotation, translation))
        mesh.compute_vertex_normals()
        ok = o3d.io.write_triangle_mesh(str(output_surface), mesh, write_ascii=False, compressed=False)
        point_count = int(len(mesh.vertices))
        triangle_count = int(len(mesh.triangles))
    else:
        pcd = o3d.io.read_point_cloud(str(source_surface))
        if len(pcd.points) == 0:
            raise RuntimeError(f"Surface is neither a mesh nor a pointcloud with points: {source_surface}")
        points = np.asarray(pcd.points, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(_transform_points(points, scale, rotation, translation))
        ok = o3d.io.write_point_cloud(str(output_surface), pcd, write_ascii=False, compressed=False)
        point_count = int(len(pcd.points))
        triangle_count = 0
    if not ok:
        raise RuntimeError(f"Failed to write {output_surface}")

    summary = {
        "task": "transform_surface_between_vggt_worlds",
        "truthful_status": "coordinate_bridge_only_requires_strict_teacher_gate",
        "source_surface": str(source_surface.resolve()),
        "output_surface": str(output_surface.resolve()),
        "source_kind": source_kind,
        "point_count": point_count,
        "triangle_count": triangle_count,
        "source_predictions_npz": str(Path(args.source_predictions_npz).resolve()),
        "target_predictions_npz": str(Path(args.target_predictions_npz).resolve()),
        "source_scene_dir": str(Path(args.source_scene_dir).resolve()),
        "target_scene_dir": str(Path(args.target_scene_dir).resolve()),
        "similarity": {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "residual_percentiles": sim["residual_percentiles"],
            "correspondence_count": sim["correspondence_count"],
            "used_axes": sim["used_axes"],
            "axis_scale": sim["axis_scale"],
            "shared_camera_ids": sim["shared_camera_ids"],
        },
        "warning": "This bridge is not a pass. Run audit_headface_teacher_surface.py and explicit Open3D review.",
    }
    summary_path = Path(args.output_summary) if args.output_summary else output_surface.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
