from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_open3d_pointcloud import _save_open3d_camera_renders, _save_open3d_renders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a teacher mesh/pointcloud for explicit visual review.")
    parser.add_argument("--teacher-mesh", default="")
    parser.add_argument("--teacher-pointcloud", default="")
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-points", type=int, default=160000)
    parser.add_argument("--camera-view-indices", default="3")
    parser.add_argument("--roi", choices=("full", "head", "face", "hands"), default="head")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1000)
    return parser.parse_args()


def parse_camera_indices(spec: str, count: int) -> list[int]:
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= count:
            raise IndexError(f"camera index {idx} outside [0,{count})")
        out.append(idx)
    return sorted(set(out))


def load_surface_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import open3d as o3d

    if args.teacher_mesh:
        mesh_path = Path(args.teacher_mesh)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.triangles) == 0:
            raise RuntimeError(f"mesh has no triangles: {mesh_path}")
        mesh.compute_vertex_normals()
        sampled = mesh.sample_points_uniformly(number_of_points=int(args.sample_points), use_triangle_normal=False)
        points = np.asarray(sampled.points, dtype=np.float32)
        normals = np.asarray(sampled.normals, dtype=np.float32)
        if normals.shape != points.shape:
            sampled.estimate_normals()
            normals = np.asarray(sampled.normals, dtype=np.float32)
        colors = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return points, colors, {"source_kind": "mesh", "source_path": str(mesh_path.resolve())}

    pcd_path = Path(args.teacher_pointcloud)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors_f = np.asarray(pcd.colors, dtype=np.float32)
    if colors_f.shape[0] == points.shape[0]:
        colors = np.clip(colors_f * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full(points.shape, 160, dtype=np.uint8)
    return points, colors, {"source_kind": "pointcloud", "source_path": str(pcd_path.resolve())}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.predictions_npz, allow_pickle=False) as payload:
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
    points, colors, source = load_surface_points(args)
    camera_indices = parse_camera_indices(str(args.camera_view_indices), int(extrinsic.shape[0]))
    screenshots = []
    screenshots.extend(
        _save_open3d_renders(
            points=points,
            colors=colors,
            output_dir=output_dir,
            roi=str(args.roi),
            width=int(args.width),
            height=int(args.height),
            point_size=float(args.point_size),
            interactive=False,
        )
    )
    screenshots.extend(
        _save_open3d_camera_renders(
            points=points,
            colors=colors,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            output_dir=output_dir,
            camera_indices=camera_indices,
            point_size=float(args.point_size),
            render_size=int(max(518, min(args.width, args.height))),
        )
    )
    summary = {
        **source,
        "predictions_npz": str(Path(args.predictions_npz).resolve()),
        "output_dir": str(output_dir.resolve()),
        "sampled_points": int(points.shape[0]),
        "camera_view_indices": camera_indices,
        "screenshots": screenshots,
        "truthful_note": "Visual review aid only. It is not a teacher gate pass.",
    }
    (output_dir / "teacher_surface_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
