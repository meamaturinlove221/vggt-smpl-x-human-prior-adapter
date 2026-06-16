from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge diagnostic teacher meshes into one candidate surface. "
            "The merged surface must still pass strict numeric and visual gates."
        )
    )
    parser.add_argument("--mesh", action="append", required=True, help="Input mesh path. Can be passed multiple times.")
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional vertex clustering size after merge.")
    parser.add_argument("--smooth-iterations", type=int, default=0)
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


def main() -> int:
    import open3d as o3d

    args = parse_args()
    output_mesh = Path(args.output_mesh).expanduser().resolve()
    if output_mesh.exists() and not bool(args.overwrite):
        raise FileExistsError(output_mesh)
    merged = o3d.geometry.TriangleMesh()
    input_rows = []
    for raw in args.mesh:
        path = Path(raw).expanduser().resolve()
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise RuntimeError(f"Mesh has no triangles: {path}")
        mesh.compute_vertex_normals()
        input_rows.append(
            {
                "path": str(path),
                "vertices": int(len(mesh.vertices)),
                "triangles": int(len(mesh.triangles)),
            }
        )
        merged += mesh
    merged.remove_degenerate_triangles()
    merged.remove_duplicated_triangles()
    merged.remove_duplicated_vertices()
    merged.remove_non_manifold_edges()
    if float(args.voxel_size) > 0:
        merged = merged.simplify_vertex_clustering(
            voxel_size=float(args.voxel_size),
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
    if int(args.smooth_iterations) > 0:
        merged = merged.filter_smooth_simple(number_of_iterations=int(args.smooth_iterations))
    merged.remove_degenerate_triangles()
    merged.remove_duplicated_triangles()
    merged.remove_duplicated_vertices()
    merged.remove_non_manifold_edges()
    merged.compute_vertex_normals()
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_mesh), merged, write_ascii=False, compressed=False)
    vertices = np.asarray(merged.vertices, dtype=np.float32)
    summary = {
        "task": "merge_teacher_meshes",
        "truthful_status": "teacher_candidate_requires_strict_gate",
        "inputs": input_rows,
        "output_mesh": str(output_mesh),
        "voxel_size": float(args.voxel_size),
        "smooth_iterations": int(args.smooth_iterations),
        "merged_vertices": int(len(merged.vertices)),
        "merged_triangles": int(len(merged.triangles)),
        "bounds": {
            "min": vertices.min(axis=0).tolist() if vertices.size else [],
            "max": vertices.max(axis=0).tolist() if vertices.size else [],
        },
        "warning": (
            "A merged diagnostic mesh is not a teacher pass. Numeric gate and explicit "
            "Open3D visual review must both pass before any overfit or training."
        ),
    }
    summary_path = output_mesh.with_name(output_mesh.stem + "_summary.json")
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
