from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove tiny disconnected triangle components from a teacher mesh. "
            "This is a visual/geometry hygiene step only; the cleaned mesh must "
            "still pass strict teacher-gate before any training."
        )
    )
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest triangle component.")
    parser.add_argument("--min-triangles", type=int, default=100)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--smooth-iterations", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> int:
    args = parse_args()
    import open3d as o3d

    input_mesh = Path(args.input_mesh)
    output_mesh = Path(args.output_mesh)
    if output_mesh.exists() and not args.overwrite:
        raise FileExistsError(output_mesh)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(input_mesh))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Bad input mesh: {input_mesh}")
    mesh.compute_vertex_normals()
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels_np = np.asarray(labels, dtype=np.int64)
    counts_np = np.asarray(counts, dtype=np.int64)
    areas_np = np.asarray(areas, dtype=np.float64)
    if counts_np.size == 0:
        raise RuntimeError("No connected triangle clusters found.")

    if bool(args.keep_largest):
        keep_clusters = {int(np.argmax(counts_np))}
    else:
        keep_clusters = {
            int(idx)
            for idx, (count, area) in enumerate(zip(counts_np.tolist(), areas_np.tolist()))
            if int(count) >= int(args.min_triangles) and float(area) >= float(args.min_area)
        }
        if not keep_clusters:
            keep_clusters = {int(np.argmax(counts_np))}

    remove_mask = np.asarray([int(label) not in keep_clusters for label in labels_np.tolist()], dtype=bool)
    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(remove_mask.tolist())
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_degenerate_triangles()
    cleaned.remove_duplicated_triangles()
    cleaned.remove_duplicated_vertices()
    cleaned.remove_non_manifold_edges()
    cleaned.compute_vertex_normals()
    if int(args.smooth_iterations) > 0:
        cleaned = cleaned.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))
        cleaned.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(output_mesh), cleaned, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write {output_mesh}")

    top_clusters = []
    for idx in np.argsort(counts_np)[::-1][:20].tolist():
        top_clusters.append(
            {
                "cluster": int(idx),
                "triangles": int(counts_np[idx]),
                "area": float(areas_np[idx]),
                "kept": int(idx) in keep_clusters,
            }
        )
    summary = {
        "task": "clean_teacher_mesh_components",
        "truthful_status": "mesh_cleanup_only_requires_strict_teacher_gate",
        "input_mesh": str(input_mesh.resolve()),
        "output_mesh": str(output_mesh.resolve()),
        "input_vertices": int(len(mesh.vertices)),
        "input_triangles": int(len(mesh.triangles)),
        "output_vertices": int(len(cleaned.vertices)),
        "output_triangles": int(len(cleaned.triangles)),
        "cluster_count": int(counts_np.size),
        "kept_cluster_count": int(len(keep_clusters)),
        "removed_triangles": int(remove_mask.sum()),
        "keep_largest": bool(args.keep_largest),
        "min_triangles": int(args.min_triangles),
        "min_area": float(args.min_area),
        "smooth_iterations": int(args.smooth_iterations),
        "top_clusters": top_clusters,
        "warning": "Cleanup is not a pass. Run audit_headface_teacher_surface.py and explicit Open3D visual review.",
    }
    summary_path = Path(args.output_summary) if args.output_summary else output_mesh.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
