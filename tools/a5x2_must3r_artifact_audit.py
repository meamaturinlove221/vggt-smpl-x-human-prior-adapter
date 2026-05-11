from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUST3R_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend"
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "output/surface_research_preflight_local/connected_payload_self_describing/connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend_audit"
DEFAULT_REPORT_JSON = REPO_ROOT / "reports/20260507_v9_a5x2_must3r_artifact_audit.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports/20260507_v9_a5x2_must3r_artifact_audit.md"


PART_NAMES = {
    "full_body": None,
    "left_hand": (1,),
    "right_hand": (2,),
    "face_core": (3,),
    "hairline": (4,),
    "head": (3, 4),
}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def ply_vertex_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                return None
    return None


def load_points(path: Path) -> np.ndarray:
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float64)
    except Exception:
        return load_ascii_or_binary_ply_minimal(path)


def load_ascii_or_binary_ply_minimal(path: Path) -> np.ndarray:
    # Fallback intentionally only handles the binary little-endian layout emitted
    # by trimesh for this MUSt3R run: xyz rgb alpha.
    import struct

    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("utf-8", errors="ignore").splitlines()
    count = 0
    for line in header:
        if line.startswith("element vertex "):
            count = int(line.split()[-1])
            break
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    stride = 3 * 4 + 4
    pts = np.zeros((count, 3), dtype=np.float64)
    offset = header_end
    for idx in range(count):
        pts[idx] = struct.unpack_from("<fff", data, offset)
        offset += stride
    return pts


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def bbox(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {"valid": False}
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    return {
        "valid": True,
        "p01": lo.tolist(),
        "p99": hi.tolist(),
        "extent_p01_p99": (hi - lo).tolist(),
        "center_p01_p99": ((hi + lo) * 0.5).tolist(),
    }


def template_regions(template_path: Path) -> dict[str, np.ndarray]:
    with np.load(template_path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float64)
        part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for name, ids in PART_NAMES.items():
        if ids is None:
            out[name] = vertices
        else:
            mask = np.isin(part_ids[: vertices.shape[0]], list(ids))
            out[name] = vertices[mask]
    return out


def region_coverage(points: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, Any]:
    if points.size == 0:
        return {name: {"region_point_count": int(region.shape[0]), "near_must3r_count": 0} for name, region in regions.items()}
    # Use a bounding-box overlap proxy, because MUSt3R world scale is not guaranteed
    # aligned to the SMPL-X canonical frame yet.
    p_lo = np.percentile(points, 1, axis=0)
    p_hi = np.percentile(points, 99, axis=0)
    out: dict[str, Any] = {}
    for name, region in regions.items():
        if region.size == 0:
            out[name] = {"region_point_count": 0, "bbox_overlap_proxy": 0.0}
            continue
        r_lo = np.percentile(region, 1, axis=0)
        r_hi = np.percentile(region, 99, axis=0)
        overlap = np.maximum(0.0, np.minimum(p_hi, r_hi) - np.maximum(p_lo, r_lo))
        denom = np.maximum(1e-8, np.maximum(r_hi, p_hi) - np.minimum(r_lo, p_lo))
        out[name] = {
            "region_point_count": int(region.shape[0]),
            "template_bbox_extent": (r_hi - r_lo).tolist(),
            "bbox_overlap_proxy": float(np.prod(overlap / denom)),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit true MUSt3R A5-X2 artifact for non-empty teacher-intake readiness.")
    parser.add_argument("--must3r-dir", type=Path, default=DEFAULT_MUST3R_DIR)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    best = summary["best_artifact"]
    lines = [
        "# V9 A5-X2 MUSt3R Artifact Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This audit checks the real MUSt3R backend output only. It cannot write strict teacher/candidate/pass state.",
        "",
        "## Best Artifact",
        "",
        f"- path: `{best.get('path')}`",
        f"- vertices: `{best.get('vertex_count')}`",
        f"- finite_ratio: `{best.get('finite_ratio')}`",
        "",
        "## Decision",
        "",
        summary["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = args.must3r_dir / "summary.json"
    backend_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    ply_files = sorted((args.must3r_dir / "must3r_run").glob("*.ply"))
    artifacts = []
    best: dict[str, Any] = {}
    best_points = np.zeros((0, 3), dtype=np.float64)
    for path in ply_files:
        count = ply_vertex_count(path)
        item = {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "vertex_count": count}
        if count and count > 0:
            points = load_points(path)
            finite = np.isfinite(points).all(axis=1) if points.size else np.zeros((0,), dtype=bool)
            item.update(
                {
                    "loaded_point_count": int(points.shape[0]),
                    "finite_ratio": float(np.mean(finite)) if finite.size else 0.0,
                    "bbox": bbox(points[finite]),
                    "radius_stats": scalar_stats(np.linalg.norm(points[finite], axis=1)) if finite.any() else {},
                }
            )
            if int(count) > int(best.get("vertex_count", 0) or 0):
                best = dict(item)
                best_points = points[finite]
        artifacts.append(item)
    regions = template_regions(args.template_payload)
    region_scores = region_coverage(best_points, regions)
    ready_for_weak_pool = bool(best.get("vertex_count", 0) and best_points.shape[0] > 10000)
    strict_teacher_ready = False
    summary = {
        "task": "v9_a5x2_must3r_artifact_audit",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "a5x2_true_backend_artifact_nonempty_weak_pool_only" if ready_for_weak_pool else "a5x2_true_backend_artifact_not_usable",
        "contract": {
            "research_only": True,
            "no_export": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_write": True,
        },
        "backend_summary": {
            "status": backend_summary.get("status"),
            "checkpoint_name": backend_summary.get("checkpoint_name"),
            "input_image_count": backend_summary.get("input_image_count"),
            "max_ply_vertex_count": backend_summary.get("max_ply_vertex_count"),
            "modal_run_artifact": str(summary_path.resolve()),
        },
        "artifacts": artifacts,
        "best_artifact": best,
        "region_overlap_proxy": region_scores,
        "ready_for_weak_teacher_pool": ready_for_weak_pool,
        "strict_teacher_ready": strict_teacher_ready,
        "blockers": [
            "MUSt3R export is not known-camera aligned to original 4K4D camera frame yet",
            "default confidence thresholds exported zero vertices; usable output requires low confidence threshold",
            "no depth residual / original 6-view reprojection audit has passed",
            "no full/head/face/hairline/hands Open3D visual gate has passed",
        ],
        "decision": (
            "A5X2_MUST3R_WEAK_POOL_ONLY: real MUSt3R backend produced a non-empty pointcloud, but it is not a strict teacher. Keep it as weak external teacher evidence until known-camera alignment and region visual gates pass."
            if ready_for_weak_pool
            else "A5X2_MUST3R_NOT_USABLE: no non-empty real pointcloud artifact was available."
        ),
    }
    write_json(out / "summary.json", summary)
    write_markdown(out / "report.md", summary)
    write_json(args.report_json.expanduser().resolve(), summary)
    write_markdown(args.report_md.expanduser().resolve(), summary)
    print(json.dumps({"status": summary["status"], "best_vertices": best.get("vertex_count", 0), "strict_teacher_ready": False}, ensure_ascii=False))
    return 0 if ready_for_weak_pool else 2


if __name__ == "__main__":
    raise SystemExit(main())
