from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v29_normal_utils import (
    REGION_NAMES,
    REPORTS,
    V29_OUT,
    all_regions_nonempty,
    load_npz,
    utc_now,
    vector_length_stats,
    write_json,
    write_md,
)


DEFAULT_TEACHER = V29_OUT / "v29_teacher_normals_world.npz"
DEFAULT_TEMPORAL = V29_OUT / "v29_temporal_normals_world.npz"
DEFAULT_CANDIDATE = V29_OUT / "v29_candidate_geometric_normals.npz"
DEFAULT_JSON = REPORTS / "20260508_v29_normal_route_rescue.json"
DEFAULT_MD = REPORTS / "20260508_v29_normal_route_rescue.md"
DEFAULT_SUMMARY = V29_OUT / "summary.json"


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def support_pass(summary: dict[str, Any]) -> bool:
    support = summary.get("region_normal_support", {})
    return all(int(support.get(name, 0)) > 0 for name in REGION_NAMES)


def normal_file_stats(path: Path, normal_key: str, valid_key: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    z = load_npz(path, allow_pickle=True)
    if normal_key not in z:
        return {"exists": True, "missing_key": normal_key}
    normals = np.asarray(z[normal_key], dtype=np.float32)
    valid = None
    if valid_key and valid_key in z:
        valid = np.asarray(z[valid_key], dtype=bool)
    return {
        "exists": True,
        "shape": list(normals.shape),
        "normal_length_stats": vector_length_stats(normals, valid),
        "valid_count": int(valid.sum()) if valid is not None else None,
    }


def candidate_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    z = load_npz(path, allow_pickle=True)
    frame_keys = [str(x) for x in np.asarray(z.get("frame_keys", np.asarray([]))).tolist()]
    out: dict[str, Any] = {"exists": True, "frame_keys": frame_keys, "frames": {}}
    for frame in frame_keys:
        key = f"{frame}_candidate_geometric_normals"
        valid_key = f"{frame}_candidate_normal_valid"
        if key in z:
            out["frames"][frame] = {
                "shape": list(np.asarray(z[key]).shape),
                "normal_length_stats": vector_length_stats(np.asarray(z[key]), np.asarray(z.get(valid_key), dtype=bool) if valid_key in z else None),
                "valid_count": int(np.asarray(z[valid_key], dtype=bool).sum()) if valid_key in z else None,
                "source": "geometric_from_candidate_points",
                "not_model_normal_head": True,
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V29 teacher, temporal, and candidate normal evidence.")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--temporal", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    teacher_summary = load_summary(V29_OUT / "v29_teacher_normal_route_rescue_summary.json")
    temporal_summary = load_summary(V29_OUT / "v29_temporal_normal_audit.json")
    candidate_summary = load_summary(V29_OUT / "v29_candidate_geometric_normal_summary.json")

    teacher_pass = support_pass(teacher_summary)
    temporal_pass = support_pass(temporal_summary)
    candidate_pass = support_pass(candidate_summary)
    normal_depth_consistency = (
        teacher_summary.get("teacher_vs_geometric_normal_consistency", {}).get("median_abs_dot", 0.0) >= 0.5
        and teacher_summary.get("normal_valid_count", 0) > 0
    )

    status = "DONE_PASS" if teacher_pass and temporal_pass and candidate_pass and normal_depth_consistency else "DONE_FAIL_ROUTED"
    payload = {
        "status": status,
        "created_at": utc_now(),
        "teacher_normal_available": teacher_pass,
        "temporal_normal_available": temporal_pass,
        "candidate_geometric_normal_available": candidate_pass,
        "normal_depth_consistency_research_gate": bool(normal_depth_consistency),
        "teacher_summary": teacher_summary,
        "temporal_summary": temporal_summary,
        "candidate_summary": candidate_summary,
        "normal_file_stats": {
            "teacher": normal_file_stats(args.teacher, "v29_teacher_normals_world", "v29_teacher_normal_valid"),
            "temporal": normal_file_stats(args.temporal, "v29_temporal_normals_world", "v29_temporal_normal_valid"),
            "candidate": candidate_stats(args.candidate),
        },
        "forbidden_writes": {
            "predictions_npz": False,
            "candidate_package": False,
            "teacher_package": False,
            "strict_registry": False,
            "strict_pass": False,
        },
        "outputs": {
            "teacher_normals": args.teacher,
            "temporal_normals": args.temporal,
            "candidate_geometric_normals": args.candidate,
            "json_report": args.output_json,
            "md_report": args.output_md,
        },
    }
    write_json(args.output_json, payload)
    write_json(args.summary, payload)

    lines = [
        f"- status: `{status}`",
        f"- teacher_normal_available: `{teacher_pass}`",
        f"- temporal_normal_available: `{temporal_pass}`",
        f"- candidate_geometric_normal_available: `{candidate_pass}`",
        f"- normal_depth_consistency_research_gate: `{normal_depth_consistency}`",
        "",
        "## Region Support",
        f"- teacher: `{teacher_summary.get('region_normal_support', {})}`",
        f"- temporal: `{temporal_summary.get('region_normal_support', {})}`",
        f"- candidate: `{candidate_summary.get('region_normal_support', {})}`",
        "",
        "## Sources",
        "- teacher normals: V24 `teacher_normals_world`, propagated and normalized.",
        "- temporal normals: geometric finite differences from V26 `target_frame_points`; V26 itself declared `normal_available=false`.",
        "- candidate normals: geometric finite differences from V25 `research_points_world.npz`; marked `not_model_normal_head=true`.",
        "",
        "## Guard",
        "- No `predictions.npz`, candidate package, teacher package, strict registry, or strict pass was written.",
    ]
    write_md(args.output_md, "V29 Normal Route Rescue", lines)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
