from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, REPO_ROOT, json_ready, read_ply_header, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ply_vertices(path: Path) -> int | None:
    if not path.is_file() or path.suffix.lower() != ".ply":
        return None
    try:
        return int(read_ply_header(path)[1])
    except Exception:
        return None


def _summary(path: Path) -> dict[str, Any]:
    return read_summary(path) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V13 artifact truth registry.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v13_artifact_truth_registry.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v13_artifact_truth_registry.md")
    args = parser.parse_args()
    entries = {
        "v10_unified_surface_v10": {
            "path": LOCAL_ROOT / "V10_unified_surface_merge_precheck/unified_surface_v10.ply",
            "truth_class": "failed_candidate_forbidden_parent",
            "forbidden_for_promotion": True,
            "reason": "V10 D-line blocked candidate promotion.",
        },
        "v11_g3_2dgs_anchor": {
            "path": LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply",
            "truth_class": "research_anchor_surface",
            "forbidden_for_promotion": False,
            "allowed_use": "surface extraction / teacher-intake evidence only",
        },
        "v11_hhand_b": {
            "path": LOCAL_ROOT / "V11_HHand_B_vggt_decoder/b_hand11_left_surface.ply",
            "truth_class": "bounded_positive_not_hand_surface",
            "forbidden_for_promotion": True,
            "reason": "Bounded overfit positive signal only; not verified hand ownership.",
        },
        "v11_hhair_b": {
            "path": LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian/b_hair4_strands.ply",
            "truth_class": "diagnostic_strands_not_hair_topology",
            "forbidden_for_promotion": True,
            "reason": "Diagnostic strand output; V11/V12 D-line did not accept hair ownership.",
        },
        "v12_tmf": {
            "path": LOCAL_ROOT / "V12_TMF_unified_surface_precheck/unified_surface_v12_tmf.ply",
            "truth_class": "procedural_temporal_diagnostic",
            "forbidden_for_promotion": True,
            "reason": "V12 TMF produced procedural diagnostic surfaces; not_proxy=false.",
        },
        "v12_kinect_tsdf": {
            "path": LOCAL_ROOT / "V12_Kinect_TSDF_VGGT_world_frame0000/kinect_tsdf_vggt_world.ply",
            "truth_class": "real_dense_sensor_surface_weak_teacher_pool",
            "forbidden_for_promotion": True,
            "reason": "Real dense sensor surface, but V12 teacher intake failed residual/reprojection/region gates.",
        },
        "v12_kinect_tsdf_real_world": {
            "path": LOCAL_ROOT / "V12_Kinect_TSDF_frame0000/kinect_tsdf_real_world.ply",
            "truth_class": "real_sensor_surface_unaligned",
            "forbidden_for_promotion": True,
            "reason": "Real-world Kinect coordinates only; needs strict RGB known-camera alignment.",
        },
    }
    for item in entries.values():
        path = Path(item["path"])
        item["exists"] = path.exists()
        item["vertex_count"] = _ply_vertices(path)
    source_summaries = {
        "v11_rollup": _summary(REPORTS / "20260508_v11_execution_rollup.json"),
        "v11_dline": _summary(REPORTS / "20260508_v11_dline_promotion_report.json"),
        "v12_rollup": _summary(REPORTS / "20260508_v12_tmf_execution_rollup.json"),
        "v12_dline": _summary(REPORTS / "20260508_v12_tmf_dline_promotion_report.json"),
        "v12_kinect": _summary(REPORTS / "20260508_v12_kinect_tsdf_teacher_intake_audit.json"),
    }
    payload = {
        "task": "v13_artifact_truth_registry",
        "created_utc": utc_now(),
        "status": "v13_truth_registry_ready",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "entries": entries,
        "source_status": {name: data.get("status") for name, data in source_summaries.items()},
        "decision": "V13 may use real/anchor evidence for new extraction and alignment, but proxy/failed artifacts are forbidden as pass sources.",
        "blockers": [],
    }
    write_json(args.output_json, payload)
    write_report(args.output_md, "V13 Artifact Truth Registry", payload)
    print(json.dumps(json_ready({"status": payload["status"], "output": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
