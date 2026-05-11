from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_source(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": data.get("status"),
        "decision": data.get("decision"),
        "strict_candidate_passes": data.get("strict_candidate_passes"),
        "strict_teacher_passes": data.get("strict_teacher_passes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write V13 execution rollup.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v13_execution_rollup.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v13_execution_rollup.md")
    args = parser.parse_args()
    sources = {
        "truth_registry": read_summary(REPORTS / "20260508_v13_artifact_truth_registry.json"),
        "k1_autopsy": read_summary(REPORTS / "20260508_v13_kinect_alignment_autopsy.json"),
        "k2_alignment_candidates": read_summary(REPORTS / "20260508_v13_k2_kinect_alignment_candidates.json"),
        "k2b_temporal_sweep": read_summary(REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.json"),
        "k3_depth_targets": read_summary(Path("output/surface_research_preflight_local/V13_K3_kinect_depth_teacher_targets_camera_axes/kinect_teacher_summary.json")),
        "g1_2dgs": read_summary(REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.json"),
        "sapiens_2d": read_summary(REPORTS / "20260508_v13_sapiens_evidence_summary.json"),
        "hline": read_summary(REPORTS / "20260508_v13_hline_external_readiness.json"),
        "dline": read_summary(REPORTS / "20260508_v13_dline_promotion_report.json"),
        "strict_registry_refresh": read_summary(REPORTS / "20260508_v13_strict_gate_registry_refresh.json"),
    }
    dline = sources["dline"]
    strict_candidate_passes = int(dline.get("strict_candidate_passes", 0) or 0)
    strict_teacher_passes = int(dline.get("strict_teacher_passes", 0) or 0)
    summary = {
        "task": "v13_execution_rollup",
        "created_utc": utc_now(),
        "status": "v13_terminal_state_no_strict_pass",
        "strict_candidate_passes": strict_candidate_passes,
        "strict_teacher_passes": strict_teacher_passes,
        "formal_cloud_unblocked": bool(dline.get("formal_cloud_unblocked")),
        "promotion_status": dline.get("status"),
        "promotion_blockers": dline.get("blockers", []),
        "key_outcomes": {
            "kinect_alignment_candidate_ready": dline.get("gate_results", {}).get("kinect_alignment_candidate_ready"),
            "kinect_teacher_gate_pass": dline.get("gate_results", {}).get("kinect_teacher_gate_pass"),
            "kinect_temporal_sweep_teacher_gate_pass": dline.get("gate_results", {}).get("kinect_temporal_sweep_teacher_gate_pass"),
            "g1_2dgs_strict_teacher_precheck_pass": dline.get("gate_results", {}).get("g1_2dgs_strict_teacher_precheck_pass"),
            "sapiens_2d_supervision_ready": dline.get("gate_results", {}).get("sapiens_2d_supervision_ready"),
            "hline_ownership_ready": dline.get("gate_results", {}).get("hline_ownership_ready"),
        },
        "artifacts": {
            "dline_report": str((REPORTS / "20260508_v13_dline_promotion_report.json").resolve()),
            "k2_best_surface": sources["k2_alignment_candidates"].get("best_surface"),
            "k2b_summary": str((REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.json").resolve()),
            "g1_surface": sources["g1_2dgs"].get("artifacts", {}).get("surface"),
            "g1_contact_sheet": sources["g1_2dgs"].get("artifacts", {}).get("contact_sheet"),
            "sapiens_normals": sources["sapiens_2d"].get("normal", {}).get("npz"),
            "sapiens_depths": sources["sapiens_2d"].get("depth", {}).get("npz"),
            "strict_registry_refresh": str((REPORTS / "20260508_v13_strict_gate_registry_refresh.json").resolve()),
        },
        "sources": {name: compact_source(data) for name, data in sources.items()},
        "decision": (
            "V13 ran to D-line terminal state. Promotion remains blocked because no strict teacher/candidate pass exists. "
            "Sapiens 2D normal/depth supervision is available for later training, but it is not a 3D teacher."
        ),
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V13 Execution Rollup", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
