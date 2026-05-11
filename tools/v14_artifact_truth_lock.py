from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, json_ready, read_ply_header, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ply_vertices(path: Path) -> int | None:
    if not path.is_file() or path.suffix.lower() != ".ply":
        return None
    try:
        return int(read_ply_header(path)[1])
    except Exception:
        return None


def entry(
    name: str,
    path: Path,
    truth_class: str,
    *,
    forbidden: bool,
    allowed_use: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.resolve()),
        "exists": path.exists(),
        "vertex_count": ply_vertices(path),
        "truth_class": truth_class,
        "forbidden_for_promotion": bool(forbidden),
        "allowed_use": allowed_use,
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V14 artifact truth lock.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v14_truth_registry.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v14_truth_registry.md")
    args = parser.parse_args()

    entries = [
        entry(
            "V10 unified_surface_v10",
            LOCAL_ROOT / "V10_unified_surface_merge_precheck/unified_surface_v10.ply",
            "failed_unified_candidate",
            forbidden=True,
            allowed_use="none",
            reason="V10 D-line blocked promotion; forbidden as V14 parent/source.",
        ),
        entry(
            "V11 hand11",
            LOCAL_ROOT / "V11_HHand_B_vggt_decoder/b_hand11_left_surface.ply",
            "diagnostic_or_bounded_hand_signal",
            forbidden=True,
            allowed_use="diagnostic reference only",
            reason="V11 hand route did not establish dense hand ownership.",
        ),
        entry(
            "V11 hair4",
            LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian/b_hair4_strands.ply",
            "diagnostic_hair_strands",
            forbidden=True,
            allowed_use="diagnostic reference only",
            reason="V11 hair route was not accepted as topology ownership.",
        ),
        entry(
            "V12 procedural TMF",
            LOCAL_ROOT / "V12_TMF_unified_surface_precheck/unified_surface_v12_tmf.ply",
            "procedural_tmf_diagnostic",
            forbidden=True,
            allowed_use="none",
            reason="V12 TMF was procedural diagnostic, not canonical teacher.",
        ),
        entry(
            "V13 failed G1 2DGS surface",
            LOCAL_ROOT / "V13_G1_2DGS_strict_surface/g1_2dgs_strict_surface_trimmed.ply",
            "weak_anchor_only",
            forbidden=False,
            allowed_use="weak geometry anchor / protocol debugging only",
            reason="G1 strict teacher precheck failed; cannot be promotion source.",
        ),
        entry(
            "V13 Kinect K2 best surface",
            LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search/k2_best_kinect_tsdf_vggt_world.ply",
            "weak_teacher_until_protocol_pass",
            forbidden=False,
            allowed_use="K14 protocol alignment/autopsy input only",
            reason="K2 nearest-neighbor residual was good, but official teacher gate failed.",
        ),
        entry(
            "V13 Sapiens normal/depth",
            Path("output/surface_research_cloud_preflight/V13_Sapiens_Normal/sapiens_normals.npz"),
            "supervision_only",
            forbidden=False,
            allowed_use="2D normal/depth supervision only",
            reason="Sapiens produces 2D supervision, not a 3D surface teacher.",
        ),
        {
            "name": "V13 Sapiens depth",
            "path": str(Path("output/surface_research_cloud_preflight/V13_Sapiens_Depth/sapiens_depths.npz").resolve()),
            "exists": Path("output/surface_research_cloud_preflight/V13_Sapiens_Depth/sapiens_depths.npz").is_file(),
            "vertex_count": None,
            "truth_class": "supervision_only",
            "forbidden_for_promotion": False,
            "allowed_use": "2D normal/depth supervision only",
            "reason": "Sapiens depth is 2D supervision, not 3D teacher.",
        },
    ]
    source_reports = {
        "v13_rollup": read_summary(REPORTS / "20260508_v13_execution_rollup.json"),
        "v13_dline": read_summary(REPORTS / "20260508_v13_dline_promotion_report.json"),
        "v13_k2": read_summary(REPORTS / "20260508_v13_k2_kinect_alignment_candidates.json"),
        "v13_k2b": read_summary(REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.json"),
        "v13_g1": read_summary(REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.json"),
        "v13_sapiens": read_summary(REPORTS / "20260508_v13_sapiens_evidence_summary.json"),
        "v13_hline": read_summary(REPORTS / "20260508_v13_hline_external_readiness.json"),
    }
    payload = {
        "task": "v14_artifact_truth_lock",
        "created_utc": utc_now(),
        "status": "v14_truth_lock_ready",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "entries": entries,
        "forbidden_names": [item["name"] for item in entries if item.get("forbidden_for_promotion")],
        "source_status": {name: data.get("status") for name, data in source_reports.items()},
        "decision": "V14 promotion must fail if any forbidden source enters a candidate/teacher package.",
    }
    write_json(args.output_json, payload)
    write_report(args.output_md, "V14 Artifact Truth Lock", payload)
    print(json.dumps(json_ready({"status": payload["status"], "output": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
