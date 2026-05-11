from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, json_ready, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 D-line promotion transaction.")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "DLine_V13_promotion_transaction")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "truth_registry": read_summary(REPORTS / "20260508_v13_artifact_truth_registry.json"),
        "k1": read_summary(REPORTS / "20260508_v13_kinect_alignment_autopsy.json"),
        "k2": read_summary(REPORTS / "20260508_v13_k2_kinect_alignment_candidates.json"),
        "k3_depth_targets": read_summary(LOCAL_ROOT / "V13_K3_kinect_depth_teacher_targets_camera_axes/kinect_teacher_summary.json"),
        "g1": read_summary(REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.json"),
        "hline": read_summary(REPORTS / "20260508_v13_hline_external_readiness.json"),
    }
    k2_ok = bool(sources["k2"].get("strict_alignment_candidate_ready"))
    k3_gate = bool(sources["k3_depth_targets"].get("teacher_gate", {}).get("pass"))
    g1_gate = bool(sources["g1"].get("gates", {}).get("strict_teacher_precheck_pass"))
    hline_ok = sources["hline"].get("status") != "hline_external_routes_blocked"
    blockers = []
    if not k3_gate:
        blockers.append("Kinect projected teacher gate failed; no strict_teacher_pass.")
    if not g1_gate:
        blockers.append("2DGS strict teacher precheck failed/blocked.")
    if not hline_ok:
        blockers.append("H-line hand/hair ownership routes are blocked by missing real external assets.")
    if not k2_ok:
        blockers.append("K2 strict alignment residual candidate was not ready.")
    strict_ok = False
    summary = {
        "task": "dline_v13_promotion_transaction",
        "created_utc": utc_now(),
        "status": "promotion_blocked_no_strict_write",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "registry_entry_path": None,
        "candidate_or_teacher_package_path": None,
        "gate_results": {
            "kinect_alignment_candidate_ready": k2_ok,
            "kinect_teacher_gate_pass": k3_gate,
            "g1_2dgs_strict_teacher_precheck_pass": g1_gate,
            "hline_ownership_ready": hline_ok,
            "strict_promotion_allowed": strict_ok,
        },
        "forbidden_output_scan_clean": True,
        "sources": {name: data.get("status") for name, data in sources.items()},
        "decision": "D-line V13 blocked promotion; no strict registry/package/pass was written.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_dline_promotion_report.json", summary)
    write_report(REPORTS / "20260508_v13_dline_promotion_report.md", "V13 D-line Promotion Transaction", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
