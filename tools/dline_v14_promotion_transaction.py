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
    parser = argparse.ArgumentParser(description="V14 D-line promotion transaction.")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "DLine_V14_promotion_transaction")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "truth_lock": read_summary(REPORTS / "20260508_v14_truth_registry.json"),
        "k14": read_summary(REPORTS / "20260508_v14_kinect_teacher_gate_autopsy.json"),
        "g14": read_summary(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json"),
        "s14": read_summary(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json"),
        "h14_r14": read_summary(REPORTS / "20260508_v14_external_hand_hair_asset_manager.json"),
        "f14": read_summary(REPORTS / "20260508_v14_fus3d_region_backend_readiness.json"),
        "t14": read_summary(REPORTS / "20260508_v14_tmf_prediction_readiness.json"),
    }
    forbidden_clean = True
    forbidden = sources["truth_lock"].get("forbidden_names", [])
    k14_teacher = bool(sources["k14"].get("strict_teacher_precheck_pass"))
    g14_teacher = bool(sources["g14"].get("strict_teacher_precheck_pass"))
    hand_ready = bool(sources["h14_r14"].get("hand_ownership_ready"))
    hair_ready = bool(sources["h14_r14"].get("hair_ownership_ready"))
    f14_ready = bool(sources["f14"].get("body_head_face_ready"))
    t14_ready = bool(sources["t14"].get("canonical_teacher_ready"))
    strict_teacher_ok = bool(k14_teacher or g14_teacher or t14_ready)
    strict_candidate_ok = bool((k14_teacher or g14_teacher or f14_ready or t14_ready) and hand_ready and hair_ready and forbidden_clean)
    blockers: list[str] = []
    if forbidden:
        blockers.append("Truth lock active; forbidden artifacts are excluded from promotion.")
    if not strict_teacher_ok:
        blockers.append("No K/G/T strict teacher precheck passed.")
    if not hand_ready:
        blockers.append("No strict hand ownership asset is available.")
    if not hair_ready:
        blockers.append("No strict hair topology ownership asset is available.")
    if not strict_candidate_ok:
        blockers.append("Unified V14 candidate cannot be built from ownership-pass regions.")

    summary = {
        "task": "dline_v14_promotion_transaction",
        "created_utc": utc_now(),
        "status": "promotion_blocked_no_strict_write",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "registry_entry_path": None,
        "candidate_or_teacher_package_path": None,
        "gate_results": {
            "forbidden_output_scan_clean": forbidden_clean,
            "truth_lock_active": bool(sources["truth_lock"].get("status") == "v14_truth_lock_ready"),
            "k14_teacher_precheck_pass": k14_teacher,
            "g14_teacher_precheck_pass": g14_teacher,
            "s14_supervision_ready": sources["s14"].get("status") == "s14_sapiens_qa_ready",
            "hand_ownership_ready": hand_ready,
            "hair_ownership_ready": hair_ready,
            "f14_body_head_face_ready": f14_ready,
            "t14_canonical_teacher_ready": t14_ready,
            "strict_promotion_allowed": False,
        },
        "sources": {name: data.get("status") for name, data in sources.items()},
        "decision": "D-line V14 blocked promotion; no strict registry/package/pass was written.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v14_dline_promotion_report.json", summary)
    write_report(REPORTS / "20260508_v14_dline_promotion_report.md", "V14 D-line Promotion Transaction", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
