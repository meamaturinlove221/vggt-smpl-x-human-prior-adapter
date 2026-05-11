from __future__ import annotations

from pathlib import Path

from v44_v50_common import LOCAL_OUT, REPORTS, read_json, scan_forbidden, utc_now, write_json, write_md


JSON = REPORTS / "20260509_v44_v50_completion_audit.json"
MD = REPORTS / "20260509_v44_v50_completion_audit.md"


STAGES = {
    "v44": {
        "report": REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V44_strict_visual_pre_promotion_gate" / "summary.json",
        ],
    },
    "v45": {
        "report": REPORTS / "20260509_v45_head_face_correction.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V45_head_face_correction" / "summary.json",
        ],
    },
    "v46": {
        "report": REPORTS / "20260509_v46_hand_correction.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V46_hand_correction" / "summary.json",
        ],
    },
    "v47": {
        "report": REPORTS / "20260509_v47_60view_correction.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V47_60view_correction" / "summary.json",
        ],
    },
    "v48": {
        "report": REPORTS / "20260509_v48_temporal_correction.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V48_temporal_correction" / "summary.json",
        ],
    },
    "v49": {
        "report": REPORTS / "20260509_v49_package_dry_run.json",
        "allowed_statuses": {"DONE_PASS", "DONE_FAIL_ROUTED"},
        "required_outputs": [
            LOCAL_OUT / "V49_package_dry_run" / "summary.json",
            LOCAL_OUT / "V49_package_dry_run" / "registry_entry_dryrun.json",
        ],
    },
    "v50": {
        "report": REPORTS / "20260509_v50_final_promotion_transaction.json",
        "allowed_statuses": {"DONE_PASS", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
        "required_outputs": [
            LOCAL_OUT / "V50_final_promotion_transaction" / "summary.json",
        ],
    },
}


def main() -> int:
    blockers: list[str] = []
    stage_rows = {}
    for stage, spec in STAGES.items():
        report = spec["report"]
        data = read_json(report)
        status = str(data.get("status", "MISSING"))
        row = {
            "report": str(report),
            "report_exists": report.is_file(),
            "status": status,
            "required_outputs": {},
        }
        if not report.is_file():
            blockers.append(f"{stage}_report_missing")
        if status not in spec["allowed_statuses"]:
            blockers.append(f"{stage}_unexpected_status_{status}")
        for output in spec["required_outputs"]:
            exists = output.is_file()
            row["required_outputs"][str(output)] = exists
            if not exists:
                blockers.append(f"{stage}_required_output_missing:{output}")
        stage_rows[stage] = row
    current_roots = [LOCAL_OUT / f"V{num}_{name}" for num, name in [
        (44, "strict_visual_pre_promotion_gate"),
        (45, "head_face_correction"),
        (46, "hand_correction"),
        (47, "60view_correction"),
        (48, "temporal_correction"),
        (49, "package_dry_run"),
        (50, "final_promotion_transaction"),
    ]]
    forbidden = scan_forbidden(current_roots)
    if forbidden:
        blockers.append("forbidden_output_hit_in_v44_v50_roots")
    status = "COMPLETE_AUDIT_PASS" if not blockers else "COMPLETE_AUDIT_FAIL"
    summary = {
        "task": "v44_v50_completion_audit",
        "created_utc": utc_now(),
        "status": status,
        "stage_rows": stage_rows,
        "forbidden_hit_count": len(forbidden),
        "forbidden_hits": forbidden,
        "blockers": blockers,
        "all_required_branches_executed": not blockers,
    }
    write_json(JSON, summary)
    write_md(MD, "V44-V50 Completion Audit", summary, [
        f"- checked_stages: `{len(STAGES)}`",
        f"- forbidden_hit_count: `{len(forbidden)}`",
    ])
    print({"status": status, "blockers": blockers, "forbidden_hit_count": len(forbidden)})
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
