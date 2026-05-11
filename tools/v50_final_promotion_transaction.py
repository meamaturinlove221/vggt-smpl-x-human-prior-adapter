from __future__ import annotations

from pathlib import Path

from v44_v50_common import (
    LOCAL_OUT,
    REPORTS,
    base_stage_statuses,
    read_json,
    scan_forbidden,
    utc_now,
    v30_prior_prediction_ready,
    write_json,
    write_md,
)


OUT = LOCAL_OUT / "V50_final_promotion_transaction"
JSON = REPORTS / "20260509_v50_final_promotion_transaction.json"
MD = REPORTS / "20260509_v50_final_promotion_transaction.md"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stage_files = {
        "v44": REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.json",
        "v45": REPORTS / "20260509_v45_head_face_correction.json",
        "v46": REPORTS / "20260509_v46_hand_correction.json",
        "v47": REPORTS / "20260509_v47_60view_correction.json",
        "v48": REPORTS / "20260509_v48_temporal_correction.json",
        "v49": REPORTS / "20260509_v49_package_dry_run.json",
    }
    stages = {name: read_json(path).get("status", "MISSING") for name, path in stage_files.items()}
    blockers: list[str] = []
    missing = [name for name, path in stage_files.items() if not path.is_file()]
    if missing:
        blockers.append(f"missing V44-V49 reports: {missing}")
    for name, status in stages.items():
        if status != "DONE_PASS":
            blockers.append(f"{name} not pass-ready: {status}")
    v30_ready, v30_blockers, _ = v30_prior_prediction_ready()
    if not v30_ready:
        blockers.extend(v30_blockers)
    visual_pass = Path("output/surface_research_preflight_local/V44_strict_visual_pre_promotion_gate/visual_review_codex_pass.json")
    if not visual_pass.is_file():
        blockers.append("visual_review_codex_pass.json missing")
    forbidden = scan_forbidden([OUT])
    if forbidden:
        blockers.append("forbidden current V50 output detected")
    strict_pass = not blockers
    status = "DONE_PASS" if strict_pass else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
    candidate_package_path = OUT / "candidate_package_v50" if strict_pass else None
    registry_entry_path = OUT / "strict_registry_entry_v50.json" if strict_pass else None
    if strict_pass:
        # This branch is intentionally unreachable unless all strict preconditions pass.
        candidate_package_path.mkdir(parents=True, exist_ok=True)
        write_json(registry_entry_path, {"strict_candidate_pass": True, "created_utc": utc_now()})
    summary = {
        "task": "v50_final_promotion_transaction",
        "created_utc": utc_now(),
        "status": status,
        "all_routes_executed": not missing,
        "strict_candidate_passes": 1 if strict_pass else 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": strict_pass,
        "candidate_package_path": candidate_package_path,
        "teacher_package_path": None,
        "registry_entry_path": registry_entry_path,
        "writes_package": strict_pass,
        "writes_strict_registry": strict_pass,
        "writes_strict_pass": strict_pass,
        "mentor_decision_needed": not strict_pass,
        "stage_statuses": stages,
        "prior_stage_statuses": base_stage_statuses(),
        "remaining_blockers": blockers,
        "forbidden_hit_count": len(forbidden),
        "decision": (
            "V50 strict promotion passed and wrote formal candidate package/registry."
            if strict_pass
            else "V50 executed all V44-V49 routes and failed with hard evidence. No formal package/registry/pass was written."
        ),
    }
    write_json(JSON, summary)
    write_json(OUT / "summary.json", summary)
    write_md(MD, "V50 Final Promotion Transaction", summary, [
        f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        f"- formal_cloud_unblocked: `{summary['formal_cloud_unblocked']}`",
        f"- candidate_package_path: `{candidate_package_path}`",
        f"- registry_entry_path: `{registry_entry_path}`",
    ])
    print({"status": status, "strict_candidate_passes": summary["strict_candidate_passes"], "strict_teacher_passes": summary["strict_teacher_passes"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
