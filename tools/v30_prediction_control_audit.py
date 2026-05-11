from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v30_prior_common import (
    CONTROLS,
    LOCAL_ROOT,
    REPORT_JSON,
    REPORT_MD,
    REPO_ROOT,
    load_json,
    scan_forbidden,
    utc_now,
    write_json,
    write_report_markdown,
)


REPORT_CONTROL_JSON = REPO_ROOT / "reports/20260508_v30_prediction_control_audit.json"


def _control_status(verifier: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    usable_ckpt = bool(verifier.get("usable_prior_enabled_checkpoint_exists"))
    intake_pass = intake.get("status") == "DONE_PASS"
    for control in CONTROLS:
        if usable_ckpt and intake_pass:
            controls[control] = {
                "status": "DONE_PASS",
                "reason": "Control payload accepted by intake.",
                "depth_point_normal_metrics_available": True,
            }
        else:
            controls[control] = {
                "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
                "reason": "No usable prior-enabled VGGT checkpoint exists; generating this control would require base VGGT or random prior-enabled weights.",
                "depth_point_normal_metrics_available": False,
            }
    return controls


def run_control_audit() -> dict[str, Any]:
    verifier = load_json(REPO_ROOT / "reports/20260508_v30_prior_channel_verifier.json")
    intake = load_json(REPO_ROOT / "reports/20260508_v30_prior_enabled_prediction_intake.json")
    controls = _control_status(verifier, intake)
    blockers = []
    if not verifier.get("usable_prior_enabled_checkpoint_exists"):
        blockers.append("V30 cannot run real/zero/shuffle/random-region/prior-dropout prediction controls without a prior-enabled checkpoint.")
    if intake.get("status") != "DONE_PASS":
        blockers.append("V30 prediction intake did not find complete prior-enabled research outputs.")

    all_controls_pass = all(item.get("status") == "DONE_PASS" for item in controls.values())
    summary = {
        "task": "v30_prediction_control_audit",
        "created_utc": utc_now(),
        "status": "DONE_PASS" if all_controls_pass else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
        "decision": (
            "All prior-enabled controls have accepted prediction payloads."
            if all_controls_pass
            else "Controls fail closed with evidence; no base-VGGT or random-weight substitute is accepted for V30."
        ),
        "controls": controls,
        "verifier": {
            "status": verifier.get("status"),
            "code_supports_prior_adapter": verifier.get("code_supports_prior_adapter"),
            "usable_prior_enabled_checkpoint_exists": verifier.get("usable_prior_enabled_checkpoint_exists"),
            "base_hf_model_allowed_for_key_predictions": verifier.get("base_hf_model_allowed_for_key_predictions"),
        },
        "intake": {
            "status": intake.get("status"),
            "no_predictions_npz": intake.get("no_predictions_npz"),
            "no_package_registry_or_strict_pass": intake.get("no_package_registry_or_strict_pass"),
        },
        "blockers": blockers,
        "forbidden_findings": scan_forbidden(LOCAL_ROOT),
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
    }
    write_json(REPORT_CONTROL_JSON, summary)
    write_json(LOCAL_ROOT / "v30_prediction_control_audit.json", summary)
    return summary


def write_aggregate_report(control_summary: dict[str, Any]) -> dict[str, Any]:
    verifier = load_json(REPO_ROOT / "reports/20260508_v30_prior_channel_verifier.json")
    intake = load_json(REPO_ROOT / "reports/20260508_v30_prior_enabled_prediction_intake.json")
    final_status = (
        "DONE_PASS"
        if control_summary.get("status") == "DONE_PASS"
        and intake.get("status") == "DONE_PASS"
        and verifier.get("usable_prior_enabled_checkpoint_exists")
        else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
    )
    outputs = {
        "modal_entrypoint": str((REPO_ROOT / "modal_v30_prior_enabled_vggt_predictions.py").resolve()),
        "verifier_json": str((REPO_ROOT / "reports/20260508_v30_prior_channel_verifier.json").resolve()),
        "intake_json": str((REPO_ROOT / "reports/20260508_v30_prior_enabled_prediction_intake.json").resolve()),
        "control_audit_json": str((REPO_ROOT / "reports/20260508_v30_prediction_control_audit.json").resolve()),
        "local_output_dir": str(LOCAL_ROOT.resolve()),
    }
    blockers = []
    blockers.extend(verifier.get("blockers") or [])
    blockers.extend(intake.get("blockers") or [])
    blockers.extend(control_summary.get("blockers") or [])
    aggregate = {
        "task": "v30_prior_enabled_vggt_predictions",
        "created_utc": utc_now(),
        "status": final_status,
        "decision": (
            "V30 prior-enabled predictions are complete."
            if final_status == "DONE_PASS"
            else "V30 fail-closed: modified VGGT code supports HumanPriorAdapter, but no usable prior-enabled checkpoint exists for legal research predictions."
        ),
        "verifier": verifier,
        "intake": intake,
        "control_audit": control_summary,
        "blockers": blockers,
        "outputs": outputs,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "no_predictions_npz": intake.get("no_predictions_npz", True),
        "no_formal_package_registry_or_pass": intake.get("no_package_registry_or_strict_pass", True)
        and not control_summary.get("forbidden_findings"),
    }
    write_json(REPORT_JSON, aggregate)
    write_json(LOCAL_ROOT / "summary.json", aggregate)
    write_report_markdown(REPORT_MD, aggregate)
    write_report_markdown(LOCAL_ROOT / "summary.md", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    control_summary = run_control_audit()
    if args.aggregate:
        aggregate = write_aggregate_report(control_summary)
        print(aggregate["status"])
    else:
        print(control_summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
