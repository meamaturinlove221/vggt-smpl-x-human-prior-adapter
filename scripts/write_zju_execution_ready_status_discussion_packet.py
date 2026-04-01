import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

BASELINE_DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
BASELINE_VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
BOUNDARY_JSON = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
SMOKE_SPEC_JSON = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
INTEGRATION_CHECK_JSON = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CHECKLIST_JSON = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"
PREP_PLAN_JSON = OUTPUT_ROOT / "execution_ready_prep_plan.camera_focal_objective_isolation.20260331.json"
PREPARATION_DESIGN_JSON = OUTPUT_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.json"
DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"
STATUS_DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"

PACKET_JSON = OUTPUT_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.json"
PACKET_MD = OUTPUT_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.md"

NEXT_REQUIREMENT = "manual_review_execution_ready_status_discussion_before_any_execution_ready_promotion"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def render_md(payload: dict) -> str:
    lines = [
        "# Execution-Ready Status Discussion Packet: Camera Focal Objective Isolation (2026-04-01)",
        "",
        f"- state: `{payload['state']}`",
        f"- manual_action_kind: `{payload['manual_action_kind']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- first_allowed_next_step: `{payload['first_allowed_next_step']}`",
        "",
        "## Discussion Objective",
        "",
        f"- {payload['discussion_objective']}",
        "",
        "## Current Truth Assertions",
        "",
    ]
    for item in payload["current_truth_assertions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Review Questions", ""])
    for item in payload["review_questions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Still Forbidden", ""])
    for item in payload["still_forbidden"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Reviewed Packet Components", ""])
    for key, value in payload["reviewed_packet_components"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Supporting Refs", ""])
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    baseline_decision = load_json(BASELINE_DECISION_JSON)
    baseline_validation = load_json(BASELINE_VALIDATION_JSON)
    boundary = load_json(BOUNDARY_JSON)
    gate = load_json(GATE_JSON)
    smoke_spec = load_json(SMOKE_SPEC_JSON)
    integration_check = load_json(INTEGRATION_CHECK_JSON)
    boundary_checklist = load_json(BOUNDARY_CHECKLIST_JSON)
    prep_plan = load_json(PREP_PLAN_JSON)
    preparation_design = load_json(PREPARATION_DESIGN_JSON)
    discussion_decision = load_json(DISCUSSION_DECISION_JSON)
    status_discussion_decision = load_json(STATUS_DISCUSSION_DECISION_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_status_discussion_packet",
        "family": "camera_focal_objective_isolation",
        "state": "manual_review_execution_ready_status_discussion_packet_ready",
        "manual_action_kind": "manual_review",
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "discussion_objective": (
            "Package the accepted loss.py-only execution-prep baseline, the bounded non-training packet, "
            "and the status-discussion decision into one reviewable manual status-discussion handoff."
        ),
        "current_truth_assertions": [
            "training/loss.py remains the only allowed authored patch surface for this family.",
            "The validated execution-prep baseline and stronger local validation remain the accepted baseline.",
            "The family is in execution-ready status discussion only; ready_for_execution stays false.",
            "Arm, run, training, config materialization, and cloud usage all remain disabled.",
        ],
        "review_questions": [
            "Is the loss.py-only boundary still the honest baseline for any later execution-ready promotion?",
            "Is the smoke, integration, and boundary packet complete enough for future promotion decisions without generating config or running training?",
            "Should the repo remain in status-discussion hold with no auto-open until a separate explicit execution-ready promotion decision is made?",
        ],
        "still_forbidden": [
            "Do not mark ready_for_execution=true automatically.",
            "Do not arm any approved problem.",
            "Do not run candidate or training flows.",
            "Do not materialize config or yaml.",
            "Do not use cloud or Modal.",
        ],
        "first_allowed_next_step": str(status_discussion_decision.get("first_allowed_next_step", "")).strip(),
        "reviewed_packet_components": {
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "execution_prep_baseline_validation": repo_rel(BASELINE_VALIDATION_JSON),
            "execution_ready_boundary": repo_rel(BOUNDARY_JSON),
            "execution_ready_gate": repo_rel(GATE_JSON),
            "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
            "execution_ready_prep_plan": repo_rel(PREP_PLAN_JSON),
            "execution_ready_preparation_design": repo_rel(PREPARATION_DESIGN_JSON),
            "execution_ready_discussion_decision": repo_rel(DISCUSSION_DECISION_JSON),
            "execution_ready_status_discussion_decision": repo_rel(STATUS_DISCUSSION_DECISION_JSON),
        },
        "supporting_refs": {
            "baseline_decision_value": baseline_decision.get("decision", ""),
            "baseline_validation_status": baseline_validation.get("overall_status", ""),
            "boundary_target_file": boundary.get("current_baseline_definition", {}).get("target_file", ""),
            "gate_recommended_decision": gate.get("recommended_decision", ""),
            "smoke_scope": smoke_spec.get("scope", ""),
            "integration_why_not_execution": integration_check.get("why_this_is_not_execution", ""),
            "boundary_checklist_state": boundary_checklist.get("state", ""),
            "prep_plan_objective": prep_plan.get("plan_objective", ""),
            "preparation_design_state": preparation_design.get("state", ""),
            "discussion_decision_value": discussion_decision.get("decision", ""),
            "status_discussion_decision_value": status_discussion_decision.get("decision", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(PACKET_JSON, payload)
    write_text(PACKET_MD, render_md(payload))

    print(
        json.dumps(
            {"execution_ready_status_discussion_packet": repo_rel(PACKET_JSON)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
