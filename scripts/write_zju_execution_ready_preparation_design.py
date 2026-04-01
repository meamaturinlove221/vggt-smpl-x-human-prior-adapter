import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

BASELINE_DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
BOUNDARY_JSON = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
SMOKE_SPEC_JSON = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
INTEGRATION_CHECK_JSON = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CHECKLIST_JSON = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"
DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"
PREP_PLAN_JSON = OUTPUT_ROOT / "execution_ready_prep_plan.camera_focal_objective_isolation.20260331.json"

DESIGN_JSON = OUTPUT_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.json"
DESIGN_MD = OUTPUT_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.md"

FAMILY = "camera_focal_objective_isolation"
TARGET_FILE = "training/loss.py"
TARGET_OBJECT = "loss_FL inside compute_camera_loss / camera loss aggregation"
NEXT_REQUIREMENT = "manual_review_execution_ready_preparation_design_before_any_execution_ready_status_decision"


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
        "# Execution-Ready Preparation Design: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- manual_action_kind: `{payload['manual_action_kind']}`",
        f"- target_file: `{payload['target_file']}`",
        f"- target_object: `{payload['target_object']}`",
        "",
        "## Minimal Local Preparations",
        "",
    ]
    for item in payload["minimal_local_preparations_if_execution_ready_is_considered"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Local-Only / Non-Training Checks", ""])
    for item in payload["local_only_nontraining_checks_that_remain"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Still Forbidden", ""])
    for item in payload["still_explicitly_forbidden"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## First Allowed Step After Discussion Promotion",
            "",
            f"- {payload['first_allowed_step_after_execution_ready_discussion_promotion']}",
            "",
            "## Supporting Refs",
            "",
        ]
    )
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    baseline_decision = load_json(BASELINE_DECISION_JSON)
    boundary = load_json(BOUNDARY_JSON)
    gate = load_json(GATE_JSON)
    smoke_spec = load_json(SMOKE_SPEC_JSON)
    integration_check = load_json(INTEGRATION_CHECK_JSON)
    boundary_checklist = load_json(BOUNDARY_CHECKLIST_JSON)
    discussion_decision = load_json(DISCUSSION_DECISION_JSON)
    prep_plan = load_json(PREP_PLAN_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_preparation_design",
        "family": FAMILY,
        "state": "manual_review_execution_ready_preparation_design_only",
        "manual_action_kind": "manual_review",
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "target_file": TARGET_FILE,
        "target_object": TARGET_OBJECT,
        "minimal_local_preparations_if_execution_ready_is_considered": [
            "Keep the target surface locked to training/loss.py only and preserve the current loss_camera/loss_T/loss_R/loss_FL return contract.",
            "Carry forward the existing local-only smoke packet so default identity semantics and signature safety can still be checked without training.",
            "Carry forward the non-training integration review so future config touchpoints stay documented without generating yaml or invoking trainer paths.",
            "Keep the boundary checklist active so no trainer, dataset, config, runner, or cloud surface gets widened during this design stage.",
        ],
        "local_only_nontraining_checks_that_remain": [
            "Import smoke for training/loss.py and compute_camera_loss only.",
            "Signature/default-identity review for the dormant loss_FL isolation hook.",
            "Return-contract review for loss_camera, loss_T, loss_R, and loss_FL only.",
            "Future config-touchpoint review remains documentation-only and does not materialize yaml.",
            "Boundary confirmation remains a manual artifact review and not a candidate or trainer run.",
        ],
        "still_explicitly_forbidden": [
            "Do not edit training/trainer.py, training/data/*, training/config/*, runner scripts, or cloud paths.",
            "Do not materialize candidate yaml or config files.",
            "Do not arm any approved problem.",
            "Do not run candidate, smoke training, or full training flows.",
            "Do not use cloud or Modal.",
            "Do not mark ready_for_execution=true in this design round.",
        ],
        "first_allowed_step_after_execution_ready_discussion_promotion": (
            "Design a minimal execution-ready prep check plan for the validated training/loss.py patch. "
            "That step remains local-only and non-training; it is not a candidate run, smoke run, training run, or cloud action."
        ),
        "supporting_refs": {
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "execution_ready_boundary": repo_rel(BOUNDARY_JSON),
            "execution_ready_gate": repo_rel(GATE_JSON),
            "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
            "execution_ready_discussion_decision": repo_rel(DISCUSSION_DECISION_JSON),
            "execution_ready_prep_plan": repo_rel(PREP_PLAN_JSON),
            "baseline_decision_value": baseline_decision.get("decision", ""),
            "gate_recommended_decision": gate.get("recommended_decision", ""),
            "discussion_decision_value": discussion_decision.get("decision", ""),
            "boundary_target_file": boundary.get("current_baseline_definition", {}).get("target_file", ""),
            "smoke_scope": smoke_spec.get("scope", ""),
            "integration_why_not_execution": integration_check.get("why_this_is_not_execution", ""),
            "boundary_state": boundary_checklist.get("state", ""),
            "prep_plan_objective": prep_plan.get("plan_objective", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(DESIGN_JSON, payload)
    write_text(DESIGN_MD, render_md(payload))

    print(
        json.dumps(
            {"execution_ready_preparation_design": repo_rel(DESIGN_JSON)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
