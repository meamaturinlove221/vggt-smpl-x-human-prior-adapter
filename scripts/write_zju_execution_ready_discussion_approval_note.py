import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

BASELINE_DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
BASELINE_VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_BOUNDARY_JSON = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_SMOKE_SPEC_JSON = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
NONTRAINING_INTEGRATION_CHECK_JSON = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CHECKLIST_JSON = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"

APPROVAL_NOTE_JSON = OUTPUT_ROOT / "execution_ready_discussion_approval_note.camera_focal_objective_isolation.20260331.json"
APPROVAL_NOTE_MD = OUTPUT_ROOT / "execution_ready_discussion_approval_note.camera_focal_objective_isolation.20260331.md"

NEXT_REQUIREMENT = "decide_keep_execution_prep_baseline_only_or_promote_to_execution_ready_discussion"


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
        "# Execution-Ready Discussion Approval Note: Camera Focal Objective Isolation (2026-03-31)",
        "",
        "## Why Now",
        "",
        f"- {payload['why_now_for_execution_ready_discussion']}",
        "",
        "## What Exactly Is Being Approved",
        "",
        f"- {payload['what_exactly_is_being_approved']}",
        "",
        "## Still Forbidden After Discussion Promotion",
        "",
    ]
    for item in payload["still_forbidden_after_discussion_promotion"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## First Allowed Step If Discussion Is Approved",
            "",
            f"- {payload['first_allowed_step_if_discussion_is_approved']}",
            "",
            "## Allowed Buttons",
            "",
        ]
    )
    for item in payload["allowed_buttons"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            f"- recommended_decision: `{payload['recommended_decision']}`",
            f"- recommended_reason: {payload['recommended_reason']}",
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
    baseline_validation = load_json(BASELINE_VALIDATION_JSON)
    boundary = load_json(EXECUTION_READY_BOUNDARY_JSON)
    gate = load_json(EXECUTION_READY_GATE_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_discussion_approval_note",
        "family": "camera_focal_objective_isolation",
        "why_now_for_execution_ready_discussion": (
            "The training/loss.py baseline patch is already validated, the stronger local validation is complete, "
            "and the smoke spec, non-training integration check, and boundary checklist are now all packaged, "
            "so the remaining step is a manual decision on whether this packet may enter execution-ready discussion."
        ),
        "what_exactly_is_being_approved": (
            "This is not arm approval, not run approval, not training approval, and not cloud approval. "
            "It only approves entering execution-ready discussion for the existing validated training/loss.py packet."
        ),
        "still_forbidden_after_discussion_promotion": [
            "Do not materialize config files.",
            "Do not arm any approved problem.",
            "Do not run candidate or training flows.",
            "Do not use cloud or Modal.",
            "Do not widen the surface to trainer, dataset, or runner paths.",
        ],
        "first_allowed_step_if_discussion_is_approved": (
            "Enter execution-ready preparation design by writing the bounded local-only design for the validated training/loss.py patch, while still not executing anything."
        ),
        "allowed_buttons": [
            "KEEP_EXECUTION_PREP_BASELINE_ONLY",
            "PROMOTE_TO_EXECUTION_READY_DISCUSSION",
        ],
        "recommended_decision": "PROMOTE_TO_EXECUTION_READY_DISCUSSION",
        "recommended_reason": (
            "The validated baseline and the smoke/integration/checklist packet are already complete enough for a bounded manual discussion step, "
            "and discussion promotion still leaves ready_for_execution=false with all execution paths disabled."
        ),
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "supporting_refs": {
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "execution_prep_baseline_validation": repo_rel(BASELINE_VALIDATION_JSON),
            "execution_ready_boundary": repo_rel(EXECUTION_READY_BOUNDARY_JSON),
            "execution_ready_gate": repo_rel(EXECUTION_READY_GATE_JSON),
            "execution_ready_smoke_spec": repo_rel(EXECUTION_READY_SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(NONTRAINING_INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
            "gate_recommended_decision": gate.get("recommended_decision", ""),
            "boundary_target_file": boundary.get("current_baseline_definition", {}).get("target_file", ""),
            "baseline_decision_value": baseline_decision.get("decision", ""),
            "validation_status": baseline_validation.get("overall_status", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(APPROVAL_NOTE_JSON, payload)
    write_text(APPROVAL_NOTE_MD, render_md(payload))

    print(
        json.dumps(
            {"execution_ready_discussion_approval_note": repo_rel(APPROVAL_NOTE_JSON)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
