import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

BASELINE_DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
MINIMAL_SURFACE_JSON = OUTPUT_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.json"
DESIGN_CONTRACT_JSON = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
PATCH_BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
IMPLEMENTATION_SKETCH_JSON = OUTPUT_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.json"
SMOKE_SPEC_JSON = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
INTEGRATION_CHECK_JSON = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CHECKLIST_JSON = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"
BOUNDARY_JSON = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
DISCUSSION_APPROVAL_NOTE_JSON = OUTPUT_ROOT / "execution_ready_discussion_approval_note.camera_focal_objective_isolation.20260331.json"
DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"

PLAN_JSON = OUTPUT_ROOT / "execution_ready_prep_plan.camera_focal_objective_isolation.20260331.json"
PLAN_MD = OUTPUT_ROOT / "execution_ready_prep_plan.camera_focal_objective_isolation.20260331.md"

FAMILY = "camera_focal_objective_isolation"
TARGET_FILE = "training/loss.py"
TARGET_OBJECT = "loss_FL inside compute_camera_loss / camera loss aggregation"
NEXT_REQUIREMENT = "design_execution_ready_preparation_design_for_validated_loss_py_patch"


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
        "# Execution-Ready Prep Plan: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- manual_action_kind: `{payload['manual_action_kind']}`",
        f"- target_file: `{payload['target_file']}`",
        f"- target_object: `{payload['target_object']}`",
        "",
        "## Plan Objective",
        "",
        f"- {payload['plan_objective']}",
        "",
        "## Baseline Anchor",
        "",
    ]
    for key, value in payload["baseline_anchor"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Workstreams", ""])
    for item in payload["workstreams"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- title: `{item['title']}`",
                f"- status: `{item['status']}`",
                f"- goal: {item['goal']}",
                f"- deliverable: {item['deliverable']}",
                f"- completion_rule: {item['completion_rule']}",
                "",
            ]
        )
        lines.append("Supporting Inputs:")
        for ref in item["supporting_inputs"]:
            lines.append(f"- `{ref}`")
        lines.append("")
    lines.extend(["## Freeze Guards", ""])
    for item in payload["freeze_guards"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Completion Definition", ""])
    for item in payload["completion_definition"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Review Packet Components", ""])
    for key, value in payload["review_packet_components"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Supporting Refs", ""])
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    baseline_decision = load_json(BASELINE_DECISION_JSON)
    minimal_surface = load_json(MINIMAL_SURFACE_JSON)
    design_contract = load_json(DESIGN_CONTRACT_JSON)
    patch_boundary = load_json(PATCH_BOUNDARY_JSON)
    implementation_sketch = load_json(IMPLEMENTATION_SKETCH_JSON)
    smoke_spec = load_json(SMOKE_SPEC_JSON)
    integration_check = load_json(INTEGRATION_CHECK_JSON)
    boundary_checklist = load_json(BOUNDARY_CHECKLIST_JSON)
    boundary = load_json(BOUNDARY_JSON)
    gate = load_json(GATE_JSON)
    discussion_approval = load_json(DISCUSSION_APPROVAL_NOTE_JSON)
    discussion_decision = load_json(DISCUSSION_DECISION_JSON)

    payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_ready_prep_plan",
        "family": FAMILY,
        "state": "manual_review_execution_ready_prep_plan_only",
        "manual_action_kind": "manual_review",
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "target_file": TARGET_FILE,
        "target_object": TARGET_OBJECT,
        "plan_objective": (
            "Turn the discussion-promoted, validated training/loss.py-only baseline into a "
            "reviewable execution-ready preparation packet that stays entirely local-only, "
            "preserves the single-file boundary, and still does not authorize execution."
        ),
        "baseline_anchor": {
            "baseline_decision": baseline_decision.get("decision", ""),
            "discussion_decision": discussion_decision.get("decision", ""),
            "recommended_gate_decision": gate.get("recommended_decision", ""),
            "recommended_discussion_decision": discussion_approval.get("recommended_decision", ""),
            "minimal_write_surface": minimal_surface.get("recommended_minimal_write_surface", ""),
            "allowed_write_surface": design_contract.get("allowed_write_surface", []),
            "target_file": patch_boundary.get("target_file", TARGET_FILE),
            "target_object": patch_boundary.get("target_object", TARGET_OBJECT),
        },
        "workstreams": [
            {
                "id": "baseline_lock",
                "title": "lock the validated loss.py-only baseline and boundary",
                "status": "packaged",
                "goal": (
                    "Keep the execution-prep starting point fixed to the current validated "
                    "training/loss.py baseline and prevent any widening to trainer, dataset, "
                    "config, runner, or cloud surfaces."
                ),
                "supporting_inputs": [
                    repo_rel(BASELINE_DECISION_JSON),
                    repo_rel(DESIGN_CONTRACT_JSON),
                    repo_rel(PATCH_BOUNDARY_JSON),
                    repo_rel(BOUNDARY_JSON),
                ],
                "deliverable": "A frozen single-file starting packet for any later execution-ready review.",
                "completion_rule": (
                    "The prep plan must still point to training/loss.py only and must keep "
                    "ready_for_execution=false."
                ),
            },
            {
                "id": "local_smoke_packet",
                "title": "carry the local-only smoke packet forward",
                "status": "packaged",
                "goal": (
                    "Use the existing smoke spec as the non-executing verification surface for "
                    "the validated patch, without materializing yaml, starting trainer flows, or "
                    "opening any candidate."
                ),
                "supporting_inputs": [
                    repo_rel(SMOKE_SPEC_JSON),
                    repo_rel(IMPLEMENTATION_SKETCH_JSON),
                ],
                "deliverable": "A local-only smoke review packet tied to the validated patch boundary.",
                "completion_rule": (
                    "All smoke checks remain describable and reviewable without arm/run/training/cloud."
                ),
            },
            {
                "id": "integration_review_packet",
                "title": "package the future integration touchpoints without turning them into execution",
                "status": "packaged",
                "goal": (
                    "Translate the future config-touchpoint review into a bounded manual packet so "
                    "a later reviewer can see what would need watching without generating config files."
                ),
                "supporting_inputs": [
                    repo_rel(INTEGRATION_CHECK_JSON),
                    repo_rel(BOUNDARY_CHECKLIST_JSON),
                    repo_rel(BOUNDARY_JSON),
                ],
                "deliverable": "A non-training integration review packet that stays local-only.",
                "completion_rule": (
                    "The packet names future touchpoints but still forbids config materialization "
                    "and any trainer or candidate execution."
                ),
            },
            {
                "id": "manual_status_review_handoff",
                "title": "define the later human review handoff",
                "status": "packaged",
                "goal": (
                    "Collect the gate and discussion artifacts into one prep-plan handoff so the "
                    "next human review can decide what to do with the plan without re-opening old families."
                ),
                "supporting_inputs": [
                    repo_rel(GATE_JSON),
                    repo_rel(DISCUSSION_APPROVAL_NOTE_JSON),
                    repo_rel(DISCUSSION_DECISION_JSON),
                ],
                "deliverable": "A reviewable execution-ready prep packet with clear governance hold points.",
                "completion_rule": (
                    "The handoff still says ready_for_execution=false and keeps arm/run/cloud disabled."
                ),
            },
        ],
        "freeze_guards": [
            "Do not mark ready_for_execution=true.",
            "Do not arm any approved problem.",
            "Do not run candidate or training flows.",
            "Do not materialize config or yaml.",
            "Do not widen the write surface beyond training/loss.py in this prep-plan round.",
            "Do not reopen tail-contract derivative families or two-stage cousins.",
            "Do not use cloud or Modal.",
        ],
        "completion_definition": [
            "The validated baseline, smoke spec, integration check, boundary checklist, boundary, gate, and discussion decision are all cross-referenced as one prep packet.",
            "The packet is explicit enough for later human review without requiring any new execution step.",
            "The route ends with a packaged manual-review prep plan and not with execution-ready status.",
        ],
        "review_packet_components": {
            "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
            "execution_ready_boundary": repo_rel(BOUNDARY_JSON),
            "execution_ready_gate": repo_rel(GATE_JSON),
            "execution_ready_discussion_approval_note": repo_rel(DISCUSSION_APPROVAL_NOTE_JSON),
            "execution_ready_discussion_decision": repo_rel(DISCUSSION_DECISION_JSON),
        },
        "supporting_refs": {
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "minimal_write_surface_recommendation": repo_rel(MINIMAL_SURFACE_JSON),
            "design_contract": repo_rel(DESIGN_CONTRACT_JSON),
            "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
            "execution_prep_implementation_sketch": repo_rel(IMPLEMENTATION_SKETCH_JSON),
            "discussion_first_allowed_next_step": discussion_decision.get("first_allowed_next_step", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(PLAN_JSON, payload)
    write_text(PLAN_MD, render_md(payload))

    print(json.dumps({"execution_ready_prep_plan": repo_rel(PLAN_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
