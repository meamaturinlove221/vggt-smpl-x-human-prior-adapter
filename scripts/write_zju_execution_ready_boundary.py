import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

BASELINE_DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
BASELINE_VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
AUTHORED_PATCH_REVIEW_JSON = OUTPUT_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.json"
HYGIENE_REVIEW_JSON = OUTPUT_ROOT / "hygiene_review.camera_focal_objective_isolation.20260331.json"
PATCH_BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"

BOUNDARY_JSON = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
BOUNDARY_MD = OUTPUT_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.md"
GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
GATE_MD = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.md"
SMOKE_SPEC_JSON = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
SMOKE_SPEC_MD = OUTPUT_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.md"
INTEGRATION_CHECK_JSON = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
INTEGRATION_CHECK_MD = OUTPUT_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.md"
BOUNDARY_CHECKLIST_JSON = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CHECKLIST_MD = OUTPUT_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.md"

FAMILY = "camera_focal_objective_isolation"
TARGET_FILE = "training/loss.py"
PATCH_TARGET_OBJECT = "loss_FL isolation inside compute_camera_loss"
BOUNDARY_TARGET_OBJECT = "loss_FL inside compute_camera_loss / camera loss aggregation"
NEXT_REQUIREMENT = "decide_keep_execution_prep_baseline_only_or_promote_to_execution_ready_discussion"
CURRENT_BASELINE_STATE = "execution_prep_baseline"
GATE_STATE = "execution_prep_baseline_manual_review_only"
KEEP_BASELINE_DECISION = "KEEP_EXECUTION_PREP_BASELINE_ONLY"
PROMOTE_DISCUSSION_DECISION = "PROMOTE_TO_EXECUTION_READY_DISCUSSION"
ALLOWED_BUTTONS = [
    KEEP_BASELINE_DECISION,
    PROMOTE_DISCUSSION_DECISION,
]
RECOMMENDED_DECISION = PROMOTE_DISCUSSION_DECISION
RECOMMENDED_REASON = (
    "The execution-prep baseline is validated and the smoke, integration, and "
    "boundary-checklist materials are now complete enough to enter execution-ready "
    "discussion while still keeping ready_for_execution=false and all execution "
    "paths disabled."
)
FIRST_ALLOWED_STEP_AFTER_EXECUTION_READY = (
    "Only design a minimal smoke/integration check plan that would verify the "
    "authored training/loss.py patch can safely connect to a future candidate "
    "config, while still not executing training."
)


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


def render_boundary_md(payload: dict) -> str:
    lines = [
        "# Execution-Ready Boundary: Camera Focal Objective Isolation (2026-03-31)",
        "",
        "## Current Baseline Definition",
        "",
    ]
    for key, value in payload["current_baseline_definition"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Minimal Entry Conditions for Execution-Ready", ""])
    for item in payload["minimal_entry_conditions_for_execution_ready"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- condition: `{item['condition']}`",
                f"- status: `{item['status']}`",
                f"- note: {item['note']}",
                "",
            ]
        )
    lines.extend(["## Remaining Blockers Before Execution-Ready", ""])
    for item in payload["remaining_blockers_before_execution_ready"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Still Forbidden After Execution-Ready", ""])
    for item in payload["still_forbidden_after_execution_ready"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## First Allowed Step After Execution-Ready",
            "",
            f"- {payload['first_allowed_step_after_execution_ready']}",
            "",
            "## Supporting Refs",
            "",
        ]
    )
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def render_smoke_spec_md(payload: dict) -> str:
    lines = [
        "# Execution-Ready Smoke Spec: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- target_file: `{payload['target_file']}`",
        f"- target_object: `{payload['target_object']}`",
        f"- scope: `{payload['scope']}`",
        "",
        "## Smoke Items",
        "",
    ]
    for item in payload["smoke_items"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- name: `{item['name']}`",
                f"- purpose: {item['purpose']}",
                f"- local_only_check: {item['local_only_check']}",
                "",
            ]
        )
    lines.extend(["## Pass Criteria", ""])
    for item in payload["pass_criteria"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Forbidden Actions", ""])
    for item in payload["forbidden_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def render_integration_md(payload: dict) -> str:
    lines = [
        "# Non-Training Integration Check: Camera Focal Objective Isolation (2026-03-31)",
        "",
        "## Future Config Touchpoints to Watch",
        "",
    ]
    for item in payload["future_config_touchpoints_to_watch"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Contract Break Risks", ""])
    for item in payload["contract_break_risks"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Checks That Remain Local Only", ""])
    for item in payload["checks_that_remain_local_only"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Why This Is Not Execution", "", f"- {payload['why_this_is_not_execution']}", "", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def render_boundary_checklist_md(payload: dict) -> str:
    lines = [
        "# Boundary Confirmation Checklist: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        "",
        "## Checklist",
        "",
    ]
    for item in payload["checks"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- label: `{item['label']}`",
                f"- status: `{item['status']}`",
                f"- note: {item['note']}",
                "",
            ]
        )
    lines.extend(["## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def render_gate_md(payload: dict) -> str:
    lines = [
        "# Execution-Ready Gate: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        f"- recommended_decision: `{payload['recommended_decision']}`",
        f"- recommended_reason: {payload['recommended_reason']}",
        "",
        "## Allowed Buttons",
        "",
    ]
    for item in payload["allowed_buttons"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Checks", ""])
    for check in payload["checks"]:
        lines.extend(
            [
                f"### {check['id']}",
                "",
                f"- label: `{check['label']}`",
                f"- status: `{check['status']}`",
                f"- note: {check['note']}",
                "",
            ]
        )
    lines.extend(["## Supporting Refs", ""])
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    baseline_decision = load_json(BASELINE_DECISION_JSON)
    baseline_validation = load_json(BASELINE_VALIDATION_JSON)
    authored_patch_review = load_json(AUTHORED_PATCH_REVIEW_JSON)
    hygiene_review = load_json(HYGIENE_REVIEW_JSON)
    patch_boundary = load_json(PATCH_BOUNDARY_JSON)
    checked_at = now_iso()

    smoke_spec_payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_ready_smoke_spec",
        "family": FAMILY,
        "target_file": TARGET_FILE,
        "target_object": PATCH_TARGET_OBJECT,
        "scope": "local_only_nontraining",
        "smoke_items": [
            {
                "id": "import_smoke",
                "name": "import smoke",
                "purpose": "Confirm the authored training/loss.py patch still imports in a local environment without invoking training.",
                "local_only_check": "Import the module and resolve compute_camera_loss without launching trainer or runner paths.",
            },
            {
                "id": "signature_smoke",
                "name": "signature smoke",
                "purpose": "Confirm compute_camera_loss still exposes the intended loss_fl_isolation_scale hook and accepts the current call surface.",
                "local_only_check": "Inspect the callable signature and verify the dormant hook remains present with identity default.",
            },
            {
                "id": "default_identity_equivalence_smoke",
                "name": "default identity equivalence smoke",
                "purpose": "Confirm default loss_camera semantics remain equivalent when loss_fl_isolation_scale stays at identity.",
                "local_only_check": "Compare the default formula path against loss_T/loss_R/loss_FL weighted assembly without training.",
            },
            {
                "id": "return_contract_smoke",
                "name": "return-contract smoke",
                "purpose": "Confirm the public return keys remain stable for downstream callers.",
                "local_only_check": "Check that loss_camera, loss_T, loss_R, and loss_FL all remain present and finite in local-only calls.",
            },
            {
                "id": "manifest_train_only_gate_smoke",
                "name": "manifest/train-only gate smoke",
                "purpose": "Confirm manifest-related and train-only gating paths remain callable and locally verifiable without trainer execution.",
                "local_only_check": "Exercise the manifest/train-only branches in a local synthetic batch and verify the val path stays gated.",
            },
        ],
        "pass_criteria": [
            "All five smoke items can be described and evaluated without materializing config files.",
            "The checks remain local-only and do not call trainer, candidate runner, or cloud paths.",
            "The authored training/loss.py patch preserves default semantics and return-contract expectations across the smoke surface.",
        ],
        "forbidden_actions": [
            "Do not materialize candidate yaml/config.",
            "Do not arm any approved problem.",
            "Do not run candidate/training flows.",
            "Do not use cloud or Modal.",
        ],
        "next_requirement": NEXT_REQUIREMENT,
    }

    integration_check_payload = {
        "checked_at": checked_at,
        "artifact_kind": "nontraining_integration_check",
        "family": FAMILY,
        "future_config_touchpoints_to_watch": [
            "Any future candidate config must continue to supply camera-loss weights without assuming new return keys.",
            "Any future config-facing hook must preserve the loss_fl_isolation_scale identity default unless an explicit later manual decision changes it.",
            "Any future config should observe whether loss_camera aggregation semantics remain consistent with current logging and comparison expectations.",
        ],
        "contract_break_risks": [
            "A future config could incorrectly assume a new external loss key instead of the existing loss_camera/loss_T/loss_R/loss_FL contract.",
            "A future config could widen the change surface beyond training/loss.py by requiring trainer-side plumbing too early.",
            "A future config could accidentally reinterpret manifest/train-only gating in a way that changes the current local validation contract.",
        ],
        "checks_that_remain_local_only": [
            "Do not generate yaml; only document which config touchpoints would need observation later.",
            "Do not invoke trainer; reason only about contract compatibility at the loss interface.",
            "Do not invoke candidate runner; keep the check at the level of future config-touchpoint review.",
        ],
        "why_this_is_not_execution": (
            "This package only describes what a future config would need to watch so the current loss contract stays stable. "
            "It does not create configs, does not launch trainer or candidate runner code, and does not authorize execution."
        ),
        "next_requirement": NEXT_REQUIREMENT,
    }

    boundary_checklist_payload = {
        "checked_at": checked_at,
        "artifact_kind": "boundary_confirmation_checklist",
        "family": FAMILY,
        "state": "manual_review_only",
        "checks": [
            {
                "id": "single_file_boundary",
                "label": "still limited to training/loss.py only",
                "status": "confirmed",
                "note": "The authored patch, baseline decision, and smoke spec all stay inside training/loss.py.",
            },
            {
                "id": "no_trainer_dataset_config_prereq",
                "label": "still does not require trainer/dataset/config pre-change",
                "status": "confirmed",
                "note": "The current correctness argument remains local to the loss contract and non-training checks.",
            },
            {
                "id": "no_tail_family_reopen",
                "label": "still does not reopen tail families",
                "status": "confirmed",
                "note": "The boundary stays on objective isolation and does not reactivate tail-contract derivative families.",
            },
            {
                "id": "no_cloud",
                "label": "still keeps cloud disabled",
                "status": "confirmed",
                "note": "All newly defined checks remain local-only and cloud_must_remain_off stays true.",
            },
            {
                "id": "not_execution_ready_yet",
                "label": "still not execution-ready",
                "status": "confirmed",
                "note": "The new packet only defines the remaining smoke/integration materials for review and does not flip ready_for_execution.",
            },
        ],
        "next_requirement": NEXT_REQUIREMENT,
    }

    boundary_payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_ready_boundary",
        "family": FAMILY,
        "current_baseline_definition": {
            "target_file": TARGET_FILE,
            "target_object": BOUNDARY_TARGET_OBJECT,
            "current_state": CURRENT_BASELINE_STATE,
            "ready_for_execution": False,
            "baseline_decision": baseline_decision.get("decision", ""),
        },
        "minimal_entry_conditions_for_execution_ready": [
            {
                "id": "loss_py_single_file_boundary_still_holds",
                "condition": "training/loss.py single-file boundary remains sufficient",
                "status": "required",
                "note": "Do not widen beyond the current loss.py-only boundary or reopen older families.",
            },
            {
                "id": "default_numeric_semantics_remain_equivalent",
                "condition": "default numeric semantics remain equivalent",
                "status": "required",
                "note": "The default formula path must stay identical to the pre-isolation behavior.",
            },
            {
                "id": "return_contract_remains_unchanged",
                "condition": "loss_camera/loss_T/loss_R/loss_FL return contract remains unchanged",
                "status": "required",
                "note": "No trainer-side or caller-side contract break may be introduced.",
            },
            {
                "id": "stronger_local_validation_passes",
                "condition": "stronger local validation passes in full",
                "status": "satisfied",
                "note": f"Current baseline validation result is {baseline_validation.get('overall_status', '')}.",
            },
            {
                "id": "no_trainer_dataset_config_prereq_needed_for_patch_correctness",
                "condition": "trainer/dataset/config changes are not required to verify patch correctness",
                "status": "required",
                "note": "Execution-ready discussion should begin only if patch correctness can still be argued from the current isolated surface.",
            },
            {
                "id": "minimal_execution_ready_smoke_plan_defined_locally",
                "condition": "a minimal execution-ready smoke plan is defined and remains local-only",
                "status": "satisfied",
                "note": "The smoke spec is now written and packaged as a local-only manual-review artifact.",
            },
        ],
        "remaining_blockers_before_execution_ready": [
            "A final manual decision is still required to keep the execution-prep baseline or promote this packet into execution-ready discussion.",
            "No execution-ready discussion promotion may authorize config materialization, training, arm/run, or cloud use.",
        ],
        "still_forbidden_after_execution_ready": [
            "Execution-ready does not equal arm approval.",
            "Execution-ready does not equal run approval.",
            "Execution-ready does not equal training approval.",
            "Execution-ready does not equal cloud approval.",
        ],
        "first_allowed_step_after_execution_ready": FIRST_ALLOWED_STEP_AFTER_EXECUTION_READY,
        "supporting_refs": {
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "execution_prep_baseline_validation": repo_rel(BASELINE_VALIDATION_JSON),
            "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_JSON),
            "hygiene_review": repo_rel(HYGIENE_REVIEW_JSON),
            "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
            "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
            "baseline_decision_value": baseline_decision.get("decision", ""),
            "validation_status": baseline_validation.get("overall_status", ""),
            "default_semantics_preserved": authored_patch_review.get("default_semantics_preserved"),
            "return_contract_preserved": authored_patch_review.get("return_contract_preserved"),
            "patch_boundary_target_file": patch_boundary.get("target_file", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    gate_payload = {
        "checked_at": boundary_payload["checked_at"],
        "artifact_kind": "execution_ready_gate",
        "family": FAMILY,
        "state": GATE_STATE,
        "allowed_buttons": ALLOWED_BUTTONS,
        "checks": [
            {
                "id": "execution_prep_baseline_exists",
                "label": "validated execution-prep baseline exists",
                "status": "supported",
                "note": f"Baseline decision is {baseline_decision.get('decision', '')} with ready_for_execution still false.",
            },
            {
                "id": "loss_py_only_boundary_remains_intact",
                "label": "loss.py-only boundary remains intact",
                "status": "supported",
                "note": "The current reviewed patch boundary is still limited to training/loss.py and loss_FL isolation.",
            },
            {
                "id": "stronger_local_validation_complete",
                "label": "stronger local validation is complete",
                "status": "supported",
                "note": f"Validation currently reports {baseline_validation.get('overall_status', '')}.",
            },
            {
                "id": "execution_ready_review_packet_pending_manual_review",
                "label": "execution-ready smoke/integration packet is fully defined for manual decision",
                "status": "supported",
                "note": "The smoke spec, non-training integration check, and boundary checklist are now all packaged, so the next step is an explicit keep-vs-promote manual decision.",
            },
            {
                "id": "manual_review_governance_still_active",
                "label": "manual review governance remains active",
                "status": "required",
                "note": "Even a later execution-ready discussion would still not authorize arm/run/training/cloud.",
            },
        ],
        "recommended_decision": RECOMMENDED_DECISION,
        "recommended_reason": RECOMMENDED_REASON,
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "supporting_refs": {
            "execution_ready_boundary": repo_rel(BOUNDARY_JSON),
            "execution_prep_baseline_decision": repo_rel(BASELINE_DECISION_JSON),
            "execution_prep_baseline_validation": repo_rel(BASELINE_VALIDATION_JSON),
            "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_JSON),
            "hygiene_review": repo_rel(HYGIENE_REVIEW_JSON),
            "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
            "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
            "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(SMOKE_SPEC_JSON, smoke_spec_payload)
    write_text(SMOKE_SPEC_MD, render_smoke_spec_md(smoke_spec_payload))
    write_json(INTEGRATION_CHECK_JSON, integration_check_payload)
    write_text(INTEGRATION_CHECK_MD, render_integration_md(integration_check_payload))
    write_json(BOUNDARY_CHECKLIST_JSON, boundary_checklist_payload)
    write_text(BOUNDARY_CHECKLIST_MD, render_boundary_checklist_md(boundary_checklist_payload))
    write_json(BOUNDARY_JSON, boundary_payload)
    write_text(BOUNDARY_MD, render_boundary_md(boundary_payload))
    write_json(GATE_JSON, gate_payload)
    write_text(GATE_MD, render_gate_md(gate_payload))

    print(
        json.dumps(
            {
                "execution_ready_smoke_spec": repo_rel(SMOKE_SPEC_JSON),
                "nontraining_integration_check": repo_rel(INTEGRATION_CHECK_JSON),
                "boundary_confirmation_checklist": repo_rel(BOUNDARY_CHECKLIST_JSON),
                "execution_ready_boundary": repo_rel(BOUNDARY_JSON),
                "execution_ready_gate": repo_rel(GATE_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
