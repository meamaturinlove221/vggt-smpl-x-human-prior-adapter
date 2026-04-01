import json
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"
GUARD_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
RESEARCH_STATUS_MD = RESEARCH_ROOT / "research_loop_status.md"
CANDIDATE_PATCH_PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.json"
CANDIDATE_PATCH_PLAN_MD = RESEARCH_ROOT / "candidate_patch_plan.md"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
TASK_PLAN_JSON = RAWPOOL_STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = RAWPOOL_STATUS_ROOT / "task_plan.md"
SUMMARY_MD = RAWPOOL_STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
LATEST_GUARD_JSON = GUARD_ROOT / "latest_guard_snapshot.json"
CANDIDATE_VERDICT_JSON = RESEARCH_ROOT / "candidate_verdict.json"
AXIS_CLOSED_JSON = RESEARCH_ROOT / "two_stage_axis_closed.20260330.json"
LOCALIZATION_JSON = RESEARCH_ROOT / "early_fl_tax_localization.20260330.json"
OBJECT_MATRIX_JSON = RESEARCH_ROOT / "fl_tax_object_alignment_matrix.20260330.json"
FAILURE_JSON = RESEARCH_ROOT / "two_stage_failure_interpretation.20260330.json"
SEED_JSON = RESEARCH_ROOT / "approved_problem.seed.camera_focal_objective_isolation.json"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
FAMILY_PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"
FAMILY_PLAN_MD = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.md"
DRAFT_JSON = RESEARCH_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.json"
REVIEW_NOTE_JSON = RESEARCH_ROOT / "review_note.camera_focal_objective_isolation.20260330.json"
CHECKLIST_JSON = RESEARCH_ROOT / "decision_checklist.camera_focal_objective_isolation.20260330.json"
DESIGN_SKETCH_JSON = RESEARCH_ROOT / "execution_prep_design_sketch.camera_focal_objective_isolation.20260330.json"
SURFACE_CANDIDATES_JSON = RESEARCH_ROOT / "minimal_write_surface_candidates.camera_focal_objective_isolation.20260330.json"
SURFACE_RECOMMEND_JSON = RESEARCH_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.json"
DESIGN_CONTRACT_JSON = RESEARCH_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
PATCH_BOUNDARY_JSON = RESEARCH_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
IMPLEMENTATION_SKETCH_JSON = RESEARCH_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.json"
PSEUDODIFF_MAP_JSON = RESEARCH_ROOT / "pseudodiff_map.camera_focal_objective_isolation.20260331.json"
PATCH_AUTHORING_GATE_JSON = RESEARCH_ROOT / "patch_authoring_gate.camera_focal_objective_isolation.20260331.json"
PATCH_AUTHORING_APPROVAL_NOTE_JSON = RESEARCH_ROOT / "patch_authoring_approval_note.camera_focal_objective_isolation.20260331.json"
AUTHORED_PATCH_REVIEW_NOTE_JSON = RESEARCH_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.json"
HYGIENE_REVIEW_JSON = RESEARCH_ROOT / "hygiene_review.camera_focal_objective_isolation.20260331.json"
EXECUTION_PREP_PROMOTION_NOTE_JSON = RESEARCH_ROOT / "execution_prep_promotion_note.camera_focal_objective_isolation.20260331.json"
EXECUTION_PREP_BASELINE_VALIDATION_JSON = RESEARCH_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
EXECUTION_PREP_BASELINE_DECISION_JSON = RESEARCH_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_BOUNDARY_JSON = RESEARCH_ROOT / "execution_ready_boundary.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_GATE_JSON = RESEARCH_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_SMOKE_SPEC_JSON = RESEARCH_ROOT / "execution_ready_smoke_spec.camera_focal_objective_isolation.20260331.json"
NONTRAINING_INTEGRATION_CHECK_JSON = RESEARCH_ROOT / "nontraining_integration_check.camera_focal_objective_isolation.20260331.json"
BOUNDARY_CONFIRMATION_CHECKLIST_JSON = RESEARCH_ROOT / "boundary_confirmation_checklist.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_DISCUSSION_APPROVAL_NOTE_JSON = RESEARCH_ROOT / "execution_ready_discussion_approval_note.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_DISCUSSION_DECISION_JSON = RESEARCH_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_PREP_PLAN_JSON = RESEARCH_ROOT / "execution_ready_prep_plan.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_PREPARATION_DESIGN_JSON = RESEARCH_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.json"
EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON = RESEARCH_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"
EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON = RESEARCH_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.json"
STATUS_DISCUSSION_GUARD_LATEST_REPORT_JSON = RESEARCH_ROOT / "status_discussion_guard_only" / "latest_report.json"

PREFERRED_FAMILY = "camera_focal_objective_isolation"
PREFERRED_SHAPE = "fl_only_camera_objective_isolation_audit"
DISCUSSION_PROMOTION_DECISION = "PROMOTE_TO_EXECUTION_READY_DISCUSSION"
STATUS_DISCUSSION_PROMOTION_DECISION = "PROMOTE_TO_EXECUTION_READY_STATUS_DISCUSSION"
EXECUTION_READY_STATUS_DISCUSSION_STATUS = "execution_ready_status_discussion_promoted_not_execution_ready"
FAMILY_STATUS_DISCUSSION_PLAN_STATE = "manual_review_execution_ready_status_discussion_only"
TASK_MODE_STATUS = "completed"
TASK_MODE_FOCUS = "execution_ready_status_discussion_promoted_cloud_off"
MANUAL_REVIEW_REASON = (
    "The two-stage dead_same_day verdict closes automatic execution for today. "
    "camera_focal_objective_isolation remains the preferred next manual-review family, "
    "the validated loss.py-only patch is approved as an execution-prep baseline, "
    "the packet is promoted into execution-ready discussion, the execution-ready preparation design is accepted as the manual-review baseline, "
    "the family is now promoted into execution-ready status discussion only, and the manual status-discussion packet is packaged. "
    "The next bounded step is to hold that status discussion while ready_for_execution=false "
    "still blocks any arm/run/cloud action."
)
NEXT_REQUIREMENT = "manual_review_execution_ready_status_discussion_before_any_execution_ready_promotion"
RESEARCH_IDLE_REASON = (
    "No approved_problem.json is present; research remains idle while the "
    "discussion-promoted camera_focal_objective_isolation packet, the packaged manual status-discussion packet, "
    "and the execution-ready status discussion wait for manual review only."
)
CURRENT_PRIORITY_REASON = (
    "Manual review only: execution-ready status discussion is now active for "
    "camera_focal_objective_isolation, and the packaged status-discussion packet is now ready for the human review itself."
)
CURRENT_FRONTIER_HINT = (
    "Manual discussion only: camera_focal_objective_isolation is promoted into "
    "execution-ready status discussion, the status-discussion packet is ready, and execution is still disabled."
)
CURRENT_FRONTIER_PRIORITY = (
    "manual-only: keep research in IDLE_GUARD, keep cloud off, and conduct the "
    "manual execution-ready status discussion for the discussion-promoted loss.py-only "
    "execution-prep packet using the packaged status-discussion handoff."
)
WATCH_CONCLUSION = (
    "guard is green and research remains idle; camera_focal_objective_isolation is "
    "in execution-ready status discussion only, the manual status-discussion packet is ready, "
    "still not execution-ready, and still blocked from arm/run/cloud"
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


def repo_rel(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def upsert_by_key(items: list[dict], payload: dict, key: str) -> list[dict]:
    result = []
    inserted = False
    for item in items:
        if item.get(key) == payload.get(key):
            result.append(payload)
            inserted = True
        else:
            result.append(item)
    if not inserted:
        result.append(payload)
    return result


def list_modal_apps() -> list[dict]:
    result = subprocess.run(
        ["modal", "app", "list", "--json"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def render_research_status_md(payload: dict) -> str:
    lines = [
        "# ZJU Source Policy Research Loop Status",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- state: `{payload['state']}`",
        f"- reason: `{payload['reason']}`",
        f"- approved_problem_present: `{payload['approved_problem_present']}`",
        f"- preferred_first_family: `{payload['preferred_first_family']}`",
        f"- preferred_first_family_reason: `{payload['preferred_first_family_reason']}`",
        f"- current_priority_family: `{payload['current_priority_family']}`",
        f"- current_priority_reason: `{payload['current_priority_reason']}`",
        f"- same_family_retry_forbidden: `{payload['same_family_retry_forbidden']}`",
        f"- next_requirement: `{payload['next_requirement']}`",
        "",
        "## Manual Review Artifacts",
        "",
    ]
    for key, value in (payload.get("manual_review_artifacts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def render_candidate_patch_plan_md(payload: dict) -> str:
    lines = [
        "# ZJU Source Policy Candidate Patch Plan",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- state: `{payload['state']}`",
        f"- preferred_first_family: `{payload['preferred_first_family']}`",
        f"- preferred_first_family_reason: `{payload['preferred_first_family_reason']}`",
        f"- preferred_first_candidate_shape: `{payload['preferred_first_candidate_shape']}`",
        f"- preferred_first_candidate_config: `{payload['preferred_first_candidate_config']}`",
        f"- preferred_first_candidate_execution_note: `{payload['preferred_first_candidate_execution_note']}`",
        f"- same_family_retry_forbidden: `{payload['same_family_retry_forbidden']}`",
        f"- next_requirement: `{payload['next_requirement']}`",
        "",
        "## Manual Review Family",
        "",
        f"- family_blueprints: `{payload.get('family_blueprints', {})}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_family_candidate_patch_plan_md(payload: dict) -> str:
    lines = [
        "# Candidate Patch Plan: Camera Focal Objective Isolation",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- state: `{payload['state']}`",
        f"- current_review_stage: `{payload['current_review_stage']}`",
        f"- current_stable_lead_config: `{payload['current_stable_lead_config']}`",
        f"- first_candidate_shape: `{payload['first_candidate_shape']}`",
        f"- first_candidate_config: `{payload['first_candidate_config']}`",
        f"- status_discussion_execution_note: `{payload['status_discussion_execution_note']}`",
        f"- next_requirement: `{payload['next_requirement']}`",
        "",
        "## Readiness",
        "",
        f"- readiness: `{payload['readiness']}`",
        "",
        "## Review Write Surface",
        "",
    ]
    lines.extend([f"- `{item}`" for item in payload.get("write_surface", [])])
    lines.extend(["", "## Supporting Review Artifacts", ""])
    for item in payload.get("supporting_review_artifacts", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def render_task_plan_md(payload: dict) -> str:
    lines = [
        f"# ZJU Source-Policy Rawpool Task Plan ({payload['checked_at'][:10]})",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- task_mode_status: `{payload['task_mode_status']}`",
        f"- current_mode: `{payload['current_mode']}`",
        f"- research_loop_mode: `{payload['research_loop_mode']}`",
        f"- task_mode_focus: `{payload['task_mode_focus']}`",
        "",
        "## Current State",
        "",
        "- Two-stage objective decoupling is formally closed as dead_same_day.",
        "- Guard and research remain idle/clean with no active approval.",
        "- camera_focal_objective_isolation is now in execution-ready status discussion only; the manual status-discussion packet is ready, ready_for_execution remains false, and no ticket may be armed or run.",
        "",
        "## Current Local Lead",
        "",
        f"- family: `{payload['current_local_lead']['family']}`",
        f"- first_candidate_shape: `{payload['current_local_lead']['first_candidate_shape']}`",
        f"- config: `{payload['current_local_lead']['config']}`",
        "",
        "## Fastest Next Path",
        "",
    ]
    lines.extend([f"- {item}" for item in payload.get("fastest_next_path", [])])
    lines.append("")
    return "\n".join(lines)


def render_summary_md(payload: dict, guard: dict) -> str:
    lines = [
        f"# ZJU Source-Policy Rawpool Status ({payload['checked_at'][:10]})",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- current_status: `{payload['current_mode']}`",
        f"- research_loop_status: `{payload['research_loop_mode']}`",
        f"- current_lead_config: `{payload['current_local_lead']['config']}`",
        f"- consistency_ok: `{bool(guard.get('consistency_ok', True))}`",
        f"- cloud_gate: `{bool(guard.get('state_cloud_gate', False))}`",
        f"- launch_cloud_now: `{bool(guard.get('state_launch_cloud_now', False))}`",
        f"- active_modal_app_count: `{int(guard.get('active_modal_app_count', 0) or 0)}`",
        f"- repo_process_count: `{int(guard.get('repo_process_count', 0) or 0)}`",
        "",
        "## Current Conclusion",
        "",
        "- The two-stage objective-decoupling axis is closed as a formal dead_same_day failure.",
        "- Research remains in IDLE_GUARD with no active approval and an empty allowlist.",
        "- camera_focal_objective_isolation is now in execution-ready status discussion only, the manual status-discussion packet is ready, and execution remains disabled.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    research_status = load_json(RESEARCH_STATUS_JSON)
    candidate_patch_plan = load_json(CANDIDATE_PATCH_PLAN_JSON)
    frontier_ledger = load_json(FRONTIER_LEDGER_JSON)
    family_stop_reason = load_json(FAMILY_STOP_REASON_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)
    latest_guard = load_json(LATEST_GUARD_JSON)
    candidate_verdict = load_json(CANDIDATE_VERDICT_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    family_plan = load_json(FAMILY_PLAN_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    execution_ready_gate = load_json(EXECUTION_READY_GATE_JSON)
    execution_ready_discussion_decision = load_json(EXECUTION_READY_DISCUSSION_DECISION_JSON)
    execution_ready_prep_plan = load_json(EXECUTION_READY_PREP_PLAN_JSON)
    execution_ready_preparation_design = load_json(EXECUTION_READY_PREPARATION_DESIGN_JSON)
    execution_ready_status_discussion_decision = load_json(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON)
    execution_ready_status_discussion_packet = load_json(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON)

    active_modal_apps = [row for row in list_modal_apps() if str(row.get("State", "")).lower() != "stopped"]
    gate_recommended_decision = str(execution_ready_gate.get("recommended_decision", "")).strip()
    discussion_decision = str(execution_ready_discussion_decision.get("decision", "")).strip()
    status_discussion_decision = str(execution_ready_status_discussion_decision.get("decision", "")).strip()
    next_requirement = (
        str(execution_ready_status_discussion_decision.get("next_requirement", NEXT_REQUIREMENT)).strip() or NEXT_REQUIREMENT
    )

    if gate_recommended_decision != DISCUSSION_PROMOTION_DECISION:
        raise RuntimeError(
            "execution_ready_gate must recommend PROMOTE_TO_EXECUTION_READY_DISCUSSION before live sync can proceed."
        )
    if discussion_decision != DISCUSSION_PROMOTION_DECISION:
        raise RuntimeError(
            "execution_ready_discussion_decision must be PROMOTE_TO_EXECUTION_READY_DISCUSSION before live sync can proceed."
        )
    if bool(execution_ready_gate.get("ready_for_execution")) or bool(execution_ready_discussion_decision.get("ready_for_execution")):
        raise RuntimeError("execution-ready discussion promotion must keep ready_for_execution=false.")
    if str(execution_ready_prep_plan.get("artifact_kind", "")).strip() != "execution_ready_prep_plan":
        raise RuntimeError("execution_ready_prep_plan artifact must exist before final live sync can proceed.")
    if bool(execution_ready_prep_plan.get("ready_for_execution")):
        raise RuntimeError("execution_ready_prep_plan must keep ready_for_execution=false.")
    if str(execution_ready_preparation_design.get("artifact_kind", "")).strip() != "execution_ready_preparation_design":
        raise RuntimeError("execution_ready_preparation_design artifact must exist before final live sync can proceed.")
    if bool(execution_ready_preparation_design.get("ready_for_execution")):
        raise RuntimeError("execution_ready_preparation_design must keep ready_for_execution=false.")
    if status_discussion_decision != STATUS_DISCUSSION_PROMOTION_DECISION:
        raise RuntimeError(
            "execution_ready_status_discussion_decision must be PROMOTE_TO_EXECUTION_READY_STATUS_DISCUSSION before final live sync can proceed."
        )
    if bool(execution_ready_status_discussion_decision.get("ready_for_execution")):
        raise RuntimeError("execution_ready_status_discussion_decision must keep ready_for_execution=false.")
    if str(execution_ready_status_discussion_packet.get("artifact_kind", "")).strip() != "execution_ready_status_discussion_packet":
        raise RuntimeError("execution_ready_status_discussion_packet artifact must exist before final live sync can proceed.")
    if bool(execution_ready_status_discussion_packet.get("ready_for_execution")):
        raise RuntimeError("execution_ready_status_discussion_packet must keep ready_for_execution=false.")

    updated_research_status = deepcopy(research_status)
    updated_research_status.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "reason": RESEARCH_IDLE_REASON,
            "research_loop_entrypoint": str((REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py").resolve()),
            "approved_problem_path": str((RESEARCH_ROOT / "approved_problem.json").resolve()),
            "repo_process_allowlist_path": str(ALLOWLIST_JSON.resolve()),
            "approved_problem_archive_root": str((RESEARCH_ROOT / "approved_problem_archive").resolve()),
            "gate_reference_logs_path": str((RESEARCH_ROOT / "gate_reference_logs.json").resolve()),
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "manual_action_required": True,
            "manual_action_kind": "manual_review",
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
            "current_priority_family": "",
            "current_priority_reason": CURRENT_PRIORITY_REASON,
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "preferred_first_candidate_shape": PREFERRED_SHAPE,
            "preferred_first_candidate_shape_reason": "This is the first manual-review shape, but execution remains disabled until a later explicit manual decision.",
            "preferred_first_candidate_config": "",
            "preferred_first_candidate_requires_code_patch": False,
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "next_requirement": next_requirement,
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": "two_stage_objective_decoupling already spent its single-ticket family budget and is formally closed after dead_same_day.",
            "current_cloud_blocker": (
                "The manually promoted hybrid-ring local lead remains the current local lead, cloud must stay off, "
                "and the execution-ready status discussion for camera_focal_objective_isolation still does not authorize arm/run/training/cloud."
            ),
            "current_frontier_hint": CURRENT_FRONTIER_HINT,
            "current_frontier_priority": CURRENT_FRONTIER_PRIORITY,
            "manual_review_artifacts": {
                "two_stage_axis_closed": repo_rel(AXIS_CLOSED_JSON),
                "early_fl_tax_localization": repo_rel(LOCALIZATION_JSON),
                "fl_tax_object_alignment_matrix": repo_rel(OBJECT_MATRIX_JSON),
                "two_stage_failure_interpretation": repo_rel(FAILURE_JSON),
                "seed": repo_rel(SEED_JSON),
                "blueprint": repo_rel(BLUEPRINT_JSON),
                "candidate_patch_plan": repo_rel(FAMILY_PLAN_JSON),
                "draft": repo_rel(DRAFT_JSON),
                "review_note": repo_rel(REVIEW_NOTE_JSON),
                "decision_checklist": repo_rel(CHECKLIST_JSON),
                "execution_prep_design_sketch": repo_rel(DESIGN_SKETCH_JSON),
                "minimal_write_surface_candidates": repo_rel(SURFACE_CANDIDATES_JSON),
                "minimal_write_surface_recommendation": repo_rel(SURFACE_RECOMMEND_JSON),
                "design_contract": repo_rel(DESIGN_CONTRACT_JSON),
                "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
                "execution_prep_implementation_sketch": repo_rel(IMPLEMENTATION_SKETCH_JSON),
                "pseudodiff_map": repo_rel(PSEUDODIFF_MAP_JSON),
                "patch_authoring_gate": repo_rel(PATCH_AUTHORING_GATE_JSON),
                "patch_authoring_approval_note": repo_rel(PATCH_AUTHORING_APPROVAL_NOTE_JSON),
                "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_NOTE_JSON),
                "hygiene_review": repo_rel(HYGIENE_REVIEW_JSON),
                "execution_prep_promotion_note": repo_rel(EXECUTION_PREP_PROMOTION_NOTE_JSON),
                "execution_prep_baseline_validation": repo_rel(EXECUTION_PREP_BASELINE_VALIDATION_JSON),
                "execution_prep_baseline_decision": repo_rel(EXECUTION_PREP_BASELINE_DECISION_JSON),
                "execution_ready_smoke_spec": repo_rel(EXECUTION_READY_SMOKE_SPEC_JSON),
                "nontraining_integration_check": repo_rel(NONTRAINING_INTEGRATION_CHECK_JSON),
                "boundary_confirmation_checklist": repo_rel(BOUNDARY_CONFIRMATION_CHECKLIST_JSON),
                "execution_ready_boundary": repo_rel(EXECUTION_READY_BOUNDARY_JSON),
                "execution_ready_gate": repo_rel(EXECUTION_READY_GATE_JSON),
                "execution_ready_discussion_approval_note": repo_rel(EXECUTION_READY_DISCUSSION_APPROVAL_NOTE_JSON),
                "execution_ready_discussion_decision": repo_rel(EXECUTION_READY_DISCUSSION_DECISION_JSON),
                "execution_ready_prep_plan": repo_rel(EXECUTION_READY_PREP_PLAN_JSON),
                "execution_ready_preparation_design": repo_rel(EXECUTION_READY_PREPARATION_DESIGN_JSON),
                "execution_ready_status_discussion_decision": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON),
                "execution_ready_status_discussion_packet": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
                "status_discussion_guard_latest_report": repo_rel(STATUS_DISCUSSION_GUARD_LATEST_REPORT_JSON),
            },
        }
    )
    write_json(RESEARCH_STATUS_JSON, updated_research_status)
    write_text(RESEARCH_STATUS_MD, render_research_status_md(updated_research_status))

    updated_candidate_patch_plan = deepcopy(candidate_patch_plan)
    updated_candidate_patch_plan.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "manual_action_required": True,
            "manual_action_kind": "manual_review",
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
            "current_priority_family": "",
            "current_priority_reason": CURRENT_PRIORITY_REASON,
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": (
                "Do not auto-arm another ticket. Conduct the manual execution-ready "
                "status discussion for camera_focal_objective_isolation using the packaged status-discussion handoff only."
            ),
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "preferred_first_candidate_shape": PREFERRED_SHAPE,
            "preferred_first_candidate_shape_reason": "First manual-review-only shape after the early FL-tax audit.",
            "preferred_first_candidate_config": "",
            "preferred_first_candidate_requires_code_patch": False,
            "preferred_first_candidate_write_surface": family_plan.get("write_surface", []),
            "preferred_first_candidate_execution_note": "Manual review only: execution is intentionally disabled, no config has been materialized, and the live route is the packaged status-discussion handoff.",
            "preferred_first_candidate_arm_command": "",
            "preferred_first_candidate_run_command": "",
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "family_blueprints": {
                PREFERRED_FAMILY: {
                    "status": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
                    "ready_for_execution": False,
                    "blueprint_path": repo_rel(BLUEPRINT_JSON),
                    "draft_path": repo_rel(DRAFT_JSON),
                    "plan_path": repo_rel(FAMILY_PLAN_JSON),
                    "review_note_path": repo_rel(REVIEW_NOTE_JSON),
                    "decision_checklist_path": repo_rel(CHECKLIST_JSON),
                    "execution_prep_design_sketch_path": repo_rel(DESIGN_SKETCH_JSON),
                    "minimal_write_surface_candidates_path": repo_rel(SURFACE_CANDIDATES_JSON),
                    "minimal_write_surface_recommendation_path": repo_rel(SURFACE_RECOMMEND_JSON),
                    "design_contract_path": repo_rel(DESIGN_CONTRACT_JSON),
                    "patch_boundary_note_path": repo_rel(PATCH_BOUNDARY_JSON),
                    "execution_prep_implementation_sketch_path": repo_rel(IMPLEMENTATION_SKETCH_JSON),
                    "pseudodiff_map_path": repo_rel(PSEUDODIFF_MAP_JSON),
                    "patch_authoring_gate_path": repo_rel(PATCH_AUTHORING_GATE_JSON),
                    "patch_authoring_approval_note_path": repo_rel(PATCH_AUTHORING_APPROVAL_NOTE_JSON),
                    "authored_patch_review_note_path": repo_rel(AUTHORED_PATCH_REVIEW_NOTE_JSON),
                    "hygiene_review_path": repo_rel(HYGIENE_REVIEW_JSON),
                    "execution_prep_promotion_note_path": repo_rel(EXECUTION_PREP_PROMOTION_NOTE_JSON),
                    "execution_prep_baseline_validation_path": repo_rel(EXECUTION_PREP_BASELINE_VALIDATION_JSON),
                    "execution_prep_baseline_decision_path": repo_rel(EXECUTION_PREP_BASELINE_DECISION_JSON),
                    "execution_ready_smoke_spec_path": repo_rel(EXECUTION_READY_SMOKE_SPEC_JSON),
                    "nontraining_integration_check_path": repo_rel(NONTRAINING_INTEGRATION_CHECK_JSON),
                    "boundary_confirmation_checklist_path": repo_rel(BOUNDARY_CONFIRMATION_CHECKLIST_JSON),
                    "execution_ready_boundary_path": repo_rel(EXECUTION_READY_BOUNDARY_JSON),
                    "execution_ready_gate_path": repo_rel(EXECUTION_READY_GATE_JSON),
                    "execution_ready_discussion_approval_note_path": repo_rel(EXECUTION_READY_DISCUSSION_APPROVAL_NOTE_JSON),
                    "execution_ready_discussion_decision_path": repo_rel(EXECUTION_READY_DISCUSSION_DECISION_JSON),
                    "execution_ready_prep_plan_path": repo_rel(EXECUTION_READY_PREP_PLAN_JSON),
                    "execution_ready_preparation_design_path": repo_rel(EXECUTION_READY_PREPARATION_DESIGN_JSON),
                    "execution_ready_status_discussion_decision_path": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON),
                    "execution_ready_status_discussion_packet_path": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
                    "status_discussion_guard_latest_report_path": repo_rel(STATUS_DISCUSSION_GUARD_LATEST_REPORT_JSON),
                    "gate_recommended_decision": gate_recommended_decision,
                    "discussion_decision": discussion_decision,
                    "status_discussion_decision": status_discussion_decision,
                    "next_requirement": next_requirement,
                }
            },
            "current_frontier_hint": CURRENT_FRONTIER_HINT,
            "current_frontier_priority": CURRENT_FRONTIER_PRIORITY,
            "next_requirement": next_requirement,
        }
    )
    write_json(CANDIDATE_PATCH_PLAN_JSON, updated_candidate_patch_plan)
    write_text(CANDIDATE_PATCH_PLAN_MD, render_candidate_patch_plan_md(updated_candidate_patch_plan))

    updated_blueprint = deepcopy(blueprint)
    updated_blueprint.update(
        {
            "checked_at": checked_at,
            "family": PREFERRED_FAMILY,
            "status": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
            "ready_for_manual_approval": True,
            "ready_for_execution": False,
            "why_now": (
                "The validated loss.py-only execution-prep baseline has already been promoted through execution-ready "
                "discussion and accepted into execution-ready status discussion; the next honest move is the bounded "
                "manual status discussion itself."
            ),
            "reference_evidence": [
                repo_rel(LOCALIZATION_JSON),
                repo_rel(OBJECT_MATRIX_JSON),
                repo_rel(FAILURE_JSON),
                repo_rel(EXECUTION_PREP_BASELINE_DECISION_JSON),
                repo_rel(EXECUTION_READY_GATE_JSON),
                repo_rel(EXECUTION_READY_PREP_PLAN_JSON),
                repo_rel(EXECUTION_READY_PREPARATION_DESIGN_JSON),
                repo_rel(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON),
                repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
            ],
            "signal_definition": (
                "Treat the live question as a bounded execution-ready status discussion only: confirm the loss.py-only "
                "packet remains the honest baseline for any later execution-ready promotion without enabling execution today."
            ),
            "scope_definition": (
                "manual execution-ready status discussion only; do not materialize candidate yaml, do not authorize arm/run, "
                "do not reopen tail-contract derivative cousins, and do not use cloud."
            ),
            "first_candidate_shape": PREFERRED_SHAPE,
            "first_candidate_config": "",
            "first_candidate_requires_code_patch": False,
            "first_candidate_write_surface": ["training/loss.py"],
            "first_candidate_execution_note": (
                "Execution remains intentionally disabled. The live route is the manual execution-ready status discussion "
                "for the bounded loss.py-only packet."
            ),
            "required_exclusions": [
                "not execution-ready",
                "not arm or run",
                "not config materialization",
                "not tail-contract derivative reopen",
                "not two-stage retry",
                "not cloud",
            ],
            "current_contract_budget": {
                "max_approved_problems_per_night": 1,
                "max_candidates_per_problem": 1,
                "same_night_execution_note": "This artifact is status-discussion-only. No same-night execution is allowed.",
            },
            "cloud_must_remain_off": True,
            "current_review_artifacts": {
                "execution_ready_prep_plan": repo_rel(EXECUTION_READY_PREP_PLAN_JSON),
                "execution_ready_preparation_design": repo_rel(EXECUTION_READY_PREPARATION_DESIGN_JSON),
                "execution_ready_status_discussion_decision": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON),
                "execution_ready_status_discussion_packet": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
            },
            "next_requirement": next_requirement,
        }
    )
    write_json(BLUEPRINT_JSON, updated_blueprint)

    updated_family_plan = deepcopy(family_plan)
    updated_family_plan.update(
        {
            "checked_at": checked_at,
            "state": FAMILY_STATUS_DISCUSSION_PLAN_STATE,
            "current_review_stage": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
            "approved_problem_present": False,
            "current_stable_lead_config": str(updated_research_status.get("current_stable_lead_config", "")),
            "family": PREFERRED_FAMILY,
            "first_candidate_shape": PREFERRED_SHAPE,
            "first_candidate_config": "",
            "arm_command": "",
            "run_command": "",
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "cloud_must_remain_off": True,
            "same_night_second_candidate_forbidden": True,
            "same_night_cousin_sweep_forbidden": True,
            "status_discussion_execution_note": (
                "Manual review only: conduct the execution-ready status discussion using the packaged loss.py-only handoff. "
                "Do not arm, run, train, materialize config, or use cloud."
            ),
            "readiness": {
                "ready_for_manual_review": True,
                "ready_for_status_discussion": True,
                "status_discussion_packet_ready": True,
                "ready_for_execution": False,
                "requires_new_manual_approval": True,
                "requires_execution_ready_promotion_decision": True,
                "do_not_auto_open_ticket": True,
            },
            "write_surface": ["training/loss.py"],
            "supporting_review_artifacts": [
                repo_rel(EXECUTION_PREP_BASELINE_DECISION_JSON),
                repo_rel(EXECUTION_READY_GATE_JSON),
                repo_rel(EXECUTION_READY_PREP_PLAN_JSON),
                repo_rel(EXECUTION_READY_PREPARATION_DESIGN_JSON),
                repo_rel(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON),
                repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
            ],
            "next_requirement": next_requirement,
        }
    )
    write_json(FAMILY_PLAN_JSON, updated_family_plan)
    write_text(FAMILY_PLAN_MD, render_family_candidate_patch_plan_md(updated_family_plan))

    updated_frontier_ledger = deepcopy(frontier_ledger)
    updated_frontier_ledger.update(
        {
            "checked_at": checked_at,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
            "recommended_next_families": [PREFERRED_FAMILY],
            "recommended_family_order": [PREFERRED_FAMILY],
            "current_priority_family": "",
            "current_priority_reason": CURRENT_PRIORITY_REASON,
        }
    )
    family_readout = deepcopy(updated_frontier_ledger.get("family_readout", {}))
    family_readout["two_stage_objective_decoupling"] = {
        "status": "dead_same_day",
        "stop_reason": candidate_verdict.get("reason", ""),
    }
    family_readout["tail_contract_derivative_batch"] = {
        "status": "closed_to_execution_after_two_stage_failure",
        "stop_reason": "Two-stage failure plus daytime FL-tax localization closes automatic tail-contract derivative execution for today. Move only to manual diagnosis.",
    }
    family_readout["default_stream_intrinsics_counterbalance"] = {
        "status": "superseded_by_manual_diagnosis_only",
        "stop_reason": "The daytime FL-tax localization widened the question beyond a single default-stream focal counterweight. Keep it as historical evidence, not as the next ticket.",
    }
    family_readout[PREFERRED_FAMILY] = {
        "status": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
        "stop_reason": (
            "The execution-ready preparation design is accepted as the baseline, the family is now promoted into execution-ready status discussion only, "
            "the manual status-discussion packet is ready, ready_for_execution=false, no auto-open is allowed, and the next step is the manual status discussion."
        ),
    }
    updated_frontier_ledger["family_readout"] = family_readout
    write_json(FRONTIER_LEDGER_JSON, updated_frontier_ledger)

    updated_family_stop_reason = deepcopy(family_stop_reason)
    updated_family_stop_reason.update(
        {
            "checked_at": checked_at,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
            "current_priority_family": "",
            "current_priority_reason": CURRENT_PRIORITY_REASON,
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": "two_stage_objective_decoupling is formally closed after dead_same_day, and no same-family retry is allowed.",
            "next_requirement": next_requirement,
        }
    )
    updated_family_stop_reason["tail_contract_derivative_batch"] = {
        "status": "closed_to_execution_after_two_stage_failure",
        "stop_rule": "Do not auto-open default_stream_intrinsics_counterbalance or any other tail-contract derivative family today. Move only to camera_focal_objective_isolation manual review.",
        "closed_families": list(
            dict.fromkeys(
                list((updated_family_stop_reason.get("tail_contract_derivative_batch", {}) or {}).get("closed_families", []))
                + ["default_stream_intrinsics_counterbalance"]
            )
        ),
        "only_eligible_next_ticket": "",
    }
    updated_family_stop_reason["latest_family_outcomes"] = deepcopy(updated_family_stop_reason.get("latest_family_outcomes", {}))
    updated_family_stop_reason["latest_family_outcomes"][PREFERRED_FAMILY] = {
        "latest_status": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
        "problem_id": "camera_focal_objective_isolation_v1",
        "first_candidate_shape": PREFERRED_SHAPE,
        "active_candidate": "",
        "reason": (
            "The execution-ready preparation design is accepted, the family is promoted into execution-ready status discussion, "
            "the manual status-discussion packet is ready, ready_for_execution=false, and no execution path is enabled."
        ),
        "gate_stage_reached": "execution_ready_status_discussion_promoted_not_execution_ready",
        "approved_problem_archive_path": "",
    }
    write_json(FAMILY_STOP_REASON_JSON, updated_family_stop_reason)

    updated_task_plan = deepcopy(task_plan)
    updated_task_plan.update(
        {
            "checked_at": checked_at,
            "latest_guard_checked_at": str(latest_guard.get("checked_at", "")),
            "task_mode_status": TASK_MODE_STATUS,
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": TASK_MODE_FOCUS,
        }
    )
    updated_task_plan["research_loop"] = deepcopy(updated_task_plan.get("research_loop", {}))
    updated_task_plan["research_loop"].update(
        {
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "state": "IDLE_GUARD",
            "current_priority_family": "",
            "auto_next_ticket_enabled": False,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
        }
    )
    updated_task_plan["research_loop_contract"] = deepcopy(updated_task_plan.get("research_loop_contract", {}))
    updated_task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": MANUAL_REVIEW_REASON,
            "current_priority_family": "",
            "current_priority_reason": CURRENT_PRIORITY_REASON,
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": "two_stage_objective_decoupling is formally closed after dead_same_day, and no same-family retry is allowed.",
            "next_requirement": next_requirement,
        }
    )
    updated_task_plan["problem_definition_progress"] = deepcopy(updated_task_plan.get("problem_definition_progress", {}))
    updated_task_plan["problem_definition_progress"].update(
        {
            "status": EXECUTION_READY_STATUS_DISCUSSION_STATUS,
            "newest_boundary_fact": "The early FL-tax audit now has a reviewed authored training/loss.py patch with stronger local validation, an explicit execution-prep baseline decision, a complete execution-ready smoke/integration/checklist packet, a manual decision that promotes the packet into execution-ready discussion, a packaged execution-ready preparation design, a manual decision that promotes the family into execution-ready status discussion only, and a packaged manual status-discussion handoff packet.",
            "next_requirement": "Hold the manual execution-ready status discussion before any future execution-ready promotion.",
        }
    )
    updated_task_plan["active_tasks"] = []
    for phase in [
        {
            "id": "phase_23_two_stage_axis_closed",
            "status": "completed",
            "details": "Formally closed the two-stage objective-decoupling axis as dead_same_day and confirmed no same-family retry.",
        },
        {
            "id": "phase_24_early_fl_tax_localization",
            "status": "completed",
            "details": "Daytime audit localized the residual camera tax to an early FL-dominant objective pattern and generated object-level evidence.",
        },
        {
            "id": "phase_25_manual_only_problem_packaged",
            "status": "completed",
            "details": "Packaged camera_focal_objective_isolation for manual review and kept the bounded loss.py-only surface ready_for_execution=false.",
        },
        {
            "id": "phase_26_sync_only_planning_close",
            "status": "completed",
            "details": "Synced research/status/task artifacts to the discussion-promoted manual planning route with manual_action_kind=manual_review and no active approval.",
        },
        {
            "id": "phase_27_execution_ready_boundary_gate_packet",
            "status": "completed",
            "details": "Wrote the execution-ready smoke spec, non-training integration check, boundary checklist, boundary, and gate while keeping ready_for_execution=false.",
        },
        {
            "id": "phase_28_execution_ready_discussion_promotion",
            "status": "completed",
            "details": "Completed the execution-ready discussion approval and decision with PROMOTE_TO_EXECUTION_READY_DISCUSSION while arm/run/training/cloud stayed disabled.",
        },
        {
            "id": "phase_29_post_promotion_sync",
            "status": "completed",
            "details": "Refreshed live planning/watch artifacts after discussion promotion while execution remained disabled.",
        },
        {
            "id": "phase_30_execution_ready_prep_plan_packaged",
            "status": "completed",
            "details": "Packaged the execution-ready prep plan and advanced live planning to manual review of that plan while ready_for_execution stayed false.",
        },
        {
            "id": "phase_31_execution_ready_preparation_design_packaged",
            "status": "completed",
            "details": "Packaged the execution-ready preparation design and advanced live planning to manual review of that design while ready_for_execution stayed false.",
        },
        {
            "id": "phase_32_execution_ready_status_discussion_promoted",
            "status": "completed",
            "details": "Accepted the preparation design as the manual-review baseline and promoted the family into execution-ready status discussion while ready_for_execution stayed false.",
        },
        {
            "id": "phase_33_execution_ready_status_discussion_packet_packaged",
            "status": "completed",
            "details": "Packaged the manual execution-ready status-discussion handoff so the next review has one current packet while ready_for_execution stayed false.",
        },
    ]:
        updated_task_plan["completed_this_round"] = upsert_by_key(list(updated_task_plan.get("completed_this_round", []) or []), phase, "id")
    updated_task_plan["blocked_tasks"] = upsert_by_key(
        list(updated_task_plan.get("blocked_tasks", []) or []),
        {
            "id": "arm_next_ticket",
            "status": "blocked_on_status_discussion_not_execution_ready",
            "details": "camera_focal_objective_isolation is now in execution-ready status discussion only but is still not execution-ready, so no ticket may be armed.",
        },
        "id",
    )
    updated_task_plan["blocked_tasks"] = upsert_by_key(
        list(updated_task_plan.get("blocked_tasks", []) or []),
        {
            "id": "reopen_tail_contract_derivative_batch",
            "status": "blocked_by_daytime_manual_diagnosis_contract",
            "details": "Tail-contract derivative execution is closed for today after the two-stage failure and the FL-tax localization audit.",
        },
        "id",
    )
    updated_task_plan["fastest_next_path"] = [
        "Keep guard and research idle/clean.",
        "Conduct the manual execution-ready status discussion for camera_focal_objective_isolation using the packaged status-discussion packet as the baseline.",
        "Keep the validated execution-prep baseline, smoke packet, integration packet, preparation design, status-discussion packet, and governance holds intact while ready_for_execution stays false.",
        "Do not arm, do not run, and do not touch cloud today.",
    ]
    updated_task_plan["summary_conclusion"] = [
        "The two-stage objective-decoupling axis is formally closed as dead_same_day.",
        "Research remains in IDLE_GUARD with no active approval, no modal apps, and an empty allowlist.",
        "camera_focal_objective_isolation now has one validated training/loss.py patch approved as an execution-prep baseline, promoted into execution-ready discussion, accepted into execution-ready status discussion, and packaged into one manual status-discussion handoff, but it is still not execution-ready and must not be armed.",
    ]
    updated_task_plan["current_state_notes"] = [
        "The two-stage objective-decoupling axis is closed for this round.",
        "The single-file training/loss.py validated patch is approved as an execution-prep baseline, the family is now in execution-ready status discussion, and the manual status-discussion packet is ready while execution remains disabled.",
        "Cloud remains off and no execution may start from this packet.",
    ]
    updated_task_plan["machine_readable_sync"] = {
        "status": "completed",
        "synced_files": [
            str(BLUEPRINT_JSON.resolve()),
            str(FAMILY_PLAN_JSON.resolve()),
            str(FAMILY_PLAN_MD.resolve()),
            str(EXECUTION_READY_PREP_PLAN_JSON.resolve()),
            str(EXECUTION_READY_PREPARATION_DESIGN_JSON.resolve()),
            str(EXECUTION_READY_STATUS_DISCUSSION_DECISION_JSON.resolve()),
            str(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON.resolve()),
            str(RESEARCH_STATUS_JSON.resolve()),
            str(CANDIDATE_PATCH_PLAN_JSON.resolve()),
            str(CANDIDATE_PATCH_PLAN_MD.resolve()),
            str(FAMILY_STOP_REASON_JSON.resolve()),
            str(FRONTIER_LEDGER_JSON.resolve()),
            str(TASK_PLAN_JSON.resolve()),
            str(TASK_PLAN_MD.resolve()),
            str(SUMMARY_MD.resolve()),
            str(LATEST_WATCH_JSON.resolve()),
        ],
    }
    write_json(TASK_PLAN_JSON, updated_task_plan)
    write_text(TASK_PLAN_MD, render_task_plan_md(updated_task_plan))
    write_text(SUMMARY_MD, render_summary_md(updated_task_plan, latest_guard))

    updated_watch = deepcopy(latest_watch)
    updated_watch["checked_at"] = checked_at
    research_node = deepcopy(updated_watch.get("research", {}))
    research_node["research_status_path"] = str(RESEARCH_STATUS_JSON.resolve())
    research_node["approved_problem_path"] = str((RESEARCH_ROOT / "approved_problem.json").resolve())
    research_node["allowlist_path"] = str(ALLOWLIST_JSON.resolve())
    research_node["research_status"] = updated_research_status
    research_node["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "allowlist_empty": len((allowlist.get("allowed_markers", []) or [])) == 0,
        "allowlist_status": str(allowlist.get("status", "")),
        "active_modal_app_count": len(active_modal_apps),
        "runtime_process_count": 0,
        "manual_action_required": True,
        "manual_action_kind": "manual_review",
        "status_discussion_packet_ready": True,
        "current_review_packet": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
    }
    research_node["allowlist"] = allowlist
    updated_watch["research"] = research_node
    updated_watch["modal_apps"] = active_modal_apps
    updated_watch["research_runtime_processes"] = []
    updated_watch["watch_conclusion"] = WATCH_CONCLUSION
    write_json(LATEST_WATCH_JSON, updated_watch)

    print(
        json.dumps(
            {
                "research_loop_status": repo_rel(RESEARCH_STATUS_JSON),
                "candidate_patch_plan": repo_rel(CANDIDATE_PATCH_PLAN_JSON),
                "family_blueprint": repo_rel(BLUEPRINT_JSON),
                "family_candidate_patch_plan": repo_rel(FAMILY_PLAN_JSON),
                "execution_ready_status_discussion_packet": repo_rel(EXECUTION_READY_STATUS_DISCUSSION_PACKET_JSON),
                "frontier_ledger": repo_rel(FRONTIER_LEDGER_JSON),
                "family_stop_reason": repo_rel(FAMILY_STOP_REASON_JSON),
                "task_plan": repo_rel(TASK_PLAN_JSON),
                "latest_watch_snapshot": repo_rel(LATEST_WATCH_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
