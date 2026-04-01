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

SEED_JSON = RESEARCH_ROOT / "approved_problem.seed.camera_focal_objective_isolation.json"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
FAMILY_PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"
FAMILY_PLAN_MD = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.md"
DRAFT_JSON = RESEARCH_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.json"
DRAFT_MD = RESEARCH_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.md"

LOCALIZATION_JSON = RESEARCH_ROOT / "early_fl_tax_localization.20260330.json"
OBJECT_ALIGNMENT_JSON = RESEARCH_ROOT / "fl_tax_object_alignment_matrix.20260330.json"
BASELINE_VALIDATION_JSON = RESEARCH_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
STATUS_DISCUSSION_DECISION_JSON = RESEARCH_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"
STATUS_DISCUSSION_PACKET_JSON = RESEARCH_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.json"
PROMOTION_DECISION_JSON = RESEARCH_ROOT / "execution_ready_promotion_decision.camera_focal_objective_isolation.20260401.json"
PROMOTION_GUARD_LATEST_REPORT_JSON = RESEARCH_ROOT / "execution_ready_promotion_guard_only" / "latest_report.json"

PREFERRED_FAMILY = "camera_focal_objective_isolation"
READY_STATUS = "ready_for_execution"
FAMILY_PLAN_STATE = "execution_ready_pending_arm"
TASK_MODE_STATUS = "active"
TASK_MODE_FOCUS = "camera_focal_objective_isolation_execution_ready_pending_arm_cloud_off"
MANUAL_ACTION_KIND = "manual_approval"
PROMOTION_DECISION = "PROMOTE_TO_EXECUTION_READY"


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


def render_json_md(title: str, payload: dict) -> str:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: `{sub_value}`")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"## {key}")
            lines.append("")
            for item in value:
                lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`" if isinstance(item, dict) else f"- {item}")
            lines.append("")
        else:
            lines.append(f"- {key}: `{value}`")
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


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


def remove_blocked_task(items: list[dict], target_id: str) -> list[dict]:
    return [item for item in items if str(item.get("id", "")) != target_id]


def main() -> int:
    checked_at = now_iso()
    research_status = load_json(RESEARCH_STATUS_JSON)
    candidate_patch_plan = load_json(CANDIDATE_PATCH_PLAN_JSON)
    frontier_ledger = load_json(FRONTIER_LEDGER_JSON)
    family_stop_reason = load_json(FAMILY_STOP_REASON_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)
    latest_guard = load_json(LATEST_GUARD_JSON)
    seed = load_json(SEED_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    family_plan = load_json(FAMILY_PLAN_JSON)
    draft = load_json(DRAFT_JSON)
    promotion_decision = load_json(PROMOTION_DECISION_JSON)
    status_discussion_decision = load_json(STATUS_DISCUSSION_DECISION_JSON)
    status_discussion_packet = load_json(STATUS_DISCUSSION_PACKET_JSON)
    baseline_validation = load_json(BASELINE_VALIDATION_JSON)
    localization = load_json(LOCALIZATION_JSON)
    object_alignment = load_json(OBJECT_ALIGNMENT_JSON)

    if str(promotion_decision.get("decision", "")).strip() != PROMOTION_DECISION:
        raise RuntimeError("execution_ready_promotion_decision must be PROMOTE_TO_EXECUTION_READY before sync can proceed.")
    if not bool(promotion_decision.get("ready_for_execution")):
        raise RuntimeError("execution_ready_promotion_decision must set ready_for_execution=true.")
    if str(status_discussion_decision.get("decision", "")).strip() != "PROMOTE_TO_EXECUTION_READY_STATUS_DISCUSSION":
        raise RuntimeError("status discussion decision must already be promoted before execution-ready sync can proceed.")
    if str(status_discussion_packet.get("state", "")).strip() != "manual_review_execution_ready_status_discussion_packet_ready":
        raise RuntimeError("status discussion packet must be ready before execution-ready sync can proceed.")
    if str(baseline_validation.get("overall_status", "")).strip() != "PASS_STRONGER_LOCAL_VALIDATION":
        raise RuntimeError("baseline validation must pass before execution-ready sync can proceed.")

    first_candidate_shape = str(promotion_decision.get("first_candidate_shape", "")).strip()
    first_candidate_config = str(promotion_decision.get("first_candidate_config", "")).strip()
    next_requirement = str(promotion_decision.get("next_requirement", "")).strip()
    if not first_candidate_shape or not first_candidate_config:
        raise RuntimeError("promotion decision must provide first_candidate_shape and first_candidate_config.")
    if not (REPO_ROOT / first_candidate_config).exists():
        raise RuntimeError("promotion decision candidate config is missing on disk.")

    current_stable_lead_config = str(research_status.get("current_stable_lead_config", "")).strip()
    active_modal_apps = [row for row in list_modal_apps() if str(row.get("State", "")).lower() != "stopped"]

    preferred_first_family_reason = (
        "camera_focal_objective_isolation is now execution-ready. One prebuilt stablelead_global_lossfl_isolation0 "
        "candidate may be armed locally once explicitly approved, while cloud still remains off."
    )
    current_priority_reason = (
        "execution-ready: arm exactly one camera_focal_objective_isolation candidate when manually approved, run the normal "
        "local smoke/10x5/100x20 path, and keep cloud off."
    )
    current_frontier_hint = (
        "One execution-ready camera_focal_objective_isolation candidate is available for explicit manual arm approval."
    )
    current_frontier_priority = (
        "execution-ready local-only: arm exactly one prebuilt lossflisolation0 candidate, keep the current stable lead fixed, "
        "open no second candidate, and keep cloud off."
    )
    watch_conclusion = (
        "guard is green and research remains idle; camera_focal_objective_isolation is execution-ready pending explicit "
        "manual arm approval for one local lossflisolation0 candidate, and cloud remains off"
    )

    seed_payload = deepcopy(seed)
    seed_payload.update(
        {
            "approved": False,
            "approved_at": "",
            "problem_id": "camera_focal_objective_isolation_v1",
            "problem_title": "Camera/focal objective isolation after execution-ready promotion",
            "family": PREFERRED_FAMILY,
            "family_options_allowed": [],
            "preferred_first_family": "",
            "preferred_first_family_reason": "This family is execution-ready and may be armed directly once explicitly approved.",
            "problem_statement": (
                "Launch exactly one execution-ready candidate that isolates the FL contribution inside loss_camera "
                "on top of the current stable lead to test whether the remaining camera tax is caused by global FL pressure."
            ),
            "why_genuinely_new": (
                "This is no longer a manual-review-only diagnosis packet. It is one concrete execution-ready "
                "objective-isolation candidate using the already validated loss_fl_isolation_scale hook."
            ),
            "why_not_reopening_frozen_family": (
                "It does not reopen tail-contract derivative families, source-policy cousins, or two-stage cousins. "
                "It keeps the stable lead fixed and changes only the FL isolation scale in loss_camera."
            ),
            "first_candidate_hint": (
                "Launch only the stablelead_global_lossfl_isolation0 candidate by keeping the stable lead fixed and "
                "setting loss.camera.loss_fl_isolation_scale=0.0."
            ),
            "first_candidate_shape": first_candidate_shape,
            "first_candidate_config": first_candidate_config,
            "first_candidate_requires_code_patch": False,
            "first_candidate_write_surface": ["training/loss.py", first_candidate_config],
            "first_candidate_knobs": {
                "loss.camera.loss_fl_isolation_scale": 0.0,
                "keep_current_stable_lead_fixed": True,
            },
            "historical_prior": "GLOBAL_OBJECTIVE_CONFLICT plus early FL-dominant localization under the current stable lead.",
            "avoid_patterns": [
                "tail-contract derivative reopen",
                "two-stage retry",
                "second candidate",
                "cousin sweep",
                "cloud action",
            ],
            "max_approved_problems_per_night": 1,
            "candidate_budget": 1,
            "max_candidates_per_night": 1,
            "long_gate_required_for_promotion": True,
            "cloud_must_remain_off": True,
            "requires_dataset_or_routing_change": False,
            "requires_supervision_audit": False,
            "mutation_dsl": {
                "allow_camera_focal_objective_isolation": True,
                "require_existing_loss_fl_isolation_hook": True,
                "allow_global_loss_fl_isolation_only": True,
                "keep_existing_depth_routing_unchanged": True,
                "disallow_tail_contract_derivative_reopen": True,
                "disallow_two_stage_same_family_retry": True,
                "disallow_cloud": True,
                "disallow_wholefg_scalar": True,
                "disallow_wholefg_decoupled": True,
                "disallow_edge_band_scalar": True,
                "disallow_edge_band_decoupled": True,
                "disallow_hard_depth_conf_threshold": True,
                "disallow_plain_anchor_view_only": True,
            },
        }
    )
    write_json(SEED_JSON, seed_payload)

    draft_payload = deepcopy(draft)
    draft_payload.update(
        {
            "checked_at": checked_at,
            "draft_kind": "new_manual_problem",
            "status": "execution_ready_pending_arm",
            "family": PREFERRED_FAMILY,
            "first_candidate_shape": first_candidate_shape,
            "candidate_config": first_candidate_config,
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "why_now": [
                "The execution-ready status-discussion packet passed with no remaining contract drift.",
                "The reviewed loss_fl_isolation_scale hook already passed stronger local validation.",
                "The narrowest honest first candidate is now the prebuilt lossflisolation0 config on the stable lead.",
            ],
            "readiness_artifact": str(PROMOTION_DECISION_JSON.resolve()),
            "hypothesis": (
                "If the remaining camera tax is driven by global FL pressure inside loss_camera, zeroing that FL contribution "
                "inside loss_camera should move the local gate in the right direction while preserving the standalone loss_FL readout."
            ),
        }
    )
    write_json(DRAFT_JSON, draft_payload)
    write_text(DRAFT_MD, render_json_md("Camera Focal Objective Isolation Draft", draft_payload))

    blueprint_payload = deepcopy(blueprint)
    blueprint_payload.update(
        {
            "checked_at": checked_at,
            "family": PREFERRED_FAMILY,
            "status": READY_STATUS,
            "ready_for_manual_approval": True,
            "ready_for_execution": True,
            "why_now": (
                "The status-discussion packet passed, the loss.py-only hook is locally validated, and one prebuilt "
                "lossflisolation0 candidate now exists as the narrowest direct execution-ready test."
            ),
            "reference_evidence": [
                repo_rel(STATUS_DISCUSSION_DECISION_JSON),
                repo_rel(STATUS_DISCUSSION_PACKET_JSON),
                repo_rel(BASELINE_VALIDATION_JSON),
                repo_rel(LOCALIZATION_JSON),
                repo_rel(OBJECT_ALIGNMENT_JSON),
                repo_rel(PROMOTION_DECISION_JSON),
            ],
            "signal_definition": (
                "Keep the current stable lead fixed and isolate the FL contribution inside loss_camera by setting "
                "loss_fl_isolation_scale=0.0 while preserving the standalone loss_FL readout."
            ),
            "scope_definition": "objective isolation only; do not reopen cousins, dataset routing, or cloud paths.",
            "first_candidate_shape": first_candidate_shape,
            "first_candidate_config": first_candidate_config,
            "first_candidate_requires_code_patch": False,
            "first_candidate_write_surface": ["training/loss.py", first_candidate_config],
            "first_candidate_execution_note": (
                "This candidate is execution-ready on the current repo and may be armed directly once explicitly approved."
            ),
            "required_exclusions": [
                "not tail-contract derivative reopen",
                "not two-stage retry",
                "not second candidate",
                "not cousin sweep",
                "not cloud",
            ],
            "current_contract_budget": {
                "max_approved_problems_per_night": 1,
                "max_candidates_per_problem": 1,
                "same_night_execution_note": "Only this single lossflisolation0 candidate is eligible under the current contract.",
            },
            "cloud_must_remain_off": True,
            "current_review_artifacts": {
                "execution_ready_status_discussion_decision": repo_rel(STATUS_DISCUSSION_DECISION_JSON),
                "execution_ready_status_discussion_packet": repo_rel(STATUS_DISCUSSION_PACKET_JSON),
                "execution_ready_promotion_decision": repo_rel(PROMOTION_DECISION_JSON),
            },
            "next_requirement": next_requirement,
        }
    )
    write_json(BLUEPRINT_JSON, blueprint_payload)

    family_plan_payload = deepcopy(family_plan)
    family_plan_payload.pop("status_discussion_execution_note", None)
    family_plan_payload.update(
        {
            "checked_at": checked_at,
            "state": FAMILY_PLAN_STATE,
            "current_review_stage": READY_STATUS,
            "approved_problem_present": False,
            "current_stable_lead_config": current_stable_lead_config,
            "family": PREFERRED_FAMILY,
            "first_candidate_shape": first_candidate_shape,
            "first_candidate_config": first_candidate_config,
            "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_focal_objective_isolation",
            "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "do_not_arm_now": False,
            "do_not_run_candidate_now": False,
            "cloud_must_remain_off": True,
            "same_night_second_candidate_forbidden": True,
            "same_night_cousin_sweep_forbidden": True,
            "execution_ready_execution_note": (
                "Execution-ready and pending explicit manual approval to arm one local lossflisolation0 candidate while cloud remains off."
            ),
            "readiness": {
                "ready_for_manual_review": True,
                "ready_for_execution": True,
                "requires_new_manual_approval": True,
                "do_not_auto_open_ticket": False,
            },
            "execution_contract": {
                "single_problem_single_candidate": True,
                "gate_sequence": ["SMOKE_1x1", "TIGHT_GATE_10x5", "LONG_GATE_100x20", "RETURN_TO_GUARD"],
                "cloud_must_remain_off": True,
            },
            "write_surface": ["training/loss.py", first_candidate_config],
            "supporting_review_artifacts": [
                repo_rel(STATUS_DISCUSSION_DECISION_JSON),
                repo_rel(STATUS_DISCUSSION_PACKET_JSON),
                repo_rel(BASELINE_VALIDATION_JSON),
                repo_rel(PROMOTION_DECISION_JSON),
            ],
            "next_requirement": next_requirement,
        }
    )
    write_json(FAMILY_PLAN_JSON, family_plan_payload)
    write_text(FAMILY_PLAN_MD, render_json_md("Camera Focal Objective Isolation Candidate Plan", family_plan_payload))

    updated_research_status = deepcopy(research_status)
    updated_research_status.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "reason": (
                "No approved_problem.json is present; camera_focal_objective_isolation is execution-ready and waiting "
                "for explicit manual approval to arm one prebuilt local candidate while cloud remains off."
            ),
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [PREFERRED_FAMILY],
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
            "current_priority_family": PREFERRED_FAMILY,
            "current_priority_reason": current_priority_reason,
            "current_priority_candidate_shape": first_candidate_shape,
            "current_priority_candidate_config": first_candidate_config,
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": ["training/loss.py", first_candidate_config],
            "current_priority_candidate_execution_note": (
                "Execution-ready and pending explicit manual arm approval: arm exactly one approved lossflisolation0 ticket, "
                "run it locally through smoke/10x5/100x20, and keep cloud off."
            ),
            "current_priority_arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_focal_objective_isolation",
            "current_priority_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "preferred_first_candidate_shape": first_candidate_shape,
            "preferred_first_candidate_shape_reason": "The status-discussion packet passed, and the narrowest first execution-ready test is the prebuilt lossflisolation0 config.",
            "preferred_first_candidate_config": first_candidate_config,
            "preferred_first_candidate_requires_code_patch": False,
            "ready_for_execution": True,
            "do_not_arm_now": False,
            "do_not_run_candidate_now": False,
            "manual_action_required": True,
            "manual_action_kind": MANUAL_ACTION_KIND,
            "same_family_retry_forbidden": False,
            "same_family_retry_reason": "",
            "next_requirement": next_requirement,
            "current_cloud_blocker": (
                "camera_focal_objective_isolation is now execution-ready for one local candidate, but cloud must remain off. "
                "Any cloud action still requires a later local long-gate outcome and a separate downstream decision."
            ),
            "current_frontier_hint": current_frontier_hint,
            "current_frontier_priority": current_frontier_priority,
            "manual_review_artifacts": {
                **(updated_research_status.get("manual_review_artifacts", {}) or {}),
                "execution_ready_status_discussion_decision": repo_rel(STATUS_DISCUSSION_DECISION_JSON),
                "execution_ready_status_discussion_packet": repo_rel(STATUS_DISCUSSION_PACKET_JSON),
                "execution_ready_promotion_decision": repo_rel(PROMOTION_DECISION_JSON),
                "execution_ready_promotion_guard_latest_report": repo_rel(PROMOTION_GUARD_LATEST_REPORT_JSON),
                "seed": repo_rel(SEED_JSON),
                "blueprint": repo_rel(BLUEPRINT_JSON),
                "candidate_patch_plan": repo_rel(FAMILY_PLAN_JSON),
                "draft": repo_rel(DRAFT_JSON),
            },
        }
    )
    write_json(RESEARCH_STATUS_JSON, updated_research_status)
    write_text(RESEARCH_STATUS_MD, render_json_md("ZJU Source Policy Research Loop Status", updated_research_status))

    updated_candidate_patch_plan = deepcopy(candidate_patch_plan)
    updated_candidate_patch_plan.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [PREFERRED_FAMILY],
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
            "current_priority_family": PREFERRED_FAMILY,
            "current_priority_reason": current_priority_reason,
            "current_priority_candidate_shape": first_candidate_shape,
            "current_priority_candidate_config": first_candidate_config,
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": ["training/loss.py", first_candidate_config],
            "current_priority_candidate_execution_note": "Execution-ready: arm exactly one approved lossflisolation0 candidate and run the normal local gate path.",
            "current_priority_arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_focal_objective_isolation",
            "current_priority_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "preferred_first_candidate_shape": first_candidate_shape,
            "preferred_first_candidate_shape_reason": "The first execution-ready candidate is the prebuilt lossflisolation0 config.",
            "preferred_first_candidate_config": first_candidate_config,
            "preferred_first_candidate_requires_code_patch": False,
            "preferred_first_candidate_write_surface": ["training/loss.py", first_candidate_config],
            "preferred_first_candidate_execution_note": "Execution-ready and pending explicit manual approval to arm one local candidate while cloud remains off.",
            "preferred_first_candidate_arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_focal_objective_isolation",
            "preferred_first_candidate_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "manual_action_required": True,
            "manual_action_kind": MANUAL_ACTION_KIND,
            "ready_for_execution": True,
            "do_not_arm_now": False,
            "do_not_run_candidate_now": False,
            "same_family_retry_forbidden": False,
            "same_family_retry_reason": "",
            "current_frontier_hint": current_frontier_hint,
            "current_frontier_priority": current_frontier_priority,
            "next_requirement": next_requirement,
            "family_blueprints": {
                PREFERRED_FAMILY: {
                    "status": READY_STATUS,
                    "ready_for_execution": True,
                    "blueprint_path": repo_rel(BLUEPRINT_JSON),
                    "draft_path": repo_rel(DRAFT_JSON),
                    "plan_path": repo_rel(FAMILY_PLAN_JSON),
                    "execution_ready_status_discussion_decision_path": repo_rel(STATUS_DISCUSSION_DECISION_JSON),
                    "execution_ready_status_discussion_packet_path": repo_rel(STATUS_DISCUSSION_PACKET_JSON),
                    "execution_ready_promotion_decision_path": repo_rel(PROMOTION_DECISION_JSON),
                    "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_focal_objective_isolation",
                    "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
                    "next_requirement": next_requirement,
                }
            },
        }
    )
    write_json(CANDIDATE_PATCH_PLAN_JSON, updated_candidate_patch_plan)
    write_text(CANDIDATE_PATCH_PLAN_MD, render_json_md("ZJU Source Policy Candidate Patch Plan", updated_candidate_patch_plan))

    updated_frontier_ledger = deepcopy(frontier_ledger)
    updated_frontier_ledger.update(
        {
            "checked_at": checked_at,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
            "recommended_next_families": [PREFERRED_FAMILY],
            "recommended_family_order": [PREFERRED_FAMILY],
            "current_priority_family": PREFERRED_FAMILY,
            "current_priority_reason": current_priority_reason,
            "same_family_retry_forbidden": False,
            "same_family_retry_reason": "",
        }
    )
    family_readout = deepcopy(updated_frontier_ledger.get("family_readout", {}))
    family_readout[PREFERRED_FAMILY] = {
        "status": READY_STATUS,
        "stop_reason": (
            "The status-discussion packet passed review and camera_focal_objective_isolation is now execution-ready pending "
            "explicit manual arm approval of one local lossflisolation0 candidate while cloud remains off."
        ),
    }
    updated_frontier_ledger["family_readout"] = family_readout
    write_json(FRONTIER_LEDGER_JSON, updated_frontier_ledger)

    updated_family_stop_reason = deepcopy(family_stop_reason)
    updated_family_stop_reason.update(
        {
            "checked_at": checked_at,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
            "current_priority_family": PREFERRED_FAMILY,
            "current_priority_reason": current_priority_reason,
            "current_priority_candidate_shape": first_candidate_shape,
            "current_priority_candidate_config": first_candidate_config,
            "same_family_retry_forbidden": False,
            "same_family_retry_reason": "",
            "next_requirement": next_requirement,
        }
    )
    updated_family_stop_reason["latest_family_outcomes"] = deepcopy(updated_family_stop_reason.get("latest_family_outcomes", {}))
    updated_family_stop_reason["latest_family_outcomes"][PREFERRED_FAMILY] = {
        "latest_status": READY_STATUS,
        "problem_id": "camera_focal_objective_isolation_v1",
        "first_candidate_shape": first_candidate_shape,
        "active_candidate": first_candidate_config,
        "reason": "PROMOTE_TO_EXECUTION_READY selected one execution-ready lossflisolation0 candidate; it may now be armed locally once explicitly approved while cloud remains off.",
        "gate_stage_reached": "execution_ready_pending_arm",
        "approved_problem_archive_path": "",
    }
    write_json(FAMILY_STOP_REASON_JSON, updated_family_stop_reason)

    updated_task_plan = deepcopy(task_plan)
    updated_task_plan.update(
        {
            "checked_at": checked_at,
            "latest_guard_checked_at": str(latest_guard.get("checked_at", "")),
            "task_mode_status": TASK_MODE_STATUS,
            "current_mode": "execution_ready_pending_manual_arm",
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
            "current_priority_family": PREFERRED_FAMILY,
            "auto_next_ticket_enabled": False,
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
        }
    )
    updated_task_plan["research_loop_contract"] = deepcopy(updated_task_plan.get("research_loop_contract", {}))
    updated_task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": PREFERRED_FAMILY,
            "preferred_first_family_reason": preferred_first_family_reason,
            "current_priority_family": PREFERRED_FAMILY,
            "current_priority_reason": current_priority_reason,
            "same_family_retry_forbidden": False,
            "same_family_retry_reason": "",
            "next_requirement": next_requirement,
        }
    )
    updated_task_plan["problem_definition_progress"] = deepcopy(updated_task_plan.get("problem_definition_progress", {}))
    updated_task_plan["problem_definition_progress"].update(
        {
            "status": READY_STATUS,
            "newest_boundary_fact": "The execution-ready status-discussion packet passed review, PROMOTE_TO_EXECUTION_READY selected one prebuilt lossflisolation0 candidate, and camera_focal_objective_isolation is now execution-ready pending manual arm.",
            "next_requirement": "Manually approve and arm the single execution-ready camera_focal_objective_isolation candidate before any run.",
        }
    )
    updated_task_plan["completed_this_round"] = upsert_by_key(
        list(updated_task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_34_execution_ready_promoted",
            "status": "completed",
            "details": "Reviewed the status-discussion packet, passed it, and promoted camera_focal_objective_isolation into execution-ready pending manual arm approval.",
        },
        "id",
    )
    updated_task_plan["active_tasks"] = [
        {
            "id": "manual_approval_to_arm_camera_focal_objective_isolation",
            "status": "active",
            "details": "Manually approve and arm exactly one stablelead_global_lossfl_isolation0 candidate on the current repo.",
        },
        {
            "id": "keep_cloud_off",
            "status": "active",
            "details": "Keep cloud_gate=false and launch_cloud_now=false while the execution-ready candidate remains local-only.",
        },
    ]
    blocked = remove_blocked_task(list(updated_task_plan.get("blocked_tasks", []) or []), "arm_next_ticket")
    blocked = upsert_by_key(
        blocked,
        {
            "id": "open_second_candidate",
            "status": "blocked_by_single_candidate_contract",
            "details": "The execution-ready route authorizes exactly one lossflisolation0 candidate and no same-night cousin sweep.",
        },
        "id",
    )
    blocked = upsert_by_key(
        blocked,
        {
            "id": "cloud_action",
            "status": "blocked_on_local_long_gate_and_downstream_decision",
            "details": "Even after execution-ready promotion, cloud remains off until a later local long-gate outcome and a separate downstream decision exist.",
        },
        "id",
    )
    updated_task_plan["blocked_tasks"] = blocked
    updated_task_plan["fastest_next_path"] = [
        "Keep guard and research idle/clean.",
        "Manually approve and arm exactly one camera_focal_objective_isolation lossflisolation0 candidate.",
        "Run the normal local smoke/10x5/100x20 gate path after arm.",
        "Keep cloud off and do not open a second candidate.",
    ]
    updated_task_plan["summary_conclusion"] = [
        "The status-discussion packet passed review and camera_focal_objective_isolation is now execution-ready.",
        "Exactly one stablelead_global_lossfl_isolation0 candidate is pending explicit manual arm approval on the current repo.",
        "Cloud remains off and no second candidate may be opened.",
    ]
    updated_task_plan["current_state_notes"] = [
        "camera_focal_objective_isolation is execution-ready pending manual arm approval.",
        "The current stable lead remains the local reference until the execution-ready candidate is run and judged.",
        "Cloud remains off and the single-candidate contract still applies.",
    ]
    updated_task_plan["machine_readable_sync"] = {
        "status": "completed",
        "synced_files": [
            str(SEED_JSON.resolve()),
            str(BLUEPRINT_JSON.resolve()),
            str(FAMILY_PLAN_JSON.resolve()),
            str(FAMILY_PLAN_MD.resolve()),
            str(DRAFT_JSON.resolve()),
            str(DRAFT_MD.resolve()),
            str(PROMOTION_DECISION_JSON.resolve()),
            str(RESEARCH_STATUS_JSON.resolve()),
            str(CANDIDATE_PATCH_PLAN_JSON.resolve()),
            str(CANDIDATE_PATCH_PLAN_MD.resolve()),
            str(FRONTIER_LEDGER_JSON.resolve()),
            str(FAMILY_STOP_REASON_JSON.resolve()),
            str(TASK_PLAN_JSON.resolve()),
            str(TASK_PLAN_MD.resolve()),
            str(SUMMARY_MD.resolve()),
            str(LATEST_WATCH_JSON.resolve()),
        ],
    }
    write_json(TASK_PLAN_JSON, updated_task_plan)
    write_text(TASK_PLAN_MD, render_json_md("ZJU Source-Policy Rawpool Task Plan", updated_task_plan))
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in updated_task_plan["summary_conclusion"]]) + "\n")

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
        "manual_action_kind": MANUAL_ACTION_KIND,
        "ready_for_execution": True,
        "current_review_packet": repo_rel(PROMOTION_DECISION_JSON),
        "current_ready_candidate_config": first_candidate_config,
    }
    research_node["allowlist"] = allowlist
    updated_watch["research"] = research_node
    updated_watch["modal_apps"] = active_modal_apps
    updated_watch["research_runtime_processes"] = []
    updated_watch["watch_conclusion"] = watch_conclusion
    write_json(LATEST_WATCH_JSON, updated_watch)

    print(
        json.dumps(
            {
                "execution_ready_promotion_decision": repo_rel(PROMOTION_DECISION_JSON),
                "seed": repo_rel(SEED_JSON),
                "family_blueprint": repo_rel(BLUEPRINT_JSON),
                "family_candidate_patch_plan": repo_rel(FAMILY_PLAN_JSON),
                "research_loop_status": repo_rel(RESEARCH_STATUS_JSON),
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
