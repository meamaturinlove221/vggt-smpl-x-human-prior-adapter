import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

VERDICT_JSON = RESEARCH_ROOT / "candidate_verdict.json"
RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
FAMILY_PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"
FAMILY_PLAN_MD = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.md"
DRAFT_JSON = RESEARCH_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.json"
DRAFT_MD = RESEARCH_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.md"
TASK_PLAN_JSON = RAWPOOL_STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = RAWPOOL_STATUS_ROOT / "task_plan.md"
SUMMARY_MD = RAWPOOL_STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"


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


def main() -> int:
    checked_at = now_iso()
    verdict = load_json(VERDICT_JSON)
    research_status = load_json(RESEARCH_STATUS_JSON)
    frontier_ledger = load_json(FRONTIER_LEDGER_JSON)
    family_stop_reason = load_json(FAMILY_STOP_REASON_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    family_plan = load_json(FAMILY_PLAN_JSON)
    draft = load_json(DRAFT_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)

    family = str(verdict.get("family", "")).strip()
    status = str(verdict.get("status", "")).strip()
    if family != "camera_focal_objective_isolation":
        raise RuntimeError("candidate_verdict family must be camera_focal_objective_isolation for post-verdict sync.")
    if status != "dead_same_day":
        raise RuntimeError("post-verdict sync currently expects dead_same_day.")

    reason = str(verdict.get("reason", "")).strip()
    next_requirement = (
        "Return to IDLE_GUARD. Do not auto-select a second ticket. "
        "If another forward move is needed, it must come from fresh manual diagnosis evidence rather than an automatic family reopen."
    )

    updated_blueprint = deepcopy(blueprint)
    updated_blueprint.update(
        {
            "checked_at": checked_at,
            "status": "dead_same_day_closed_for_round",
            "ready_for_execution": False,
            "why_now": "The single execution-ready lossflisolation0 candidate has now been executed and closed with a dead_same_day short-gate verdict.",
            "latest_verdict": {
                "status": status,
                "reason": reason,
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
                "candidate_config": verdict.get("active_candidate", ""),
            },
            "next_requirement": next_requirement,
        }
    )
    write_json(BLUEPRINT_JSON, updated_blueprint)

    updated_family_plan = deepcopy(family_plan)
    updated_family_plan.update(
        {
            "checked_at": checked_at,
            "state": "dead_same_day_closed_no_same_family_retry",
            "current_review_stage": "dead_same_day",
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "readiness": {
                "ready_for_manual_review": False,
                "ready_for_execution": False,
                "requires_new_manual_approval": True,
                "do_not_auto_open_ticket": True,
            },
            "execution_ready_execution_note": (
                "The single execution-ready lossflisolation0 candidate has already been executed and closed for this round."
            ),
            "latest_verdict": {
                "status": status,
                "reason": reason,
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
                "candidate_config": verdict.get("active_candidate", ""),
            },
            "next_requirement": next_requirement,
        }
    )
    write_json(FAMILY_PLAN_JSON, updated_family_plan)
    write_text(FAMILY_PLAN_MD, json.dumps(updated_family_plan, ensure_ascii=False, indent=2) + "\n")

    updated_draft = deepcopy(draft)
    updated_draft.update(
        {
            "checked_at": checked_at,
            "status": "executed_dead_same_day",
            "ready_for_execution": False,
            "latest_verdict": {
                "status": status,
                "reason": reason,
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
            },
        }
    )
    write_json(DRAFT_JSON, updated_draft)
    write_text(DRAFT_MD, json.dumps(updated_draft, ensure_ascii=False, indent=2) + "\n")

    updated_frontier = deepcopy(frontier_ledger)
    family_readout = deepcopy(updated_frontier.get("family_readout", {}))
    family_readout["camera_focal_objective_isolation"] = {
        "status": "dead_same_day",
        "stop_reason": reason,
    }
    updated_frontier["family_readout"] = family_readout
    write_json(FRONTIER_LEDGER_JSON, updated_frontier)

    updated_family_stop = deepcopy(family_stop_reason)
    updated_family_stop["latest_family_outcomes"] = deepcopy(updated_family_stop.get("latest_family_outcomes", {}))
    updated_family_stop["latest_family_outcomes"]["camera_focal_objective_isolation"] = {
        "latest_status": status,
        "problem_id": verdict.get("problem_id", ""),
        "first_candidate_shape": verdict.get("first_candidate_shape", ""),
        "active_candidate": verdict.get("active_candidate", ""),
        "reason": reason,
        "gate_stage_reached": verdict.get("gate_stage_reached", ""),
        "approved_problem_archive_path": verdict.get("approved_problem_archive_path", ""),
    }
    write_json(FAMILY_STOP_REASON_JSON, updated_family_stop)

    updated_research_status = deepcopy(research_status)
    updated_research_status.update(
        {
            "checked_at": checked_at,
            "reason": (
                "No approved_problem.json is present; the single camera_focal_objective_isolation candidate has already been "
                "executed and closed as dead_same_day, so research remains idle in guard."
            ),
            "current_cloud_blocker": (
                "camera_focal_objective_isolation closed as dead_same_day at short_gate_10x5, and cloud must remain off. "
                "Any later forward move requires a fresh manual problem and a separate explicit downstream decision."
            ),
            "allowed_families": [],
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
            "current_priority_family": "",
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "preferred_first_candidate_shape": "",
            "preferred_first_candidate_config": "",
            "preferred_first_candidate_requires_code_patch": False,
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
        }
    )
    write_json(RESEARCH_STATUS_JSON, updated_research_status)

    updated_task_plan = deepcopy(task_plan)
    updated_task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "completed",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": "camera_focal_objective_isolation_dead_same_day_cloud_off",
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
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
        }
    )
    updated_task_plan["research_loop_contract"] = deepcopy(updated_task_plan.get("research_loop_contract", {}))
    updated_task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
            "current_priority_family": "",
            "current_priority_reason": research_status.get("current_priority_reason", ""),
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": research_status.get("same_family_retry_reason", ""),
            "next_requirement": next_requirement,
        }
    )
    updated_task_plan["problem_definition_progress"] = deepcopy(updated_task_plan.get("problem_definition_progress", {}))
    updated_task_plan["problem_definition_progress"].update(
        {
            "status": "camera_focal_objective_isolation_dead_same_day",
            "newest_boundary_fact": "The single execution-ready lossflisolation0 candidate improved camera at short gate but still failed the multi-metric stable-lead rule because T regressed and depth terms did not improve, so the family is closed for this round.",
            "next_requirement": "Hold in IDLE_GUARD and wait for a fresh manual problem only if later needed.",
        }
    )
    updated_task_plan["completed_this_round"] = upsert_by_key(
        list(updated_task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_35_camera_focal_execution_closed_dead_same_day",
            "status": "completed",
            "details": "Ran the single execution-ready lossflisolation0 candidate and closed camera_focal_objective_isolation as dead_same_day at short_gate_10x5.",
        },
        "id",
    )
    updated_task_plan["completed_this_round"] = upsert_by_key(
        list(updated_task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_36_camera_focal_post_verdict_sync",
            "status": "completed",
            "details": "Synced family/task/watch truth to the dead_same_day post-verdict state and kept cloud off.",
        },
        "id",
    )
    updated_task_plan["active_tasks"] = []
    blocked = [item for item in list(updated_task_plan.get("blocked_tasks", []) or []) if str(item.get("id", "")) != "manual_approval_to_arm_camera_focal_objective_isolation"]
    blocked = upsert_by_key(
        blocked,
        {
            "id": "rearm_camera_focal_objective_isolation",
            "status": "blocked_by_formal_dead_same_day_verdict",
            "details": "The first camera_focal_objective_isolation ticket already spent the current family budget and may not be re-armed this round.",
        },
        "id",
    )
    updated_task_plan["blocked_tasks"] = blocked
    updated_task_plan["fastest_next_path"] = [
        "Keep guard and research idle/clean.",
        "Do not arm a second camera_focal_objective_isolation ticket.",
        "Keep cloud off.",
        "If another move is needed later, define a fresh manual problem first.",
    ]
    updated_task_plan["summary_conclusion"] = [
        "camera_focal_objective_isolation / stablelead_global_lossfl_isolation0 finished with a dead_same_day verdict at short_gate_10x5.",
        "loss_camera improved versus the stable lead, but loss_T regressed and depth terms did not improve, so the candidate did not satisfy the promotion rule.",
        "Research is back in IDLE_GUARD, no approved problem remains, and cloud stays off.",
    ]
    updated_task_plan["current_state_notes"] = [
        "The single execution-ready camera_focal_objective_isolation candidate has been executed and closed for this round.",
        "No second same-family ticket may be opened automatically.",
        "Cloud remains off and no current family is selected.",
    ]
    write_json(TASK_PLAN_JSON, updated_task_plan)
    write_text(TASK_PLAN_MD, json.dumps(updated_task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in updated_task_plan["summary_conclusion"]]) + "\n")

    updated_watch = deepcopy(latest_watch)
    updated_watch["checked_at"] = checked_at
    research_node = deepcopy(updated_watch.get("research", {}))
    research_node["research_status"] = updated_research_status
    research_node["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "manual_action_required": False,
        "manual_action_kind": "",
        "ready_for_execution": False,
        "current_review_packet": repo_rel(VERDICT_JSON),
    }
    updated_watch["research"] = research_node
    updated_watch["watch_conclusion"] = (
        "guard is green and research remains idle; camera_focal_objective_isolation closed as dead_same_day at short_gate_10x5, "
        "no second ticket is active, and cloud remains off"
    )
    write_json(LATEST_WATCH_JSON, updated_watch)

    print(
        json.dumps(
            {
                "candidate_verdict": repo_rel(VERDICT_JSON),
                "research_loop_status": repo_rel(RESEARCH_STATUS_JSON),
                "frontier_ledger": repo_rel(FRONTIER_LEDGER_JSON),
                "family_stop_reason": repo_rel(FAMILY_STOP_REASON_JSON),
                "family_blueprint": repo_rel(BLUEPRINT_JSON),
                "family_candidate_patch_plan": repo_rel(FAMILY_PLAN_JSON),
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
