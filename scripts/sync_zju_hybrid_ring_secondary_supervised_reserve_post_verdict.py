import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

VERDICT_JSON = RESEARCH_ROOT / "candidate_verdict.json"
RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.hybrid_ring_secondary_supervised_reserve.20260403.json"
PLAN_MD = RESEARCH_ROOT / "candidate_patch_plan.hybrid_ring_secondary_supervised_reserve.20260403.md"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.hybrid_ring_secondary_supervised_reserve.20260403.json"
DRAFT_JSON = RESEARCH_ROOT / "next_manual_problem_draft.hybrid_ring_secondary_supervised_reserve.20260403.json"
DRAFT_MD = RESEARCH_ROOT / "next_manual_problem_draft.hybrid_ring_secondary_supervised_reserve.20260403.md"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
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
    if verdict.get("family") != "hybrid_ring_secondary_supervised_reserve":
        raise RuntimeError("candidate_verdict is not for hybrid_ring_secondary_supervised_reserve")
    if verdict.get("status") != "dead_same_day":
        raise RuntimeError("post-verdict sync currently expects dead_same_day")

    research = load_json(RESEARCH_STATUS_JSON)
    frontier = load_json(FRONTIER_LEDGER_JSON)
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    plan = load_json(PLAN_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    draft = load_json(DRAFT_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)

    updated_blueprint = deepcopy(blueprint)
    updated_blueprint.update(
        {
            "checked_at": checked_at,
            "status": "dead_same_day_closed_for_round",
            "ready_for_execution": False,
            "latest_verdict": {
                "status": verdict.get("status", ""),
                "reason": verdict.get("reason", ""),
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
                "candidate_config": verdict.get("active_candidate", ""),
            },
        }
    )
    write_json(BLUEPRINT_JSON, updated_blueprint)

    updated_plan = deepcopy(plan)
    updated_plan.update(
        {
            "checked_at": checked_at,
            "state": "dead_same_day_closed_no_same_family_retry",
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "latest_verdict": {
                "status": verdict.get("status", ""),
                "reason": verdict.get("reason", ""),
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
                "candidate_config": verdict.get("active_candidate", ""),
            },
        }
    )
    write_json(PLAN_JSON, updated_plan)
    write_text(PLAN_MD, json.dumps(updated_plan, ensure_ascii=False, indent=2) + "\n")

    updated_draft = deepcopy(draft)
    updated_draft.update(
        {
            "checked_at": checked_at,
            "status": "executed_dead_same_day",
            "ready_for_execution": False,
            "latest_verdict": {
                "status": verdict.get("status", ""),
                "reason": verdict.get("reason", ""),
                "gate_stage_reached": verdict.get("gate_stage_reached", ""),
            },
        }
    )
    write_json(DRAFT_JSON, updated_draft)
    write_text(DRAFT_MD, json.dumps(updated_draft, ensure_ascii=False, indent=2) + "\n")

    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"]["hybrid_ring_secondary_supervised_reserve"] = {
        "status": "dead_same_day",
        "stop_reason": verdict.get("reason", ""),
    }
    frontier["latest_formal_verdict"] = verdict
    frontier["latest_family_outcomes"] = {
        "hybrid_ring_secondary_supervised_reserve": {
            "latest_status": verdict.get("status", ""),
            "problem_id": verdict.get("problem_id", ""),
            "first_candidate_shape": verdict.get("first_candidate_shape", ""),
            "active_candidate": verdict.get("active_candidate", ""),
            "reason": verdict.get("reason", ""),
            "gate_stage_reached": verdict.get("gate_stage_reached", ""),
            "approved_problem_archive_path": verdict.get("approved_problem_archive_path", ""),
        }
    }
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop["latest_family_outcomes"] = deepcopy(family_stop.get("latest_family_outcomes", {}))
    family_stop["latest_family_outcomes"]["hybrid_ring_secondary_supervised_reserve"] = {
        "latest_status": verdict.get("status", ""),
        "problem_id": verdict.get("problem_id", ""),
        "first_candidate_shape": verdict.get("first_candidate_shape", ""),
        "active_candidate": verdict.get("active_candidate", ""),
        "reason": verdict.get("reason", ""),
        "gate_stage_reached": verdict.get("gate_stage_reached", ""),
        "approved_problem_archive_path": verdict.get("approved_problem_archive_path", ""),
    }
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    research.update(
        {
            "checked_at": checked_at,
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [],
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
            "current_priority_family": "",
            "current_priority_reason": "The first hybrid_ring_secondary_supervised_reserve ticket already produced a terminal research verdict (dead_same_day) at the local gate, so this derived family is closed for the round.",
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": "Do not auto-arm another ticket. The first hybrid_ring_secondary_supervised_reserve launch already spent the family budget.",
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": "The first hybrid_ring_secondary_supervised_reserve ticket already consumed the family budget, so same-family retry would violate the single-ticket contract.",
            "next_requirement": "Return to IDLE_GUARD. Do not auto-select a second ticket. Any later move must come from a fresh manual problem.",
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "latest_formal_verdict": verdict,
        }
    )
    write_json(RESEARCH_STATUS_JSON, research)

    task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "completed",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": "hybrid_ring_secondary_supervised_reserve_dead_same_day_cloud_off",
        }
    )
    task_plan["research_loop"] = deepcopy(task_plan.get("research_loop", {}))
    task_plan["research_loop"].update(
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
    task_plan["active_tasks"] = []
    task_plan["completed_this_round"] = upsert_by_key(
        list(task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_49_hybrid_ring_secondary_supervised_reserve_execution_closed_dead_same_day",
            "status": "completed",
            "details": "Ran the single hybrid_ring_secondary_supervised_reserve candidate and closed it as dead_same_day at short_gate_10x5.",
        },
        "id",
    )
    task_plan["summary_conclusion"] = [
        "hybrid_ring_secondary_supervised_reserve / anchor_plus_secondary_supervised_uniform_tail finished with a dead_same_day verdict at short_gate_10x5.",
        "The derived family did not improve the full gate: camera and T regressed sharply despite a conf_depth gain.",
        "Research is back in IDLE_GUARD, no approved problem remains, and cloud stays off.",
    ]
    task_plan["current_state_notes"] = [
        "The first hybrid_ring_secondary_supervised_reserve ticket is closed for this round.",
        "No same-family retry may be opened automatically.",
        "The next move must come from a fresh manual problem, not from this derived family.",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    latest_watch["checked_at"] = checked_at
    latest_watch["research"] = {
        "summary": {
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "current_review_packet": "output/zju_source_policy_research_loop/candidate_verdict.json",
        },
        "research_status": research,
        "allowlist": load_json(ALLOWLIST_JSON),
    }
    latest_watch["watch_conclusion"] = "hybrid_ring_secondary_supervised_reserve closed as dead_same_day at short_gate_10x5, no active cloud app, and no active local family run is open"
    write_json(LATEST_WATCH_JSON, latest_watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
