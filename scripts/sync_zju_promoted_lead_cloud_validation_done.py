import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
TASK_PLAN_JSON = RAWPOOL_STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = RAWPOOL_STATUS_ROOT / "task_plan.md"
SUMMARY_MD = RAWPOOL_STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
RESULT_JSON = RESEARCH_ROOT / "cloud_validation_result.source_policy_hybrid_ring_regularization.20260403.json"
RUNTIME_JSON = RESEARCH_ROOT / "cloud_runtime_state.20260403.json"


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
    research_status = load_json(RESEARCH_STATUS_JSON)
    frontier_ledger = load_json(FRONTIER_LEDGER_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)
    result = load_json(RESULT_JSON)
    runtime = load_json(RUNTIME_JSON)

    updated_research_status = deepcopy(research_status)
    updated_research_status.update(
        {
            "checked_at": checked_at,
            "current_cloud_blocker": (
                "The promoted hybrid-ring stable lead already completed one downstream cloud validation cleanly. "
                "The latest camera_depth_objective_coupling_audit ticket still closed as dead_same_day locally, so no candidate-escalation cloud path is open."
            ),
            "latest_promoted_lead_cloud_validation": {
                "status": "cloud_validation_done_clean",
                "family": result.get("family", ""),
                "mode": result.get("mode", ""),
                "result_path": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                "runtime_path": str(RUNTIME_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                "output_subdir": result.get("output_subdir", ""),
                "modal_app_id": result.get("modal_app_id", ""),
                "active_modal_app_count_after_finish": result.get("active_modal_app_count_after_finish", 0),
            },
        }
    )
    write_json(RESEARCH_STATUS_JSON, updated_research_status)

    updated_frontier = deepcopy(frontier_ledger)
    family_readout = deepcopy(updated_frontier.get("family_readout", {}))
    family_readout["source_policy_hybrid_ring_regularization_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "stop_reason": (
            "The currently promoted hybrid-ring local lead finished one downstream Modal validation cleanly; this does not reopen or advance the closed camera_depth_objective_coupling_audit line."
        ),
    }
    updated_frontier["family_readout"] = family_readout
    progression = list(updated_frontier.get("frontier_progression", []) or [])
    progression = upsert_by_key(
        progression,
        {
            "family": "source_policy_hybrid_ring_regularization_cloud_validation",
            "label": "promoted_lead_cloud_validation",
            "verdict": "cloud_validation_done_clean",
            "gate_stage_reached": "modal_cloud_validation_100x20",
            "val_loss_camera": result.get("val", {}).get("loss_camera"),
            "val_loss_T": result.get("val", {}).get("loss_T"),
            "val_loss_conf_depth": result.get("val", {}).get("loss_conf_depth"),
            "val_loss_reg_depth": result.get("val", {}).get("loss_reg_depth"),
        },
        "family",
    )
    updated_frontier["frontier_progression"] = progression
    updated_frontier["latest_promoted_lead_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "result_path": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "runtime_path": str(RUNTIME_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    write_json(FRONTIER_LEDGER_JSON, updated_frontier)

    updated_task_plan = deepcopy(task_plan)
    updated_task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "completed",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": "promoted_hybrid_ring_cloud_validation_done_clean__camera_depth_audit_closed",
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
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a fresh manual problem before any future approval.",
        }
    )
    updated_task_plan["research_loop_contract"] = deepcopy(updated_task_plan.get("research_loop_contract", {}))
    updated_task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. The latest camera-depth audit already closed for this round.",
            "current_priority_family": "",
            "current_priority_reason": research_status.get("current_priority_reason", ""),
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": research_status.get("same_family_retry_reason", ""),
            "next_requirement": research_status.get("next_requirement", ""),
        }
    )
    updated_task_plan["problem_definition_progress"] = deepcopy(updated_task_plan.get("problem_definition_progress", {}))
    updated_task_plan["problem_definition_progress"].update(
        {
            "status": "camera_depth_objective_coupling_audit_closed__promoted_lead_cloud_validation_done_clean",
            "newest_boundary_fact": (
                "The latest camera_depth_objective_coupling_audit ticket already closed as dead_same_day locally, while the separate promoted hybrid-ring stable-lead cloud validation finished cleanly."
            ),
            "next_requirement": "Treat the cloud line as promoted-lead validation only. Any future local move still requires a fresh manual problem.",
        }
    )
    updated_task_plan["active_tasks"] = []
    updated_task_plan["completed_this_round"] = upsert_by_key(
        list(updated_task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_44_promoted_hybrid_ring_cloud_validation_done_clean",
            "status": "completed",
            "details": "Ran the currently promoted hybrid-ring stable lead on Modal, wrote checkpoints, and finished with zero active apps remaining.",
        },
        "id",
    )
    updated_task_plan["blocked_tasks"] = upsert_by_key(
        list(updated_task_plan.get("blocked_tasks", []) or []),
        {
            "id": "rearm_camera_depth_objective_coupling_audit",
            "status": "blocked_by_formal_dead_same_day_verdict",
            "details": "The first camera_depth_objective_coupling_audit ticket already spent its family budget and may not be re-armed this round.",
        },
        "id",
    )
    updated_task_plan["promoted_lead_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "family": result.get("family", ""),
        "mode": result.get("mode", ""),
        "output_subdir": result.get("output_subdir", ""),
        "modal_app_id": result.get("modal_app_id", ""),
        "result_path": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "runtime_path": str(RUNTIME_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cleanup_ok": bool(runtime.get("cleanup_ok")),
    }
    updated_task_plan["fastest_next_path"] = [
        "Keep guard and research idle/clean.",
        "Treat the promoted hybrid-ring cloud run as complete downstream validation of the existing stable lead.",
        "Do not reinterpret that cloud result as permission to reopen camera_depth_objective_coupling_audit.",
        "If another move is needed later, define a fresh higher-level manual problem first.",
    ]
    updated_task_plan["summary_conclusion"] = [
        "source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail finished one downstream cloud validation cleanly.",
        "camera_depth_objective_coupling_audit still closed as dead_same_day at short_gate_10x5; the cloud run was not its continuation.",
        "Research is back in IDLE_GUARD, no approved problem remains, and there are zero active Modal apps.",
    ]
    updated_task_plan["current_state_notes"] = [
        "The promoted hybrid-ring local lead now has a completed clean cloud-validation record.",
        "The latest camera_depth_objective_coupling_audit ticket is closed for this round and may not be auto-reopened.",
        "Cloud cleanup finished cleanly; the historical app record is stopped, not active.",
    ]
    write_json(TASK_PLAN_JSON, updated_task_plan)
    write_text(TASK_PLAN_MD, json.dumps(updated_task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in updated_task_plan["summary_conclusion"]]) + "\n")

    updated_watch = deepcopy(latest_watch)
    updated_watch["checked_at"] = checked_at
    updated_watch["cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "family": result.get("family", ""),
        "mode": result.get("mode", ""),
        "modal_app_id": result.get("modal_app_id", ""),
        "output_subdir": result.get("output_subdir", ""),
        "result_path": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "runtime_path": str(RUNTIME_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cleanup_ok": bool(runtime.get("cleanup_ok")),
        "active_modal_app_count_after_finish": result.get("active_modal_app_count_after_finish", 0),
    }
    research_node = deepcopy(updated_watch.get("research", {}))
    research_node["research_status"] = updated_research_status
    research_node["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "manual_action_required": False,
        "manual_action_kind": "",
        "ready_for_execution": False,
        "current_review_packet": "output/zju_source_policy_research_loop/candidate_verdict.json",
    }
    updated_watch["research"] = research_node
    updated_watch["watch_conclusion"] = (
        "guard is green; promoted hybrid-ring cloud validation is done clean with zero active apps, and the latest camera_depth_objective_coupling_audit ticket remains locally closed as dead_same_day"
    )
    write_json(LATEST_WATCH_JSON, updated_watch)

    print(
        json.dumps(
            {
                "research_loop_status": "output/zju_source_policy_research_loop/research_loop_status.json",
                "frontier_ledger": "output/zju_source_policy_research_loop/frontier_ledger.json",
                "task_plan": "output/zju_source_policy_rawpool_status_20260326_current/task_plan.json",
                "latest_watch_snapshot": "output/zju_source_policy_research_watch/latest_watch_snapshot.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
