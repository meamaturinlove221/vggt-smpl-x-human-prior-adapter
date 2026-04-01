import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
TASK_PLAN_JSON = RAWPOOL_STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = RAWPOOL_STATUS_ROOT / "task_plan.md"
SUMMARY_MD = RAWPOOL_STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
DECISION_JSON = RESEARCH_ROOT / "execution_ready_promotion_decision.camera_translation_objective_isolation.20260401.json"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.camera_translation_objective_isolation.json"
PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.camera_translation_objective_isolation.json"


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
    task_plan = load_json(TASK_PLAN_JSON)
    latest_watch = load_json(LATEST_WATCH_JSON)
    decision = load_json(DECISION_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    family_plan = load_json(PLAN_JSON)

    family = "camera_translation_objective_isolation"
    shape = str(decision.get("first_candidate_shape", "")).strip()
    config = str(decision.get("first_candidate_config", "")).strip()

    updated_task_plan = deepcopy(task_plan)
    updated_task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "active",
            "current_mode": "execution_ready_pending_manual_arm",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": "camera_translation_objective_isolation_execution_ready_pending_arm_cloud_off",
        }
    )
    updated_task_plan["research_loop"] = deepcopy(updated_task_plan.get("research_loop", {}))
    updated_task_plan["research_loop"].update(
        {
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "state": "IDLE_GUARD",
            "current_priority_family": family,
            "auto_next_ticket_enabled": False,
            "preferred_first_family": family,
            "preferred_first_family_reason": research_status.get("preferred_first_family_reason", ""),
        }
    )
    updated_task_plan["research_loop_contract"] = deepcopy(updated_task_plan.get("research_loop_contract", {}))
    updated_task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": family,
            "preferred_first_family_reason": research_status.get("preferred_first_family_reason", ""),
            "current_priority_family": family,
            "current_priority_reason": research_status.get("current_priority_reason", ""),
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": research_status.get("same_family_retry_reason", ""),
            "next_requirement": research_status.get("next_requirement", ""),
        }
    )
    updated_task_plan["problem_definition_progress"] = deepcopy(updated_task_plan.get("problem_definition_progress", {}))
    updated_task_plan["problem_definition_progress"].update(
        {
            "status": "ready_for_execution",
            "newest_boundary_fact": "camera_translation_objective_isolation is now packaged to execution-ready pending manual arm using the single losstisolation0 candidate on the stable lead.",
            "next_requirement": "Manually approve and arm the single execution-ready camera_translation_objective_isolation candidate before any run.",
        }
    )
    updated_task_plan["active_tasks"] = [
        {
            "id": "manual_approval_to_arm_camera_translation_objective_isolation",
            "status": "active",
            "details": "Manually approve and arm exactly one stablelead_global_losst_isolation0 candidate on the current repo.",
        },
        {
            "id": "keep_cloud_off",
            "status": "active",
            "details": "Keep cloud_gate=false and launch_cloud_now=false while the execution-ready translation-isolation candidate remains local-only.",
        },
    ]
    updated_task_plan["completed_this_round"] = upsert_by_key(
        list(updated_task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_37_camera_translation_execution_ready_packaged",
            "status": "completed",
            "details": "Packaged camera_translation_objective_isolation to execution-ready pending manual arm with one losstisolation0 candidate.",
        },
        "id",
    )
    updated_task_plan["fastest_next_path"] = [
        "Keep guard and research idle/clean.",
        "Manually approve and arm exactly one camera_translation_objective_isolation losstisolation0 candidate.",
        "Run the normal local smoke/10x5/100x20 gate path after arm.",
        "Keep cloud off and do not open a second candidate.",
    ]
    updated_task_plan["summary_conclusion"] = [
        "camera_translation_objective_isolation is now the next execution-ready family.",
        "Exactly one stablelead_global_losst_isolation0 candidate is pending explicit manual arm approval on the current repo.",
        "Cloud remains off and no second candidate may be opened.",
    ]
    updated_task_plan["current_state_notes"] = [
        "camera_translation_objective_isolation is execution-ready pending manual arm approval.",
        "The translation-isolation ticket keeps the stable lead fixed and touches only training/loss.py plus one config.",
        "Cloud remains off and the single-candidate contract still applies.",
    ]
    write_json(TASK_PLAN_JSON, updated_task_plan)
    write_text(TASK_PLAN_MD, json.dumps(updated_task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in updated_task_plan["summary_conclusion"]]) + "\n")

    updated_watch = deepcopy(latest_watch)
    updated_watch["checked_at"] = checked_at
    research_node = deepcopy(updated_watch.get("research", {}))
    research_node["research_status"] = research_status
    research_node["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "manual_action_required": True,
        "manual_action_kind": "manual_approval",
        "ready_for_execution": True,
        "current_review_packet": "output/zju_source_policy_research_loop/execution_ready_promotion_decision.camera_translation_objective_isolation.20260401.json",
        "current_ready_candidate_config": config,
    }
    updated_watch["research"] = research_node
    updated_watch["watch_conclusion"] = (
        "guard is green and research remains idle; camera_translation_objective_isolation is execution-ready pending explicit "
        "manual arm approval for one local losstisolation0 candidate, and cloud remains off"
    )
    write_json(LATEST_WATCH_JSON, updated_watch)

    print(
        json.dumps(
            {
                "research_loop_status": "output/zju_source_policy_research_loop/research_loop_status.json",
                "task_plan": "output/zju_source_policy_rawpool_status_20260326_current/task_plan.json",
                "latest_watch_snapshot": "output/zju_source_policy_research_watch/latest_watch_snapshot.json",
                "family_blueprint": "output/zju_source_policy_research_loop/family_blueprint.camera_translation_objective_isolation.json",
                "family_candidate_patch_plan": "output/zju_source_policy_research_loop/candidate_patch_plan.camera_translation_objective_isolation.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
