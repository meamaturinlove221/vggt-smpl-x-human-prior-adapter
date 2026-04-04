import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
TASK_PLAN_JSON = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
TASK_PLAN_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.md"
SUMMARY_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "summary.md"
WATCH_JSON = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_JSON = RESEARCH_ROOT / "family_stop_reason.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-tag", default="20260404")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checked_at = datetime.now().astimezone().isoformat()
    date_tag = str(args.date_tag)
    family = "teacher_fixed_visual_lift_cloud_deliverable_completion"

    smoke_failure_json = RESEARCH_ROOT / f"cloud_smoke_failure_packet.{family}.{date_tag}.json"
    smoke_failure_md = RESEARCH_ROOT / f"cloud_smoke_failure_packet.{family}.{date_tag}.md"
    result_json = RESEARCH_ROOT / f"{family}_result.{date_tag}.json"
    result_md = RESEARCH_ROOT / f"{family}_result.{date_tag}.md"
    next_draft_json = RESEARCH_ROOT / f"next_manual_problem_draft.teacher_fixed_visual_lift_scheduler_audit.{date_tag}.json"
    next_draft_md = RESEARCH_ROOT / f"next_manual_problem_draft.teacher_fixed_visual_lift_scheduler_audit.{date_tag}.md"

    failure_payload = {
        "checked_at": checked_at,
        "family": family,
        "status": "hard_blocker_before_smoke_execution_start",
        "selected_variant": "mask_hole_fill_plus_guided",
        "cloud_contract_go_no_go_path": "output/zju_source_policy_research_loop/cloud_contract_go_no_go.teacher_fixed_visual_lift_benchmark.20260404.json",
        "smoke_attempts": [
            {
                "app_id": "ap-DFELqQu0Qw2hh6i8cKIIws",
                "gpu_target": "A100-80GB",
                "created_at": "2026-04-04 04:09:29+08:00",
                "stopped_at": "2026-04-04 04:12:56+08:00",
                "output_subdir": "zju_source_policy_research_loop/cloud_runs/20260404_teacher_fixed_visual_lift_smoke_v2",
                "output_root_materialized": False,
                "heartbeat_materialized": False,
                "stdout_materialized": False,
                "reason": "The smoke app stayed in scheduler-side pending/empty-task behavior and never reached the remote launch-artifact write step.",
            },
            {
                "app_id": "ap-HJdhi457AMoqI99PlmMuVC",
                "gpu_target": "A10G",
                "created_at": "2026-04-04 04:17:03+08:00",
                "stopped_at": "2026-04-04 04:19:04+08:00",
                "output_subdir": "zju_source_policy_research_loop/cloud_runs/20260404_teacher_fixed_visual_lift_smoke_a10g_v1",
                "output_root_materialized": False,
                "heartbeat_materialized": False,
                "stdout_materialized": False,
                "reason": "Changing GPU tier did not break the scheduler-side pending/empty-task behavior; the smoke app still never reached the remote launch-artifact write step.",
            },
        ],
        "active_modal_app_count_after_cleanup": 0,
        "repo_process_allowlist_empty": True,
        "second_full_benchmark_run_opened": False,
        "hard_blocker_reason": "The cloud execution contract is locally repaired and passes rehearsal, but the actual smoke cloud runs never start remote execution on Modal, so no output root or heartbeat is ever materialized.",
    }
    _write_json(smoke_failure_json, failure_payload)
    _write_md(smoke_failure_md, json.dumps(failure_payload, ensure_ascii=False, indent=2))

    result_payload = {
        "checked_at": checked_at,
        "family": family,
        "status": "hard_blocker_before_cloud_deliverable_completion",
        "contract_audit_passed": True,
        "local_benchmark_passed": True,
        "selected_variant": "mask_hole_fill_plus_guided",
        "full_benchmark_cloud_run_opened": False,
        "smoke_failure_packet": str(smoke_failure_json.relative_to(REPO_ROOT)),
    }
    _write_json(result_json, result_payload)
    _write_md(result_md, json.dumps(result_payload, ensure_ascii=False, indent=2))

    next_draft_payload = {
        "checked_at": checked_at,
        "family": "teacher_fixed_visual_lift_scheduler_audit",
        "status": "fresh_manual_problem_candidate",
        "question": "Why do detached/background Modal submissions for the visual-lift benchmark create ephemeral apps but never reach the remote launch-artifact write step, even after the contract wrapper itself passes local rehearsal?",
        "forbidden_actions": [
            "do not open another cloud smoke run automatically tonight",
            "do not change the frozen teacher or the accepted local variant",
        ],
    }
    _write_json(next_draft_json, next_draft_payload)
    _write_md(next_draft_md, json.dumps(next_draft_payload, ensure_ascii=False, indent=2))

    research = _load_json(RESEARCH_STATUS_JSON)
    research["checked_at"] = checked_at
    research["state"] = "IDLE_GUARD"
    research["reason"] = "teacher_fixed_visual_lift_benchmark passed local acceptance and the cloud contract passed rehearsal, but smoke cloud runs never reached remote execution, so the system returned cleanly to guard."
    research["approved_problem_present"] = False
    research["approved_problem_ready"] = False
    research["current_cloud_blocker"] = failure_payload["hard_blocker_reason"]
    research["next_requirement"] = "Return to IDLE_GUARD. Do not auto-open another smoke/full cloud app. Review the scheduler-audit draft first."
    research["latest_visual_lift_benchmark"] = {
        "family": "teacher_fixed_visual_lift_benchmark",
        "shape": "hybridring_frozen_teacher_decoder_visual_lift_v1",
        "selected_variant": "mask_hole_fill_plus_guided",
        "local_benchmark_passed": True,
        "cloud_contract_audit_passed": True,
        "smoke_cloud_started": False,
        "smoke_failure_packet": str(smoke_failure_json.relative_to(REPO_ROOT)),
    }
    _write_json(RESEARCH_STATUS_JSON, research)

    frontier = _load_json(FRONTIER_JSON)
    frontier["checked_at"] = checked_at
    family_readout = dict(frontier.get("family_readout", {}))
    family_readout["teacher_fixed_visual_lift_benchmark"] = {
        "status": "local_acceptance_passed",
        "stop_reason": "The frozen teacher plus mask_hole_fill_plus_guided variant already passed the local benchmark stack.",
    }
    family_readout[family] = {
        "status": "scheduler_blocked_before_smoke_materialization",
        "stop_reason": failure_payload["hard_blocker_reason"],
    }
    family_readout["teacher_fixed_visual_lift_scheduler_audit"] = {
        "status": "fresh_manual_problem_candidate",
        "stop_reason": "The next honest move is to explain the Modal scheduler-side non-start behavior before opening more cloud runs.",
    }
    frontier["family_readout"] = family_readout
    frontier["recommended_next_families"] = ["teacher_fixed_visual_lift_scheduler_audit"]
    frontier["recommended_family_order"] = ["teacher_fixed_visual_lift_scheduler_audit"]
    frontier["latest_visual_lift_benchmark"] = research["latest_visual_lift_benchmark"]
    _write_json(FRONTIER_JSON, frontier)

    family_stop = _load_json(FAMILY_STOP_JSON)
    family_stop["checked_at"] = checked_at
    latest = dict(family_stop.get("latest_family_outcomes", {}))
    latest[family] = {
        "latest_status": "scheduler_blocked_before_smoke_materialization",
        "selected_variant": "mask_hole_fill_plus_guided",
        "reason": failure_payload["hard_blocker_reason"],
    }
    family_stop["latest_family_outcomes"] = latest
    family_stop["latest_visual_lift_benchmark"] = research["latest_visual_lift_benchmark"]
    _write_json(FAMILY_STOP_JSON, family_stop)

    task_plan = _load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = checked_at
    task_plan["task_mode_status"] = "hard_blocker"
    task_plan["task_mode_focus"] = "teacher_fixed_visual_lift_contract_passed_smoke_scheduler_blocked_clean"
    completed = list(task_plan.get("completed_this_round", []))
    if not any(item.get("id") == "phase_51_teacher_fixed_visual_lift_cloud_contract_audit_passed" for item in completed):
        completed.append(
            {
                "id": "phase_51_teacher_fixed_visual_lift_cloud_contract_audit_passed",
                "status": "completed",
                "details": "Patched the visual-lift cloud contract and verified launch/materialization/heartbeat/failure-preservation locally with success and failure rehearsals.",
            }
        )
    task_plan["completed_this_round"] = completed
    blocked = list(task_plan.get("blocked_tasks", []))
    if not any(item.get("id") == "phase_52_teacher_fixed_visual_lift_smoke_scheduler_blocked" for item in blocked):
        blocked.append(
            {
                "id": "phase_52_teacher_fixed_visual_lift_smoke_scheduler_blocked",
                "status": "blocked_on_cloud_scheduler",
                "details": "Multiple smoke cloud submissions were watched and cleanly stopped after they never reached remote execution or materialized any output root.",
            }
        )
    task_plan["blocked_tasks"] = blocked
    _write_json(TASK_PLAN_JSON, task_plan)
    _write_md(
        TASK_PLAN_MD,
        "\n".join(
            [
                "# Task Plan",
                "",
                "- task_mode_status: `hard_blocker`",
                "- task_mode_focus: `teacher_fixed_visual_lift_contract_passed_smoke_scheduler_blocked_clean`",
                "- local benchmark: passed",
                "- cloud contract audit: passed",
                "- smoke cloud run: blocked before remote execution start",
                "- no active Modal app remains",
                "- repo_process_allowlist remains empty",
            ]
        ),
    )

    _write_md(
        SUMMARY_MD,
        "\n".join(
            [
                "# ZJU Source-Policy Rawpool Status",
                "",
                "- teacher_fixed_visual_lift_benchmark / hybridring_frozen_teacher_decoder_visual_lift_v1 still stands as a strong local visual-lift result with `mask_hole_fill_plus_guided`.",
                "- The cloud contract itself now passes local rehearsal, but smoke cloud runs never reached remote execution and never materialized an output root on Modal.",
                "- Research is back in IDLE_GUARD, no approved problem remains, no active Modal app remains, and the allowlist is empty.",
            ]
        ),
    )

    watch = _load_json(WATCH_JSON)
    watch["checked_at"] = checked_at
    watch["research"]["summary"]["state"] = "IDLE_GUARD"
    watch["research"]["summary"]["approved_problem_present"] = False
    watch["research"]["summary"]["approved_problem_ready"] = False
    watch["research"]["summary"]["manual_action_required"] = False
    watch["research"]["summary"]["manual_action_kind"] = ""
    watch["research"]["summary"]["ready_for_execution"] = False
    watch["research"]["summary"]["current_review_packet"] = str(result_json.relative_to(REPO_ROOT))
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["watch_conclusion"] = "teacher_fixed_visual_lift cloud contract passed rehearsal, smoke cloud runs stayed scheduler-blocked before remote start, no active cloud app remains, and no active local family run is open"
    _write_json(WATCH_JSON, watch)

    print(smoke_failure_json)
    print(result_json)
    print(RESEARCH_STATUS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
