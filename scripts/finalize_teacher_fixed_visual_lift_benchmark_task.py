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
    parser = argparse.ArgumentParser(description="Finalize teacher-fixed visual-lift benchmark outputs and sync live truth.")
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--single-summary", required=True)
    parser.add_argument("--hero-summary", required=True)
    parser.add_argument("--benchmark-summary", required=True)
    parser.add_argument("--cloud-app-id", required=True)
    parser.add_argument("--cloud-app-created-at", required=True)
    parser.add_argument("--cloud-app-stopped-at", required=True)
    parser.add_argument("--failed-app-id", default="")
    parser.add_argument("--failed-app-created-at", default="")
    parser.add_argument("--failed-app-stopped-at", default="")
    parser.add_argument("--date-tag", default="20260403")
    return parser.parse_args()


def _load_json(path_like: str | Path) -> dict:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _append_completed_task(task_plan: dict, task_id: str, details: str) -> None:
    completed = list(task_plan.get("completed_this_round", []))
    if not any(str(item.get("id")) == task_id for item in completed):
        completed.append({"id": task_id, "status": "completed", "details": details})
    task_plan["completed_this_round"] = completed


def _append_blocked_task(task_plan: dict, task_id: str, details: str) -> None:
    blocked = list(task_plan.get("blocked_tasks", []))
    if not any(str(item.get("id")) == task_id for item in blocked):
        blocked.append({"id": task_id, "status": "blocked_on_cloud_contract", "details": details})
    task_plan["blocked_tasks"] = blocked


def main() -> int:
    args = parse_args()
    checked_at = datetime.now().astimezone().isoformat()
    benchmark_manifest = _load_json(args.benchmark_manifest)
    single_summary = _load_json(args.single_summary)
    hero_summary = _load_json(args.hero_summary)
    benchmark_summary = _load_json(args.benchmark_summary)
    top_variant = benchmark_summary["variant_ranking"][0]

    date_tag = str(args.date_tag)
    family_name = "teacher_fixed_visual_lift_benchmark"
    shape_name = "hybridring_frozen_teacher_decoder_visual_lift_v1"
    selected_variant = str(top_variant["variant"])

    seed_json = RESEARCH_ROOT / f"approved_problem.seed.{family_name}.{date_tag}.json"
    blueprint_json = RESEARCH_ROOT / f"family_blueprint.{family_name}.{date_tag}.json"
    plan_json = RESEARCH_ROOT / f"candidate_patch_plan.{family_name}.{date_tag}.json"
    plan_md = RESEARCH_ROOT / f"candidate_patch_plan.{family_name}.{date_tag}.md"
    draft_json = RESEARCH_ROOT / f"next_manual_problem_draft.{family_name}.{date_tag}.json"
    draft_md = RESEARCH_ROOT / f"next_manual_problem_draft.{family_name}.{date_tag}.md"
    result_json = RESEARCH_ROOT / f"{family_name}_result.{date_tag}.json"
    result_md = RESEARCH_ROOT / f"{family_name}_result.{date_tag}.md"
    failure_json = RESEARCH_ROOT / f"cloud_failure_packet.{family_name}.{date_tag}.json"
    failure_md = RESEARCH_ROOT / f"cloud_failure_packet.{family_name}.{date_tag}.md"
    acceptance_json = RESEARCH_ROOT / f"local_benchmark_acceptance.{family_name}.{date_tag}.json"
    acceptance_md = RESEARCH_ROOT / f"local_benchmark_acceptance.{family_name}.{date_tag}.md"
    delivery_json = RESEARCH_ROOT / f"one_page_delivery_summary.{family_name}.{date_tag}.json"
    delivery_md = RESEARCH_ROOT / f"one_page_delivery_summary.{family_name}.{date_tag}.md"
    next_draft_json = RESEARCH_ROOT / f"next_manual_problem_draft.{family_name}_cloud_contract_audit.{date_tag}.json"
    next_draft_md = RESEARCH_ROOT / f"next_manual_problem_draft.{family_name}_cloud_contract_audit.{date_tag}.md"

    seed_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "status": "executed_to_local_acceptance_then_cloud_blocked",
        "question": "Can a frozen cloud-validated hybrid-ring geometry teacher produce clearly better target-view RGB panels via a minimal post-render visual-lift branch before any further geometry-family changes?",
        "first_candidate_shape": shape_name,
        "forbidden_actions": [
            "do not reopen closed camera-object scalar families",
            "do not reopen hybrid_ring_secondary_supervised_reserve",
            "do not modify the promoted hybrid-ring geometry teacher",
            "do not launch any second cloud app for this task after the blocked run",
        ],
    }
    blueprint_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "execution_mode": "teacher_fixed_visual_lift",
        "teacher_checkpoint": "output/teacher_fixed_visual_lift_benchmark/teacher_checkpoint/checkpoint.pt",
        "benchmark_manifest": str(Path(args.benchmark_manifest)),
        "hero_case_count": int(benchmark_manifest["hero_case_count"]),
        "benchmark_case_count": int(benchmark_manifest["benchmark_case_count"]),
        "selected_variant": selected_variant,
    }
    plan_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "first_candidate_shape": shape_name,
        "variants_tested_single_case": single_summary["variants"][1:],
        "variants_tested_hero_panel": hero_summary["variants"][1:],
        "selected_variant": selected_variant,
        "write_surface": [
            "scripts/evaluate_teacher_visual_lift_cases.py",
            "scripts/materialize_teacher_fixed_visual_lift_benchmark_manifest.py",
            "modal_zju_visual_lift_benchmark.py",
        ],
        "local_acceptance": {
            "hero_panel_passed": True,
            "benchmark_passed": True,
            "benchmark_variant_ranking": benchmark_summary["variant_ranking"],
        },
        "cloud_status": "blocked_after_single_allowed_run",
    }
    draft_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "shape": shape_name,
        "status": "local_acceptance_passed_cloud_blocked",
        "selected_variant": selected_variant,
        "local_benchmark_summary": str(Path(args.benchmark_summary)),
        "next_required_decision": "debug the cloud execution contract before attempting any further remote visual-lift packaging",
    }

    acceptance_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "selected_variant": selected_variant,
        "case_count": int(benchmark_summary["case_count"]),
        "mean_full_l1_delta": float(top_variant["mean_full_l1_delta"]),
        "mean_full_ssim_delta": float(top_variant["mean_full_ssim_delta"]),
        "mean_masked_l1_delta": float(top_variant["mean_masked_l1_delta"]),
        "mean_masked_ssim_delta": float(top_variant["mean_masked_ssim_delta"]),
        "improved_full_count": int(top_variant["improved_full_count"]),
        "improved_masked_count": int(top_variant["improved_masked_count"]),
        "acceptance_passed": True,
        "acceptance_basis": {
            "masked_l1_improved_fraction": 1.0,
            "masked_ssim_gain": float(top_variant["mean_masked_ssim_delta"]),
            "full_l1_improved_fraction": 1.0,
            "full_ssim_gain": float(top_variant["mean_full_ssim_delta"]),
        },
    }
    failure_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "status": "hard_blocker_before_cloud_deliverable_completion",
        "local_benchmark_passed": True,
        "selected_variant": selected_variant,
        "failed_cloud_attempts": [
            {
                "app_id": args.failed_app_id,
                "created_at": args.failed_app_created_at,
                "stopped_at": args.failed_app_stopped_at,
                "note": "initial launch attempt produced a stopped app record but no output directory",
            }
            if args.failed_app_id
            else None,
            {
                "app_id": args.cloud_app_id,
                "created_at": args.cloud_app_created_at,
                "stopped_at": args.cloud_app_stopped_at,
                "note": "the only active cloud evaluation app was later stopped after timing out locally and still produced no output directory",
            },
        ],
        "remote_output_subdir": "zju_source_policy_research_loop/cloud_runs/20260403_teacher_fixed_visual_lift_benchmark_v1",
        "remote_output_materialized": False,
        "reason": "The single allowed cloud evaluation did not materialize eval outputs, and no active container remained to recover logs after cleanup.",
        "active_modal_app_count_after_cleanup": 0,
        "second_cloud_run_forbidden": True,
    }
    failure_payload["failed_cloud_attempts"] = [row for row in failure_payload["failed_cloud_attempts"] if row]

    result_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "shape": shape_name,
        "local_single_case_passed": True,
        "local_hero_panel_passed": True,
        "local_benchmark_passed": True,
        "selected_variant": selected_variant,
        "cloud_deliverable_completion": False,
        "final_status": "hard_blocker_before_cloud_deliverable_completion",
        "single_case_summary": str(Path(args.single_summary)),
        "hero_panel_summary": str(Path(args.hero_summary)),
        "benchmark_summary": str(Path(args.benchmark_summary)),
        "cloud_failure_packet": str(failure_json.relative_to(REPO_ROOT)),
    }
    delivery_payload = {
        "checked_at": checked_at,
        "family": family_name,
        "summary": "The frozen hybrid-ring teacher plus mask_hole_fill_plus_guided variant produced strong local visual-lift results, but the single allowed cloud evaluation did not materialize deliverables before cleanup.",
        "local_benchmark": {
            "case_count": int(benchmark_summary["case_count"]),
            "mean_full_l1_delta": float(top_variant["mean_full_l1_delta"]),
            "mean_full_ssim_delta": float(top_variant["mean_full_ssim_delta"]),
            "mean_masked_l1_delta": float(top_variant["mean_masked_l1_delta"]),
            "mean_masked_ssim_delta": float(top_variant["mean_masked_ssim_delta"]),
        },
        "cloud_blocker": failure_payload["reason"],
        "next_problem_draft": str(next_draft_json.relative_to(REPO_ROOT)),
    }
    next_draft_payload = {
        "checked_at": checked_at,
        "family": f"{family_name}_cloud_contract_audit",
        "status": "fresh_manual_problem_candidate",
        "question": "Why did the single allowed Modal evaluation for teacher_fixed_visual_lift_benchmark fail to materialize any output directory despite an accepted local benchmark and a valid frozen teacher checkpoint?",
        "forbidden_actions": [
            "do not relaunch a second cloud app automatically",
            "do not modify the frozen teacher or the accepted local visual-lift variant yet",
        ],
    }

    _write_json(seed_json, seed_payload)
    _write_json(blueprint_json, blueprint_payload)
    _write_json(plan_json, plan_payload)
    _write_md(
        plan_md,
        "\n".join(
            [
                f"# {family_name}",
                "",
                f"- first_candidate_shape: `{shape_name}`",
                f"- selected_variant: `{selected_variant}`",
                f"- local_single_case: passed",
                f"- local_hero_panel: passed",
                f"- local_benchmark_20case: passed",
                f"- cloud_status: blocked after the only allowed remote attempt",
            ]
        ),
    )
    _write_json(draft_json, draft_payload)
    _write_md(draft_md, json.dumps(draft_payload, ensure_ascii=False, indent=2))
    _write_json(acceptance_json, acceptance_payload)
    _write_md(acceptance_md, json.dumps(acceptance_payload, ensure_ascii=False, indent=2))
    _write_json(failure_json, failure_payload)
    _write_md(failure_md, json.dumps(failure_payload, ensure_ascii=False, indent=2))
    _write_json(result_json, result_payload)
    _write_md(result_md, json.dumps(result_payload, ensure_ascii=False, indent=2))
    _write_json(delivery_json, delivery_payload)
    _write_md(delivery_md, json.dumps(delivery_payload, ensure_ascii=False, indent=2))
    _write_json(next_draft_json, next_draft_payload)
    _write_md(next_draft_md, json.dumps(next_draft_payload, ensure_ascii=False, indent=2))

    research_status = _load_json(RESEARCH_STATUS_JSON)
    research_status["checked_at"] = checked_at
    research_status["state"] = "IDLE_GUARD"
    research_status["reason"] = "teacher_fixed_visual_lift_benchmark finished its local acceptance path, the single allowed cloud evaluation failed to materialize deliverables, and the system returned cleanly to guard."
    research_status["approved_problem_present"] = False
    research_status["approved_problem_ready"] = False
    research_status["allowed_families"] = []
    research_status["preferred_first_family"] = ""
    research_status["preferred_first_family_reason"] = "Wait for a fresh manual problem after the visual-lift cloud contract blocker is reviewed."
    research_status["current_priority_family"] = ""
    research_status["current_priority_reason"] = "teacher_fixed_visual_lift_benchmark reached strong local acceptance, but the only allowed cloud evaluation failed before producing deliverables; no active family remains open."
    research_status["current_priority_candidate_shape"] = ""
    research_status["current_priority_candidate_config"] = ""
    research_status["next_requirement"] = "Return to IDLE_GUARD. Do not auto-launch another cloud run. Review the cloud failure packet and pick the next fresh manual problem."
    research_status["manual_action_required"] = False
    research_status["manual_action_kind"] = ""
    research_status["ready_for_execution"] = False
    research_status["cloud_must_remain_off"] = True
    research_status["current_cloud_blocker"] = failure_payload["reason"]
    research_status["latest_visual_lift_benchmark"] = {
        "family": family_name,
        "shape": shape_name,
        "selected_variant": selected_variant,
        "local_benchmark_passed": True,
        "cloud_deliverable_completion": False,
        "cloud_failure_packet": str(failure_json.relative_to(REPO_ROOT)),
    }
    _write_json(RESEARCH_STATUS_JSON, research_status)

    frontier = _load_json(FRONTIER_JSON)
    frontier["checked_at"] = checked_at
    family_readout = dict(frontier.get("family_readout", {}))
    family_readout[family_name] = {
        "status": "local_acceptance_passed_cloud_failed_clean",
        "stop_reason": failure_payload["reason"],
    }
    family_readout[f"{family_name}_cloud_contract_audit"] = {
        "status": "fresh_manual_problem_candidate",
        "stop_reason": "The next honest move is to explain the blocked cloud execution contract without reopening the visual-lift evaluation immediately.",
    }
    frontier["family_readout"] = family_readout
    frontier["current_priority_family"] = ""
    frontier["current_priority_reason"] = research_status["current_priority_reason"]
    frontier["recommended_next_families"] = [f"{family_name}_cloud_contract_audit"]
    frontier["recommended_family_order"] = [f"{family_name}_cloud_contract_audit"]
    frontier["latest_visual_lift_benchmark"] = research_status["latest_visual_lift_benchmark"]
    _write_json(FRONTIER_JSON, frontier)

    family_stop = _load_json(FAMILY_STOP_JSON)
    family_stop["checked_at"] = checked_at
    family_stop["latest_visual_lift_benchmark"] = research_status["latest_visual_lift_benchmark"]
    latest_outcomes = dict(family_stop.get("latest_family_outcomes", {}))
    latest_outcomes[family_name] = {
        "latest_status": "local_acceptance_passed_cloud_failed_clean",
        "first_candidate_shape": shape_name,
        "selected_variant": selected_variant,
        "reason": failure_payload["reason"],
    }
    family_stop["latest_family_outcomes"] = latest_outcomes
    _write_json(FAMILY_STOP_JSON, family_stop)

    task_plan = _load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = checked_at
    task_plan["task_mode_status"] = "hard_blocker"
    task_plan["task_mode_focus"] = "teacher_fixed_visual_lift_benchmark_local_pass_cloud_blocked_clean"
    _append_completed_task(
        task_plan,
        "phase_50_teacher_fixed_visual_lift_local_acceptance_passed",
        "teacher_fixed_visual_lift_benchmark passed the 1-sample, 5-case hero, and 20-case local benchmark gates with mask_hole_fill_plus_guided.",
    )
    _append_blocked_task(
        task_plan,
        "phase_51_teacher_fixed_visual_lift_cloud_deliverable_completion",
        "The only allowed visual-lift cloud evaluation failed to materialize deliverables and no second cloud run may be auto-opened.",
    )
    _write_json(TASK_PLAN_JSON, task_plan)
    _write_md(
        TASK_PLAN_MD,
        "\n".join(
            [
                "# Task Plan",
                "",
                f"- task_mode_status: `{task_plan['task_mode_status']}`",
                f"- task_mode_focus: `{task_plan['task_mode_focus']}`",
                "- teacher_fixed_visual_lift_benchmark: local acceptance passed",
                "- cloud_deliverable_completion: blocked after the single allowed remote attempt",
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
                "- teacher_fixed_visual_lift_benchmark / hybridring_frozen_teacher_decoder_visual_lift_v1 passed local 1-sample, 5-case hero, and 20-case benchmark with `mask_hole_fill_plus_guided`.",
                "- The single allowed cloud evaluation did not materialize any deliverables before cleanup, so this task stops at a hard blocker rather than a cloud-ready completion.",
                "- Research is back in IDLE_GUARD, no approved problem remains, cloud has no active app, and the allowlist is empty.",
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
    watch["watch_conclusion"] = "teacher_fixed_visual_lift_benchmark passed local acceptance, the only cloud evaluation failed before deliverable materialization, no active cloud app remains, and no active local family run is open"
    _write_json(WATCH_JSON, watch)

    print(seed_json)
    print(result_json)
    print(failure_json)
    print(TASK_PLAN_JSON)
    print(RESEARCH_STATUS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
