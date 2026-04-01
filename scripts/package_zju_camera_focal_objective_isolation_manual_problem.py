import argparse
import json
from datetime import datetime
from pathlib import Path


FAMILY = "camera_focal_objective_isolation"
FIRST_CANDIDATE_SHAPE = "fl_only_camera_objective_isolation_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the camera_focal_objective_isolation family into manual-review-only artifacts."
    )
    parser.add_argument("--research-status-json", type=Path, required=True)
    parser.add_argument("--localization-json", type=Path, required=True)
    parser.add_argument("--object-matrix-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path, required=True)
    parser.add_argument("--blueprint-json", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-md", type=Path, required=True)
    parser.add_argument("--draft-json", type=Path, required=True)
    parser.add_argument("--draft-md", type=Path, required=True)
    return parser.parse_args()


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


def build_seed(localization: dict, failure: dict) -> dict:
    return {
        "approved": False,
        "approved_at": "",
        "problem_id": "camera_focal_objective_isolation_v1",
        "problem_title": "Camera/focal objective isolation after the two-stage dead_same_day result",
        "family": FAMILY,
        "family_options_allowed": [],
        "preferred_first_family": FAMILY,
        "preferred_first_family_reason": (
            "This is the preferred next manual-review family after the two-stage failure, but it remains packaging-only "
            "and must not be armed or auto-opened because ready_for_execution=false."
        ),
        "problem_statement": (
            "Define exactly one new manual-only problem that isolates the FL-dominant camera tax as an objective-layer "
            "issue after depth lock, without reopening tail-contract derivative tickets or launching training."
        ),
        "why_genuinely_new": (
            "This reframes the question at the object layer: the target is no longer another family cousin or another "
            "scalar schedule, but a bounded isolation of the camera/focal objective itself."
        ),
        "why_not_reopening_frozen_family": (
            "The new problem is explicitly outside the closed tail-contract derivative batch and outside the failed "
            "two-stage schedule family. It asks for objective isolation evidence, not another ticket."
        ),
        "first_candidate_hint": (
            "Keep the first shape at fl_only_camera_objective_isolation_audit, but do not define or materialize an "
            "execution config today. First decide during manual review whether the isolation should land as a loss-only "
            "audit, a train/val logging patch, or a minimal config-controlled objective fork."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [],
        "first_candidate_knobs": {
            "target_component": localization.get("dominant_component", "loss_FL"),
            "target_scope": localization.get("most_supported_scope", "global"),
            "execution_disabled_until_manual_review": True,
            "no_candidate_yaml_materialized_today": True,
        },
        "historical_prior": failure.get("conclusion", ""),
        "avoid_patterns": [
            "arm approved problem",
            "run candidate",
            "cloud action",
            "tail-contract derivative reopen",
            "same-night cousin sweep",
            "execution-ready config materialization",
        ],
        "max_approved_problems_per_night": 1,
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "requires_dataset_or_routing_change": False,
        "requires_supervision_audit": True,
        "mutation_dsl": {
            "allow_camera_focal_objective_isolation": True,
            "require_manual_review_only": True,
            "disallow_execution_ready_without_fresh_manual_decision": True,
            "disallow_arm_now": True,
            "disallow_run_now": True,
            "disallow_cloud": True,
            "disallow_tail_contract_derivative_reopen": True,
            "disallow_two_stage_same_family_retry": True,
        },
    }


def render_draft_md(draft: dict) -> str:
    lines = [
        "# Camera Focal Objective Isolation Draft (2026-03-30)",
        "",
        f"- family: `{draft['family']}`",
        f"- status: `{draft['status']}`",
        f"- first_candidate_shape: `{draft['first_candidate_shape']}`",
        f"- candidate_config: `{draft['candidate_config']}`",
        f"- ready_for_manual_review: `{draft['ready_for_manual_review']}`",
        f"- ready_for_execution: `{draft['ready_for_execution']}`",
        "",
        "## Why This Family",
        "",
    ]
    lines.extend([f"- {item}" for item in draft["why_now"]])
    return "\n".join(lines).rstrip() + "\n"


def render_plan_md(plan: dict) -> str:
    lines = [
        "# Candidate Patch Plan: Camera Focal Objective Isolation",
        "",
        f"- checked_at: `{plan['checked_at']}`",
        f"- state: `{plan['state']}`",
        f"- approved_problem_present: `{plan['approved_problem_present']}`",
        f"- current_stable_lead_config: `{plan['current_stable_lead_config']}`",
        "",
        "## Candidate",
        "",
        f"- family: `{plan['family']}`",
        f"- first_candidate_shape: `{plan['first_candidate_shape']}`",
        f"- first_candidate_config: `{plan['first_candidate_config']}`",
        f"- arm_command: `{plan['arm_command']}`",
        f"- run_command: `{plan['run_command']}`",
        "",
        "## Guard Contract",
        "",
        f"- do_not_arm_now: `{plan['do_not_arm_now']}`",
        f"- do_not_run_candidate_now: `{plan['do_not_run_candidate_now']}`",
        f"- cloud_must_remain_off: `{plan['cloud_must_remain_off']}`",
        f"- same_night_second_candidate_forbidden: `{plan['same_night_second_candidate_forbidden']}`",
        f"- same_night_cousin_sweep_forbidden: `{plan['same_night_cousin_sweep_forbidden']}`",
        "",
        "## Readiness",
        "",
        f"- readiness: `{plan['readiness']}`",
        "",
        "## Write Surface",
        "",
    ]
    lines.extend([f"- `{item}`" for item in plan["write_surface"]])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    research_status = load_json(args.research_status_json)
    localization = load_json(args.localization_json)
    failure = load_json(args.failure_json)

    seed = build_seed(localization, failure)
    blueprint = {
        "checked_at": now_iso(),
        "family": FAMILY,
        "status": "manual_review_only",
        "ready_for_manual_approval": True,
        "ready_for_execution": False,
        "why_now": (
            "Two-stage objective decoupling failed, and the daytime audit localizes the residual tax to an early "
            "FL-dominant global objective pattern rather than to a single stream-local focal underweight."
        ),
        "reference_evidence": [
            str(args.localization_json).replace("\\", "/"),
            str(args.object_matrix_json).replace("\\", "/"),
            str(args.failure_json).replace("\\", "/"),
        ],
        "signal_definition": (
            "Treat the next question as objective isolation only: define how to separate FL-dominant camera pressure "
            "from the depth-win regime before any later execution-ready ticket is considered."
        ),
        "scope_definition": (
            "manual review only; do not materialize candidate yaml, do not authorize arm/run, and do not reopen "
            "tail-contract derivative cousins."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [],
        "first_candidate_execution_note": (
            "Execution is intentionally disabled today. The first candidate remains a named placeholder until manual "
            "review decides what concrete implementation surface is honest."
        ),
        "required_exclusions": [
            "not execution-ready",
            "not tail-contract derivative reopen",
            "not two-stage retry",
            "not cloud",
        ],
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "This artifact is review-only. No same-night execution is allowed.",
        },
        "cloud_must_remain_off": True,
    }
    draft = {
        "checked_at": now_iso(),
        "draft_kind": "new_manual_problem",
        "status": "ready_for_manual_review",
        "family": FAMILY,
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "candidate_config": "",
        "ready_for_manual_review": True,
        "ready_for_execution": False,
        "requires_new_manual_approval": True,
        "do_not_auto_open_ticket": True,
        "why_now": [
            "Two-stage objective decoupling is now formally closed as dead_same_day.",
            "The daytime audit localizes the early camera tax to a loss_FL-dominant global objective pattern rather than a single default/hardtail/reserve underweight.",
            "The next honest move is a manual-only camera/focal objective isolation problem, not another automatic training ticket.",
        ],
        "readiness_artifacts": [
            str(args.localization_json).replace("\\", "/"),
            str(args.object_matrix_json).replace("\\", "/"),
            str(args.failure_json).replace("\\", "/"),
        ],
        "hypothesis": (
            "If a later family is justified, it should first isolate the FL-only camera objective after depth lock rather than rely on another scalar schedule or another tail cousin."
        ),
    }
    plan = {
        "checked_at": now_iso(),
        "state": str(research_status.get("state", "")),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": str(research_status.get("current_stable_lead_config", "")),
        "family": FAMILY,
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": "",
        "arm_command": "",
        "run_command": "",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": False,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "write_surface": [
            "scripts/audit_zju_early_fl_tax_localization_daytime_20260330.py",
            "output/zju_source_policy_research_loop/early_fl_tax_localization.20260330.json",
            "output/zju_source_policy_research_loop/fl_tax_object_alignment_matrix.20260330.json",
            "output/zju_source_policy_research_loop/two_stage_failure_interpretation.20260330.json",
        ],
    }

    write_json(args.seed_json, seed)
    write_json(args.blueprint_json, blueprint)
    write_json(args.plan_json, plan)
    write_text(args.plan_md, render_plan_md(plan))
    write_json(args.draft_json, draft)
    write_text(args.draft_md, render_draft_md(draft))
    print(args.seed_json)
    print(args.blueprint_json)
    print(args.plan_json)
    print(args.draft_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
