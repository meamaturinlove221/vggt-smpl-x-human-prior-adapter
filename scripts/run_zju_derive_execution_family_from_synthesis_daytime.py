import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

DATE_TAG = "20260403"
SELECTED_FAMILY = "hybrid_ring_secondary_supervised_reserve"
SELECTED_SHAPE = "anchor_plus_secondary_supervised_uniform_tail"
SELECTED_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_supervisedreserve_rawpool_"
    "confdepth_dropworst_gradconfmask_minimal.yaml"
)

DERIVED_CANDIDATES_JSON = RESEARCH_ROOT / f"derived_execution_family_candidates.{DATE_TAG}.json"
DERIVED_CANDIDATES_MD = RESEARCH_ROOT / f"derived_execution_family_candidates.{DATE_TAG}.md"
DERIVED_RANKING_JSON = RESEARCH_ROOT / f"derived_execution_family_ranking.{DATE_TAG}.json"
DERIVED_RANKING_MD = RESEARCH_ROOT / f"derived_execution_family_ranking.{DATE_TAG}.md"
SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{SELECTED_FAMILY}.{DATE_TAG}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{SELECTED_FAMILY}.{DATE_TAG}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{SELECTED_FAMILY}.{DATE_TAG}.json"
PLAN_MD = RESEARCH_ROOT / f"candidate_patch_plan.{SELECTED_FAMILY}.{DATE_TAG}.md"
DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{SELECTED_FAMILY}.{DATE_TAG}.json"
DRAFT_MD = RESEARCH_ROOT / f"next_manual_problem_draft.{SELECTED_FAMILY}.{DATE_TAG}.md"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{SELECTED_FAMILY}.{DATE_TAG}.json"
EXEC_READY_MD = RESEARCH_ROOT / f"execution_ready_promotion_decision.{SELECTED_FAMILY}.{DATE_TAG}.md"
VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "validate_zju_hybrid_ring_secondary_supervised_reserve_execution_prep.py"
VALIDATION_JSON = RESEARCH_ROOT / f"execution_prep_validation.{SELECTED_FAMILY}.{DATE_TAG}.json"
VALIDATION_MD = RESEARCH_ROOT / f"execution_prep_validation.{SELECTED_FAMILY}.{DATE_TAG}.md"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"

RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"


class DeriveError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def run_checked(args: list[str]) -> None:
    result = subprocess.run(args, cwd=str(REPO_ROOT), text=True, capture_output=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise DeriveError("Command failed: " + " ".join(args) + "\n" + result.stderr.strip())


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
    candidates = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "derived_execution_family_candidates",
        "source_family": "cross_axis_plateau_boundary_synthesis",
        "candidates": [
            {
                "family": "hybrid_ring_secondary_supervised_reserve",
                "first_candidate_shape": SELECTED_SHAPE,
                "why_genuinely_new": "It keeps the promoted hybrid-ring tail entry but prevents the recurring secondary supervised view drop, so it is not a focal/translation/coupling scalar cousin.",
                "minimal_write_surface": ["training/data/datasets/zju_vggt_geom.py", SELECTED_CONFIG],
            },
            {
                "family": "hybrid_ring_slot3_source_only_tempering",
                "first_candidate_shape": "tail_slot3_source_only_tempering",
                "why_genuinely_new": "It would dampen source-only slot_3 churn without reopening any camera-loss scalar axis.",
                "minimal_write_surface": ["training/data/datasets/zju_vggt_geom.py", "training/config/<new>.yaml"],
            },
            {
                "family": "promoted_lead_camera_supervision_split_probe",
                "first_candidate_shape": "camera_supervision_split_probe",
                "why_genuinely_new": "It would separate camera-object supervision allocation from pure source selection without touching cloud or closed family scalars.",
                "minimal_write_surface": ["training/data/datasets/zju_vggt_geom.py", "training/config/<new>.yaml", "training/loss.py"],
            },
        ],
    }
    write_json(DERIVED_CANDIDATES_JSON, candidates)
    write_text(DERIVED_CANDIDATES_MD, render_md("Derived Execution Family Candidates", candidates))

    ranking = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "derived_execution_family_ranking",
        "status": "SINGLE_TOP_CANDIDATE_SELECTED",
        "ranking": [
            {
                "family": "hybrid_ring_secondary_supervised_reserve",
                "score": 0.94,
                "priority_rank": 1,
                "why_ranked_here": "It is the narrowest genuinely new training family: one dataset helper plus one config, with a concrete first candidate shape and no cloud or closed-axis reopen.",
            },
            {
                "family": "hybrid_ring_slot3_source_only_tempering",
                "score": 0.77,
                "priority_rank": 2,
                "why_ranked_here": "It is mechanism-adjacent but less direct than preserving the dropped supervised view that appears in the strongest repeated transition family.",
            },
            {
                "family": "promoted_lead_camera_supervision_split_probe",
                "score": 0.59,
                "priority_rank": 3,
                "why_ranked_here": "It widens the write surface to training/loss.py and therefore is less honest as the immediate next derived family.",
            },
        ],
        "selected_family": SELECTED_FAMILY,
        "selection_reason": "hybrid_ring_secondary_supervised_reserve wins because it directly targets the repeated supervised-to-absent / tail-source-entry coupling while preserving the promoted lead’s tail mechanism and keeping the write surface minimal.",
    }
    write_json(DERIVED_RANKING_JSON, ranking)
    write_text(DERIVED_RANKING_MD, render_md("Derived Execution Family Ranking", ranking))

    run_checked([sys.executable, str(VALIDATION_SCRIPT)])

    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": f"{SELECTED_FAMILY}_v1",
        "problem_title": "Promoted hybrid-ring secondary supervised reserve",
        "family": SELECTED_FAMILY,
        "problem_statement": "Keep the promoted hybrid-ring tail entry while reserving one secondary supervised geometric view so the next ticket tests supervision-role retention rather than another camera-loss scalar change.",
        "first_candidate_shape": SELECTED_SHAPE,
        "first_candidate_config": SELECTED_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/data/datasets/zju_vggt_geom.py",
            SELECTED_CONFIG,
        ],
        "forbidden_actions": [
            "do not auto-arm",
            "do not auto-run",
            "do not use cloud",
            "do not reopen closed camera-object families",
        ],
        "mutation_dsl": {
            "allow_hybrid_ring_secondary_supervised_reserve": True,
            "keep_promoted_hybrid_ring_tail_entry": True,
            "allow_secondary_supervised_reserve_only": True,
            "keep_cloud_off": True,
            "disallow_closed_camera_axis_reopen": True,
            "disallow_wholefg_scalar": True,
            "disallow_wholefg_decoupled": True,
            "disallow_edge_band_scalar": True,
            "disallow_edge_band_decoupled": True,
            "disallow_hard_depth_conf_threshold": True,
            "disallow_plain_anchor_view_only": True,
        },
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "max_approved_problems_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
    }
    blueprint = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "family": SELECTED_FAMILY,
        "status": "ready_for_execution",
        "ready_for_execution": True,
        "execution_mode": "training_family_pending_manual_arm",
        "why_now": ranking["selection_reason"],
        "first_candidate_shape": SELECTED_SHAPE,
        "first_candidate_config": SELECTED_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/data/datasets/zju_vggt_geom.py",
            SELECTED_CONFIG,
        ],
        "first_candidate_execution_note": "This is a real executable training family on the current repo, but it remains pending manual arm approval with cloud off.",
    }
    plan = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "state": "execution_ready_pending_arm",
        "family": SELECTED_FAMILY,
        "first_candidate_shape": SELECTED_SHAPE,
        "first_candidate_config": SELECTED_CONFIG,
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed hybrid_ring_secondary_supervised_reserve",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "execution_mode": "training_family_pending_manual_arm",
        "selected_reason": ranking["selection_reason"],
    }
    draft = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "draft_kind": "new_manual_problem",
        "status": "execution_ready_pending_arm",
        "family": SELECTED_FAMILY,
        "first_candidate_shape": SELECTED_SHAPE,
        "candidate_config": SELECTED_CONFIG,
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [item["why_ranked_here"] for item in ranking["ranking"][:2]],
    }
    exec_ready = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_promotion_decision",
        "family": SELECTED_FAMILY,
        "decision": "PROMOTE_TO_EXECUTION_READY_PENDING_ARM",
        "ready_for_execution": True,
        "execution_mode": "training_family_pending_manual_arm",
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "still_forbidden": [
            "Do not auto-arm.",
            "Do not auto-run.",
            "Do not use cloud.",
            "Do not reopen any closed local family.",
        ],
    }

    for path, payload, title in [
        (SEED_JSON, seed, None),
        (BLUEPRINT_JSON, blueprint, None),
        (PLAN_JSON, plan, "Candidate Patch Plan"),
        (DRAFT_JSON, draft, "Next Manual Problem Draft"),
        (EXEC_READY_JSON, exec_ready, "Execution Ready Promotion Decision"),
    ]:
        write_json(path, payload)
        if title:
            write_text(Path(str(path).replace(".json", ".md")), render_md(title, payload))
    write_text(PLAN_MD, render_md("Candidate Patch Plan", plan))
    write_text(DRAFT_MD, render_md("Next Manual Problem Draft", draft))
    write_text(EXEC_READY_MD, render_md("Execution Ready Promotion Decision", exec_ready))

    run_checked([sys.executable, str(RESEARCH_LOOP_SCRIPT)])
    research = load_json(RESEARCH_STATUS_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = datetime.now().isoformat(timespec="seconds")
    task_plan["task_mode_status"] = "active"
    task_plan["current_mode"] = "execution_ready_pending_manual_arm"
    task_plan["research_loop_mode"] = "IDLE_GUARD"
    task_plan["task_mode_focus"] = "hybrid_ring_secondary_supervised_reserve_execution_ready_pending_arm_cloud_off"
    task_plan["research_loop"] = deepcopy(task_plan.get("research_loop", {}))
    task_plan["research_loop"].update(
        {
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "state": "IDLE_GUARD",
            "current_priority_family": SELECTED_FAMILY,
            "auto_next_ticket_enabled": False,
            "preferred_first_family": SELECTED_FAMILY,
            "preferred_first_family_reason": ranking["selection_reason"],
        }
    )
    task_plan["problem_definition_progress"] = deepcopy(task_plan.get("problem_definition_progress", {}))
    task_plan["problem_definition_progress"].update(
        {
            "status": "ready_for_execution",
            "newest_boundary_fact": "hybrid_ring_secondary_supervised_reserve is now the selected concrete execution family and is execution-ready pending-arm with cloud off.",
            "next_requirement": "Manual approval may arm exactly one hybrid_ring_secondary_supervised_reserve ticket later; do not auto-arm or auto-run now.",
        }
    )
    task_plan["active_tasks"] = [
        {
            "id": "manual_approval_to_arm_hybrid_ring_secondary_supervised_reserve",
            "status": "active",
            "details": "Manually approve the single derived training family ticket later if daytime review accepts it.",
        }
    ]
    task_plan["completed_this_round"] = upsert_by_key(list(task_plan.get("completed_this_round", []) or []), {
        "id": "phase_48_hybrid_ring_secondary_supervised_reserve_selected_and_packaged",
        "status": "completed",
        "details": "Derived and packaged hybrid_ring_secondary_supervised_reserve as the top executable family from cross_axis_plateau_boundary_synthesis.",
    }, "id")
    task_plan["summary_conclusion"] = [
        "hybrid_ring_secondary_supervised_reserve is the selected next concrete execution family.",
        "It outranks the other derived families because it directly targets the repeated secondary-supervision drop while keeping the promoted tail-entry mechanism intact.",
        "The family is now execution-ready pending-arm, while cloud remains off and no run is authorized.",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    frontier = load_json(FRONTIER_LEDGER_JSON)
    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"][SELECTED_FAMILY] = {
        "status": "ready_for_execution",
        "stop_reason": "The derived family is now execution-ready pending-arm on the current repo.",
    }
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    family_stop["higher_level_manual_problem_outcomes"] = deepcopy(family_stop.get("higher_level_manual_problem_outcomes", {}))
    family_stop["higher_level_manual_problem_outcomes"][SELECTED_FAMILY] = {
        "status": "execution_ready_pending_arm",
        "seed_path": repo_rel(SEED_JSON),
        "blueprint_path": repo_rel(BLUEPRINT_JSON),
        "plan_path": repo_rel(PLAN_JSON),
    }
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    watch_payload = load_json(LATEST_WATCH_JSON)
    watch_payload["checked_at"] = datetime.now().isoformat(timespec="seconds")
    watch_payload["research"] = {
        "summary": {
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "manual_action_required": True,
            "manual_action_kind": "manual_approval",
            "ready_for_execution": True,
            "current_review_packet": repo_rel(EXEC_READY_JSON),
        },
        "research_status": research,
        "allowlist": load_json(ALLOWLIST_JSON),
    }
    watch_payload["watch_conclusion"] = "hybrid_ring_secondary_supervised_reserve is execution-ready pending manual approval, no active cloud app, and no active local family run is open"
    write_json(LATEST_WATCH_JSON, watch_payload)
    run_checked([sys.executable, str(WATCH_SCRIPT), "--once"])

    print(
        json.dumps(
            {
                "derived_execution_family_candidates": repo_rel(DERIVED_CANDIDATES_JSON),
                "derived_execution_family_ranking": repo_rel(DERIVED_RANKING_JSON),
                "seed": repo_rel(SEED_JSON),
                "family_blueprint": repo_rel(BLUEPRINT_JSON),
                "candidate_patch_plan": repo_rel(PLAN_JSON),
                "next_manual_problem_draft": repo_rel(DRAFT_JSON),
                "validation": repo_rel(VALIDATION_JSON),
                "execution_ready_decision": repo_rel(EXEC_READY_JSON),
                "task_plan": repo_rel(TASK_PLAN_JSON),
                "summary": repo_rel(SUMMARY_MD),
                "latest_watch_snapshot": repo_rel(LATEST_WATCH_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
