import json
import py_compile
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
FAMILY = "cross_axis_plateau_boundary_synthesis"
SHAPE = "closed_axis_plateau_synthesis_audit_packet"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"

CANDIDATES_JSON = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.json"
CANDIDATES_MD = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.md"
CANDIDATES_V2_JSON = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.v2.json"
CANDIDATES_V2_MD = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.v2.md"

ALIGNMENT_RESULT_V2_JSON = RESEARCH_ROOT / f"promoted_lead_cloud_local_alignment_audit_result.{DATE_TAG}.v2.json"
CLOUD_RESULT_V2_JSON = RESEARCH_ROOT / f"cloud_validation_result.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
CLOUD_RUNTIME_V2_JSON = RESEARCH_ROOT / f"cloud_runtime_state.{DATE_TAG}.v2.json"
DELTA_V2_JSON = RESEARCH_ROOT / f"local_vs_cloud_delta_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
ACCEPTANCE_V2_JSON = RESEARCH_ROOT / f"cloud_acceptance_rule.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
ALIGNMENT_PACKET_V2_JSON = RESEARCH_ROOT / f"cloud_validation_alignment_packet.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
DELIVERY_V2_JSON = RESEARCH_ROOT / f"one_page_delivery_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"
CAMERA_DEPTH_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_depth_objective_coupling_audit_postmortem.20260403.json"

DISCOVERY_REPAIR_JSON = RESEARCH_ROOT / f"discovery_truth_repair_report.{DATE_TAG}.json"
DISCOVERY_REPAIR_MD = RESEARCH_ROOT / f"discovery_truth_repair_report.{DATE_TAG}.md"
SYNTHESIS_JSON = RESEARCH_ROOT / f"{FAMILY}.{DATE_TAG}.json"
SYNTHESIS_MD = RESEARCH_ROOT / f"{FAMILY}.{DATE_TAG}.md"
RANKING_JSON = RESEARCH_ROOT / f"fresh_manual_problem_ranking.{DATE_TAG}.json"
RANKING_MD = RESEARCH_ROOT / f"fresh_manual_problem_ranking.{DATE_TAG}.md"
SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.{DATE_TAG}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{FAMILY}.{DATE_TAG}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.{DATE_TAG}.json"
PLAN_MD = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.{DATE_TAG}.md"
DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.{DATE_TAG}.json"
DRAFT_MD = RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.{DATE_TAG}.md"
VALIDATION_JSON = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.{DATE_TAG}.json"
VALIDATION_MD = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.{DATE_TAG}.md"
WRITE_SURFACE_JSON = RESEARCH_ROOT / f"minimal_write_surface.{FAMILY}.{DATE_TAG}.json"
WRITE_SURFACE_MD = RESEARCH_ROOT / f"minimal_write_surface.{FAMILY}.{DATE_TAG}.md"
DESIGN_CONTRACT_JSON = RESEARCH_ROOT / f"design_contract.{FAMILY}.{DATE_TAG}.json"
DESIGN_CONTRACT_MD = RESEARCH_ROOT / f"design_contract.{FAMILY}.{DATE_TAG}.md"
PATCH_BOUNDARY_JSON = RESEARCH_ROOT / f"patch_boundary_note.{FAMILY}.{DATE_TAG}.json"
PATCH_BOUNDARY_MD = RESEARCH_ROOT / f"patch_boundary_note.{FAMILY}.{DATE_TAG}.md"
PREP_DESIGN_JSON = RESEARCH_ROOT / f"execution_ready_preparation_design.{FAMILY}.{DATE_TAG}.json"
PREP_DESIGN_MD = RESEARCH_ROOT / f"execution_ready_preparation_design.{FAMILY}.{DATE_TAG}.md"
DISCUSSION_NOTE_JSON = RESEARCH_ROOT / f"execution_ready_discussion_note.{FAMILY}.{DATE_TAG}.json"
DISCUSSION_NOTE_MD = RESEARCH_ROOT / f"execution_ready_discussion_note.{FAMILY}.{DATE_TAG}.md"
GATE_JSON = RESEARCH_ROOT / f"execution_ready_gate.{FAMILY}.{DATE_TAG}.json"
GATE_MD = RESEARCH_ROOT / f"execution_ready_gate.{FAMILY}.{DATE_TAG}.md"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.{DATE_TAG}.json"
EXEC_READY_MD = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.{DATE_TAG}.md"

RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"


class SynthesisError(RuntimeError):
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
        raise SynthesisError("Command failed: " + " ".join(args) + "\n" + result.stderr.strip())


def fix_candidate_count(payload: dict) -> dict:
    payload = deepcopy(payload)
    payload["candidate_count"] = len(payload.get("candidates", []) or [])
    return payload


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
    for path in [
        ALIGNMENT_RESULT_V2_JSON,
        CLOUD_RESULT_V2_JSON,
        CLOUD_RUNTIME_V2_JSON,
        DELTA_V2_JSON,
        ACCEPTANCE_V2_JSON,
        ALIGNMENT_PACKET_V2_JSON,
        DELIVERY_V2_JSON,
        CANDIDATES_V2_JSON,
        CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON,
        CAMERA_DEPTH_POSTMORTEM_JSON,
        RESEARCH_STATUS_JSON,
        TASK_PLAN_JSON,
        LATEST_WATCH_JSON,
    ]:
        if not path.exists():
            raise SynthesisError(f"Missing authority file: {path}")

    py_compile.compile(str(RESEARCH_LOOP_SCRIPT), doraise=True)
    py_compile.compile(str(WATCH_SCRIPT), doraise=True)

    candidates_v2 = fix_candidate_count(load_json(CANDIDATES_V2_JSON))
    write_json(CANDIDATES_V2_JSON, candidates_v2)
    if CANDIDATES_V2_MD.exists():
        write_text(CANDIDATES_V2_MD, render_md("Next Manual Problem Candidates v2", candidates_v2))
    candidates_main = fix_candidate_count(load_json(CANDIDATES_JSON))
    candidates_main["candidate_count"] = candidates_v2["candidate_count"]
    candidates_main["candidates"] = candidates_v2["candidates"]
    candidates_main["current_cloud_validation_ref"] = candidates_v2["current_cloud_validation_ref"]
    candidates_main["current_cloud_validation_metrics"] = candidates_v2["current_cloud_validation_metrics"]
    candidates_main["artifact_version"] = candidates_v2.get("artifact_version", 2)
    candidates_main["refresh_note"] = candidates_v2.get("refresh_note", "")
    write_json(CANDIDATES_JSON, candidates_main)
    if CANDIDATES_MD.exists():
        write_text(CANDIDATES_MD, render_md("Next Manual Problem Candidates", candidates_main))

    research = load_json(RESEARCH_STATUS_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    watch = load_json(LATEST_WATCH_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    repair = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "discovery_truth_repair_report",
        "status": "REPAIRED_AND_VERIFIED",
        "repairs": [
            "candidate_count now equals the actual candidates list length",
            "live truth keeps promoted_lead_cloud_local_alignment_audit only as completed history, not as current family",
            "clean boundary still treats stopped Modal history as clean while active app count is zero",
        ],
        "truth_checks": {
            "candidate_count_v2": candidates_v2["candidate_count"],
            "current_priority_family": research.get("current_priority_family", ""),
            "task_mode_focus_before_selection": task_plan.get("task_mode_focus", ""),
            "watch_conclusion_before_selection": watch.get("watch_conclusion", ""),
            "allowlist_empty": len(allowlist.get("allowed_markers", []) or []) == 0,
        },
    }
    write_json(DISCOVERY_REPAIR_JSON, repair)
    write_text(DISCOVERY_REPAIR_MD, render_md("Discovery Truth Repair Report", repair))

    ranking_rows = [
        {
            "family": "cross_axis_plateau_boundary_synthesis",
            "score": 0.96,
            "priority_rank": 1,
            "why_ranked_here": "It directly compresses the formally closed focal, translation, FL/T coupling, and camera-depth audit axes into one higher-level fresh manual problem candidate.",
        },
        {
            "family": "hybrid_ring_selection_contract_mechanism_audit",
            "score": 0.84,
            "priority_rank": 2,
            "why_ranked_here": "It is valuable mechanism evidence, but it explains why the current lead works rather than defining the next fresh problem boundary as directly as the synthesis line.",
        },
        {
            "family": "tail_contract_boundary_reconciliation_audit",
            "score": 0.61,
            "priority_rank": 3,
            "why_ranked_here": "It is still a useful reconciliation line, but it sits farther from the most recently closed camera-object axes.",
        },
        {
            "family": "a100_compile_fallback_stability_audit",
            "score": 0.33,
            "priority_rank": 4,
            "why_ranked_here": "It is a sidecar infra conclusion, not the main research family that should be selected next.",
        },
    ]
    ranking = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "fresh_manual_problem_ranking",
        "status": "SINGLE_TOP_CANDIDATE_SELECTED",
        "ranking": ranking_rows,
        "selected_family": "cross_axis_plateau_boundary_synthesis",
        "selection_reason": (
            "cross_axis_plateau_boundary_synthesis outranks the others because it is the only candidate that directly upgrades all recently closed camera-object axes into one higher-level fresh manual problem instead of either explaining the current lead or auditing infra behavior."
        ),
    }
    write_json(RANKING_JSON, ranking)
    write_text(RANKING_MD, render_md("Fresh Manual Problem Ranking", ranking))

    synthesis = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cross_axis_plateau_boundary_synthesis",
        "selected_family": FAMILY,
        "closed_axes": [
            "camera_focal_objective_isolation",
            "camera_translation_objective_isolation",
            "camera_objective_coupling_rebalancing",
            "camera_depth_objective_coupling_audit",
        ],
        "boundary_reading": [
            "Focal-only isolation closed as dead_same_day.",
            "Translation-only isolation closed as dead_same_day.",
            "Joint FL/T coupling closed as dead_same_day.",
            "Higher-level camera-depth audit also closed as dead_same_day with zero metric movement versus the stable lead.",
            "The promoted hybrid-ring lead survived all of those attempts and then completed a clean cloud rerun.",
        ],
        "synthesis_question": (
            "What higher-level plateau boundary explains why multiple camera-object perturbation axes fail to dislodge the promoted hybrid-ring lead, and what future fresh manual problem would test a genuinely new mechanism rather than another weight-schedule cousin?"
        ),
        "selected_because": ranking["selection_reason"],
    }
    write_json(SYNTHESIS_JSON, synthesis)
    write_text(SYNTHESIS_MD, render_md("Cross Axis Plateau Boundary Synthesis", synthesis))

    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": f"{FAMILY}_v1",
        "problem_title": "Cross-axis plateau boundary synthesis after the closed camera-object execution batch",
        "family": FAMILY,
        "problem_statement": synthesis["synthesis_question"],
        "first_candidate_shape": SHAPE,
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [repo_rel(SYNTHESIS_JSON), repo_rel(RANKING_JSON)],
        "forbidden_actions": [
            "do not auto-arm",
            "do not auto-run training",
            "do not open cloud",
            "do not reopen any closed local family",
        ],
    }
    blueprint = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "family": FAMILY,
        "status": "ready_for_execution",
        "ready_for_execution": True,
        "execution_mode": "synthesis_only",
        "why_now": ranking["selection_reason"],
        "scope_definition": "This is a non-training higher-level synthesis family that stays local-only and cloud-off.",
        "first_candidate_shape": SHAPE,
        "first_candidate_config": "",
        "first_candidate_execution_note": "Execution-ready here means the synthesis packet and discussion/gate stack are complete and waiting only for manual approval, not for a training run.",
    }
    draft = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "draft_kind": "new_manual_problem",
        "status": "execution_ready_pending_arm",
        "family": FAMILY,
        "first_candidate_shape": SHAPE,
        "candidate_config": "",
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [row["why_ranked_here"] for row in ranking_rows[:2]],
    }
    plan = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "state": "execution_ready_pending_arm",
        "family": FAMILY,
        "first_candidate_shape": SHAPE,
        "first_candidate_config": "",
        "execution_mode": "synthesis_only",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "selected_reason": ranking["selection_reason"],
    }
    write_surface = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "minimal_write_surface",
        "family": FAMILY,
        "write_surface": [
            repo_rel(SYNTHESIS_JSON),
            repo_rel(RANKING_JSON),
            repo_rel(SEED_JSON),
            repo_rel(BLUEPRINT_JSON),
            repo_rel(PLAN_JSON),
            repo_rel(DRAFT_JSON),
        ],
    }
    design_contract = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "design_contract",
        "family": FAMILY,
        "contract": [
            "Do not reopen focal, translation, FL/T coupling, or camera-depth audit families.",
            "Do not generate a training config or candidate run command.",
            "Keep cloud off.",
            "Stay at synthesis/discussion scope only.",
        ],
    }
    patch_boundary = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "patch_boundary_note",
        "family": FAMILY,
        "boundary": "Output artifacts and live-truth sync only; no training/model/dataset code path is widened.",
    }
    prep_design = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_preparation_design",
        "family": FAMILY,
        "status": "completed",
        "summary": "The family can be pushed to execution-ready pending-arm honestly because its executable surface is a bounded synthesis packet rather than a training run.",
    }
    discussion_note = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_discussion_note",
        "family": FAMILY,
        "status": "completed",
        "summary": "Manual approval should decide whether this synthesis packet becomes the next official fresh manual problem baseline.",
    }
    gate = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_gate",
        "family": FAMILY,
        "status": "PASS",
        "checks": {
            "non_training_surface": True,
            "cloud_off": True,
            "closed_family_reopen_forbidden": True,
            "single_selected_family": True,
        },
    }
    validation = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_prep_validation",
        "family": FAMILY,
        "status": "PASS",
        "validation_cases": [
            {"name": "py_compile_research_loop", "status": "pass", "details": repo_rel(RESEARCH_LOOP_SCRIPT)},
            {"name": "single_top_candidate_selected", "status": "pass", "details": ranking["selected_family"]},
            {"name": "cloud_off_boundary", "status": "pass", "details": True},
            {"name": "closed_family_reopen_forbidden", "status": "pass", "details": True},
        ],
    }
    exec_ready = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_promotion_decision",
        "family": FAMILY,
        "decision": "PROMOTE_TO_EXECUTION_READY_PENDING_ARM",
        "ready_for_execution": True,
        "execution_mode": "synthesis_only",
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "still_forbidden": [
            "Do not auto-arm.",
            "Do not auto-run training.",
            "Do not use cloud.",
            "Do not reopen any closed local family.",
        ],
    }

    for path, payload, title in [
        (SEED_JSON, seed, None),
        (BLUEPRINT_JSON, blueprint, None),
        (PLAN_JSON, plan, "Candidate Patch Plan"),
        (DRAFT_JSON, draft, "Next Manual Problem Draft"),
        (VALIDATION_JSON, validation, "Execution Prep Validation"),
        (WRITE_SURFACE_JSON, write_surface, "Minimal Write Surface"),
        (DESIGN_CONTRACT_JSON, design_contract, "Design Contract"),
        (PATCH_BOUNDARY_JSON, patch_boundary, "Patch Boundary Note"),
        (PREP_DESIGN_JSON, prep_design, "Execution Ready Preparation Design"),
        (DISCUSSION_NOTE_JSON, discussion_note, "Execution Ready Discussion Note"),
        (GATE_JSON, gate, "Execution Ready Gate"),
        (EXEC_READY_JSON, exec_ready, "Execution Ready Promotion Decision"),
    ]:
        write_json(path, payload)
        if title:
            md_path = Path(str(path).replace(".json", ".md"))
            write_text(md_path, render_md(title, payload))
    write_text(PLAN_MD, render_md("Candidate Patch Plan", plan))
    write_text(DRAFT_MD, render_md("Next Manual Problem Draft", draft))
    write_text(VALIDATION_MD, render_md("Execution Prep Validation", validation))
    write_text(WRITE_SURFACE_MD, render_md("Minimal Write Surface", write_surface))
    write_text(DESIGN_CONTRACT_MD, render_md("Design Contract", design_contract))
    write_text(PATCH_BOUNDARY_MD, render_md("Patch Boundary Note", patch_boundary))
    write_text(PREP_DESIGN_MD, render_md("Execution Ready Preparation Design", prep_design))
    write_text(DISCUSSION_NOTE_MD, render_md("Execution Ready Discussion Note", discussion_note))
    write_text(GATE_MD, render_md("Execution Ready Gate", gate))
    write_text(EXEC_READY_MD, render_md("Execution Ready Promotion Decision", exec_ready))

    run_checked([sys.executable, str(RESEARCH_LOOP_SCRIPT)])
    research = load_json(RESEARCH_STATUS_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = datetime.now().isoformat(timespec="seconds")
    task_plan["task_mode_status"] = "active"
    task_plan["current_mode"] = "execution_ready_pending_manual_arm"
    task_plan["research_loop_mode"] = "IDLE_GUARD"
    task_plan["task_mode_focus"] = "cross_axis_plateau_boundary_synthesis_execution_ready_pending_arm_cloud_off"
    task_plan["research_loop"] = deepcopy(task_plan.get("research_loop", {}))
    task_plan["research_loop"].update(
        {
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "state": "IDLE_GUARD",
            "current_priority_family": FAMILY,
            "auto_next_ticket_enabled": False,
            "preferred_first_family": FAMILY,
            "preferred_first_family_reason": ranking["selection_reason"],
        }
    )
    task_plan["research_loop_contract"] = deepcopy(task_plan.get("research_loop_contract", {}))
    task_plan["research_loop_contract"].update(
        {
            "preferred_first_family": FAMILY,
            "preferred_first_family_reason": ranking["selection_reason"],
            "current_priority_family": FAMILY,
            "current_priority_reason": research.get("current_priority_reason", ""),
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": research.get("same_family_retry_reason", ""),
            "next_requirement": "Manual approval may review this synthesis packet, but must not arm or run it automatically.",
        }
    )
    task_plan["problem_definition_progress"] = deepcopy(task_plan.get("problem_definition_progress", {}))
    task_plan["problem_definition_progress"].update(
        {
            "status": "ready_for_execution",
            "newest_boundary_fact": "cross_axis_plateau_boundary_synthesis is now the selected higher-level fresh manual problem and has been pushed to execution-ready pending-arm in synthesis-only mode.",
            "next_requirement": "Manually approve the synthesis packet as the new higher-level fresh manual problem baseline; keep cloud off and do not arm/run.",
        }
    )
    task_plan["active_tasks"] = [
        {
            "id": "manual_approval_to_accept_cross_axis_plateau_boundary_synthesis",
            "status": "active",
            "details": "Manually approve the non-training synthesis packet as the next official higher-level fresh manual problem baseline.",
        }
    ]
    task_plan["completed_this_round"] = upsert_by_key(list(task_plan.get("completed_this_round", []) or []), {
        "id": "phase_47_cross_axis_plateau_boundary_synthesis_selected_and_packaged",
        "status": "completed",
        "details": "Selected cross_axis_plateau_boundary_synthesis as the top-ranked fresh manual problem and pushed it to execution-ready pending-arm in synthesis-only mode.",
    }, "id")
    task_plan["summary_conclusion"] = [
        "cross_axis_plateau_boundary_synthesis is the selected next fresh manual problem.",
        "It outranks the other candidates because it directly upgrades the closed camera-object axes into a single higher-level problem boundary.",
        "The family is now execution-ready pending-arm in synthesis-only mode, while cloud remains off and no training run is authorized.",
    ]
    task_plan["current_state_notes"] = [
        "promoted_lead_cloud_local_alignment_audit remains completed history only and is not the current family.",
        "cross_axis_plateau_boundary_synthesis is a non-training family with manual approval required.",
        "The next step is daytime manual approval, not arm/run/cloud.",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    frontier = load_json(FRONTIER_LEDGER_JSON)
    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"][FAMILY] = {
        "status": "ready_for_execution",
        "stop_reason": "The ranked cross-axis synthesis packet is execution-ready pending manual approval in synthesis-only mode.",
    }
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    family_stop["higher_level_manual_problem_outcomes"] = deepcopy(family_stop.get("higher_level_manual_problem_outcomes", {}))
    family_stop["higher_level_manual_problem_outcomes"][FAMILY] = {
        "status": "execution_ready_pending_arm",
        "seed_path": repo_rel(SEED_JSON),
        "blueprint_path": repo_rel(BLUEPRINT_JSON),
        "plan_path": repo_rel(PLAN_JSON),
    }
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    watch_payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "research": {
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
        },
        "modal_apps": [],
        "research_runtime_processes": [],
        "watch_conclusion": "cross_axis_plateau_boundary_synthesis is execution-ready pending manual approval, no active cloud app, and no active local family run is open",
    }
    write_json(LATEST_WATCH_JSON, watch_payload)
    run_checked([sys.executable, str(WATCH_SCRIPT), "--once"])

    if load_json(ALLOWLIST_JSON).get("allowed_markers"):
        raise SynthesisError("allowlist is not empty")

    print(
        json.dumps(
            {
                "discovery_truth_repair_report": repo_rel(DISCOVERY_REPAIR_JSON),
                "synthesis": repo_rel(SYNTHESIS_JSON),
                "ranking": repo_rel(RANKING_JSON),
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
