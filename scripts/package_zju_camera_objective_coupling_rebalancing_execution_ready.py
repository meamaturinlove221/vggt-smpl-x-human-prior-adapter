import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

POSTMORTEM_JSON = OUTPUT_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"
VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_objective_coupling_rebalancing.20260402.json"

DECISION_JSON = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_objective_coupling_rebalancing.20260402.json"
DECISION_MD = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_objective_coupling_rebalancing.20260402.md"
SEED_JSON = OUTPUT_ROOT / "approved_problem.seed.camera_objective_coupling_rebalancing.json"
BLUEPRINT_JSON = OUTPUT_ROOT / "family_blueprint.camera_objective_coupling_rebalancing.json"
DRAFT_JSON = OUTPUT_ROOT / "next_manual_problem_draft.camera_objective_coupling_rebalancing.20260402.json"
DRAFT_MD = OUTPUT_ROOT / "next_manual_problem_draft.camera_objective_coupling_rebalancing.20260402.md"
PLAN_JSON = OUTPUT_ROOT / "candidate_patch_plan.camera_objective_coupling_rebalancing.json"
PLAN_MD = OUTPUT_ROOT / "candidate_patch_plan.camera_objective_coupling_rebalancing.md"

CANDIDATE_SHAPE = "stablelead_global_fl_relief090_t_boost105"
CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_"
    "confdepth_dropworst_gradconfmask_flrelief090_tboost105_minimal.yaml"
)
NEXT_REQUIREMENT = "manual_approval_to_arm_execution_ready_camera_objective_coupling_rebalancing"


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


def render_json_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    postmortem = load_json(POSTMORTEM_JSON)
    validation = load_json(VALIDATION_JSON)

    decision = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_promotion_decision",
        "family": "camera_objective_coupling_rebalancing",
        "decision": "PROMOTE_TO_EXECUTION_READY",
        "ready_for_execution": True,
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "promotion_scope": (
            "This decision promotes camera_objective_coupling_rebalancing directly to execution-ready pending manual arm approval. "
            "It authorizes one concrete local candidate while still forbidding cloud, second tickets, and cousin sweeps."
        ),
        "why_this_candidate": [
            "Both focal-only and translation-only isolation tickets already closed as dead_same_day, so the single-subobjective axis is closed for this round.",
            "The new question is no longer whether FL or T alone is toxic, but whether a bounded FL relief plus T boost can rebalance the camera object on the same stable lead.",
            "The reviewed coupling knobs are already locally validated with identity defaults and joint total-camera-only behavior.",
            "The 0.90 / 1.05 setting is the most conservative genuinely new coupled discriminator before escalating to a higher-level camera-depth coupling problem.",
        ],
        "first_allowed_next_step": (
            "Arm exactly one approved camera_objective_coupling_rebalancing ticket with the prebuilt flrelief090_tboost105 config, "
            "run the normal local smoke/10x5/100x20 gate path, and keep cloud off."
        ),
        "still_forbidden": [
            "Do not auto-arm an approved problem without an explicit manual approval step.",
            "Do not open a second candidate or cousin sweep.",
            "Do not use cloud or Modal.",
            "Do not reopen focal-only, translation-only, tail, source, or bucket families.",
        ],
        "supporting_refs": {
            "camera_subobjective_isolation_postmortem": repo_rel(POSTMORTEM_JSON),
            "execution_prep_baseline_validation": repo_rel(VALIDATION_JSON),
            "postmortem_next_problem": postmortem.get("next_problem", ""),
            "validation_status": validation.get("overall_status", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": "camera_objective_coupling_rebalancing_v1",
        "problem_title": "Camera objective coupling rebalancing after subobjective isolation failures",
        "family": "camera_objective_coupling_rebalancing",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": "This family is execution-ready and may be armed directly once explicitly approved.",
        "problem_statement": (
            "Launch exactly one execution-ready candidate that jointly relieves FL pressure and boosts T pressure inside loss_camera "
            "on top of the current stable lead."
        ),
        "why_genuinely_new": (
            "This is a coupled FL/T rebalance family, not a retry of focal-only or translation-only isolation."
        ),
        "why_not_reopening_frozen_family": (
            "It keeps the stable lead fixed and changes only the joint FL/T camera-object balance. "
            "It does not reopen focal-only, translation-only, tail-contract, source-policy, or bucket families."
        ),
        "first_candidate_hint": (
            "Launch only the stablelead_global_fl_relief090_t_boost105 candidate by keeping the stable lead fixed, "
            "setting loss.camera.loss_fl_isolation_scale=0.90, and loss.camera.loss_t_isolation_scale=1.05."
        ),
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "loss.camera.loss_fl_isolation_scale": 0.90,
            "loss.camera.loss_t_isolation_scale": 1.05,
            "keep_current_stable_lead_fixed": True,
        },
        "historical_prior": "Both focal-only and translation-only camera subobjective isolation tickets closed as dead_same_day.",
        "avoid_patterns": [
            "camera_focal same-family retry",
            "camera_translation same-family retry",
            "tail-contract derivative reopen",
            "source-policy reopen",
            "bucket cousin sweep",
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
            "allow_camera_objective_coupling_rebalancing": True,
            "require_existing_loss_fl_isolation_hook": True,
            "require_existing_loss_t_isolation_hook": True,
            "allow_joint_fl_relief_and_t_boost_only": True,
            "keep_existing_depth_routing_unchanged": True,
            "disallow_camera_focal_same_family_retry": True,
            "disallow_camera_translation_same_family_retry": True,
            "disallow_tail_contract_derivative_reopen": True,
            "disallow_cloud": True,
            "disallow_wholefg_scalar": True,
            "disallow_wholefg_decoupled": True,
            "disallow_edge_band_scalar": True,
            "disallow_edge_band_decoupled": True,
            "disallow_hard_depth_conf_threshold": True,
            "disallow_plain_anchor_view_only": True,
        },
    }

    blueprint = {
        "checked_at": now_iso(),
        "family": "camera_objective_coupling_rebalancing",
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": "Single-subobjective isolation is now closed for this round, so the next honest question is a bounded FL/T coupling rebalance on the stable lead.",
        "reference_evidence": [
            repo_rel(POSTMORTEM_JSON),
            repo_rel(VALIDATION_JSON),
            repo_rel(DECISION_JSON),
        ],
        "signal_definition": "Keep the current stable lead fixed and jointly relieve FL pressure to 0.90 while boosting T pressure to 1.05 inside loss_camera.",
        "scope_definition": "camera-object coupling only; do not reopen focal-only, translation-only, tail, source, bucket, dataset, or cloud paths.",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": ["training/loss.py", CANDIDATE_CONFIG],
        "first_candidate_execution_note": "This candidate is execution-ready on the current repo and may be armed directly once explicitly approved.",
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single FL/T coupling candidate is eligible under the current contract.",
        },
        "cloud_must_remain_off": True,
        "next_requirement": NEXT_REQUIREMENT,
    }

    draft = {
        "checked_at": now_iso(),
        "draft_kind": "new_manual_problem",
        "status": "execution_ready_pending_arm",
        "family": "camera_objective_coupling_rebalancing",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "candidate_config": CANDIDATE_CONFIG,
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [
            "Focal-only isolation already failed.",
            "Translation-only isolation already failed.",
            "The next genuinely new discriminator is one bounded FL/T joint rebalance on the same stable lead.",
        ],
        "readiness_artifact": str(DECISION_JSON.resolve()),
        "hypothesis": (
            "If the residual plateau is caused by FL/T balance rather than either subobjective alone, then a conservative FL relief + T boost "
            "may improve camera and translation together without depth regressions."
        ),
    }

    plan = {
        "checked_at": now_iso(),
        "state": "execution_ready_pending_arm",
        "approved_problem_present": False,
        "current_stable_lead_config": "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml",
        "family": "camera_objective_coupling_rebalancing",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_objective_coupling_rebalancing",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": False,
        "do_not_run_candidate_now": False,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
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
        "write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
        "supporting_review_artifacts": [
            repo_rel(POSTMORTEM_JSON),
            repo_rel(VALIDATION_JSON),
            repo_rel(DECISION_JSON),
        ],
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(DECISION_JSON, decision)
    write_text(DECISION_MD, render_json_md("Execution-Ready Promotion Decision: Camera Objective Coupling Rebalancing", decision))
    write_json(SEED_JSON, seed)
    write_json(BLUEPRINT_JSON, blueprint)
    write_json(DRAFT_JSON, draft)
    write_text(DRAFT_MD, render_json_md("Camera Objective Coupling Rebalancing Draft", draft))
    write_json(PLAN_JSON, plan)
    write_text(PLAN_MD, render_json_md("Camera Objective Coupling Rebalancing Candidate Plan", plan))

    print(
        json.dumps(
            {
                "postmortem": repo_rel(POSTMORTEM_JSON),
                "validation": repo_rel(VALIDATION_JSON),
                "promotion_decision": repo_rel(DECISION_JSON),
                "seed": repo_rel(SEED_JSON),
                "family_blueprint": repo_rel(BLUEPRINT_JSON),
                "draft": repo_rel(DRAFT_JSON),
                "candidate_patch_plan": repo_rel(PLAN_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
