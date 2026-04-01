import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

POSTMORTEM_JSON = OUTPUT_ROOT / "camera_focal_vs_translation_postmortem.20260401.json"
VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_translation_objective_isolation.20260401.json"

DECISION_JSON = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_translation_objective_isolation.20260401.json"
DECISION_MD = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_translation_objective_isolation.20260401.md"
SEED_JSON = OUTPUT_ROOT / "approved_problem.seed.camera_translation_objective_isolation.json"
BLUEPRINT_JSON = OUTPUT_ROOT / "family_blueprint.camera_translation_objective_isolation.json"
DRAFT_JSON = OUTPUT_ROOT / "next_manual_problem_draft.camera_translation_objective_isolation.20260401.json"
DRAFT_MD = OUTPUT_ROOT / "next_manual_problem_draft.camera_translation_objective_isolation.20260401.md"
PLAN_JSON = OUTPUT_ROOT / "candidate_patch_plan.camera_translation_objective_isolation.json"
PLAN_MD = OUTPUT_ROOT / "candidate_patch_plan.camera_translation_objective_isolation.md"

CANDIDATE_SHAPE = "stablelead_global_losst_isolation0"
CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_"
    "confdepth_dropworst_gradconfmask_losstisolation0_minimal.yaml"
)
NEXT_REQUIREMENT = "manual_approval_to_arm_execution_ready_camera_translation_objective_isolation"


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
        "family": "camera_translation_objective_isolation",
        "decision": "PROMOTE_TO_EXECUTION_READY",
        "ready_for_execution": True,
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "promotion_scope": (
            "This decision promotes camera_translation_objective_isolation directly to execution-ready pending manual arm approval. "
            "It authorizes one concrete local candidate while still forbidding cloud, second tickets, and cousin sweeps."
        ),
        "why_this_candidate": [
            "The focal-isolation ticket already produced the key discriminator: camera improved, but T regressed and depth stayed flat.",
            "The postmortem formalizes that the remaining gate blocker shifted from broad camera tax to translation-specific pressure.",
            "The reviewed loss_t_isolation_scale hook is already locally validated with identity default 1.0 and isolated camera-total behavior.",
            "Setting loss_t_isolation_scale=0.0 is the narrowest direct test of whether translation pressure inside loss_camera is now the blocking term.",
        ],
        "first_allowed_next_step": (
            "Arm exactly one approved camera_translation_objective_isolation ticket with the prebuilt losstisolation0 config, "
            "run the normal local smoke/10x5/100x20 gate path, and keep cloud off."
        ),
        "still_forbidden": [
            "Do not auto-arm an approved problem without an explicit manual approval step.",
            "Do not open a second candidate or cousin sweep.",
            "Do not use cloud or Modal.",
            "Do not reopen focal, tail, source, or bucket families from this ticket.",
        ],
        "supporting_refs": {
            "camera_focal_vs_translation_postmortem": repo_rel(POSTMORTEM_JSON),
            "execution_prep_baseline_validation": repo_rel(VALIDATION_JSON),
            "postmortem_verdict": postmortem.get("verdict", ""),
            "postmortem_gate_stage_reached": postmortem.get("gate_stage_reached", ""),
            "validation_status": validation.get("overall_status", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": "camera_translation_objective_isolation_v1",
        "problem_title": "Camera/translation objective isolation after focal-isolation verdict",
        "family": "camera_translation_objective_isolation",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": "This family is execution-ready and may be armed directly once explicitly approved.",
        "problem_statement": (
            "Launch exactly one execution-ready candidate that isolates the translation contribution inside loss_camera "
            "on top of the current stable lead to test whether T is now the blocking camera-object term."
        ),
        "why_genuinely_new": (
            "This is a new single-variable objective isolation family derived from the focal-isolation postmortem, not a retry of the focal ticket."
        ),
        "why_not_reopening_frozen_family": (
            "It keeps the stable lead fixed and changes only the translation contribution inside loss_camera. "
            "It does not reopen focal, tail-contract, source-policy, or bucket families."
        ),
        "first_candidate_hint": (
            "Launch only the stablelead_global_losst_isolation0 candidate by keeping the stable lead fixed and "
            "setting loss.camera.loss_t_isolation_scale=0.0."
        ),
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "loss.camera.loss_t_isolation_scale": 0.0,
            "keep_current_stable_lead_fixed": True,
        },
        "historical_prior": "camera_focal_objective_isolation dead_same_day still improved camera while T regressed.",
        "avoid_patterns": [
            "camera_focal same-family retry",
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
            "allow_camera_translation_objective_isolation": True,
            "require_existing_loss_t_isolation_hook": True,
            "allow_global_loss_t_isolation_only": True,
            "keep_existing_depth_routing_unchanged": True,
            "disallow_camera_focal_same_family_retry": True,
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
        "family": "camera_translation_objective_isolation",
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The focal-isolation ticket already compressed the camera-object failure down to translation-specific gate regression."
        ),
        "reference_evidence": [
            repo_rel(POSTMORTEM_JSON),
            repo_rel(VALIDATION_JSON),
            repo_rel(DECISION_JSON),
        ],
        "signal_definition": (
            "Keep the current stable lead fixed and isolate the translation contribution inside loss_camera by "
            "setting loss_t_isolation_scale=0.0 while preserving the standalone loss_T readout."
        ),
        "scope_definition": "objective isolation only; do not reopen focal, tail, source, bucket, dataset, or cloud paths.",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": ["training/loss.py", CANDIDATE_CONFIG],
        "first_candidate_execution_note": "This candidate is execution-ready on the current repo and may be armed directly once explicitly approved.",
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single losstisolation0 candidate is eligible under the current contract.",
        },
        "cloud_must_remain_off": True,
        "next_requirement": NEXT_REQUIREMENT,
    }

    draft = {
        "checked_at": now_iso(),
        "draft_kind": "new_manual_problem",
        "status": "execution_ready_pending_arm",
        "family": "camera_translation_objective_isolation",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "candidate_config": CANDIDATE_CONFIG,
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [
            "The focal-isolation ticket already improved camera but failed because T regressed.",
            "The translation-isolation hook is now locally validated on the same single-file loss surface.",
            "The narrowest next discriminator is one prebuilt losstisolation0 candidate on the current stable lead.",
        ],
        "readiness_artifact": str(DECISION_JSON.resolve()),
        "hypothesis": (
            "If translation pressure is now the blocking term inside loss_camera, zeroing that T contribution should further "
            "improve the local gate without reopening broader family changes."
        ),
    }

    plan = {
        "checked_at": now_iso(),
        "state": "execution_ready_pending_arm",
        "approved_problem_present": False,
        "current_stable_lead_config": "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml",
        "family": "camera_translation_objective_isolation",
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed camera_translation_objective_isolation",
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
    write_text(DECISION_MD, render_json_md("Execution-Ready Promotion Decision: Camera Translation Objective Isolation", decision))
    write_json(SEED_JSON, seed)
    write_json(BLUEPRINT_JSON, blueprint)
    write_json(DRAFT_JSON, draft)
    write_text(DRAFT_MD, render_json_md("Camera Translation Objective Isolation Draft", draft))
    write_json(PLAN_JSON, plan)
    write_text(PLAN_MD, render_json_md("Camera Translation Objective Isolation Candidate Plan", plan))

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
