import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
DEFAULT_APPROVED_PROBLEM_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.json"
DEFAULT_APPROVED_PROBLEM_TEMPLATE_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.template.json"
DEFAULT_APPROVED_PROBLEM_INTERP_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.interpolated_eligibility_shaping.json"
DEFAULT_APPROVED_PROBLEM_INTERP_SMOOTHSTEP_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.interpolated_eligibility_shaping.smoothstep_taper.json"
DEFAULT_APPROVED_PROBLEM_PARTIAL_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.partial_joint_depth_routing.json"
DEFAULT_APPROVED_PROBLEM_DISAGREEMENT_SEED_PATH = DEFAULT_OUTPUT_ROOT / "approved_problem.seed.conf_reg_disagreement_routing.json"
DEFAULT_APPROVED_PROBLEM_UNPROJECT_CONSISTENCY_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.unproject_consistency_routing.json"
)
DEFAULT_APPROVED_PROBLEM_UNPROJECT_AUX_CONFGATE_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.unproject_aux_confgate.json"
)
DEFAULT_APPROVED_PROBLEM_SOURCE_POLICY_HYBRID_RING_SEED_PATH = (
    DEFAULT_OUTPUT_ROOT / "approved_problem.seed.source_policy_hybrid_ring_regularization.json"
)
DEFAULT_INTERP_BLUEPRINT_PATH = DEFAULT_OUTPUT_ROOT / "family_blueprint.interpolated_eligibility_shaping.json"
DEFAULT_PARTIAL_BLUEPRINT_PATH = DEFAULT_OUTPUT_ROOT / "family_blueprint.partial_joint_depth_routing.json"
DEFAULT_DISAGREEMENT_BLUEPRINT_PATH = DEFAULT_OUTPUT_ROOT / "family_blueprint.conf_reg_disagreement_routing.json"
DEFAULT_UNPROJECT_CONSISTENCY_BLUEPRINT_PATH = (
    DEFAULT_OUTPUT_ROOT / "family_blueprint.unproject_consistency_routing.json"
)
DEFAULT_UNPROJECT_AUX_CONFGATE_BLUEPRINT_PATH = (
    DEFAULT_OUTPUT_ROOT / "family_blueprint.unproject_aux_confgate.json"
)
DEFAULT_SOURCE_POLICY_HYBRID_RING_BLUEPRINT_PATH = (
    DEFAULT_OUTPUT_ROOT / "family_blueprint.source_policy_hybrid_ring_regularization.json"
)
DEFAULT_FRONTIER_LEDGER_PATH = DEFAULT_OUTPUT_ROOT / "frontier_ledger.json"
DEFAULT_CANDIDATE_PATCH_PLAN_PATH = DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.json"
DEFAULT_CANDIDATE_PATCH_PLAN_MD_PATH = DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.md"
DEFAULT_CANDIDATE_VERDICT_PATH = DEFAULT_OUTPUT_ROOT / "candidate_verdict.json"
DEFAULT_FAMILY_STOP_REASON_PATH = DEFAULT_OUTPUT_ROOT / "family_stop_reason.json"
DEFAULT_RESUME_TOKEN_PATH = DEFAULT_OUTPUT_ROOT / "resume_token.json"
DEFAULT_GATE_REFERENCE_LOGS_PATH = DEFAULT_OUTPUT_ROOT / "gate_reference_logs.json"
DEFAULT_APPROVED_PROBLEM_ARCHIVE_ROOT = DEFAULT_OUTPUT_ROOT / "approved_problem_archive"
DEFAULT_REPO_PROCESS_ALLOWLIST_PATH = DEFAULT_OUTPUT_ROOT / "repo_process_allowlist.json"
DEFAULT_REPO_PROCESS_ALLOWLIST_TEMPLATE_PATH = DEFAULT_OUTPUT_ROOT / "repo_process_allowlist.template.json"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_ROOT / "research_loop_status.json"
DEFAULT_STATUS_MD_PATH = DEFAULT_OUTPUT_ROOT / "research_loop_status.md"
DEFAULT_APPROVAL_HELPER_PATH = REPO_ROOT / "scripts" / "arm_zju_source_policy_approved_problem.py"
DEFAULT_APPROVED_RUNNER_PATH = REPO_ROOT / "scripts" / "run_zju_source_policy_research_candidate.py"
DEFAULT_LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
DEFAULT_TRAINING_QUESTION_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
DEFAULT_TASK_PLAN_PATH = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
SHORT_GATE_STABLE_REFERENCE_SUMMARY_PATH = (
    REPO_ROOT
    / "output"
    / "zju_training_ablation"
    / "zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale0875_vs_lead_20260326_v1"
    / "summary.json"
)
SHORT_GATE_BASELINE_REFERENCE_SUMMARY_PATH = (
    REPO_ROOT
    / "output"
    / "zju_training_ablation"
    / "zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfp60jointdepthscale05_vs_baseline_20260326_v1"
    / "summary.json"
)
LONG_GATE_REFERENCE_STATUS_PATH = (
    REPO_ROOT
    / "output"
    / "zju_source_policy_rawpool_long_gate"
    / "20260326_002340_lead_validation_100x20"
    / "status.json"
)

STATE_IDLE_GUARD = "IDLE_GUARD"
STATE_CONTRACT_REJECTED = "CONTRACT_REJECTED"
STATE_ARMED_PROBLEM = "ARMED_PROBLEM"
STATE_SYNTHESIZE_ONE_CANDIDATE = "SYNTHESIZE_ONE_CANDIDATE"
STATE_SMOKE_1X1 = "SMOKE_1x1"
STATE_TIGHT_GATE_10X5 = "TIGHT_GATE_10x5"
STATE_LONG_GATE_100X20 = "LONG_GATE_100x20"
STATE_VERDICT_WRITEBACK = "VERDICT_WRITEBACK"
STATE_RETURN_TO_GUARD = "RETURN_TO_GUARD"

RESEARCH_STATE_MACHINE = [
    STATE_IDLE_GUARD,
    STATE_CONTRACT_REJECTED,
    STATE_ARMED_PROBLEM,
    STATE_SYNTHESIZE_ONE_CANDIDATE,
    STATE_SMOKE_1X1,
    STATE_TIGHT_GATE_10X5,
    STATE_LONG_GATE_100X20,
    STATE_VERDICT_WRITEBACK,
    STATE_RETURN_TO_GUARD,
]

ALLOWED_FAMILIES = [
    "unproject_consistency_routing",
    "unproject_aux_confgate",
    "source_policy_hybrid_ring_regularization",
]

PREFERRED_FIRST_FAMILY = "unproject_consistency_routing"
PREFERRED_FIRST_FAMILY_REASON = (
    "The first interpolated, partial, and disagreement-routing tickets all formally failed at 10/5, so the "
    "next human-approved problem must change the routing signal again rather than micro-tune another "
    "depth-branch-local cousin. unproject_consistency_routing is the next bounded family because it routes "
    "from detached depth-plus-pose geometry inconsistency on anchor-supervised pixels while staying within "
    "the existing loss/batch tensors."
)

HISTORICAL_THRESHOLD_POW2_PRIOR = {
    "doc": "docs/geometry_direction_status_20260323_threshold_and_pow2_completed.md",
    "takeaway": (
        "An earlier geometry-gate line already showed the same pattern: hard thresholding was not viable "
        "and stronger pow-like confidence weighting also moved the objective the wrong way. Combined with "
        "the recent depth_conf<=p60 rejection, this supports preferring smooth interpolated eligibility "
        "tapers over another abrupt threshold or another sharper power-style weighting."
    ),
}

FROZEN_FAMILIES = [
    "wholefg_scalar_near_neighbors",
    "wholefg_decoupled_near_neighbors",
    "edge_band_scalar_near_neighbors",
    "edge_band_decoupled_near_neighbors",
    "hard_pixel_level_depth_conf_threshold",
    "plain_anchor_view_only",
]

ALLOWED_WRITE_SCOPE = [
    "training/loss.py",
    "training/config/*.yaml",
    "training/data/datasets/zju_vggt_geom.py",
    "training/data/composed_dataset.py",
    "scripts/compare_zju_finetune_runs.py",
    "scripts/run_zju_vggt_geom_minimal_finetune.ps1",
    "scripts/run_zju_source_policy_rawpool_long_gate.py",
]

FORBIDDEN_WRITE_SCOPE = [
    "training/models/**",
    "training/model/**",
    "training/networks/**",
    "training/camera/**",
    "training/unproject/**",
    "scripts/run_zju_source_policy_rawpool_guard_daemon.py",
    "scripts/run_zju_source_policy_rawpool_overnight_watch.py",
    "scripts/run_zju_source_policy_rawpool_local_nightly.py",
    "scripts/invoke_modal_zju_cloud_*.ps1",
]

RESEARCH_PROCESS_MARKER_TEMPLATE = [
    "run_zju_source_policy_research_loop.py",
    "run_zju_vggt_geom_minimal_finetune.ps1",
    "run_zju_source_policy_rawpool_long_gate.py",
    "compare_zju_finetune_runs.py",
]

STABLE_LEAD_VAL_METRICS = {
    "camera": 0.0219,
    "T": 0.0003,
    "conf_depth": 0.2288,
    "reg_depth": 0.1759,
}

LATEST_MANUAL_GATE = {
    "candidate": (
        "training/config/"
        "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
        "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfp60jointdepthscale05_minimal.yaml"
    ),
    "verdict": "dead_same_day",
    "interpretation": (
        "The hard gt depth_conf<=p60 threshold inside the promising non-wholefg edge-band route "
        "was too aggressive: camera improved, but both depth terms regressed materially."
    ),
    "val_metrics_stable_lead": STABLE_LEAD_VAL_METRICS,
    "val_metrics_candidate": {
        "camera": 0.0184,
        "T": 0.0003,
        "conf_depth": 0.2411,
        "reg_depth": 0.1837,
    },
}

DEPTH_CONF_P60_MAX = 5.913640410988592
QUALITY_LOW = 2.4610
QUALITY_HIGH = 4.4193
DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfsmoothstepp60jointdepthscale0875_minimal.yaml"
)
DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_SHAPE = "conf_branch_smoothstep_subset"
DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5partialjointconfsmoothstepp60jointdepthscale0875_minimal.yaml"
)
DEFAULT_CONF_REG_DISAGREEMENT_SHAPE = "anchor_disagreement_joint_routing"
DEFAULT_CONF_REG_DISAGREEMENT_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_anchorb1disagreementjointconf0875reg1125_minimal.yaml"
)
DEFAULT_UNPROJECT_CONSISTENCY_SHAPE = "anchor_unproject_consistency_joint_routing"
DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_anchorb1unprojectconsistencyjointconf0875reg1125_minimal.yaml"
)
DEFAULT_UNPROJECT_AUX_CONFGATE_SHAPE = "stablelead_unproject_aux_confgate_w005"
DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_unprojectauxconfgatew005_minimal.yaml"
)
DEFAULT_SOURCE_POLICY_HYBRID_RING_SHAPE = "stablelead_nearest_plus_uniform_tail"
DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_"
    "confdepth_dropworst_gradconfmask_minimal.yaml"
)

DEFAULT_FRONTIER_PROGRESSION = [
    {
        "family": "wholefg_jointdepth_scalar",
        "label": "q>=2.75 + wholefg + joint_depth_scale0.75",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qge275wholefgjointdepthscale075_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "camera": 0.0219,
            "T": 0.0003,
            "conf_depth": 0.2546,
            "reg_depth": 0.1805,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "wholefg_jointdepth_scalar",
        "label": "qlinear + wholefg + joint_depth_scale0.75",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgjointdepthscale075_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "camera": 0.0219,
            "T": 0.0003,
            "conf_depth": 0.2458,
            "reg_depth": 0.1790,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "wholefg_jointdepth_scalar",
        "label": "qquadratic + wholefg + joint_depth_scale0.75",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgjointdepthscale075_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "camera": 0.0219,
            "T": 0.0003,
            "conf_depth": 0.2417,
            "reg_depth": 0.1783,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "wholefg_jointdepth_scalar",
        "label": "qquadratic + wholefg + joint_depth_scale0.875",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgjointdepthscale0875_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "camera": 0.0219,
            "T": 0.0003,
            "conf_depth": 0.2344,
            "reg_depth": 0.1770,
        },
        "verdict": "dead_same_day",
        "frontier_role": "best_wholefg_wrong_side_point",
    },
    {
        "family": "wholefg_decoupled",
        "label": "qquadratic + wholefg + reg0.9375/conf0.875",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgdecoupleddepthreg09375conf0875_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "camera": 0.0219,
            "T": 0.0003,
            "conf_depth": 0.2378,
            "reg_depth": 0.1776,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "nonwholefg_edge_band_jointdepth",
        "label": "qquadratic + fgedge5 + joint_depth_scale0.5",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale05_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "conf_depth": 0.2330,
            "reg_depth": 0.1767,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "nonwholefg_edge_band_jointdepth",
        "label": "qquadratic + fgedge5 + joint_depth_scale0.75",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale075_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "conf_depth": 0.2304,
            "reg_depth": 0.1762,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "nonwholefg_edge_band_jointdepth",
        "label": "qquadratic + fgedge5 + joint_depth_scale0.875",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale0875_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "conf_depth": 0.2293,
            "reg_depth": 0.1760,
        },
        "verdict": "dead_same_day",
        "frontier_role": "best_nonwholefg_wrong_side_point",
    },
    {
        "family": "nonwholefg_edge_band_decoupled",
        "label": "qquadratic + fgedge5 + reg0.9375/conf0.875",
        "candidate": (
            "training/config/"
            "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
            "confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5decoupleddepthreg09375conf0875_minimal.yaml"
        ),
        "val_metrics_candidate": {
            "conf_depth": 0.2302,
            "reg_depth": 0.1762,
        },
        "verdict": "dead_same_day",
    },
    {
        "family": "hard_pixel_depth_conf_threshold",
        "label": "qquadratic + fgedge5 + depth_conf<=p60 + joint_depth_scale0.5",
        "candidate": LATEST_MANUAL_GATE["candidate"],
        "val_metrics_candidate": LATEST_MANUAL_GATE["val_metrics_candidate"],
        "verdict": "dead_same_day",
        "frontier_role": "too_aggressive_follow_up",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate the opt-in ZJU source-policy research loop contract without disturbing guard-only steady_hold."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--approved-problem-path", type=Path, default=DEFAULT_APPROVED_PROBLEM_PATH)
    parser.add_argument("--approved-problem-template-path", type=Path, default=DEFAULT_APPROVED_PROBLEM_TEMPLATE_PATH)
    parser.add_argument("--frontier-ledger-path", type=Path, default=DEFAULT_FRONTIER_LEDGER_PATH)
    parser.add_argument("--candidate-patch-plan-path", type=Path, default=DEFAULT_CANDIDATE_PATCH_PLAN_PATH)
    parser.add_argument("--candidate-patch-plan-md-path", type=Path, default=DEFAULT_CANDIDATE_PATCH_PLAN_MD_PATH)
    parser.add_argument("--candidate-verdict-path", type=Path, default=DEFAULT_CANDIDATE_VERDICT_PATH)
    parser.add_argument("--family-stop-reason-path", type=Path, default=DEFAULT_FAMILY_STOP_REASON_PATH)
    parser.add_argument("--resume-token-path", type=Path, default=DEFAULT_RESUME_TOKEN_PATH)
    parser.add_argument("--gate-reference-logs-path", type=Path, default=DEFAULT_GATE_REFERENCE_LOGS_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--status-md-path", type=Path, default=DEFAULT_STATUS_MD_PATH)
    parser.add_argument("--local-manifest-path", type=Path, default=DEFAULT_LOCAL_MANIFEST_PATH)
    parser.add_argument("--training-question-manifest-path", type=Path, default=DEFAULT_TRAINING_QUESTION_MANIFEST_PATH)
    parser.add_argument("--task-plan-path", type=Path, default=DEFAULT_TASK_PLAN_PATH)
    parser.add_argument("--max-approved-problems-per-night", type=int, default=1)
    parser.add_argument("--max-candidates-per-problem", type=int, default=1)
    parser.add_argument("--disk-floor-gb", type=float, default=120.0)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def summarize_candidate_verdict(candidate_verdict: dict) -> dict:
    if not candidate_verdict:
        return {}
    family = str(candidate_verdict.get("family", "")).strip()
    status = str(candidate_verdict.get("status", "")).strip()
    if not family and status not in {
        "dead_same_day",
        "failed_long_gate",
        "provisional_lead",
        "contract_rejected",
        "runner_error",
        "reference_missing",
    }:
        return {}
    summary = {
        "checked_at": candidate_verdict.get("checked_at", ""),
        "status": status,
        "active_candidate": candidate_verdict.get("active_candidate", ""),
        "problem_id": candidate_verdict.get("problem_id", ""),
        "family": family,
        "first_candidate_shape": candidate_verdict.get("first_candidate_shape", ""),
        "reason": candidate_verdict.get("reason", ""),
        "gate_stage_reached": candidate_verdict.get("gate_stage_reached", ""),
        "approved_problem_archive_path": candidate_verdict.get("approved_problem_archive_path", ""),
    }
    short_gate_vs_lead = candidate_verdict.get("short_gate_vs_lead", {}) or {}
    if short_gate_vs_lead:
        summary["short_gate_vs_lead"] = short_gate_vs_lead
    return summary


def build_latest_family_outcomes(candidate_verdict: dict) -> dict:
    family = str(candidate_verdict.get("family", "")).strip()
    if not family:
        return {}
    return {
        family: {
            "latest_status": candidate_verdict.get("status", ""),
            "problem_id": candidate_verdict.get("problem_id", ""),
            "first_candidate_shape": candidate_verdict.get("first_candidate_shape", ""),
            "active_candidate": candidate_verdict.get("active_candidate", ""),
            "reason": candidate_verdict.get("reason", ""),
            "gate_stage_reached": candidate_verdict.get("gate_stage_reached", ""),
            "approved_problem_archive_path": candidate_verdict.get("approved_problem_archive_path", ""),
        }
    }


def build_auto_next_ticket_fields(current_priority: dict) -> dict:
    recommended_next_families = list(current_priority.get("recommended_next_families", []) or [])
    recommended_family_order = list(current_priority.get("recommended_family_order", []) or [])
    preferred_first_family = (
        str(recommended_family_order[0]).strip()
        if recommended_family_order
        else (str(recommended_next_families[0]).strip() if recommended_next_families else "")
    )
    if not preferred_first_family:
        next_requirement = str(current_priority.get("next_requirement", "")).strip().lower()
        waiting_for_manual_promotion = "manual promotion decision" in next_requirement
        return {
            "allowed_families": [],
            "preferred_first_family": "",
            "preferred_first_family_reason": (
                "No auto-next ticket is currently selected. Wait for a fresh manual promotion decision before "
                "any future approval."
                if waiting_for_manual_promotion
                else (
                    "No auto-next ticket is currently selected. Wait for a new manual problem before any future "
                    "approval."
                )
            ),
            "preferred_first_candidate_shape": "",
            "preferred_first_candidate_shape_reason": (
                "No preferred first candidate is currently auto-selected because the provisional local lead is "
                "on manual-promotion hold."
                if waiting_for_manual_promotion
                else "No preferred first candidate is currently auto-selected because the active family batch is closed."
            ),
            "preferred_first_candidate_config": "",
            "preferred_first_candidate_requires_code_patch": False,
            "preferred_first_candidate_write_surface": [],
            "preferred_first_candidate_execution_note": (
                "Do not auto-arm a next ticket. Wait for the manual promotion decision first."
                if waiting_for_manual_promotion
                else (
                    "Do not auto-arm a next ticket. Define a new manual problem first, then let Codex make that new "
                    "family execution-ready if it survives review."
                )
            ),
            "preferred_first_candidate_arm_command": "",
            "preferred_first_candidate_run_command": "",
            "family_blueprints": {},
        }
    return {
        "allowed_families": recommended_next_families,
        "preferred_first_family": preferred_first_family,
        "preferred_first_family_reason": PREFERRED_FIRST_FAMILY_REASON,
        "preferred_first_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
        "preferred_first_candidate_shape_reason": (
            "The first interpolated, partial, and disagreement-routing tickets already failed at 10/5, so the "
            "next approved family must switch to detached geometry-chain inconsistency rather than another "
            "depth-branch-local cousin."
        ),
        "preferred_first_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        "preferred_first_candidate_requires_code_patch": False,
        "preferred_first_candidate_write_surface": [
            DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        ],
        "preferred_first_candidate_execution_note": (
            "The first approved unprojection-consistency candidate is already executable on the current repo, so "
            "approval can go directly into smoke and gate without another code patch."
        ),
        "preferred_first_candidate_arm_command": (
            "python scripts/arm_zju_source_policy_approved_problem.py --seed unproject_consistency_routing"
        ),
        "preferred_first_candidate_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "family_blueprints": {
            "unproject_consistency_routing": str(DEFAULT_UNPROJECT_CONSISTENCY_BLUEPRINT_PATH.resolve()),
        },
    }


def build_current_priority(candidate_verdict: dict) -> dict:
    family = str(candidate_verdict.get("family", "")).strip()
    status = str(candidate_verdict.get("status", "")).strip()
    if family == "source_policy_hybrid_ring_regularization" and status == "provisional_lead":
        return {
            "current_priority_family": "",
            "current_priority_reason": (
                "The first source_policy_hybrid_ring_regularization ticket beat the stable lead through "
                "long gate and is now a provisional local lead. Research must return to guard and wait for a "
                "fresh manual promotion decision instead of auto-opening another ticket."
            ),
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": (
                "Do not auto-arm another ticket. The winning hybrid-ring ticket is already archived and now "
                "needs a manual promotion decision."
            ),
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "current_frontier_hint": "No current priority family is selected. Wait for a fresh manual promotion decision.",
            "current_frontier_priority": (
                "promotion-hold: source_policy_hybrid_ring_regularization is a provisional local lead; do not "
                "auto-open another family or any cloud action."
            ),
            "recommended_next_families": [],
            "recommended_family_order": [],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first source_policy_hybrid_ring_regularization ticket already spent the current family "
                "budget and produced a provisional lead, so same-family retry would violate the single-ticket "
                "cross-night contract."
            ),
            "next_requirement": (
                "Wait for a fresh manual promotion decision. Keep research in IDLE_GUARD, keep cloud off, and "
                "do not auto-select a new family or auto-launch cloud."
            ),
        }
    if family == "source_policy_hybrid_ring_regularization" and status in {"dead_same_day", "failed_long_gate"}:
        verdict_label = status or "formal_verdict"
        return {
            "current_priority_family": "",
            "current_priority_reason": (
                "The first source_policy_hybrid_ring_regularization ticket already produced a terminal "
                f"research verdict ({verdict_label}), so this family batch is closed. The system must now "
                "wait for a new manual problem instead of auto-selecting another ticket."
            ),
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": (
                "Do not auto-arm another ticket. The first source_policy_hybrid_ring_regularization launch "
                "has already spent the current family budget."
            ),
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "current_frontier_hint": "No current priority family is selected. Wait for a new manual problem.",
            "current_frontier_priority": (
                "manual-only: the interpolated, partial, conf_reg_disagreement, unproject_consistency, "
                "unproject_aux_confgate, and source_policy_hybrid_ring_regularization families are closed for "
                "this round; do not reopen cousins."
            ),
            "recommended_next_families": [],
            "recommended_family_order": [],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first source_policy_hybrid_ring_regularization ticket already consumed the current "
                "family budget, so same-family retry would violate the single-ticket cross-night contract."
            ),
            "next_requirement": (
                "Wait for a new manual problem. Do not auto-select a next ticket and do not reopen "
                "interpolated, partial, conf_reg_disagreement, unproject_consistency, unproject_aux_confgate, "
                "or source_policy_hybrid_ring_regularization cousins."
            ),
        }
    if family == "unproject_aux_confgate" and status in {"dead_same_day", "failed_long_gate", "provisional_lead"}:
        verdict_label = status or "formal_verdict"
        return {
            "current_priority_family": "",
            "current_priority_reason": (
                "The first unproject_aux_confgate ticket already produced a terminal research verdict "
                f"({verdict_label}), so this family batch is closed. The system must now wait for a new manual "
                "problem instead of auto-selecting another ticket."
            ),
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": (
                "Do not auto-arm another ticket. The first unproject_aux_confgate launch has already spent the "
                "current family budget."
            ),
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "current_frontier_hint": "No current priority family is selected. Wait for a new manual problem.",
            "current_frontier_priority": (
                "manual-only: the interpolated, partial, conf_reg_disagreement, unproject_consistency, and "
                "unproject_aux_confgate families are closed for this round; do not reopen cousins."
            ),
            "recommended_next_families": [],
            "recommended_family_order": [],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first unproject_aux_confgate ticket already consumed the current family budget, so "
                "same-family retry would violate the single-ticket cross-night contract."
            ),
            "next_requirement": (
                "Wait for a new manual problem. Do not auto-select a next ticket and do not reopen "
                "interpolated, partial, conf_reg_disagreement, unproject_consistency, or "
                "unproject_aux_confgate cousins."
            ),
        }
    if family == "unproject_consistency_routing" and status in {"dead_same_day", "failed_long_gate"}:
        verdict_label = status or "formal_verdict"
        return {
            "current_priority_family": "",
            "current_priority_reason": (
                "The first unproject_consistency_routing ticket already produced a formal research verdict "
                f"({verdict_label}), so this family batch is closed. The system must now wait for a new manual "
                "problem instead of auto-selecting another ticket."
            ),
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": (
                "Do not auto-arm another ticket. The first unproject_consistency_routing launch has already "
                "spent the current family budget."
            ),
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "current_frontier_hint": "No current priority family is selected. Wait for a new manual problem.",
            "current_frontier_priority": (
                "manual-only: the interpolated, partial, conf_reg_disagreement, and unproject_consistency "
                "families are closed for this round; do not reopen cousins."
            ),
            "recommended_next_families": [],
            "recommended_family_order": [],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first unproject_consistency_routing ticket was a formal research failure, so same-family "
                "retry would violate the single-ticket cross-night contract."
            ),
            "next_requirement": (
                "Wait for a new manual problem. Do not auto-select a next ticket and do not reopen "
                "interpolated, partial, conf_reg_disagreement, or unproject_consistency cousins."
            ),
        }
    if family == "conf_reg_disagreement_routing" and status in {"dead_same_day", "failed_long_gate"}:
        verdict_label = status or "formal_verdict"
        return {
            "current_priority_family": PREFERRED_FIRST_FAMILY,
            "current_priority_reason": (
                "The first conf_reg_disagreement_routing ticket already produced a formal research verdict "
                f"({verdict_label}), so the next human-approved question is now "
                "unproject_consistency_routing rather than another depth-branch-local retry."
            ),
            "current_priority_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
            "current_priority_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [
                DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            ],
            "current_priority_candidate_execution_note": (
                "No same-family retry is allowed after the first conf_reg_disagreement_routing ticket closes. "
                "The repo is now prewired for one anchor_unproject_consistency_joint_routing ticket, but "
                "research must still remain in guard until that new manual approval is armed."
            ),
            "current_priority_arm_command": (
                "python scripts/arm_zju_source_policy_approved_problem.py --seed unproject_consistency_routing"
            ),
            "current_priority_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "current_frontier_hint": (
                "The next manual approval may consider exactly one unproject_consistency_routing ticket; do not "
                "reopen disagreement or earlier cousins."
            ),
            "current_frontier_priority": (
                "cross-family only: after the first disagreement-routing verdict, switch to detached "
                "unprojection-consistency routing instead of another branch-local cousin sweep"
            ),
            "recommended_next_families": [PREFERRED_FIRST_FAMILY],
            "recommended_family_order": [PREFERRED_FIRST_FAMILY],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first conf_reg_disagreement_routing ticket was a formal research failure, so same-family "
                "micro-tuning would violate the single-ticket cross-night contract."
            ),
            "next_requirement": (
                "Prepare one human-approved unproject_consistency_routing ticket with exactly one "
                "anchor_unproject_consistency_joint_routing candidate. Do not auto-approve it and do not "
                "reopen disagreement, interpolated, or partial cousins."
            ),
        }
    if family == "partial_joint_depth_routing" and status in {"dead_same_day", "failed_long_gate"}:
        verdict_label = status or "formal_verdict"
        return {
            "current_priority_family": PREFERRED_FIRST_FAMILY,
            "current_priority_reason": (
                "The first interpolated and partial tickets already produced formal research verdicts, and both "
                f"stopped at 10/5 ({verdict_label} on the partial ticket), so the next approval must route from "
                "a genuinely new geometry-chain signal rather than keep tuning depth-branch-local cousins."
            ),
            "current_priority_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
            "current_priority_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [
                DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            ],
            "current_priority_candidate_execution_note": (
                "The executable first ticket is now anchor-only unprojection-consistency routing on the current "
                "stable lead: build a detached residual from predicted depth plus predicted pose against "
                "ground-truth world points on anchor-supervised pixels, then downscale conf-target and "
                "upweight reg-target on the same pixels."
            ),
            "current_priority_arm_command": (
                "python scripts/arm_zju_source_policy_approved_problem.py --seed unproject_consistency_routing"
            ),
            "current_priority_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
            "current_frontier_hint": (
                "The next manual approval may consider exactly one unproject_consistency_routing ticket; do not "
                "reopen interpolated or partial cousins."
            ),
            "current_frontier_priority": (
                "cross-family only: after the first interpolated and partial tickets failed, switch to the "
                "single anchor_unproject_consistency_joint_routing candidate instead of any depth_conf cousin sweep"
            ),
            "recommended_next_families": [PREFERRED_FIRST_FAMILY],
            "recommended_family_order": [PREFERRED_FIRST_FAMILY],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                "The first partial_joint_depth_routing ticket was a formal research failure, so same-family "
                "micro-tuning would violate the single-ticket cross-night contract."
            ),
            "next_requirement": (
                "Prepare one human-approved unproject_consistency_routing ticket with exactly one "
                "anchor_unproject_consistency_joint_routing candidate. Do not auto-approve it and do not "
                "reopen interpolated or partial cousins."
            ),
        }
    return {
        "current_priority_family": PREFERRED_FIRST_FAMILY,
        "current_priority_reason": PREFERRED_FIRST_FAMILY_REASON,
        "current_priority_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
        "current_priority_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        "current_priority_candidate_requires_code_patch": False,
        "current_priority_candidate_write_surface": [
            DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        ],
        "current_priority_candidate_execution_note": (
            "The first approved unprojection-consistency candidate is already executable on the current repo: "
            "loss.py provides anchor-only detached unprojection-residual routing and the first config is "
            "materialized as a single-candidate follow-up."
        ),
        "current_priority_arm_command": (
            "python scripts/arm_zju_source_policy_approved_problem.py --seed unproject_consistency_routing"
        ),
        "current_priority_run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "current_frontier_hint": (
            "The next manual question is now the single unproject_consistency_routing family; it should change "
            "the routing signal from branch-local depth scores to detached geometry-chain inconsistency."
        ),
        "current_frontier_priority": (
            "prepare only the single anchor_unproject_consistency_joint_routing first ticket; do not reopen "
            "interpolated, partial, disagreement, wholefg, edge-band, hard-threshold, pow-like, or plain "
            "anchor_view_only families"
        ),
        "recommended_next_families": list(ALLOWED_FAMILIES),
        "recommended_family_order": [
            PREFERRED_FIRST_FAMILY,
        ],
        "same_family_retry_forbidden": False,
        "same_family_retry_reason": "",
        "next_requirement": (
            "Approve only unproject_consistency_routing, keep the first candidate to one "
            "anchor_unproject_consistency_joint_routing launch, and continue to forbid cousin sweep."
        ),
    }


def build_approved_problem_template(max_approved_problems_per_night: int, max_candidates_per_problem: int) -> dict:
    return {
        "approved": False,
        "approved_at": "",
        "problem_id": "",
        "problem_title": "",
        "family": "",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. Define a new manual problem before any future approval."
        ),
        "problem_statement": "",
        "why_genuinely_new": "",
        "why_not_reopening_frozen_family": "",
        "first_candidate_hint": "",
        "first_candidate_shape": "",
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": False,
        "historical_prior": "",
        "avoid_patterns": [],
        "max_approved_problems_per_night": max_approved_problems_per_night,
        "candidate_budget": max_candidates_per_problem,
        "max_candidates_per_night": max_candidates_per_problem,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "requires_dataset_or_routing_change": False,
        "requires_supervision_audit": False,
        "mutation_dsl": {},
    }


def build_approved_problem_seed(
    family: str,
    problem_id: str,
    problem_title: str,
    problem_statement: str,
    why_genuinely_new: str,
    why_not_reopening_frozen_family: str,
    first_candidate_hint: str,
    historical_prior: str,
    avoid_patterns: list[str],
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_template(max_approved_problems_per_night, max_candidates_per_problem)
    payload.update(
        {
            "problem_id": problem_id,
            "problem_title": problem_title,
            "family": family,
            "problem_statement": problem_statement,
            "why_genuinely_new": why_genuinely_new,
            "why_not_reopening_frozen_family": why_not_reopening_frozen_family,
            "first_candidate_hint": first_candidate_hint,
            "historical_prior": historical_prior,
            "avoid_patterns": avoid_patterns,
        }
    )
    return payload


def build_interpolated_smoothstep_ready_seed(
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_seed(
        family="interpolated_eligibility_shaping",
        problem_id="camera_b1_interpolated_eligibility_shaping_v1",
        problem_title="Camera_B1 interpolated non-wholefg eligibility shaping",
        problem_statement=(
            "Design a softer non-wholefg selectivity-changing depth-loss rule that modulates eligibility "
            "continuously with per-pixel depth_conf_maps instead of reopening wholefg softening or a hard "
            "pixel threshold."
        ),
        why_genuinely_new=(
            "This changes pixel-level eligibility semantics with continuous depth_conf shaping rather than "
            "another wholefg or edge-band scalar/decoupled near-neighbor."
        ),
        why_not_reopening_frozen_family=(
            "It keeps the best qquadratic fgedge5 joint-depth scale0.875 route but replaces the rejected "
            "hard depth_conf<=p60 cutoff with a smooth taper, so it stays softer than the failed hard "
            "threshold while not reopening wholefg, edge-band near-neighbor, or anchor-view-only families."
        ),
        first_candidate_hint=(
            "Launch the prebuilt smoothstep taper candidate config directly; do not reopen hard depth_conf "
            "thresholds, pow-like sharpened weighting, or another wholefg/edge-band cousin."
        ),
        historical_prior=HISTORICAL_THRESHOLD_POW2_PRIOR["takeaway"],
        avoid_patterns=[
            "hard depth_conf threshold",
            "pow-like sharper weighting",
            "wholefg scalar reopen",
            "wholefg decoupled reopen",
            "edge-band scalar near-neighbor reopen",
            "plain anchor_view_only",
        ],
        max_approved_problems_per_night=max_approved_problems_per_night,
        max_candidates_per_problem=max_candidates_per_problem,
    )
    payload.update(
        {
            "first_candidate_shape": "smoothstep_taper",
            "first_candidate_config": DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG,
            "first_candidate_requires_code_patch": False,
            "first_candidate_write_surface": [DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG],
            "first_candidate_knobs": {
                "anchor_conditioned_reg_target_cameras": ["Camera_B1"],
                "anchor_conditioned_reg_target_scale": 0.875,
                "anchor_conditioned_reg_target_train_only": True,
                "anchor_conditioned_reg_target_quality_interp": "quadratic",
                "anchor_conditioned_reg_target_quality_low": QUALITY_LOW,
                "anchor_conditioned_reg_target_quality_high": QUALITY_HIGH,
                "anchor_conditioned_reg_target_foreground_edge_band_px": 5,
                "anchor_conditioned_reg_target_depth_conf_interp": "smoothstep",
                "anchor_conditioned_reg_target_depth_conf_low": 0.0,
                "anchor_conditioned_reg_target_depth_conf_high": DEPTH_CONF_P60_MAX,
            },
        }
    )
    return payload


def build_partial_joint_ready_seed(
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_seed(
        family="partial_joint_depth_routing",
        problem_id="camera_b1_partial_joint_depth_routing_v1",
        problem_title="Camera_B1 partial joint-depth routing",
        problem_statement=(
            "Design one partial joint-depth routing candidate that keeps the best non-wholefg reg branch "
            "route intact while narrowing only the conf branch to a smoother low-depth_conf subset."
        ),
        why_genuinely_new=(
            "This keeps the best qquadratic fgedge5 joint-depth scale0.875 reg route but changes only the "
            "conf branch selectivity, so it is a new branch-routing split rather than another interpolated "
            "eligibility retry or an edge-band scalar near-neighbor."
        ),
        why_not_reopening_frozen_family=(
            "It is not another wholefg scalar, wholefg decoupled, edge-band scalar, edge-band decoupled, "
            "hard pixel threshold, plain anchor_view_only restatement, or interpolated smoothstep cousin."
        ),
        first_candidate_hint=(
            "Launch the prebuilt partial-joint config directly: preserve the qquadratic fgedge5 "
            "joint-depth scale0.875 reg route and add a smoothstep depth_conf taper only on the conf branch "
            "inside the same edge band."
        ),
        historical_prior=HISTORICAL_THRESHOLD_POW2_PRIOR["takeaway"],
        avoid_patterns=[
            "interpolated smoothstep cousin reopen",
            "hard depth_conf threshold",
            "pow-like sharper weighting",
            "wholefg scalar reopen",
            "wholefg decoupled reopen",
            "edge-band scalar near-neighbor reopen",
            "edge-band decoupled reopen",
            "plain anchor_view_only",
        ],
        max_approved_problems_per_night=max_approved_problems_per_night,
        max_candidates_per_problem=max_candidates_per_problem,
    )
    payload.update(
        {
            "first_candidate_shape": DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_SHAPE,
            "first_candidate_config": DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_CANDIDATE_CONFIG,
            "first_candidate_requires_code_patch": False,
            "first_candidate_write_surface": [
                DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_CANDIDATE_CONFIG,
            ],
            "first_candidate_knobs": {
                "anchor_conditioned_reg_target_cameras": ["Camera_B1"],
                "anchor_conditioned_reg_target_scale": 0.875,
                "anchor_conditioned_reg_target_train_only": True,
                "anchor_conditioned_reg_target_quality_interp": "quadratic",
                "anchor_conditioned_reg_target_quality_low": QUALITY_LOW,
                "anchor_conditioned_reg_target_quality_high": QUALITY_HIGH,
                "anchor_conditioned_reg_target_foreground_edge_band_px": 5,
                "anchor_conditioned_conf_target_cameras": ["Camera_B1"],
                "anchor_conditioned_conf_target_scale": 0.875,
                "anchor_conditioned_conf_target_train_only": True,
                "anchor_conditioned_conf_target_quality_interp": "quadratic",
                "anchor_conditioned_conf_target_quality_low": QUALITY_LOW,
                "anchor_conditioned_conf_target_quality_high": QUALITY_HIGH,
                "anchor_conditioned_conf_target_foreground_edge_band_px": 5,
                "anchor_conditioned_conf_target_depth_conf_interp": "smoothstep",
                "anchor_conditioned_conf_target_depth_conf_low": 0.0,
                "anchor_conditioned_conf_target_depth_conf_high": DEPTH_CONF_P60_MAX,
            },
        }
    )
    return payload


def build_unproject_consistency_ready_seed(
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_seed(
        family="unproject_consistency_routing",
        problem_id="camera_b1_unproject_consistency_routing_v1",
        problem_title="Camera_B1 unprojection consistency routing",
        problem_statement=(
            "Design exactly one new-family candidate that routes depth supervision from detached per-pixel "
            "unprojection inconsistency between predicted depth plus predicted pose and ground-truth world "
            "points on anchor-supervised pixels, instead of reopening raw depth_conf shaping, partial "
            "conf-branch subsets, or conf-vs-reg disagreement."
        ),
        why_genuinely_new=(
            "This changes the routing basis from depth-branch-local signals to detached geometry-chain "
            "inconsistency on anchor-supervised pixels, so it is not an interpolated, partial, or "
            "conf_reg_disagreement cousin."
        ),
        why_not_reopening_frozen_family=(
            "It does not reopen wholefg, edge-band, hard-threshold, pow-like, plain anchor_view_only, "
            "interpolated, partial, or disagreement families; it changes the routing signal itself."
        ),
        first_candidate_hint=(
            "Launch only the anchor_unproject_consistency_joint_routing candidate: on anchor-supervised "
            "pixels, reconstruct world points from predicted depth plus predicted pose, compute detached "
            "normalized residual against batch world_points, then downscale conf-target and upweight "
            "reg-target on the same pixels."
        ),
        historical_prior=HISTORICAL_THRESHOLD_POW2_PRIOR["takeaway"],
        avoid_patterns=[
            "interpolated cousin reopen",
            "partial cousin reopen",
            "conf_reg_disagreement cousin reopen",
            "wholefg reopen",
            "edge-band reopen",
            "hard depth_conf threshold",
            "pow-like sharper weighting",
            "plain anchor_view_only",
        ],
        max_approved_problems_per_night=max_approved_problems_per_night,
        max_candidates_per_problem=max_candidates_per_problem,
    )
    payload.update(
        {
            "first_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
            "first_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            "first_candidate_requires_code_patch": False,
            "requires_dataset_or_routing_change": False,
            "first_candidate_write_surface": [
                DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
            ],
            "mutation_dsl": {
                "non_wholefg_selectivity_required": True,
                "allow_unproject_consistency_routing": True,
                "disallow_wholefg_scalar": True,
                "disallow_wholefg_decoupled": True,
                "disallow_edge_band_scalar": True,
                "disallow_edge_band_decoupled": True,
                "disallow_hard_depth_conf_threshold": True,
                "disallow_plain_anchor_view_only": True,
            },
            "first_candidate_knobs": {
                "anchor_conditioned_unproject_consistency_cameras": ["Camera_B1"],
                "anchor_conditioned_unproject_consistency_conf_scale": 0.875,
                "anchor_conditioned_unproject_consistency_reg_scale": 1.125,
                "anchor_conditioned_unproject_consistency_train_only": True,
            },
        }
    )
    return payload


def build_unproject_aux_confgate_ready_seed(
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_seed(
        family="unproject_aux_confgate",
        problem_id="source_policy_unproject_aux_confgate_v1",
        problem_title="Stable-lead confidence-gated auxiliary unprojection",
        problem_statement=(
            "Design exactly one new-family candidate that keeps the current source-policy stable lead intact "
            "while changing only the auxiliary unproject_geometry term from an always-on fixed weight to a "
            "lighter detached depth-confidence gate, instead of reopening any more depth-loss routing cousins."
        ),
        why_genuinely_new=(
            "This family changes the objective-layer geometry auxiliary term rather than routing conf/reg depth "
            "supervision, so it is not an interpolated, partial, conf_reg_disagreement, or "
            "unproject_consistency cousin."
        ),
        why_not_reopening_frozen_family=(
            "It does not reopen wholefg, edge-band, hard-threshold, pow-like, plain anchor_view_only, "
            "reliable-region unproject, or bottom-only unproject families; it only retunes the existing "
            "auxiliary unproject_geometry branch on the stable lead."
        ),
        first_candidate_hint=(
            "Launch only the stablelead_unproject_aux_confgate_w005 candidate: keep the current "
            "confdepth_dropworst_gradconfmask source-policy stable lead, lower the auxiliary unproject weight "
            "to 0.05, and gate that term by detached per-pixel depth confidence."
        ),
        historical_prior=(
            "Earlier geometry-side evidence already showed that detached depth-confidence gating was the "
            "strongest surviving auxiliary unproject variant, substantially closer to baseline than fixed-weight, "
            "warmup, thresholded, or pow-like alternatives."
        ),
        avoid_patterns=[
            "interpolated cousin reopen",
            "partial cousin reopen",
            "conf_reg_disagreement cousin reopen",
            "unproject_consistency cousin reopen",
            "wholefg reopen",
            "edge-band reopen",
            "reliable-region unproject reopen",
            "bottom-only unproject reopen",
            "hard depth_conf threshold",
            "pow-like sharper weighting",
            "plain anchor_view_only",
        ],
        max_approved_problems_per_night=max_approved_problems_per_night,
        max_candidates_per_problem=max_candidates_per_problem,
    )
    payload.update(
        {
            "first_candidate_shape": DEFAULT_UNPROJECT_AUX_CONFGATE_SHAPE,
            "first_candidate_config": DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG,
            "first_candidate_requires_code_patch": False,
            "requires_dataset_or_routing_change": False,
            "first_candidate_write_surface": [
                DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG,
            ],
            "mutation_dsl": {
                "allow_unproject_aux_confgate": True,
                "keep_existing_depth_routing_unchanged": True,
                "disallow_wholefg_scalar": True,
                "disallow_wholefg_decoupled": True,
                "disallow_edge_band_scalar": True,
                "disallow_edge_band_decoupled": True,
                "disallow_hard_depth_conf_threshold": True,
                "disallow_plain_anchor_view_only": True,
                "disallow_reliable_region_unproject_reopen": True,
                "disallow_bottom_only_unproject_reopen": True,
            },
            "first_candidate_knobs": {
                "loss.unproject_geometry.weight": 0.05,
                "loss.unproject_geometry.use_depth_conf_gate": True,
                "loss.unproject_geometry.detach_depth_conf": True,
                "loss.unproject_geometry.depth_conf_power": 1.0,
                "loss.unproject_geometry.depth_conf_threshold": 0.0,
            },
        }
    )
    return payload


def build_source_policy_hybrid_ring_ready_seed(
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> dict:
    payload = build_approved_problem_seed(
        family="source_policy_hybrid_ring_regularization",
        problem_id="source_policy_hybrid_ring_regularization_v1",
        problem_title="Hybrid ring source-policy regularization",
        problem_statement=(
            "Design exactly one new-family candidate that keeps the current source-policy stable lead's depth "
            "routing intact while regularizing rawpool source selection with a nearest-dominant but "
            "uniform-coverage tail, instead of reopening any more depth-loss routing cousins or old sampler "
            "variants."
        ),
        why_genuinely_new=(
            "This family changes the data-side source-view rule itself rather than loss routing, auxiliary "
            "unprojection, or previously rejected sampler toggles, so it is not an interpolated, partial, "
            "conf_reg_disagreement, unproject_consistency, or unproject_aux_confgate cousin."
        ),
        why_not_reopening_frozen_family=(
            "It does not reopen uniform-only rawpool, trainmix exposure, min_supervised_views=2, "
            "max_depth_conf anchor selection, or any wholefg/edge-band depth-routing family; it keeps one "
            "supervised anchor and the current stable loss route while changing only how rawpool source slots "
            "are filled."
        ),
        first_candidate_hint=(
            "Launch only the stablelead_nearest_plus_uniform_tail candidate: keep the current "
            "confdepth_dropworst_gradconfmask stable lead, preserve geom_plus_raw plus one supervised anchor, "
            "and change source_policy from nearest_ring to nearest_plus_uniform_tail so the last rawpool slot "
            "regularizes coverage instead of staying fully local."
        ),
        historical_prior=(
            "Earlier source-policy evidence showed that uniform coverage can improve geometry-side coverage "
            "while the current training lead still depends on nearest-ring stability. The next bounded rule is "
            "therefore a hybrid that keeps nearest views local but adds one uniform-coverage tail source."
        ),
        avoid_patterns=[
            "interpolated cousin reopen",
            "partial cousin reopen",
            "conf_reg_disagreement cousin reopen",
            "unproject_consistency cousin reopen",
            "unproject_aux_confgate cousin reopen",
            "uniform-only rawpool reopen",
            "trainmix rawpool exposure reopen",
            "min_supervised_views=2 reopen",
            "max_depth_conf anchor reopen",
            "wholefg reopen",
            "edge-band reopen",
            "hard depth_conf threshold",
            "pow-like sharper weighting",
            "plain anchor_view_only",
        ],
        max_approved_problems_per_night=max_approved_problems_per_night,
        max_candidates_per_problem=max_candidates_per_problem,
    )
    payload.update(
        {
            "first_candidate_shape": DEFAULT_SOURCE_POLICY_HYBRID_RING_SHAPE,
            "first_candidate_config": DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG,
            "first_candidate_requires_code_patch": False,
            "requires_dataset_or_routing_change": False,
            "first_candidate_write_surface": [
                DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG,
            ],
            "mutation_dsl": {
                "allow_source_policy_hybrid_ring_regularization": True,
                "keep_existing_depth_routing_unchanged": True,
                "keep_geom_plus_raw_pool": True,
                "keep_single_supervised_anchor": True,
                "disallow_uniform_only_rawpool_reopen": True,
                "disallow_trainmix_reopen": True,
                "disallow_minsup2_reopen": True,
                "disallow_max_depth_conf_anchor_reopen": True,
                "disallow_wholefg_scalar": True,
                "disallow_wholefg_decoupled": True,
                "disallow_edge_band_scalar": True,
                "disallow_edge_band_decoupled": True,
                "disallow_hard_depth_conf_threshold": True,
                "disallow_plain_anchor_view_only": True,
            },
            "first_candidate_knobs": {
                "zju_source_policy": "nearest_plus_uniform_tail",
                "zju_source_view_pool": "geom_plus_raw",
                "zju_min_supervised_views": 1,
                "loss.depth.respect_conf_mask_in_grad_conf": True,
            },
        }
    )
    return payload


def build_repo_process_allowlist_template() -> dict:
    return {
        "checked_at": iso_now(),
        "status": "template_only",
        "guard_track_must_continue": True,
        "notes": (
            "Populate active repo_process_allowlist.json only for the currently running approved "
            "research candidate, then clear it again when returning to guard."
        ),
        "allowed_markers": list(RESEARCH_PROCESS_MARKER_TEMPLATE),
    }


def build_repo_process_allowlist(approved_problem_ready: bool) -> dict:
    return {
        "checked_at": iso_now(),
        "status": "armed_but_empty_until_candidate_launch" if approved_problem_ready else "idle_empty_allowlist",
        "guard_track_must_continue": True,
        "notes": (
            "Keep this active allowlist empty by default. A future execution wrapper may populate only the "
            "current-round research markers immediately before launching a candidate and must clear them on return."
        ),
        "allowed_markers": [],
    }


def build_interpolated_eligibility_blueprint() -> dict:
    return {
        "checked_at": iso_now(),
        "family": "interpolated_eligibility_shaping",
        "status": "ready_as_next_low_friction_family",
        "why_now": (
            "The best non-wholefg frontier point already uses qquadratic Camera_B1 edge-band joint-depth "
            "routing at scale 0.875, and the hard depth_conf<=p60 follow-up proved that an abrupt pixel "
            "threshold is too aggressive inside the same promising route."
        ),
        "reference_candidate": {
            "config": (
                "training/config/"
                "zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_"
                "anchorb1qquadraticfgedge5jointdepthscale0875_minimal.yaml"
            ),
            "val_metrics": {
                "conf_depth": 0.2293,
                "reg_depth": 0.1760,
            },
            "interpretation": "Best non-wholefg scalar point; still a wrong-side near-miss."
        },
        "rejected_follow_up": {
            "config": LATEST_MANUAL_GATE["candidate"],
            "val_metrics": LATEST_MANUAL_GATE["val_metrics_candidate"],
            "interpretation": LATEST_MANUAL_GATE["interpretation"],
        },
        "preserve_knobs": {
            "anchor_conditioned_reg_target_cameras": ["Camera_B1"],
            "anchor_conditioned_reg_target_scale": 0.875,
            "anchor_conditioned_reg_target_train_only": True,
            "anchor_conditioned_reg_target_quality_interp": "quadratic",
            "anchor_conditioned_reg_target_quality_low": QUALITY_LOW,
            "anchor_conditioned_reg_target_quality_high": QUALITY_HIGH,
            "anchor_conditioned_reg_target_foreground_edge_band_px": 5,
        },
        "replace_hard_threshold_with": {
            "remove": ["anchor_conditioned_reg_target_depth_conf_max"],
            "add": {
                "anchor_conditioned_reg_target_depth_conf_interp": "smoothstep",
                "anchor_conditioned_reg_target_depth_conf_low": 0.0,
                "anchor_conditioned_reg_target_depth_conf_high": DEPTH_CONF_P60_MAX,
            }
        },
        "first_candidate_hypothesis": (
            "Keep the best qquadratic fgedge5 joint-depth scale0.875 route, but replace the rejected hard "
            "depth_conf<=p60 cutoff with a continuous smoothstep eligibility taper from full effect at very "
            "low depth_conf to no effect by the prior p60 boundary."
        ),
        "first_candidate_config": DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG,
        ],
        "first_candidate_execution_note": (
            "This first interpolated candidate is config-only on the current repo: the required smoothstep "
            "depth_conf shaping knobs already exist in training/loss.py, so approval can go straight to gate."
        ),
        "preferred_first_candidate_shape": "smoothstep",
        "historical_prior": dict(HISTORICAL_THRESHOLD_POW2_PRIOR),
        "frontier_order_if_approved": [
            {
                "order": 1,
                "label": "smoothstep taper at scale0.875 using previous p60 boundary as high cutoff",
                "knobs": {
                    "anchor_conditioned_reg_target_scale": 0.875,
                    "anchor_conditioned_reg_target_depth_conf_interp": "smoothstep",
                    "anchor_conditioned_reg_target_depth_conf_low": 0.0,
                    "anchor_conditioned_reg_target_depth_conf_high": DEPTH_CONF_P60_MAX,
                }
            },
            {
                "order": 2,
                "label": "quadratic taper at scale0.875 using the same low/high boundary",
                "knobs": {
                    "anchor_conditioned_reg_target_scale": 0.875,
                    "anchor_conditioned_reg_target_depth_conf_interp": "quadratic",
                    "anchor_conditioned_reg_target_depth_conf_low": 0.0,
                    "anchor_conditioned_reg_target_depth_conf_high": DEPTH_CONF_P60_MAX,
                }
            },
            {
                "order": 3,
                "label": "smoothstep taper softened further toward scale0.9375",
                "knobs": {
                    "anchor_conditioned_reg_target_scale": 0.9375,
                    "anchor_conditioned_reg_target_depth_conf_interp": "smoothstep",
                    "anchor_conditioned_reg_target_depth_conf_low": 0.0,
                    "anchor_conditioned_reg_target_depth_conf_high": DEPTH_CONF_P60_MAX,
                }
            }
        ],
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only frontier order 1 is eligible under the current overnight contract. Orders 2 and 3 require a fresh manual approval on a later night, and the historical threshold/pow2 prior further supports starting from the smoothstep taper instead of a sharper fallback."
        },
        "supervision_audit_expected": False,
        "requires_dataset_plumbing": False,
        "allowed_same_night_budget": 1,
        "cloud_must_remain_off": True,
    }


def build_partial_joint_depth_blueprint() -> dict:
    return {
        "checked_at": iso_now(),
        "family": "partial_joint_depth_routing",
        "status": "ready_as_next_cross_family_follow_up",
        "why_now": (
            "The first interpolated_eligibility_shaping / smoothstep_taper ticket now has a formal "
            "dead_same_day verdict at 10/5, so the next approved problem should switch families rather than "
            "reopen interpolated cousins."
        ),
        "why_not_same_family_retry": (
            "Continuing inside interpolated_eligibility_shaping would immediately slide back into same-family "
            "cousin sweep, which the current overnight contract forbids."
        ),
        "reference_candidate": {
            "config": (
                "training/config/"
                "zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_"
                "anchorb1qquadraticfgedge5jointdepthscale0875_minimal.yaml"
            ),
            "val_metrics": {
                "conf_depth": 0.2293,
                "reg_depth": 0.1760,
            },
        },
        "preserve_knobs": {
            "anchor_conditioned_reg_target_cameras": ["Camera_B1"],
            "anchor_conditioned_reg_target_quality_interp": "quadratic",
            "anchor_conditioned_reg_target_quality_low": QUALITY_LOW,
            "anchor_conditioned_reg_target_quality_high": QUALITY_HIGH,
            "anchor_conditioned_reg_target_train_only": True,
        },
        "required_new_selectivity_rule": (
            "Introduce a narrower non-wholefg branch-routing subset than global wholefg without falling back "
            "to exhausted interior, bottom20, plain edge-band scalar, hard depth_conf threshold, or plain anchor_view_only."
        ),
        "first_candidate_hypothesis": (
            "Keep the best qquadratic fgedge5 joint-depth scale0.875 reg route, but route only the conf "
            "branch through a smoothstep low-depth_conf subset inside the same edge band."
        ),
        "first_candidate_shape": DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_SHAPE,
        "first_candidate_config": DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            DEFAULT_PARTIAL_JOINT_DEPTH_ROUTING_CANDIDATE_CONFIG,
        ],
        "first_candidate_execution_note": (
            "This cross-family follow-up is config-only on the current repo: no new dataset plumbing or loss.py "
            "edit is required before dry-run or approval."
        ),
        "first_candidate_knobs": {
            "anchor_conditioned_reg_target_cameras": ["Camera_B1"],
            "anchor_conditioned_reg_target_scale": 0.875,
            "anchor_conditioned_reg_target_train_only": True,
            "anchor_conditioned_reg_target_quality_interp": "quadratic",
            "anchor_conditioned_reg_target_quality_low": QUALITY_LOW,
            "anchor_conditioned_reg_target_quality_high": QUALITY_HIGH,
            "anchor_conditioned_reg_target_foreground_edge_band_px": 5,
            "anchor_conditioned_conf_target_cameras": ["Camera_B1"],
            "anchor_conditioned_conf_target_scale": 0.875,
            "anchor_conditioned_conf_target_train_only": True,
            "anchor_conditioned_conf_target_quality_interp": "quadratic",
            "anchor_conditioned_conf_target_quality_low": QUALITY_LOW,
            "anchor_conditioned_conf_target_quality_high": QUALITY_HIGH,
            "anchor_conditioned_conf_target_foreground_edge_band_px": 5,
            "anchor_conditioned_conf_target_depth_conf_interp": "smoothstep",
            "anchor_conditioned_conf_target_depth_conf_low": 0.0,
            "anchor_conditioned_conf_target_depth_conf_high": DEPTH_CONF_P60_MAX,
        },
        "historical_prior": dict(HISTORICAL_THRESHOLD_POW2_PRIOR),
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "This family must stay single-candidate under the current overnight contract."
        },
        "supervision_audit_expected": False,
        "requires_dataset_plumbing": False,
        "cloud_must_remain_off": True,
    }


def build_unproject_consistency_blueprint() -> dict:
    return {
        "checked_at": iso_now(),
        "family": "unproject_consistency_routing",
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The first interpolated, partial, and disagreement-routing research tickets all produced formal "
            "dead_same_day verdicts at 10/5, so the next approved problem must change the routing basis again "
            "rather than keep tuning another depth-branch-local cousin."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another interpolated, partial, or disagreement near-neighbor: the routing "
            "signal changes from branch-local depth scores to detached unprojection residual from predicted "
            "depth plus predicted pose on anchor-supervised pixels."
        ),
        "reference_failures": [
            {
                "family": "interpolated_eligibility_shaping",
                "shape": "smoothstep_taper",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2291,
                    "reg_depth": 0.1759,
                    "delta_conf_depth": 0.0003,
                    "delta_reg_depth": 0.0,
                },
            },
            {
                "family": "partial_joint_depth_routing",
                "shape": "conf_branch_smoothstep_subset",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2303,
                    "reg_depth": 0.1761,
                    "delta_conf_depth": 0.0015,
                    "delta_reg_depth": 0.0002,
                },
            },
            {
                "family": "conf_reg_disagreement_routing",
                "shape": "anchor_disagreement_joint_routing",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2339,
                    "reg_depth": 0.1769,
                    "delta_conf_depth": 0.0051,
                    "delta_reg_depth": 0.0010,
                },
            },
        ],
        "signal_definition": (
            "Reconstruct world points from predicted depth plus predicted pose, compare them against "
            "batch['world_points'], and use the detached normalized residual on anchor-supervised pixels as "
            "the routing signal."
        ),
        "scope_definition": "anchor-supervised pixels only",
        "first_candidate_hypothesis": (
            "On anchor-supervised Camera_B1 pixels only, the remaining bad pixels may be the ones where "
            "predicted depth and predicted pose jointly reconstruct the wrong 3D point even when branch-local "
            "confidence signals do not mark them clearly enough. Routing from detached unprojection residual "
            "may therefore correct the residual depth gap without touching camera/T."
        ),
        "first_candidate_shape": DEFAULT_UNPROJECT_CONSISTENCY_SHAPE,
        "first_candidate_config": DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG,
        ],
        "first_candidate_execution_note": (
            "This first unprojection-consistency candidate is now executable on the repo: the required loss.py "
            "support is in place, so approval can go straight to smoke and gate."
        ),
        "first_candidate_knobs": {
            "anchor_conditioned_unproject_consistency_cameras": ["Camera_B1"],
            "anchor_conditioned_unproject_consistency_conf_scale": 0.875,
            "anchor_conditioned_unproject_consistency_reg_scale": 1.125,
            "anchor_conditioned_unproject_consistency_train_only": True,
        },
        "required_exclusions": [
            "not interpolated cousin",
            "not partial cousin",
            "not conf_reg_disagreement cousin",
            "not wholefg reopen",
            "not edge-band reopen",
            "not hard-threshold reopen",
            "not pow-like reopen",
            "not plain anchor_view_only reopen",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single first candidate is eligible under the current contract."
        },
        "cloud_must_remain_off": True,
    }


def build_unproject_aux_confgate_blueprint() -> dict:
    return {
        "checked_at": iso_now(),
        "family": "unproject_aux_confgate",
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The first interpolated, partial, disagreement-routing, and unprojection-consistency tickets all "
            "closed with formal same-day failures, so the next approved problem should stop retuning depth-loss "
            "routing and instead move to the objective-layer geometry auxiliary branch."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another routing cousin: it keeps the current stable lead's depth routing "
            "unchanged and changes only how the auxiliary unproject_geometry term is applied."
        ),
        "reference_failures": [
            {
                "family": "interpolated_eligibility_shaping",
                "shape": "smoothstep_taper",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2291,
                    "reg_depth": 0.1759,
                    "delta_conf_depth": 0.0003,
                    "delta_reg_depth": 0.0,
                },
            },
            {
                "family": "partial_joint_depth_routing",
                "shape": "conf_branch_smoothstep_subset",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2303,
                    "reg_depth": 0.1761,
                    "delta_conf_depth": 0.0015,
                    "delta_reg_depth": 0.0002,
                },
            },
            {
                "family": "conf_reg_disagreement_routing",
                "shape": "anchor_disagreement_joint_routing",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2339,
                    "reg_depth": 0.1769,
                    "delta_conf_depth": 0.0051,
                    "delta_reg_depth": 0.0010,
                },
            },
            {
                "family": "unproject_consistency_routing",
                "shape": "anchor_unproject_consistency_joint_routing",
                "short_gate_vs_lead": {
                    "conf_depth": 0.2348,
                    "reg_depth": 0.1771,
                    "delta_conf_depth": 0.0060,
                    "delta_reg_depth": 0.0012,
                },
            },
        ],
        "signal_definition": (
            "Keep the current source-policy stable lead's camera/depth routing fixed, but gate the auxiliary "
            "unproject_geometry loss by detached per-pixel depth confidence so the geometry chain concentrates "
            "on pixels the depth branch already trusts."
        ),
        "scope_definition": "the auxiliary unproject_geometry term only; do not rewrite depth-loss routing",
        "first_candidate_hypothesis": (
            "The current stable lead already fixes camera/T but still leaves a depth gap. A lighter "
            "confidence-gated auxiliary geometry term may recover geometry supervision without repeating another "
            "depth-routing cousin sweep."
        ),
        "first_candidate_shape": DEFAULT_UNPROJECT_AUX_CONFGATE_SHAPE,
        "first_candidate_config": DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG,
        ],
        "first_candidate_execution_note": (
            "This first auxiliary-geometry confgate candidate is already executable on the repo because the "
            "required loss.py support already exists; only a stable-lead-derived config is needed."
        ),
        "first_candidate_knobs": {
            "loss.unproject_geometry.weight": 0.05,
            "loss.unproject_geometry.use_depth_conf_gate": True,
            "loss.unproject_geometry.detach_depth_conf": True,
            "loss.unproject_geometry.depth_conf_power": 1.0,
            "loss.unproject_geometry.depth_conf_threshold": 0.0,
        },
        "required_exclusions": [
            "not interpolated cousin",
            "not partial cousin",
            "not conf_reg_disagreement cousin",
            "not unproject_consistency cousin",
            "not reliable-region unproject reopen",
            "not bottom-only unproject reopen",
            "not hard-threshold reopen",
            "not pow-like reopen",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single first candidate is eligible under the current contract.",
        },
        "cloud_must_remain_off": True,
    }


def build_source_policy_hybrid_ring_blueprint() -> dict:
    return {
        "checked_at": iso_now(),
        "family": "source_policy_hybrid_ring_regularization",
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The first interpolated, partial, disagreement-routing, unprojection-consistency, and "
            "auxiliary-unprojection tickets all closed with formal same-day failures, so the next approved "
            "problem should stop changing loss routing and move to the still-open source-policy rule "
            "regularization dimension that the overnight manifests kept recommending."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another loss-routing cousin: it keeps the current stable lead's depth "
            "routing intact and changes only the rawpool source-view rule from fully nearest to a "
            "nearest-dominant hybrid with a uniform tail."
        ),
        "reference_failures": [
            {
                "family": "interpolated_eligibility_shaping",
                "shape": "smoothstep_taper",
                "verdict": "dead_same_day",
            },
            {
                "family": "partial_joint_depth_routing",
                "shape": "conf_branch_smoothstep_subset",
                "verdict": "dead_same_day",
            },
            {
                "family": "conf_reg_disagreement_routing",
                "shape": "anchor_disagreement_joint_routing",
                "verdict": "dead_same_day",
            },
            {
                "family": "unproject_consistency_routing",
                "shape": "anchor_unproject_consistency_joint_routing",
                "verdict": "dead_same_day",
            },
            {
                "family": "unproject_aux_confgate",
                "shape": "stablelead_unproject_aux_confgate_w005",
                "verdict": "dead_same_day",
            },
        ],
        "signal_definition": (
            "Keep the current confdepth_dropworst_gradconfmask stable lead intact, but regularize source "
            "selection so rawpool keeps its nearest-ring stability while forcing one tail source toward wider "
            "ring coverage."
        ),
        "scope_definition": "source-policy rule only; do not rewrite depth-loss routing or auxiliary unproject loss",
        "first_candidate_hypothesis": (
            "The current stable lead already fixed camera/T and stabilized the depth path, but its rawpool "
            "source rule stays fully local. Adding one uniform-coverage tail source may improve depth-side "
            "coverage without reopening rejected sampler toggles or routing cousins."
        ),
        "first_candidate_shape": DEFAULT_SOURCE_POLICY_HYBRID_RING_SHAPE,
        "first_candidate_config": DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG,
        ],
        "first_candidate_execution_note": (
            "This first hybrid-ring candidate is executable on the current repo because the dataset policy "
            "support is now materialized locally; approval can go straight into gate with a config-only launch."
        ),
        "first_candidate_knobs": {
            "zju_source_policy": "nearest_plus_uniform_tail",
            "zju_source_view_pool": "geom_plus_raw",
            "zju_min_supervised_views": 1,
            "loss.depth.respect_conf_mask_in_grad_conf": True,
        },
        "required_exclusions": [
            "not interpolated cousin",
            "not partial cousin",
            "not conf_reg_disagreement cousin",
            "not unproject_consistency cousin",
            "not unproject_aux_confgate cousin",
            "not uniform-only rawpool reopen",
            "not trainmix reopen",
            "not min_supervised_views=2 reopen",
            "not max_depth_conf anchor reopen",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single first candidate is eligible under the current contract.",
        },
        "cloud_must_remain_off": True,
    }


def default_frontier_progression() -> list[dict]:
    return [dict(item) for item in DEFAULT_FRONTIER_PROGRESSION]


def default_latest_manual_gate() -> dict:
    return dict(LATEST_MANUAL_GATE)


def build_frontier_ledger(task_plan: dict, local_manifest: dict, candidate_verdict: dict) -> dict:
    latest_manual_gate = task_plan.get("latest_manual_gate", {}) or default_latest_manual_gate()
    stable_metrics = latest_manual_gate.get("val_metrics_stable_lead", {}) or dict(STABLE_LEAD_VAL_METRICS)
    current_lead = local_manifest.get("current_lead", {})
    current_priority = build_current_priority(candidate_verdict)
    auto_next = build_auto_next_ticket_fields(current_priority)
    return {
        "checked_at": iso_now(),
        "stable_lead_config": current_lead.get("config", ""),
        "stable_lead_val_metrics": stable_metrics,
        "frontier_progression": list(task_plan.get("frontier_progression", [])) or default_frontier_progression(),
        "frozen_families": FROZEN_FAMILIES,
        "preferred_first_family": auto_next["preferred_first_family"],
        "preferred_first_family_reason": auto_next["preferred_first_family_reason"],
        "family_readout": {
            "wholefg_jointdepth_scalar": {
                "status": "bounded_wrong_side",
                "stop_reason": "softening approached the stable lead from the wrong side without crossing"
            },
            "wholefg_decoupled": {
                "status": "bounded_wrong_side",
                "stop_reason": "first decoupled conf/reg near-neighbor still regressed both depth terms"
            },
            "nonwholefg_edge_band_jointdepth": {
                "status": "bounded_wrong_side",
                "stop_reason": "edge-band frontier got very close but still did not beat the stable lead"
            },
            "nonwholefg_edge_band_decoupled": {
                "status": "bounded_wrong_side",
                "stop_reason": "edge-band decoupled near-neighbor stayed worse than the best edge-band scalar point"
            },
            "hard_pixel_depth_conf_threshold": {
                "status": "rejected_too_aggressive",
                "stop_reason": "hard gt depth_conf<=p60 threshold worsened conf_depth and reg_depth inside the promising edge-band route"
            }
        },
        "recommended_next_families": current_priority["recommended_next_families"],
        "recommended_family_order": current_priority["recommended_family_order"],
        "current_priority_family": current_priority["current_priority_family"],
        "current_priority_reason": current_priority["current_priority_reason"],
        "current_priority_candidate_shape": current_priority["current_priority_candidate_shape"],
        "current_priority_candidate_config": current_priority["current_priority_candidate_config"],
        "same_family_retry_forbidden": current_priority["same_family_retry_forbidden"],
        "same_family_retry_reason": current_priority["same_family_retry_reason"],
        "wrong_side_asymptote_families": [
            "wholefg_jointdepth_scalar",
            "nonwholefg_edge_band_jointdepth"
        ],
        "latest_formal_verdict": summarize_candidate_verdict(candidate_verdict),
        "latest_family_outcomes": build_latest_family_outcomes(candidate_verdict),
    }


def build_family_stop_reason(candidate_verdict: dict) -> dict:
    current_priority = build_current_priority(candidate_verdict)
    auto_next = build_auto_next_ticket_fields(current_priority)
    return {
        "checked_at": iso_now(),
        "frozen_families": {
            "wholefg_scalar_near_neighbors": "bounded wrong-side frontier; do not reopen without a different selectivity argument",
            "wholefg_decoupled_near_neighbors": "first branch split already lost to the stable lead",
            "edge_band_scalar_near_neighbors": "bounded wrong-side frontier; do not keep softening near-neighbors",
            "edge_band_decoupled_near_neighbors": "first edge-band branch split stayed worse than the best edge-band scalar point",
            "hard_pixel_level_depth_conf_threshold": "too aggressive inside the promising non-wholefg edge-band route",
            "plain_anchor_view_only": "close to a no-op under zju_min_supervised_views=1 on the current rawpool recipe"
        },
        "preferred_first_family": auto_next["preferred_first_family"],
        "preferred_first_family_reason": auto_next["preferred_first_family_reason"],
        "current_priority_family": current_priority["current_priority_family"],
        "current_priority_reason": current_priority["current_priority_reason"],
        "current_priority_candidate_shape": current_priority["current_priority_candidate_shape"],
        "current_priority_candidate_config": current_priority["current_priority_candidate_config"],
        "same_family_retry_forbidden": current_priority["same_family_retry_forbidden"],
        "same_family_retry_reason": current_priority["same_family_retry_reason"],
        "next_requirement": current_priority["next_requirement"],
        "cloud_policy": "cloud must remain off regardless of research-loop state until a candidate clears long gate locally",
        "latest_formal_verdict": summarize_candidate_verdict(candidate_verdict),
        "latest_family_outcomes": build_latest_family_outcomes(candidate_verdict),
    }


def build_gate_reference_logs() -> dict:
    stable_short_summary = load_json(SHORT_GATE_STABLE_REFERENCE_SUMMARY_PATH)
    baseline_short_summary = load_json(SHORT_GATE_BASELINE_REFERENCE_SUMMARY_PATH)
    long_gate_status = load_json(LONG_GATE_REFERENCE_STATUS_PATH)

    stable_short_log = resolve_repo_path(stable_short_summary["baseline_log"])
    baseline_short_log = resolve_repo_path(baseline_short_summary["baseline_log"])
    baseline_long_log = resolve_repo_path(long_gate_status["artifacts"]["baseline_log"])
    stable_long_log = resolve_repo_path(long_gate_status["artifacts"]["previous_lead_log"])

    if not stable_short_log.exists():
        raise FileNotFoundError(f"Missing canonical short-gate stable reference log: {stable_short_log}")
    if not baseline_short_log.exists():
        raise FileNotFoundError(f"Missing canonical short-gate baseline reference log: {baseline_short_log}")
    if not baseline_long_log.exists():
        raise FileNotFoundError(f"Missing canonical long-gate baseline reference log: {baseline_long_log}")
    if not stable_long_log.exists():
        raise FileNotFoundError(f"Missing canonical long-gate stable-lead reference log: {stable_long_log}")

    return {
        "checked_at": iso_now(),
        "short_gate": {
            "stable_lead_reference_summary": str(SHORT_GATE_STABLE_REFERENCE_SUMMARY_PATH.resolve()),
            "stable_lead_reference_log": str(stable_short_log),
            "baseline_reference_summary": str(SHORT_GATE_BASELINE_REFERENCE_SUMMARY_PATH.resolve()),
            "baseline_reference_log": str(baseline_short_log),
        },
        "long_gate": {
            "reference_status": str(LONG_GATE_REFERENCE_STATUS_PATH.resolve()),
            "baseline_reference_log": str(baseline_long_log),
            "stable_lead_reference_log": str(stable_long_log),
            "stable_lead_reference_source_field": "previous_lead_log",
            "stable_lead_reference_note": (
                "The canonical 100/20 stable-lead reference comes from the previous_lead branch in the "
                "20260326 long-gate validation, because that run treated the current stable lead as the "
                "previous local lead while testing a rejected candidate as current_lead."
            ),
        },
        "reuse_policy": {
            "short_gate_reuses_existing_baseline_and_stable_logs": True,
            "long_gate_reuses_existing_baseline_and_stable_logs": True,
            "future_approved_candidate_long_gate_runs_candidate_only": True,
        },
    }


def build_candidate_patch_plan(
    args: argparse.Namespace,
    approved_problem: dict,
    local_manifest: dict,
    task_plan: dict,
    gate_reference_logs: dict,
    candidate_verdict: dict,
) -> dict:
    approved_problem_present = bool(approved_problem)
    approved_problem_ready = bool(approved_problem_present and approved_problem.get("approved"))
    validation_issues = (
        validate_approved_problem(
            approved_problem,
            args.max_approved_problems_per_night,
            args.max_candidates_per_problem,
        )
        if approved_problem_present
        else []
    )
    if approved_problem_present and validation_issues:
        state = STATE_CONTRACT_REJECTED
    else:
        state = STATE_ARMED_PROBLEM if approved_problem_ready else STATE_IDLE_GUARD
    stable_lead = local_manifest.get("current_lead", {}).get("config", "")
    current_priority = build_current_priority(candidate_verdict)
    auto_next = build_auto_next_ticket_fields(current_priority)
    return {
        "checked_at": iso_now(),
        "state": state,
        "approved_problem_present": approved_problem_present,
        "approved_problem_ready": approved_problem_ready,
        "approved_problem_validation_issues": validation_issues,
        "current_stable_lead_config": stable_lead,
        "allowed_families": auto_next["allowed_families"],
        "preferred_first_family": auto_next["preferred_first_family"],
        "preferred_first_family_reason": auto_next["preferred_first_family_reason"],
        "current_priority_family": current_priority["current_priority_family"],
        "current_priority_reason": current_priority["current_priority_reason"],
        "current_priority_candidate_shape": current_priority["current_priority_candidate_shape"],
        "current_priority_candidate_config": current_priority["current_priority_candidate_config"],
        "current_priority_candidate_requires_code_patch": current_priority["current_priority_candidate_requires_code_patch"],
        "current_priority_candidate_write_surface": current_priority["current_priority_candidate_write_surface"],
        "current_priority_candidate_execution_note": current_priority["current_priority_candidate_execution_note"],
        "current_priority_arm_command": current_priority["current_priority_arm_command"],
        "current_priority_run_command": current_priority["current_priority_run_command"],
        "preferred_first_candidate_shape": auto_next["preferred_first_candidate_shape"],
        "preferred_first_candidate_shape_reason": auto_next["preferred_first_candidate_shape_reason"],
        "preferred_first_candidate_config": auto_next["preferred_first_candidate_config"],
        "preferred_first_candidate_requires_code_patch": auto_next["preferred_first_candidate_requires_code_patch"],
        "preferred_first_candidate_write_surface": auto_next["preferred_first_candidate_write_surface"],
        "preferred_first_candidate_execution_note": auto_next["preferred_first_candidate_execution_note"],
        "gate_reference_logs_path": str(args.gate_reference_logs_path.resolve()),
        "gate_reference_reuse_note": (
            "Both short-gate and long-gate reference lines are canonicalized in gate_reference_logs.json, so "
            "future approved-candidate runs only need to execute the candidate path and compare against the "
            "stored baseline/stable logs."
        ),
        "approved_problem_archive_root": str(DEFAULT_APPROVED_PROBLEM_ARCHIVE_ROOT.resolve()),
        "approved_problem_consumption_note": (
            "A real approved-candidate run must archive and remove the active approved_problem.json on exit so "
            "the research loop returns cleanly to IDLE_GUARD."
        ),
        "short_gate_reference_logs": gate_reference_logs.get("short_gate", {}),
        "long_gate_reference_logs": gate_reference_logs.get("long_gate", {}),
        "approval_helper_path": str(DEFAULT_APPROVAL_HELPER_PATH.resolve()),
        "execution_runner_path": str(DEFAULT_APPROVED_RUNNER_PATH.resolve()),
        "preferred_first_candidate_arm_command": auto_next["preferred_first_candidate_arm_command"],
        "preferred_first_candidate_run_command": auto_next["preferred_first_candidate_run_command"],
        "family_blueprints": auto_next["family_blueprints"],
        "forbidden_families": FROZEN_FAMILIES,
        "allowed_write_scope": ALLOWED_WRITE_SCOPE,
        "forbidden_write_scope": FORBIDDEN_WRITE_SCOPE,
        "state_machine": RESEARCH_STATE_MACHINE,
        "repo_process_allowlist_path": str(DEFAULT_REPO_PROCESS_ALLOWLIST_PATH.resolve()),
        "gate_sequence": [
            STATE_SMOKE_1X1,
            STATE_TIGHT_GATE_10X5,
            STATE_LONG_GATE_100X20,
            STATE_VERDICT_WRITEBACK,
            STATE_RETURN_TO_GUARD
        ],
        "auto_problem_generation_forbidden_without_manual_approval": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "cross_night_loop_only": True,
        "same_night_stop_rules": [
            "smoke_1x1_failure_returns_to_guard",
            "tight_gate_10x5_failure_returns_to_guard",
            "long_gate_100x20_failure_returns_to_guard",
            "contract_validation_failure_returns_to_guard",
            "reference_missing_returns_to_guard",
            "compare_or_writeback_or_archive_failure_returns_to_guard",
        ],
        "promotion_rule": {
            "short_gate_must_beat_stable_lead": True,
            "long_gate_required_for_promotion": True,
            "camera_must_not_regress": True,
            "translation_must_not_regress": True,
            "conf_depth_must_improve": True,
            "reg_depth_must_improve": True,
            "cloud_must_remain_off": True
        },
        "nightly_budget": {
            "max_approved_problems_per_night": int(args.max_approved_problems_per_night),
            "max_candidates_per_problem": int(args.max_candidates_per_problem),
            "max_candidates_per_night": int(args.max_candidates_per_problem),
            "stop_family_after_two_wrong_side_near_misses": True
        },
        "disk_policy": {
            "disk_floor_gb": float(args.disk_floor_gb),
            "cleanup_only_current_round_artifacts": True,
            "cleanup_targets": [
                "training/logs/**/ckpts",
                "training/logs/**/tensorboard"
            ],
            "preserve_targets": [
                "training/logs/**/log.txt",
                "output/**/summary.md",
                "training/config/*.yaml",
                "output/zju_source_policy_research_loop/*.json"
            ]
        },
        "current_frontier_hint": current_priority["current_frontier_hint"],
        "current_frontier_priority": current_priority["current_frontier_priority"],
        "same_family_retry_forbidden": current_priority["same_family_retry_forbidden"],
        "same_family_retry_reason": current_priority["same_family_retry_reason"],
        "next_requirement": current_priority["next_requirement"],
        "repo_process_allowlist_requirement": "If a future approved problem launches repo-scoped training or compare processes, register only the current-round markers in repo_process_allowlist.json before launch and clear them again on return to guard.",
        "approved_problem": approved_problem if approved_problem_present else {}
    }


def build_candidate_verdict(
    task_plan: dict,
    approved_problem: dict,
    validation_issues: list[str],
    existing_candidate_verdict: dict,
) -> dict:
    latest = task_plan.get("latest_manual_gate", {}) or default_latest_manual_gate()
    payload = existing_candidate_verdict.copy() if existing_candidate_verdict else {}
    status = "no_active_candidate"
    if approved_problem and validation_issues:
        status = "contract_rejected"
        payload.update(
            {
                "checked_at": iso_now(),
                "status": status,
                "active_candidate": str(approved_problem.get("first_candidate_config", "")).strip(),
                "problem_id": approved_problem.get("problem_id", ""),
                "family": approved_problem.get("family", ""),
                "first_candidate_shape": approved_problem.get("first_candidate_shape", ""),
                "reason": "approved_problem.json failed contract validation before any candidate launch.",
            }
        )
    elif payload and str(payload.get("status", "")).strip() and payload.get("status") != "no_active_candidate":
        payload["checked_at"] = iso_now()
    else:
        payload = {
            "checked_at": iso_now(),
            "status": status,
            "active_candidate": "",
        }
    payload["latest_known_candidate"] = latest.get("candidate", "")
    payload["latest_known_verdict"] = latest.get("verdict", "")
    payload["latest_known_interpretation"] = latest.get("interpretation", "")
    payload["latest_known_val_metrics_stable_lead"] = latest.get("val_metrics_stable_lead", {})
    payload["latest_known_val_metrics_candidate"] = latest.get("val_metrics_candidate", {})
    payload["approved_problem_validation_issues"] = validation_issues
    payload.setdefault("approved_problem_archive_path", "")
    return payload


def build_resume_token(approved_problem: dict, validation_issues: list[str]) -> dict:
    approved_problem_present = bool(approved_problem)
    approved_problem_ready = bool(approved_problem_present and approved_problem.get("approved"))
    if approved_problem_present and validation_issues:
        state = STATE_CONTRACT_REJECTED
        next_allowed_state = STATE_IDLE_GUARD
    else:
        state = STATE_ARMED_PROBLEM if approved_problem_ready else STATE_IDLE_GUARD
        next_allowed_state = STATE_SYNTHESIZE_ONE_CANDIDATE if approved_problem_ready else STATE_ARMED_PROBLEM
    return {
        "checked_at": iso_now(),
        "state": state,
        "next_allowed_state": next_allowed_state,
        "active_problem_id": approved_problem.get("problem_id", "") if approved_problem_ready else "",
        "active_candidate_count": 0,
        "cloud_gate_must_remain_false": True,
        "notes": "Research loop stays separate from guard-only steady_hold and must return to guard after verdict writeback."
    }


def validate_approved_problem(
    approved_problem: dict,
    max_approved_problems_per_night: int,
    max_candidates_per_problem: int,
) -> list[str]:
    issues: list[str] = []
    approved = bool(approved_problem.get("approved"))
    family = str(approved_problem.get("family", ""))
    mutation_dsl = approved_problem.get("mutation_dsl", {}) or {}
    if family and family not in ALLOWED_FAMILIES:
        issues.append(f"family_not_allowed:{family}")
    candidate_budget = int(approved_problem.get("candidate_budget", 0) or 0)
    max_candidates_per_night = int(approved_problem.get("max_candidates_per_night", 0) or 0)
    max_approved_per_night = int(approved_problem.get("max_approved_problems_per_night", 0) or 0)
    if candidate_budget != max_candidates_per_problem:
        issues.append("candidate_budget_must_match_single_candidate_contract")
    if max_candidates_per_night != max_candidates_per_problem:
        issues.append("max_candidates_per_night_must_match_single_candidate_contract")
    if max_approved_per_night != max_approved_problems_per_night:
        issues.append("max_approved_problems_per_night_must_match_single_problem_contract")
    if approved_problem.get("cloud_must_remain_off") is False:
        issues.append("cloud_must_remain_off_false")
    if approved_problem.get("long_gate_required_for_promotion") is False:
        issues.append("long_gate_requirement_disabled")
    if mutation_dsl.get("disallow_wholefg_scalar") is not True:
        issues.append("mutation_dsl_must_disallow_wholefg_scalar")
    if mutation_dsl.get("disallow_wholefg_decoupled") is not True:
        issues.append("mutation_dsl_must_disallow_wholefg_decoupled")
    if mutation_dsl.get("disallow_edge_band_scalar") is not True:
        issues.append("mutation_dsl_must_disallow_edge_band_scalar")
    if mutation_dsl.get("disallow_edge_band_decoupled") is not True:
        issues.append("mutation_dsl_must_disallow_edge_band_decoupled")
    if mutation_dsl.get("disallow_hard_depth_conf_threshold") is not True:
        issues.append("mutation_dsl_must_disallow_hard_depth_conf_threshold")
    if mutation_dsl.get("disallow_plain_anchor_view_only") is not True:
        issues.append("mutation_dsl_must_disallow_plain_anchor_view_only")
    if family == "unproject_consistency_routing":
        if mutation_dsl.get("non_wholefg_selectivity_required") is not True:
            issues.append("mutation_dsl_must_require_non_wholefg_selectivity")
        if mutation_dsl.get("allow_unproject_consistency_routing") is not True:
            issues.append("mutation_dsl_must_allow_unproject_consistency_routing")
    if family == "unproject_aux_confgate":
        if mutation_dsl.get("allow_unproject_aux_confgate") is not True:
            issues.append("mutation_dsl_must_allow_unproject_aux_confgate")
        if mutation_dsl.get("keep_existing_depth_routing_unchanged") is not True:
            issues.append("mutation_dsl_must_keep_existing_depth_routing_unchanged")
        if mutation_dsl.get("disallow_reliable_region_unproject_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_reliable_region_unproject_reopen")
        if mutation_dsl.get("disallow_bottom_only_unproject_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_bottom_only_unproject_reopen")
    if family == "source_policy_hybrid_ring_regularization":
        if mutation_dsl.get("allow_source_policy_hybrid_ring_regularization") is not True:
            issues.append("mutation_dsl_must_allow_source_policy_hybrid_ring_regularization")
        if mutation_dsl.get("keep_existing_depth_routing_unchanged") is not True:
            issues.append("mutation_dsl_must_keep_existing_depth_routing_unchanged")
        if mutation_dsl.get("keep_geom_plus_raw_pool") is not True:
            issues.append("mutation_dsl_must_keep_geom_plus_raw_pool")
        if mutation_dsl.get("keep_single_supervised_anchor") is not True:
            issues.append("mutation_dsl_must_keep_single_supervised_anchor")
        if mutation_dsl.get("disallow_uniform_only_rawpool_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_uniform_only_rawpool_reopen")
        if mutation_dsl.get("disallow_trainmix_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_trainmix_reopen")
        if mutation_dsl.get("disallow_minsup2_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_minsup2_reopen")
        if mutation_dsl.get("disallow_max_depth_conf_anchor_reopen") is not True:
            issues.append("mutation_dsl_must_disallow_max_depth_conf_anchor_reopen")
    if approved:
        if not str(approved_problem.get("problem_id", "")).strip():
            issues.append("problem_id_required_for_approved_problem")
        if not str(approved_problem.get("problem_title", "")).strip():
            issues.append("problem_title_required_for_approved_problem")
        if not family.strip():
            issues.append("family_required_for_approved_problem")
        if not str(approved_problem.get("first_candidate_shape", "")).strip():
            issues.append("first_candidate_shape_required_for_approved_problem")
        first_candidate_config = str(approved_problem.get("first_candidate_config", "")).strip()
        if not first_candidate_config:
            issues.append("first_candidate_config_required_for_approved_problem")
        elif not resolve_repo_path(first_candidate_config).exists():
            issues.append("first_candidate_config_missing_on_disk")
        if family == "interpolated_eligibility_shaping":
            if str(approved_problem.get("first_candidate_shape", "")).strip() != "smoothstep_taper":
                issues.append("interpolated_first_candidate_must_be_smoothstep_taper")
            if first_candidate_config and first_candidate_config != DEFAULT_INTERP_SMOOTHSTEP_CANDIDATE_CONFIG:
                issues.append("interpolated_first_candidate_must_match_prebuilt_config")
            if bool(approved_problem.get("first_candidate_requires_code_patch", True)):
                issues.append("interpolated_first_candidate_must_stay_config_only")
        if family == "unproject_consistency_routing":
            if str(approved_problem.get("first_candidate_shape", "")).strip() != DEFAULT_UNPROJECT_CONSISTENCY_SHAPE:
                issues.append("unproject_consistency_first_candidate_must_match_anchor_unproject_consistency_joint_routing")
            if first_candidate_config and first_candidate_config != DEFAULT_UNPROJECT_CONSISTENCY_CANDIDATE_CONFIG:
                issues.append("unproject_consistency_first_candidate_must_match_prebuilt_config")
            if bool(approved_problem.get("first_candidate_requires_code_patch", True)):
                issues.append("unproject_consistency_first_candidate_must_stay_repo_ready")
        if family == "unproject_aux_confgate":
            if str(approved_problem.get("first_candidate_shape", "")).strip() != DEFAULT_UNPROJECT_AUX_CONFGATE_SHAPE:
                issues.append("unproject_aux_confgate_first_candidate_must_match_stablelead_unproject_aux_confgate_w005")
            if first_candidate_config and first_candidate_config != DEFAULT_UNPROJECT_AUX_CONFGATE_CANDIDATE_CONFIG:
                issues.append("unproject_aux_confgate_first_candidate_must_match_prebuilt_config")
            if bool(approved_problem.get("first_candidate_requires_code_patch", True)):
                issues.append("unproject_aux_confgate_first_candidate_must_stay_config_only")
        if family == "source_policy_hybrid_ring_regularization":
            if str(approved_problem.get("first_candidate_shape", "")).strip() != DEFAULT_SOURCE_POLICY_HYBRID_RING_SHAPE:
                issues.append("source_policy_hybrid_ring_first_candidate_must_match_stablelead_nearest_plus_uniform_tail")
            if first_candidate_config and first_candidate_config != DEFAULT_SOURCE_POLICY_HYBRID_RING_CANDIDATE_CONFIG:
                issues.append("source_policy_hybrid_ring_first_candidate_must_match_prebuilt_config")
            if bool(approved_problem.get("first_candidate_requires_code_patch", True)):
                issues.append("source_policy_hybrid_ring_first_candidate_must_stay_config_only")
    return issues


def build_status(
    approved_problem: dict,
    validation_issues: list[str],
    local_manifest: dict,
    task_plan: dict,
    candidate_verdict: dict,
) -> dict:
    current_priority = build_current_priority(candidate_verdict)
    auto_next = build_auto_next_ticket_fields(current_priority)
    approved_problem_present = bool(approved_problem)
    approved_problem_ready = bool(approved_problem_present and approved_problem.get("approved"))
    if not approved_problem_present:
        state = STATE_IDLE_GUARD
        reason = "No approved_problem.json is present; research loop remains idle while guard-only steady_hold continues."
    elif not approved_problem_ready:
        state = STATE_IDLE_GUARD
        reason = "approved_problem.json exists but is not marked approved=true; research loop remains idle."
    elif validation_issues:
        state = STATE_CONTRACT_REJECTED
        reason = "Approved problem failed research-loop contract validation and was rejected before any candidate synthesis."
    else:
        state = STATE_ARMED_PROBLEM
        reason = "Approved problem passes contract validation and is ready for one-candidate synthesis under the research-loop budget."
    return {
        "checked_at": iso_now(),
        "state": state,
        "reason": reason,
        "research_loop_entrypoint": str((REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py").resolve()),
        "approved_problem_path": str(DEFAULT_APPROVED_PROBLEM_PATH.resolve()),
        "approved_problem_present": approved_problem_present,
        "approved_problem_ready": approved_problem_ready,
        "approved_problem_validation_issues": validation_issues,
        "guard_mode_expected": "steady_hold",
        "guard_remains_separate": True,
        "repo_process_allowlist_path": str(DEFAULT_REPO_PROCESS_ALLOWLIST_PATH.resolve()),
        "approved_problem_archive_root": str(DEFAULT_APPROVED_PROBLEM_ARCHIVE_ROOT.resolve()),
        "current_stable_lead_config": local_manifest.get("current_lead", {}).get("config", ""),
        "current_cloud_blocker": local_manifest.get("current_cloud_blocker", ""),
        "allowed_families": auto_next["allowed_families"],
        "preferred_first_family": auto_next["preferred_first_family"],
        "preferred_first_family_reason": auto_next["preferred_first_family_reason"],
        "current_priority_family": current_priority["current_priority_family"],
        "current_priority_reason": current_priority["current_priority_reason"],
        "current_priority_candidate_shape": current_priority["current_priority_candidate_shape"],
        "current_priority_candidate_config": current_priority["current_priority_candidate_config"],
        "same_family_retry_forbidden": current_priority["same_family_retry_forbidden"],
        "same_family_retry_reason": current_priority["same_family_retry_reason"],
        "next_requirement": current_priority["next_requirement"],
        "preferred_first_candidate_shape": auto_next["preferred_first_candidate_shape"],
        "preferred_first_candidate_shape_reason": auto_next["preferred_first_candidate_shape_reason"],
        "preferred_first_candidate_config": auto_next["preferred_first_candidate_config"],
        "preferred_first_candidate_requires_code_patch": auto_next["preferred_first_candidate_requires_code_patch"],
        "latest_formal_verdict": summarize_candidate_verdict(candidate_verdict),
        "gate_reference_logs_path": str(DEFAULT_GATE_REFERENCE_LOGS_PATH.resolve()),
        "long_gate_reference_reuse_enabled": True,
        "auto_problem_generation_forbidden_without_manual_approval": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "cross_night_loop_only": True,
        "historical_prior_doc": HISTORICAL_THRESHOLD_POW2_PRIOR["doc"],
        "historical_prior_takeaway": HISTORICAL_THRESHOLD_POW2_PRIOR["takeaway"],
        "frozen_families": FROZEN_FAMILIES,
        "current_frontier_hint": current_priority["current_frontier_hint"],
        "current_frontier_priority": current_priority["current_frontier_priority"],
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True
    }


def render_status_md(status: dict) -> str:
    lines = ["# ZJU Source-Policy Research Loop Status", ""]
    for key, value in status.items():
        if isinstance(value, list):
            lines.append(f"## {key}")
            lines.append("")
            for item in value:
                lines.append(f"- `{item}`")
            lines.append("")
        else:
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def render_candidate_patch_plan_md(plan: dict) -> str:
    lines = ["# Candidate Patch Plan", ""]
    lines.append(f"- checked_at: `{plan.get('checked_at', '')}`")
    lines.append(f"- state: `{plan.get('state', '')}`")
    lines.append(f"- approved_problem_present: `{plan.get('approved_problem_present', False)}`")
    lines.append(f"- approved_problem_ready: `{plan.get('approved_problem_ready', False)}`")
    lines.append(f"- current_stable_lead_config: `{plan.get('current_stable_lead_config', '')}`")
    lines.append("")
    lines.append("## Current Priority")
    lines.append("")
    lines.append(f"- current_priority_family: `{plan.get('current_priority_family', '')}`")
    lines.append(f"- current_priority_reason: `{plan.get('current_priority_reason', '')}`")
    lines.append(f"- current_priority_candidate_shape: `{plan.get('current_priority_candidate_shape', '')}`")
    lines.append(f"- current_priority_candidate_config: `{plan.get('current_priority_candidate_config', '')}`")
    lines.append(f"- current_priority_candidate_requires_code_patch: `{plan.get('current_priority_candidate_requires_code_patch', '')}`")
    lines.append(f"- current_priority_candidate_execution_note: `{plan.get('current_priority_candidate_execution_note', '')}`")
    lines.append(f"- current_priority_arm_command: `{plan.get('current_priority_arm_command', '')}`")
    lines.append(f"- current_priority_run_command: `{plan.get('current_priority_run_command', '')}`")
    lines.append(f"- same_family_retry_forbidden: `{plan.get('same_family_retry_forbidden', '')}`")
    lines.append(f"- same_family_retry_reason: `{plan.get('same_family_retry_reason', '')}`")
    lines.append(f"- next_requirement: `{plan.get('next_requirement', '')}`")
    lines.append(f"- preferred_first_family: `{plan.get('preferred_first_family', '')}`")
    lines.append(f"- why_first: `{plan.get('preferred_first_family_reason', '')}`")
    lines.append(f"- frontier_hint: `{plan.get('current_frontier_hint', '')}`")
    lines.append(f"- frontier_priority: `{plan.get('current_frontier_priority', '')}`")
    lines.append(f"- preferred_first_candidate_shape: `{plan.get('preferred_first_candidate_shape', '')}`")
    lines.append(f"- preferred_first_candidate_shape_reason: `{plan.get('preferred_first_candidate_shape_reason', '')}`")
    lines.append(f"- preferred_first_candidate_config: `{plan.get('preferred_first_candidate_config', '')}`")
    lines.append(f"- preferred_first_candidate_requires_code_patch: `{plan.get('preferred_first_candidate_requires_code_patch', '')}`")
    lines.append(f"- preferred_first_candidate_execution_note: `{plan.get('preferred_first_candidate_execution_note', '')}`")
    lines.append(f"- gate_reference_logs_path: `{plan.get('gate_reference_logs_path', '')}`")
    lines.append(f"- gate_reference_reuse_note: `{plan.get('gate_reference_reuse_note', '')}`")
    lines.append(f"- approved_problem_archive_root: `{plan.get('approved_problem_archive_root', '')}`")
    lines.append(f"- approved_problem_consumption_note: `{plan.get('approved_problem_consumption_note', '')}`")
    lines.append(f"- auto_problem_generation_forbidden_without_manual_approval: `{plan.get('auto_problem_generation_forbidden_without_manual_approval', '')}`")
    lines.append(f"- same_night_second_candidate_forbidden: `{plan.get('same_night_second_candidate_forbidden', '')}`")
    lines.append(f"- same_night_cousin_sweep_forbidden: `{plan.get('same_night_cousin_sweep_forbidden', '')}`")
    lines.append(f"- cross_night_loop_only: `{plan.get('cross_night_loop_only', '')}`")
    lines.append(f"- approval_helper_path: `{plan.get('approval_helper_path', '')}`")
    lines.append(f"- execution_runner_path: `{plan.get('execution_runner_path', '')}`")
    lines.append(f"- preferred_first_candidate_arm_command: `{plan.get('preferred_first_candidate_arm_command', '')}`")
    lines.append(f"- preferred_first_candidate_run_command: `{plan.get('preferred_first_candidate_run_command', '')}`")
    lines.append("")
    lines.append("## Gate References")
    lines.append("")
    short_gate_refs = plan.get("short_gate_reference_logs", {}) or {}
    long_gate_refs = plan.get("long_gate_reference_logs", {}) or {}
    for key, value in short_gate_refs.items():
        lines.append(f"- short_gate.{key}: `{value}`")
    for key, value in long_gate_refs.items():
        lines.append(f"- long_gate.{key}: `{value}`")
    lines.append("")
    lines.append("## Gate Sequence")
    lines.append("")
    for step in plan.get("gate_sequence", []):
        lines.append(f"- `{step}`")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    nightly_budget = plan.get("nightly_budget", {})
    lines.append(f"- max_approved_problems_per_night: `{nightly_budget.get('max_approved_problems_per_night', '')}`")
    lines.append(f"- max_candidates_per_problem: `{nightly_budget.get('max_candidates_per_problem', '')}`")
    lines.append(f"- max_candidates_per_night: `{nightly_budget.get('max_candidates_per_night', '')}`")
    lines.append("")
    lines.append("## Write Scope")
    lines.append("")
    for item in plan.get("allowed_write_scope", []):
        lines.append(f"- allowed: `{item}`")
    for item in plan.get("forbidden_write_scope", []):
        lines.append(f"- forbidden: `{item}`")
    lines.append("")
    lines.append("## Repo Process Allowlist")
    lines.append("")
    lines.append(f"- allowlist_path: `{plan.get('repo_process_allowlist_path', '')}`")
    lines.append(f"- rule: `{plan.get('repo_process_allowlist_requirement', '')}`")
    approved_problem = plan.get("approved_problem", {}) or {}
    lines.append("")
    lines.append("## Approved Problem")
    lines.append("")
    if approved_problem:
        for key in [
            "problem_id",
            "problem_title",
            "family",
            "problem_statement",
            "why_genuinely_new",
            "why_not_reopening_frozen_family",
            "first_candidate_hint",
            "historical_prior",
        ]:
            if key in approved_problem:
                lines.append(f"- {key}: `{approved_problem.get(key, '')}`")
    else:
        lines.append("- status: `no approved_problem.json is active; research loop remains idle`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()

    ensure_dir(args.output_root)
    local_manifest = load_json(args.local_manifest_path)
    training_question_manifest = load_json(args.training_question_manifest_path)
    task_plan = load_json(args.task_plan_path)

    write_json(
        args.approved_problem_template_path,
        build_approved_problem_template(args.max_approved_problems_per_night, args.max_candidates_per_problem),
    )
    write_json(
        DEFAULT_APPROVED_PROBLEM_UNPROJECT_CONSISTENCY_SEED_PATH,
        build_unproject_consistency_ready_seed(
            args.max_approved_problems_per_night,
            args.max_candidates_per_problem,
        ),
    )
    write_json(
        DEFAULT_APPROVED_PROBLEM_UNPROJECT_AUX_CONFGATE_SEED_PATH,
        build_unproject_aux_confgate_ready_seed(
            args.max_approved_problems_per_night,
            args.max_candidates_per_problem,
        ),
    )
    write_json(
        DEFAULT_APPROVED_PROBLEM_SOURCE_POLICY_HYBRID_RING_SEED_PATH,
        build_source_policy_hybrid_ring_ready_seed(
            args.max_approved_problems_per_night,
            args.max_candidates_per_problem,
        ),
    )

    approved_problem = maybe_load_json(args.approved_problem_path)
    validation_issues = (
        validate_approved_problem(
            approved_problem,
            args.max_approved_problems_per_night,
            args.max_candidates_per_problem,
        )
        if approved_problem
        else []
    )
    approved_problem_ready = bool(approved_problem and approved_problem.get("approved") and not validation_issues)

    write_json(
        DEFAULT_REPO_PROCESS_ALLOWLIST_TEMPLATE_PATH,
        build_repo_process_allowlist_template(),
    )
    write_json(
        DEFAULT_REPO_PROCESS_ALLOWLIST_PATH,
        build_repo_process_allowlist(approved_problem_ready),
    )
    write_json(
        DEFAULT_UNPROJECT_CONSISTENCY_BLUEPRINT_PATH,
        build_unproject_consistency_blueprint(),
    )
    write_json(
        DEFAULT_UNPROJECT_AUX_CONFGATE_BLUEPRINT_PATH,
        build_unproject_aux_confgate_blueprint(),
    )
    write_json(
        DEFAULT_SOURCE_POLICY_HYBRID_RING_BLUEPRINT_PATH,
        build_source_policy_hybrid_ring_blueprint(),
    )

    existing_candidate_verdict = maybe_load_json(args.candidate_verdict_path)
    candidate_verdict = build_candidate_verdict(
        task_plan,
        approved_problem,
        validation_issues,
        existing_candidate_verdict,
    )
    frontier_ledger = build_frontier_ledger(task_plan, local_manifest, candidate_verdict)
    family_stop_reason = build_family_stop_reason(candidate_verdict)
    gate_reference_logs = build_gate_reference_logs()
    candidate_patch_plan = build_candidate_patch_plan(
        args,
        approved_problem,
        local_manifest,
        task_plan,
        gate_reference_logs,
        candidate_verdict,
    )
    resume_token = build_resume_token(approved_problem, validation_issues)
    status = build_status(approved_problem, validation_issues, local_manifest, task_plan, candidate_verdict)

    write_json(args.frontier_ledger_path, frontier_ledger)
    write_json(args.family_stop_reason_path, family_stop_reason)
    write_json(args.gate_reference_logs_path, gate_reference_logs)
    write_json(args.candidate_patch_plan_path, candidate_patch_plan)
    write_text(args.candidate_patch_plan_md_path, render_candidate_patch_plan_md(candidate_patch_plan))
    write_json(args.candidate_verdict_path, candidate_verdict)
    write_json(args.resume_token_path, resume_token)
    write_json(args.status_path, status)
    write_text(args.status_md_path, render_status_md(status))

    # Keep the loop strictly local and opt-in.
    if validation_issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
