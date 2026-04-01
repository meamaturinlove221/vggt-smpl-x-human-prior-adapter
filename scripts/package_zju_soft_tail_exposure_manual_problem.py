import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
FIRST_CANDIDATE_SHAPE = "contract_balanced_soft_tail_taper"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_anchorb13softtailtaperreg095conf095_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the soft-tail exposure rebalancing family into manual-approval artifacts."
    )
    parser.add_argument(
        "--prior-verdict-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_verdict.json",
    )
    parser.add_argument(
        "--hardtail-profile-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "official_hardtail_coverage_profile.20260329.json",
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "soft_tail_exposure_readiness.20260330.json",
    )
    parser.add_argument(
        "--research-status-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "research_loop_status.json",
    )
    parser.add_argument(
        "--draft-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "next_manual_problem_draft.soft_tail_exposure_rebalancing.20260330.json",
    )
    parser.add_argument(
        "--draft-md",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "next_manual_problem_draft.soft_tail_exposure_rebalancing.20260330.md",
    )
    parser.add_argument(
        "--seed-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "approved_problem.seed.soft_tail_exposure_rebalancing.json",
    )
    parser.add_argument(
        "--blueprint-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "family_blueprint.soft_tail_exposure_rebalancing.json",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.soft_tail_exposure_rebalancing.json",
    )
    parser.add_argument(
        "--plan-md",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.soft_tail_exposure_rebalancing.md",
    )
    return parser.parse_args()


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_seed():
    return {
        "approved": False,
        "approved_at": "",
        "problem_id": "promoted_soft_tail_exposure_rebalancing_v1",
        "problem_title": "Promoted-lead soft tail exposure rebalancing",
        "family": "soft_tail_exposure_rebalancing",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the current hybrid-ring source-policy "
            "lead and dataset sampling distribution fixed while replacing hard tail over-exposure with a mild, "
            "train-only smooth taper on the dominant Camera_B13 contract slice."
        ),
        "why_genuinely_new": (
            "This changes the tail exposure contract from hard manifest mixing to soft train-only tapering on "
            "the dominant contract slice, rather than reopening bucket composition, ratio tweaks, slot_3, or "
            "source-policy retries."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen residual_case_coverage_rebalancing, hardtail_bucket_granularity_refinement, "
            "slot_3 stabilization, source_policy_hybrid_ring_regularization, or old wholefg/edge-band families. "
            "The current lead stays fixed and only a config-only soft taper is added."
        ),
        "first_candidate_hint": (
            "Launch only the contract_balanced_soft_tail_taper candidate: keep the current hybrid-ring lead "
            "config, target Camera_B13 with train-only smoothstep quality and depth_conf tapering, keep scales "
            "mild (0.95/0.95), and do not branch into hard bucket or ratio retries."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            FIRST_CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "current_local_lead_config": (
                "training/config/"
                "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
                "gradconfmask_minimal.yaml"
            ),
            "dominant_contract_anchor_camera": "Camera_B13",
            "train_only": True,
            "quality_interp": "smoothstep",
            "quality_low": 2.0051,
            "quality_high": 2.3407,
            "depth_conf_interp": "smoothstep",
            "depth_conf_low": 0.0,
            "depth_conf_high": 5.913640410988592,
            "reg_scale": 0.95,
            "conf_scale": 0.95,
        },
        "avoid_patterns": [
            "hard bucket reopen",
            "4:1 to 3:1 or 2:1 ratio tweak",
            "slot_3 ticket reopen",
            "source_policy same-family retry",
            "same-night second ticket",
            "same-night cousin sweep",
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
            "allow_soft_tail_exposure_rebalancing": True,
            "keep_current_source_policy_lead_fixed": True,
            "keep_dataset_sampling_distribution_fixed": True,
            "allow_train_only_soft_tail_taper": True,
            "require_existing_loss_hooks_only": True,
            "require_dominant_contract_anchor_target": True,
            "disallow_hard_bucket_reopen": True,
            "disallow_ratio_tweak_retry": True,
            "disallow_slot3_ticket_reopen": True,
            "disallow_source_policy_same_family_retry": True,
            "disallow_wholefg_scalar": True,
            "disallow_wholefg_decoupled": True,
            "disallow_edge_band_scalar": True,
            "disallow_edge_band_decoupled": True,
            "disallow_hard_depth_conf_threshold": True,
            "disallow_plain_anchor_view_only": True,
        },
    }


def render_draft_md(draft: dict) -> str:
    lines = [
        "# Soft Tail Exposure Rebalancing Draft (2026-03-30)",
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
    lines.extend(
        [
            "",
            "## Target Window",
            "",
            f"- dominant_contract_anchor_camera: `{draft['dominant_contract_anchor_camera']}`",
            f"- quality_window: `{draft['quality_window']}`",
            f"- depth_conf_window: `{draft['depth_conf_window']}`",
            f"- target_scales: `{draft['target_scales']}`",
            "",
            "## Exclusions",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in draft["required_exclusions"]])
    return "\n".join(lines).rstrip() + "\n"


def render_plan_md(plan: dict) -> str:
    lines = [
        "# Candidate Patch Plan: Soft Tail Exposure Rebalancing",
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
        f"- plumbing_smoke: `{plan['plumbing_smoke']}`",
        "",
        "## Write Surface",
        "",
    ]
    lines.extend([f"- `{item}`" for item in plan["write_surface"]])
    lines.extend(["", "## Evidence", ""])
    lines.extend([f"- `{item}`" for item in plan["evidence_refs"]])
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    prior_verdict = load_json(args.prior_verdict_json)
    hardtail_profile = load_json(args.hardtail_profile_json)
    readiness = load_json(args.readiness_json)
    research_status = load_json(args.research_status_json)

    seed = build_seed()
    blueprint = {
        "checked_at": iso_now(),
        "family": seed["family"],
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The hardtail bucket family already reduced dominance but still died at short gate on camera, so "
            "the next bounded question should soften exposure rather than reopen hard bucket composition."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another hardtail_bucket_granularity_refinement retry: the train stream and "
            "current source policy stay fixed while only a config-only soft taper is added."
        ),
        "signal_definition": (
            "Keep the current hybrid-ring lead fixed while applying a mild train-only smoothstep taper to the "
            "dominant Camera_B13 contract slice to reduce hard-tail over-exposure without changing sampling."
        ),
        "scope_definition": "config only; do not rewrite dataset sampling, loss.py, or source policy",
        "first_candidate_hypothesis": (
            "If the hard bucket already narrowed dominance but still paid a camera tax, then a softer train-only "
            "taper on the same dominant contract slice may preserve camera while still helping the tail."
        ),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first soft-tail candidate is already repo-ready because it reuses existing anchor-conditioned "
            "smoothstep loss hooks with a new config only."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not hard bucket reopen",
            "not residual same-family retry",
            "not ratio tweak",
            "not slot_3 ticket",
            "not source_policy retry",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": (
                "Only this single first candidate is eligible under the current contract, and it must return "
                "to guard after verdict writeback regardless of outcome."
            ),
        },
        "gate_sequence": [
            "SMOKE_1x1",
            "TIGHT_GATE_10x5",
            "LONG_GATE_100x20",
            "VERDICT_WRITEBACK",
            "RETURN_TO_GUARD",
        ],
        "cloud_must_remain_off": True,
    }

    anchor_ranked = hardtail_profile.get("anchor_enrichment_ranked", [])
    dominant_anchor = (anchor_ranked[0] or {}).get("promoted_anchor_camera", "Camera_B13") if anchor_ranked else "Camera_B13"
    draft = {
        "checked_at": iso_now(),
        "draft_kind": "new_manual_problem",
        "status": "ready_for_manual_review_and_execution_pending_approval",
        "family": seed["family"],
        "problem_statement": seed["problem_statement"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "candidate_config": seed["first_candidate_config"],
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "do_not_auto_open_ticket": True,
        "why_now": [
            "The prior hardtail bucket family already reduced bucket dominance but still died at short gate on loss_camera.",
            "The next bounded change is to keep the current hybrid-ring lead fixed and replace hard tail over-exposure with a mild train-only smooth taper on the dominant contract slice.",
            "The first candidate is config-only and reuses existing quality/depth_conf smoothstep hooks, so it is directly executable without new plumbing.",
        ],
        "dominant_contract_anchor_camera": dominant_anchor,
        "quality_window": readiness.get("selected_quality_window", {}),
        "depth_conf_window": readiness.get("selected_depth_conf_window", {}),
        "target_scales": readiness.get("target_scales", {}),
        "required_exclusions": blueprint["required_exclusions"],
        "evidence_refs": [
            str(args.prior_verdict_json.resolve()),
            str(args.hardtail_profile_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.research_status_json.resolve()),
        ],
    }

    plan = {
        "checked_at": iso_now(),
        "state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": research_status.get("current_stable_lead_config", ""),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed soft_tail_exposure_rebalancing",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "gate_sequence": blueprint["gate_sequence"],
        "readiness": readiness.get("readiness", {}),
        "plumbing_smoke": readiness.get("plumbing_smoke", {}),
        "write_surface": [
            "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_anchorb13softtailtaperreg095conf095_minimal.yaml",
            "scripts/summarize_zju_soft_tail_exposure_readiness.py",
            "scripts/package_zju_soft_tail_exposure_manual_problem.py",
            "scripts/arm_zju_source_policy_approved_problem.py",
            "scripts/run_zju_source_policy_research_loop.py",
        ],
        "evidence_refs": draft["evidence_refs"],
    }

    write_json(args.seed_json, seed)
    write_json(args.blueprint_json, blueprint)
    write_json(args.draft_json, draft)
    write_text(args.draft_md, render_draft_md(draft))
    write_json(args.plan_json, plan)
    write_text(args.plan_md, render_plan_md(plan))

    print(f"[wrote] {args.draft_json}")
    print(f"[wrote] {args.draft_md}")
    print(f"[wrote] {args.seed_json}")
    print(f"[wrote] {args.blueprint_json}")
    print(f"[wrote] {args.plan_json}")
    print(f"[wrote] {args.plan_md}")


if __name__ == "__main__":
    main()
