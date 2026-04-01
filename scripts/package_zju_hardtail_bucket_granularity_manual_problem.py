import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
FIRST_CANDIDATE_SHAPE = "contract_segment_stratified_hardtail_bucket"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_minimal.yaml"
)
REFINED_MANIFEST_PATH = (
    "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the hardtail bucket granularity refinement family into manual-approval artifacts."
    )
    parser.add_argument(
        "--postmortem-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "residual_vs_hybrid_postmortem.20260329.json",
    )
    parser.add_argument(
        "--refined-manifest-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "hardtail_bucket_granularity_readiness.20260329.json",
    )
    parser.add_argument(
        "--research-status-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "research_loop_status.json",
    )
    parser.add_argument(
        "--gate-reference-logs-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "gate_reference_logs.json",
    )
    parser.add_argument(
        "--draft-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "next_manual_problem_draft.hardtail_bucket_granularity_refinement.20260329.json",
    )
    parser.add_argument(
        "--draft-md",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "next_manual_problem_draft.hardtail_bucket_granularity_refinement.20260329.md",
    )
    parser.add_argument(
        "--seed-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "approved_problem.seed.hardtail_bucket_granularity_refinement.json",
    )
    parser.add_argument(
        "--blueprint-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "family_blueprint.hardtail_bucket_granularity_refinement.json",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.hardtail_bucket_granularity_refinement.json",
    )
    parser.add_argument(
        "--plan-md",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.hardtail_bucket_granularity_refinement.md",
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


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_seed():
    return {
        "approved": False,
        "approved_at": "",
        "problem_id": "promoted_hardtail_bucket_granularity_refinement_v1",
        "problem_title": "Promoted-lead hard-tail bucket granularity refinement",
        "family": "hardtail_bucket_granularity_refinement",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the promoted source policy, loss, "
            "and 4:1 two-stream train contract fixed while refining the official hard-tail bucket from a coarse "
            "frame-level set into a contract-slice plus contiguous-segment stratified manifest."
        ),
        "why_genuinely_new": (
            "This changes the hard-tail bucket definition itself instead of reopening residual same-family "
            "mixing, ratio tweaks, slot_3 tickets, or source-policy retries."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen residual_case_coverage_rebalancing, source_policy_hybrid_ring_regularization, "
            "slot_3 stabilization, role reassignment, interpolated, partial, disagreement, unprojection-"
            "consistency, or auxiliary confgate families; the promoted source policy stays fixed and only the "
            "hard-tail bucket granularity changes."
        ),
        "first_candidate_hint": (
            "Launch only the contract_segment_stratified_hardtail_bucket candidate: cap the dominant "
            "Camera_B13/[Camera_B15, Camera_B14, Camera_B20] slice and its dominant contiguous segment inside "
            "the official tail bucket, refill from high-score near-tail reserve rows, keep the same 4:1 "
            "train-stream mix contract, and do not branch into residual ratio tweaks or slot_3 diagnostics."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            FIRST_CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "promoted_local_lead_config": (
                "training/config/"
                "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
                "gradconfmask_minimal.yaml"
            ),
            "refined_hardtail_manifest_path": REFINED_MANIFEST_PATH,
            "train_default_stream_len": 80000,
            "train_hardcase_stream_len": 20000,
            "default_to_hardcase_ratio": "4:1",
            "dominant_contract_cap_share": 0.47,
            "dominant_segment_cap_share": 0.21,
            "stratification_axes": [
                "contract_slice",
                "contiguous_segment",
            ],
        },
        "avoid_patterns": [
            "residual same-family retry",
            "4:1 to 3:1 or 2:1 ratio tweak",
            "slot_3 ticket reopen",
            "source_policy same-family retry",
            "loss-routing cousin reopen",
            "same-night second ticket",
            "same-night cousin sweep",
        ],
        "max_approved_problems_per_night": 1,
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "requires_dataset_or_routing_change": False,
        "requires_supervision_audit": False,
        "mutation_dsl": {
            "allow_hardtail_bucket_granularity_refinement": True,
            "keep_promoted_source_policy_fixed": True,
            "keep_existing_loss_and_sampler_logic_frozen": True,
            "require_refined_hardtail_manifest": True,
            "use_train_split_manifest_mix": True,
            "require_contract_segment_stratification": True,
            "disallow_residual_same_family_retry": True,
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
        "# Hardtail Bucket Granularity Refinement Draft (2026-03-29)",
        "",
        f"- family: `{draft['family']}`",
        f"- status: `{draft['status']}`",
        f"- first_candidate_shape: `{draft['first_candidate_shape']}`",
        f"- candidate_config: `{draft['candidate_config']}`",
        f"- refined_manifest_path: `{draft['refined_manifest_path']}`",
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
            "## Dominance Reduction",
            "",
            f"- official_dominant_contract_count: `{draft['dominance_reduction']['official_dominant_contract_count']}`",
            f"- official_dominant_contract_share: `{fmt(draft['dominance_reduction']['official_dominant_contract_share'])}`",
            f"- refined_dominant_contract_count: `{draft['dominance_reduction']['refined_dominant_contract_count']}`",
            f"- refined_dominant_contract_share: `{fmt(draft['dominance_reduction']['refined_dominant_contract_share'])}`",
            f"- official_dominant_segment_count: `{draft['dominance_reduction']['official_dominant_segment_count']}`",
            f"- official_dominant_segment_share: `{fmt(draft['dominance_reduction']['official_dominant_segment_share'])}`",
            f"- refined_dominant_segment_count: `{draft['dominance_reduction']['refined_dominant_segment_count']}`",
            f"- refined_dominant_segment_share: `{fmt(draft['dominance_reduction']['refined_dominant_segment_share'])}`",
            "",
            "## Exclusions",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in draft["required_exclusions"]])
    return "\n".join(lines).rstrip() + "\n"


def render_plan_md(plan: dict) -> str:
    lines = [
        "# Candidate Patch Plan: Hardtail Bucket Granularity Refinement",
        "",
        f"- checked_at: `{plan['checked_at']}`",
        f"- state: `{plan['state']}`",
        f"- approved_problem_present: `{plan['approved_problem_present']}`",
        f"- current_stable_lead_config: `{plan['current_stable_lead_config']}`",
        f"- preferred_first_family: `{plan['preferred_first_family']}`",
        f"- preferred_first_family_reason: `{plan['preferred_first_family_reason']}`",
        "",
        "## Candidate",
        "",
        f"- family: `{plan['family']}`",
        f"- first_candidate_shape: `{plan['first_candidate_shape']}`",
        f"- first_candidate_config: `{plan['first_candidate_config']}`",
        f"- refined_manifest_path: `{plan['refined_manifest_path']}`",
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
    postmortem = load_json(args.postmortem_json)
    refined_manifest = load_json(args.refined_manifest_json)
    readiness = load_json(args.readiness_json)
    research_status = load_json(args.research_status_json)
    gate_refs = load_json(args.gate_reference_logs_json)

    seed = build_seed()
    blueprint = {
        "checked_at": iso_now(),
        "family": seed["family"],
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The reject postmortem resolved the next direction to BUCKET_TOO_COARSE, the refined manifest is "
            "already materialized, and the new config instantiates cleanly without changing loss or sampler logic."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another residual_case_coverage_rebalancing retry: the 4:1 mix contract "
            "stays fixed and only the hard-tail bucket granularity changes."
        ),
        "signal_definition": (
            "Keep the promoted source-policy lead fixed while refining the hard-tail bucket so one contract slice "
            "and one contiguous segment no longer dominate the manifest-filtered stream."
        ),
        "scope_definition": "bucket materialization and config only; do not rewrite loss routing or source policy",
        "first_candidate_hypothesis": (
            "The residual postmortem says the old bucket was too coarse. If we cap the dominant contract slice "
            "and dominant contiguous segment, then refill from high-score near-tail reserve rows, we may retain "
            "tail focus without paying the same camera-regression tax."
        ),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first granularity-refinement candidate is already repo-ready: the refined manifest exists, the "
            "manifest-aware dataset path is live, and the config passes compose plus instantiate checks."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not residual same-family retry",
            "not 4:1 ratio tweak",
            "not slot_3 ticket",
            "not source_policy retry",
            "not loss-routing cousin reopen",
        ],
        "requires_dataset_plumbing": False,
        "compare_script_change_required": False,
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": (
                "Only this single first candidate would be eligible under the current contract, and it must "
                "return to guard after verdict writeback regardless of outcome."
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

    draft = {
        "checked_at": iso_now(),
        "draft_kind": "new_manual_problem",
        "status": "ready_for_manual_review_and_execution_pending_approval",
        "family": seed["family"],
        "problem_statement": seed["problem_statement"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "candidate_config": seed["first_candidate_config"],
        "refined_manifest_path": str(args.refined_manifest_json.resolve()),
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "do_not_auto_open_ticket": True,
        "why_now": [
            "The current-lead-aware postmortem chose BUCKET_TOO_COARSE, not MIX_CONTRACT_TOO_AGGRESSIVE.",
            "The refined manifest reduces dominance in the Camera_B13 / [Camera_B15, Camera_B14, Camera_B20] slice without reopening the residual family or changing the 4:1 mix contract.",
            "The new config already composes and instantiates cleanly while research remains in IDLE_GUARD with cloud off.",
        ],
        "dominance_reduction": readiness.get("dominance_reduction", {}),
        "required_exclusions": blueprint["required_exclusions"],
        "evidence_refs": [
            str(args.postmortem_json.resolve()),
            str(args.refined_manifest_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.research_status_json.resolve()),
        ],
    }

    plan = {
        "checked_at": iso_now(),
        "state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": research_status.get("current_stable_lead_config", ""),
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "Stay in IDLE_GUARD until a human explicitly approves this new family; do not auto-open a ticket."
        ),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "refined_manifest_path": str(args.refined_manifest_json.resolve()),
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed hardtail_bucket_granularity_refinement",
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
            "scripts/materialize_zju_contract_segment_stratified_hardtail_bucket.py",
            "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_minimal.yaml",
            "scripts/summarize_zju_hardtail_bucket_granularity_readiness.py",
            "scripts/package_zju_hardtail_bucket_granularity_manual_problem.py",
            "scripts/arm_zju_source_policy_approved_problem.py",
            "scripts/run_zju_source_policy_research_loop.py",
        ],
        "evidence_refs": [
            str(args.postmortem_json.resolve()),
            str(args.refined_manifest_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.research_status_json.resolve()),
            str(args.gate_reference_logs_json.resolve()),
        ],
    }
    short_ref = gate_refs.get("short_gate", {}).get("stable_lead_reference_summary", "")
    if short_ref:
        plan["evidence_refs"].append(str(Path(short_ref).resolve()))

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
