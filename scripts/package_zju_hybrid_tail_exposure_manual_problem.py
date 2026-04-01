import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
FIRST_CANDIDATE_SHAPE = "stratified_bucket_plus_soft_guard"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_anchorb13softguardreg095conf095_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the hybrid tail exposure balancing family into manual-approval artifacts."
    )
    parser.add_argument("--bucket-readiness-json", type=Path, required=True)
    parser.add_argument("--soft-readiness-json", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--research-status-json", type=Path, required=True)
    parser.add_argument("--draft-json", type=Path, required=True)
    parser.add_argument("--draft-md", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path, required=True)
    parser.add_argument("--blueprint-json", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-md", type=Path, required=True)
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
        "problem_id": "promoted_hybrid_tail_exposure_balancing_v1",
        "problem_title": "Promoted-lead hybrid tail exposure balancing",
        "family": "hybrid_tail_exposure_balancing",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the refined contract-segment hardtail "
            "mix fixed, then overlays a very mild train-only Camera_B13 soft guard so the bucket depth gains "
            "might survive without the short-gate camera tax."
        ),
        "why_genuinely_new": (
            "This is a hybrid family: not bucket-only and not soft-only. It combines the refined hardtail mix "
            "with an existing soft guard on the dominant contract slice."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen residual_case_coverage_rebalancing, hardtail_bucket_granularity_refinement, or "
            "soft_tail_exposure_rebalancing as same-family retries. The new question is whether the hybrid of "
            "those two signals can beat the current lead."
        ),
        "first_candidate_hint": (
            "Launch only the stratified_bucket_plus_soft_guard candidate: keep the refined 4:1 contract-segment "
            "bucket, add a mild train-only B13 smoothstep guard at 0.95/0.95, and do not branch into ratio "
            "tweaks, bucket-only retries, or soft-only retries."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            FIRST_CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "refined_hardtail_manifest_path": "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
            "train_default_stream_len": 80000,
            "train_hardcase_stream_len": 20000,
            "dominant_contract_anchor_camera": "Camera_B13",
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
            "bucket-only retry",
            "soft-only retry",
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
            "allow_hybrid_tail_exposure_balancing": True,
            "keep_refined_hardtail_mix_fixed": True,
            "allow_existing_soft_tail_hooks_only": True,
            "require_refined_hardtail_manifest": True,
            "require_dominant_contract_soft_guard": True,
            "disallow_bucket_only_retry": True,
            "disallow_soft_only_retry": True,
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
        "# Hybrid Tail Exposure Balancing Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Hybrid Tail Exposure Balancing",
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
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    bucket_readiness = load_json(args.bucket_readiness_json)
    soft_readiness = load_json(args.soft_readiness_json)
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
            "The refined bucket family had the right depth direction but paid a small camera tax, while the "
            "soft-tail family removed the camera tax but collapsed to near no-op. The next bounded question is "
            "their hybrid."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another bucket-only or soft-only retry. It keeps the refined mix and layers "
            "a mild soft guard on top."
        ),
        "signal_definition": (
            "Keep the refined 4:1 contract-segment bucket fixed and add a mild B13 smoothstep soft guard so the "
            "hybrid candidate can test whether it preserves bucket depth gains while cancelling camera tax."
        ),
        "scope_definition": "config only; do not rewrite dataset plumbing, loss.py, or source policy",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first hybrid candidate is repo-ready because both ingredients are already live: manifest-aware "
            "two-stream mixing and train-only anchor-conditioned smoothstep scaling."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not bucket-only retry",
            "not soft-only retry",
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
            "The refined bucket family gave strong depth improvements but a small short-gate camera regression.",
            "The soft-only family removed the camera regression but also erased the depth gain, ending as a near no-op.",
            "The next bounded question is whether a hybrid of those two signals can keep the depth gains while neutralizing the camera tax.",
        ],
        "evidence_refs": [
            str(args.bucket_readiness_json.resolve()),
            str(args.soft_readiness_json.resolve()),
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
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed hybrid_tail_exposure_balancing",
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
            FIRST_CANDIDATE_CONFIG,
            "scripts/summarize_zju_hybrid_tail_exposure_readiness.py",
            "scripts/package_zju_hybrid_tail_exposure_manual_problem.py",
            "scripts/arm_zju_source_policy_approved_problem.py",
            "scripts/run_zju_source_policy_research_loop.py",
        ],
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
