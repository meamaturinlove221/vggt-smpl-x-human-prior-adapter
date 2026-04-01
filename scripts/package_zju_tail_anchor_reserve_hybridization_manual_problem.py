import argparse
import json
from datetime import datetime
from pathlib import Path


FIRST_CANDIDATE_SHAPE = "maxdepthanchor_plus_balance_reserve"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the tail anchor reserve hybridization family into manual-approval artifacts."
    )
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
        "problem_id": "promoted_tail_anchor_reserve_hybridization_v1",
        "problem_title": "Promoted-lead tail anchor reserve hybridization",
        "family": "tail_anchor_reserve_hybridization",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the refined hardtail and reserve cohort "
            "manifests fixed, then changes only the hardtail stream anchor policy to max_depth_conf so the larger "
            "depth gains from anchor stabilization can be tested together with the broader reserve coverage."
        ),
        "why_genuinely_new": (
            "This is a bounded hybrid family rather than a same-family retry: it combines the strongest anchor-policy "
            "signal with the reserve cohort companion stream in one new config-only contract."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen tail_anchor_stabilization or tail_counterbalance_cohort_mixing as same-family retries. "
            "The new question is whether the remaining camera tax disappears only when anchor stabilization and "
            "reserve coverage are applied together."
        ),
        "first_candidate_hint": (
            "Launch only the maxdepthanchor_plus_balance_reserve candidate: keep the total train budget at 100k with "
            "an 80k/10k/10k split, keep the reserve stream unchanged, and set only the refined hardtail stream to "
            "source_anchor_policy=max_depth_conf."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            FIRST_CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "refined_hardtail_manifest_path": "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
            "balance_reserve_manifest_path": "output/zju_source_policy_research_loop/anchor_balance_reserve_manifest.promotedlead.v1.json",
            "hardtail_source_anchor_policy": "max_depth_conf",
            "reserve_source_anchor_policy": "current_lead_default",
            "train_default_stream_len": 80000,
            "train_hardtail_stream_len": 10000,
            "train_balance_reserve_stream_len": 10000,
        },
        "avoid_patterns": [
            "branch-decoupling retry",
            "bucket-only retry",
            "source-pool retry",
            "soft-tail no-op retry",
            "slot_3 ticket reopen",
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
            "allow_tail_anchor_reserve_hybridization": True,
            "keep_refined_hardtail_and_reserve_manifests_fixed": True,
            "allow_maxdepthanchor_on_hardtail_stream_only": True,
            "allow_reserve_stream_only_as_companion": True,
            "require_refined_hardtail_manifest": True,
            "require_anchor_balance_reserve_manifest": True,
            "disallow_branch_only_retry": True,
            "disallow_bucket_only_retry": True,
            "disallow_source_pool_retry": True,
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
        "# Tail Anchor Reserve Hybridization Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Tail Anchor Reserve Hybridization",
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
        f"- reserve_manifest_summary: `{plan['reserve_manifest_summary']}`",
        f"- plumbing_smoke: `{plan['plumbing_smoke']}`",
        "",
        "## Write Surface",
        "",
    ]
    lines.extend([f"- `{item}`" for item in plan["write_surface"]])
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
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
            "The max-depth-anchor hardtail ticket was the strongest depth-improving variant so far, and the reserve "
            "cohort ticket proved the repo can companion the dominant slice with non-dominant anchors. The next "
            "bounded question is whether those two signals must be applied together."
        ),
        "why_not_same_family_retry": (
            "This is not another isolated anchor-policy retry and not another isolated reserve-cohort retry. It is "
            "a new hybrid family that combines the strongest available contract pieces."
        ),
        "signal_definition": (
            "Keep the refined hardtail and reserve manifests fixed, set only the hardtail stream anchor policy to "
            "max_depth_conf, and keep the reserve stream on the current lead anchor policy."
        ),
        "scope_definition": "config only; do not rewrite loss routing, source policy, dataset code, or cloud flow",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first hybridization candidate is repo-ready because both manifests already exist and the composed "
            "three-stream training path is already exercised."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not branch-decoupling retry",
            "not bucket-only retry",
            "not source-pool retry",
            "not slot_3 ticket",
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
        "status": "ready_for_manual_review",
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "candidate_config": seed["first_candidate_config"],
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [
            "The strongest refined-tail result so far came from hardcase max-depth anchor stabilization.",
            "The reserve-cohort result broadened anchor coverage but still left a small camera tax.",
            "The next bounded question is whether those two strongest mechanisms need to be combined in one ticket.",
        ],
        "supporting_readiness": readiness,
        "blueprint_path": str(args.blueprint_json),
        "seed_path": str(args.seed_json),
    }

    plan = {
        "checked_at": iso_now(),
        "state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": research_status.get("current_stable_lead_config", ""),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed tail_anchor_reserve_hybridization",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "readiness": readiness.get("readiness", {}),
        "reserve_manifest_summary": readiness.get("reserve_manifest_summary", {}),
        "plumbing_smoke": readiness.get("plumbing_smoke", {}),
        "write_surface": seed["first_candidate_write_surface"],
    }

    write_json(args.seed_json, seed)
    write_json(args.blueprint_json, blueprint)
    write_json(args.draft_json, draft)
    write_text(args.draft_md, render_draft_md(draft))
    write_json(args.plan_json, plan)
    write_text(args.plan_md, render_plan_md(plan))

    print(args.seed_json)
    print(args.blueprint_json)
    print(args.draft_json)
    print(args.plan_json)


if __name__ == "__main__":
    main()
