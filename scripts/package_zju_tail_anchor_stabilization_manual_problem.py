import argparse
import json
from datetime import datetime
from pathlib import Path


FIRST_CANDIDATE_SHAPE = "stratified_bucket_hardcase_maxdepthanchor"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_hardcasemaxdepthanchor_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the tail anchor stabilization family into manual-approval artifacts."
    )
    parser.add_argument("--latest-verdict-json", type=Path, required=True)
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
        "problem_id": "promoted_tail_anchor_stabilization_v1",
        "problem_title": "Promoted-lead tail anchor stabilization",
        "family": "tail_anchor_stabilization",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the refined contract-segment hardtail "
            "mix fixed, then stabilizes the hardcase stream anchor with max_depth_conf selection so the depth "
            "gains might survive without the short-gate camera tax."
        ),
        "why_genuinely_new": (
            "This is an anchor-stabilization family: not bucket-only, not branch-local, and not source-pool "
            "tempering. It changes only the hardcase stream's anchor policy."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen hardtail_bucket_granularity_refinement, soft_tail_exposure_rebalancing, hybrid_"
            "tail_exposure_balancing, tail_conf_branch_decoupling, or tail_source_pool_tempering as same-family "
            "retries. The new question is whether low-quality hardcase anchors drive the remaining camera tax."
        ),
        "first_candidate_hint": (
            "Launch only the stratified_bucket_hardcase_maxdepthanchor candidate: keep the refined 4:1 contract-"
            "segment bucket, leave the default stream untouched, switch the hardcase stream anchor policy to "
            "max_depth_conf, and do not branch into ratio tweaks or branch-local cousins."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [FIRST_CANDIDATE_CONFIG],
        "first_candidate_knobs": {
            "refined_hardtail_manifest_path": "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
            "train_default_stream_len": 80000,
            "train_hardcase_stream_len": 20000,
            "default_stream_source_anchor_policy": "random",
            "hardcase_stream_source_anchor_policy": "max_depth_conf",
        },
        "avoid_patterns": [
            "bucket-only retry",
            "branch-local retry",
            "source-pool tempering retry",
            "4:1 to 3:1 or 2:1 ratio tweak",
            "slot_3 ticket reopen",
            "source_policy global same-family retry",
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
            "allow_tail_anchor_stabilization": True,
            "keep_refined_hardtail_mix_fixed": True,
            "keep_current_source_policy_eval_fixed": True,
            "allow_hardcase_stream_anchor_policy_only": True,
            "require_refined_hardtail_manifest": True,
            "require_hardcase_max_depth_conf_anchor": True,
            "disallow_bucket_only_retry": True,
            "disallow_branch_only_retry": True,
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
        "# Tail Anchor Stabilization Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Tail Anchor Stabilization",
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
    latest_verdict = load_json(args.latest_verdict_json)
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
            "The refined hardcase stream still pays a camera tax after branch-local decoupling and source-pool "
            "tempering, which points to hardcase supervision quality itself. The next bounded question is "
            "whether low-quality hardcase anchors are driving that tax."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another bucket-only, branch-local, or source-pool retry. It keeps the "
            "refined mix and changes only the hardcase stream's anchor policy."
        ),
        "signal_definition": (
            "Keep the refined 4:1 contract-segment bucket fixed and stabilize the hardcase stream anchor to "
            "the per-frame max-depth-conf view so the candidate can test whether hardcase anchor quality drives "
            "the short-gate camera tax."
        ),
        "scope_definition": "config only; do not rewrite dataset plumbing, loss.py, or source policy",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first anchor-stabilization candidate is repo-ready because the dataset already supports "
            "source_anchor_policy=max_depth_conf on a per-stream basis and the refined manifest mix is already live."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not bucket-only retry",
            "not branch-local retry",
            "not source-pool tempering retry",
            "not ratio tweak",
            "not slot_3 ticket",
            "not source_policy global retry",
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
            "The refined bucket family improved depth metrics but paid a small short-gate camera tax.",
            "Branch-local decoupling and hardcase source-pool tempering changed other parts of the contract but did not remove that camera tax.",
            "The next bounded question is whether low-quality hardcase anchors are the remaining source of camera drift.",
        ],
        "evidence_refs": [
            str(args.latest_verdict_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.research_status_json.resolve()),
        ],
        "latest_tail_source_pool_short_gate_deltas": readiness.get("latest_tail_source_pool_short_gate_deltas", {}),
        "hardcase_anchor_histogram_probe12": readiness.get("plumbing_smoke", {}).get("hardcase_anchor_histogram_probe12", {}),
    }

    plan = {
        "checked_at": iso_now(),
        "state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": research_status.get("current_stable_lead_config", ""),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed tail_anchor_stabilization",
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
            "scripts/summarize_zju_tail_anchor_stabilization_readiness.py",
            "scripts/package_zju_tail_anchor_stabilization_manual_problem.py",
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
