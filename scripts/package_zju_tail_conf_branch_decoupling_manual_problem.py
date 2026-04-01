import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
FIRST_CANDIDATE_SHAPE = "stratified_bucket_plus_conf_mask_drop"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_anchorb13iqrconfmaskdrop_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the tail conf-branch decoupling family into manual-approval artifacts."
    )
    parser.add_argument("--soft-readiness-json", type=Path, required=True)
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
        "problem_id": "promoted_tail_conf_branch_decoupling_v1",
        "problem_title": "Promoted-lead tail conf-branch decoupling",
        "family": "tail_conf_branch_decoupling",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the refined contract-segment hardtail "
            "mix fixed, then routes the dominant Camera_B13 interquartile-quality slice out of conf-depth "
            "supervision only so the bucket depth gains might survive without the short-gate camera tax."
        ),
        "why_genuinely_new": (
            "This is a branch-decoupling family: not bucket-only, not soft-only, and not hybrid-soft. It uses "
            "the existing conf-mask hook to remove conf-depth supervision on the dominant slice while leaving "
            "the regression branch unchanged."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen hardtail_bucket_granularity_refinement, soft_tail_exposure_rebalancing, or "
            "hybrid_tail_exposure_balancing as same-family retries. The new question is whether the short-gate "
            "camera tax is primarily a conf-branch effect on the dominant B13 slice."
        ),
        "first_candidate_hint": (
            "Launch only the stratified_bucket_plus_conf_mask_drop candidate: keep the refined 4:1 contract-"
            "segment bucket, drop conf-depth supervision on the dominant B13 interquartile-quality slice, keep "
            "the regression branch unchanged, and do not branch into ratio tweaks, bucket-only retries, or soft "
            "guard cousins."
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
            "conf_mask_quality_min": 2.0051,
            "conf_mask_quality_max": 2.3407,
            "conf_mask_train_only": True,
            "reg_branch_unchanged": True,
        },
        "avoid_patterns": [
            "bucket-only retry",
            "soft-only retry",
            "hybrid soft-guard retry",
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
            "allow_tail_conf_branch_decoupling": True,
            "keep_refined_hardtail_mix_fixed": True,
            "allow_existing_conf_mask_hooks_only": True,
            "require_refined_hardtail_manifest": True,
            "require_dominant_contract_conf_mask_drop": True,
            "require_reg_branch_unchanged": True,
            "disallow_bucket_only_retry": True,
            "disallow_soft_only_retry": True,
            "disallow_hybrid_soft_guard_retry": True,
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
        "# Tail Conf-Branch Decoupling Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Tail Conf-Branch Decoupling",
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
    soft_readiness = load_json(args.soft_readiness_json)
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
            "The refined bucket family had the right depth direction but paid a small camera tax, and the "
            "hybrid-soft family preserved the same depth gain shape without removing that camera tax. The next "
            "bounded question is whether the dominant B13 slice should leave the conf branch entirely while the "
            "reg branch stays unchanged."
        ),
        "why_not_same_family_retry": (
            "This is explicitly not another bucket-only, soft-only, or hybrid-soft retry. It keeps the refined "
            "mix and changes only the dominant slice's conf branch through the existing conf-mask hook."
        ),
        "signal_definition": (
            "Keep the refined 4:1 contract-segment bucket fixed and mask the dominant B13 interquartile-quality "
            "slice out of conf-depth supervision only so the candidate can test whether the short-gate camera "
            "tax is conf-driven."
        ),
        "scope_definition": "config only; do not rewrite dataset plumbing, loss.py, or source policy",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first conf-branch candidate is repo-ready because both ingredients are already live: "
            "manifest-aware two-stream mixing and anchor-conditioned conf-mask drop."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not bucket-only retry",
            "not soft-only retry",
            "not hybrid-soft retry",
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
            "The refined bucket family improved conf_depth and reg_depth but paid a short-gate camera tax.",
            "The hybrid-soft family kept almost the same depth gain shape and still paid the camera tax, so a softer conf scaling alone was not enough.",
            "The next bounded question is whether the dominant B13 slice should leave the conf branch entirely while the regression branch stays unchanged.",
        ],
        "evidence_refs": [
            str(args.soft_readiness_json.resolve()),
            str(args.latest_verdict_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.research_status_json.resolve()),
        ],
        "latest_hybrid_short_gate_deltas": readiness.get("latest_hybrid_short_gate_deltas", {}),
        "selected_quality_window": soft_readiness.get("selected_quality_window", {}),
    }

    plan = {
        "checked_at": iso_now(),
        "state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": research_status.get("current_stable_lead_config", ""),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed tail_conf_branch_decoupling",
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
            "scripts/summarize_zju_tail_conf_branch_decoupling_readiness.py",
            "scripts/package_zju_tail_conf_branch_decoupling_manual_problem.py",
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
