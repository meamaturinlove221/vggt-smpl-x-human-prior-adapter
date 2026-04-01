import argparse
import json
from datetime import datetime
from pathlib import Path


FIRST_CANDIDATE_SHAPE = "maxdepthanchor_plus_balance_reserve_defaultfocal105"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_"
    "defaultfocal105_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the default-stream intrinsics counterbalance family into manual-approval artifacts."
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
        "problem_id": "promoted_default_stream_intrinsics_counterbalance_v1",
        "problem_title": "Promoted-lead default-stream intrinsics counterbalance",
        "family": "default_stream_intrinsics_counterbalance",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the strongest current tail contract fixed "
            "and applies a light train-only focal counterweight to the default stream only."
        ),
        "why_genuinely_new": (
            "This is a default-stream objective-balance family rather than another tail-stream cousin. It asks "
            "whether the remaining camera tax is global balance rather than tail-contract detail."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen any tail-stream derivative. The hardtail and reserve contracts, manifests, "
            "anchor policy, replay policy, supervision count, and ratio all stay fixed."
        ),
        "first_candidate_hint": (
            "Launch only the maxdepthanchor_plus_balance_reserve_defaultfocal105 candidate: keep the refined "
            "hardtail + reserve contract fixed and apply a train-only focal scale of 1.05 to the default stream only."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            FIRST_CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "refined_hardtail_manifest_path": "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
            "balance_reserve_manifest_path": "output/zju_source_policy_research_loop/anchor_balance_reserve_manifest.promotedlead.v1.json",
            "hardtail_anchor_policy": "max_depth_conf",
            "default_stream_focal_scale": 1.05,
            "hardtail_stream_focal_scale": 1.0,
            "reserve_stream_focal_scale": 1.0,
            "no_T_or_rotation_scaling": True,
            "train_default_stream_len": 80000,
            "train_hardtail_stream_len": 10000,
            "train_balance_reserve_stream_len": 10000,
        },
        "avoid_patterns": [
            "tail-stream cousin",
            "bucket retry",
            "focal reinforcement retry on tail",
            "replay retry",
            "dual-supervision retry",
            "ratio tweak",
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
            "allow_default_stream_intrinsics_counterbalance": True,
            "keep_refined_hardtail_and_reserve_contract_fixed": True,
            "allow_default_stream_focal_scale_only": True,
            "require_default_stream_label_plumbing": True,
            "require_refined_hardtail_manifest": True,
            "require_anchor_balance_reserve_manifest": True,
            "disallow_tail_stream_cousin_retry": True,
            "disallow_tail_focal_retry": True,
            "disallow_replay_retry": True,
            "disallow_dual_supervision_retry": True,
            "disallow_ratio_tweak_retry": True,
            "disallow_slot3_ticket_reopen": True,
            "disallow_source_policy_same_family_retry": True,
            "stop_if_first_ticket_loss_camera_positive": True,
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
        "# Default-Stream Intrinsics Counterbalance Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Default-Stream Intrinsics Counterbalance",
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
            "The tail-manifest batch has already touched soft, hybrid, branch, pool, anchor, replay, and "
            "supervision-count axes, and the residual camera tax still remained. The next bounded question is "
            "global objective balance rather than another tail-stream detail."
        ),
        "why_not_same_family_retry": (
            "This is not a tail-stream cousin. The strongest hardtail + reserve contract remains fixed and only "
            "the default stream gets a light focal counterweight."
        ),
        "signal_definition": (
            "Keep the refined hardtail/reserve contract fixed and apply a train-only focal scale of 1.05 to the "
            "default stream label only."
        ),
        "scope_definition": "minimal write surface only: loss.py label-scaling support plus one config and readiness/package wiring",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first counterbalance candidate is repo-ready because stream labels now let default, hardtail, "
            "and reserve focal scales be separated without touching dataset contracts."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not tail-stream cousin",
            "not focal retry on tail",
            "not replay retry",
            "not ratio tweak",
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
        "stop_rule": (
            "If the first ticket still has loss_camera > 0 at short gate, close the whole tail-contract derivative batch."
        ),
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
            "Recent tail-manifest tickets repeatedly showed depth-side upside while leaving a small camera tax.",
            "After replay, view-set replay, and dual-supervision all failed, the next plausible axis is global objective balance rather than another tail-stream detail.",
            "This first candidate keeps the strongest tail contract fixed and only adds a light default-stream focal counterweight.",
        ],
        "readiness_artifact": str(args.readiness_json.resolve()),
        "hypothesis": readiness.get("hypothesis", ""),
    }

    plan = {
        "checked_at": iso_now(),
        "state": str(research_status.get("state", "")),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "current_stable_lead_config": str(research_status.get("current_stable_lead_config", "")),
        "family": seed["family"],
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed default_stream_intrinsics_counterbalance",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "readiness": readiness.get("readiness", {}),
        "plumbing_smoke": readiness.get("plumbing_smoke", {}),
        "write_surface": seed["first_candidate_write_surface"],
    }

    write_json(args.seed_json, seed)
    write_json(args.blueprint_json, blueprint)
    write_json(args.draft_json, draft)
    write_json(args.plan_json, plan)
    write_text(args.draft_md, render_draft_md(draft))
    write_text(args.plan_md, render_plan_md(plan))
    print(args.seed_json)
    print(args.blueprint_json)
    print(args.draft_json)
    print(args.plan_json)


if __name__ == "__main__":
    main()
