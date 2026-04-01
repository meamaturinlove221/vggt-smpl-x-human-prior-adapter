import argparse
import json
from datetime import datetime
from pathlib import Path


FIRST_CANDIDATE_SHAPE = "manifest_viewset_replay_on_tail_streams"
FIRST_CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_manifestviewsetreplay_minimal.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the tail contract view-set replay family into manual-approval artifacts."
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
        "problem_id": "promoted_tail_contract_viewset_replay_v1",
        "problem_title": "Promoted-lead tail contract view-set replay",
        "family": "tail_contract_viewset_replay",
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": (
            "No auto-next ticket is currently selected. This new family remains manual-approval-only until a "
            "human explicitly opens it."
        ),
        "problem_statement": (
            "Design exactly one genuinely new-family candidate that keeps the best current three-stream tail "
            "contract fixed, then replays the manifest-prescribed selected camera set and role split on both "
            "tail streams so the training contract matches the promoted residual evidence more faithfully."
        ),
        "why_genuinely_new": (
            "This is a full view-set contract replay family rather than another anchor-only, focal-side, or "
            "ratio-side family. It changes how manifest entries determine the ordered camera set and roles."
        ),
        "why_not_reopening_frozen_family": (
            "It does not reopen tail_contract_anchor_replay or other tail cousins as same-family retries. The "
            "new question is whether the missing ingredient is full camera-set replay, not another loss knob."
        ),
        "first_candidate_hint": (
            "Launch only the manifest_viewset_replay_on_tail_streams candidate: keep the 80k/10k/10k contract "
            "fixed, and for both hardtail and reserve streams choose the camera list from each manifest entry's "
            "`selected_camera_names` and `selected_supervised_camera_names` fields."
        ),
        "first_candidate_shape": FIRST_CANDIDATE_SHAPE,
        "first_candidate_config": FIRST_CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            FIRST_CANDIDATE_CONFIG,
            "training/data/datasets/zju_vggt_geom.py",
        ],
        "first_candidate_knobs": {
            "refined_hardtail_manifest_path": "output/zju_source_policy_research_loop/contract_segment_stratified_hardtail_bucket.promotedlead.v1.json",
            "balance_reserve_manifest_path": "output/zju_source_policy_research_loop/anchor_balance_reserve_manifest.promotedlead.v1.json",
            "sample_manifest_use_entry_anchor": True,
            "sample_manifest_use_entry_camera_set": True,
            "sample_manifest_anchor_field": "promoted_anchor_camera",
            "sample_manifest_camera_list_field": "selected_camera_names",
            "sample_manifest_supervised_camera_field": "selected_supervised_camera_names",
            "train_default_stream_len": 80000,
            "train_hardtail_stream_len": 10000,
            "train_balance_reserve_stream_len": 10000,
        },
        "avoid_patterns": [
            "anchor-only retry",
            "loss-side cousin",
            "blanket focal retry",
            "stream-selective focal retry",
            "source-pool retry",
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
        "requires_dataset_or_routing_change": True,
        "requires_supervision_audit": False,
        "mutation_dsl": {
            "allow_tail_contract_viewset_replay": True,
            "keep_three_stream_tail_contract_fixed": True,
            "allow_manifest_entry_viewset_replay_only": True,
            "require_refined_hardtail_manifest": True,
            "require_anchor_balance_reserve_manifest": True,
            "require_manifest_entry_camera_fields": True,
            "disallow_anchor_only_retry": True,
            "disallow_loss_side_retry": True,
            "disallow_blanket_focal_retry": True,
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
        "# Tail Contract View-Set Replay Draft (2026-03-30)",
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
        "# Candidate Patch Plan: Tail Contract View-Set Replay",
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
            "Anchor replay alone made the camera tax worse, which narrows the missing mechanism to the rest of "
            "the promoted tail contract: ordered camera set and supervised/source-only role split."
        ),
        "why_not_same_family_retry": (
            "This is not another anchor-only cousin. It preserves the best current contract and changes only how "
            "manifest entries determine the full view-set contract."
        ),
        "signal_definition": (
            "Keep the three-stream tail contract fixed and use each manifest entry's selected_camera_names plus "
            "selected_supervised_camera_names as the camera/role contract on the hardtail and reserve streams."
        ),
        "scope_definition": "minimal dataset/config change only; do not rewrite loss routing, source policy family, or cloud flow",
        "first_candidate_hypothesis": readiness.get("hypothesis", ""),
        "first_candidate_shape": seed["first_candidate_shape"],
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": (
            "This first view-set replay candidate is repo-ready because the manifests already carry selected "
            "camera and role fields and the dataset can now replay them."
        ),
        "first_candidate_knobs": seed["first_candidate_knobs"],
        "required_exclusions": [
            "not anchor-only retry",
            "not loss-side cousin",
            "not blanket focal retry",
            "not source-pool retry",
            "not slot_3 ticket",
        ],
        "requires_dataset_plumbing": True,
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
            "The best current tail families still improve depth while paying a small camera tax.",
            "Anchor replay alone already failed, which points to the remaining unreplayed contract fields: ordered camera set and roles.",
            "The manifests already carry those fields, so the next bounded question is full tail view-set replay rather than another loss-side cousin.",
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
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed tail_contract_viewset_replay",
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
