import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the promoted residual-case-coverage line into one manual-approval packet."
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "hardcase_bucket_entries.promotedlead.v1.json",
    )
    parser.add_argument(
        "--coverage-profile-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "official_hardtail_coverage_profile.20260329.json",
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "residual_case_coverage_readiness.20260329.json",
    )
    parser.add_argument(
        "--slot3-decision-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "selection_contract_mechanism_decision.20260329.json",
    )
    parser.add_argument(
        "--seed-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "approved_problem.seed.residual_case_coverage_rebalancing.json",
    )
    parser.add_argument(
        "--blueprint-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "family_blueprint.residual_case_coverage_rebalancing.json",
    )
    parser.add_argument(
        "--research-status-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "research_loop_status.json",
    )
    parser.add_argument(
        "--candidate-patch-plan-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "candidate_patch_plan.json",
    )
    parser.add_argument(
        "--cloud-runtime-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "cloud_runtime_state.20260329.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "manual_approval_packet.residual_case_coverage_rebalancing.20260329.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "manual_approval_packet.residual_case_coverage_rebalancing.20260329.md",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_packet(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest_json)
    coverage = load_json(args.coverage_profile_json)
    readiness = load_json(args.readiness_json)
    slot3 = load_json(args.slot3_decision_json)
    seed = load_json(args.seed_json)
    blueprint = load_json(args.blueprint_json)
    research_status = load_json(args.research_status_json)
    candidate_patch_plan = load_json(args.candidate_patch_plan_json)
    cloud_runtime = load_json(args.cloud_runtime_json) if args.cloud_runtime_json.exists() else {}

    dominant_anchor = (coverage.get("anchor_enrichment_ranked") or [{}])[0]
    dominant_source_set = (coverage.get("source_only_set_enrichment_ranked") or [{}])[0]
    largest_segment = (coverage.get("hard_tail_segments_ranked") or [{}])[0]
    readiness_flags = readiness.get("readiness", {}) or {}

    arm_command = (
        candidate_patch_plan.get("preferred_first_candidate_arm_command")
        or candidate_patch_plan.get("current_priority_arm_command")
        or "python scripts/arm_zju_source_policy_approved_problem.py --seed residual_case_coverage_rebalancing"
    )
    run_command = (
        candidate_patch_plan.get("preferred_first_candidate_run_command")
        or candidate_patch_plan.get("current_priority_run_command")
        or "python scripts/run_zju_source_policy_research_candidate.py"
    )

    return {
        "checked_at": iso_now(),
        "packet_kind": "manual_approval_packet",
        "family": "residual_case_coverage_rebalancing",
        "status": "ready_for_manual_review_and_single_ticket_execution_pending_approval",
        "problem_statement": seed.get("problem_statement", ""),
        "promoted_local_lead_config": seed.get("first_candidate_knobs", {}).get("promoted_local_lead_config", ""),
        "first_candidate_shape": seed.get("first_candidate_shape", ""),
        "first_candidate_config": seed.get("first_candidate_config", ""),
        "why_this_ticket_now": [
            "The official promoted hard-tail manifest is now frozen from real per-frame residual rows.",
            "The manifest-aware hardcasebucketmix4to1 config already instantiates cleanly on the current repo.",
            "The slot_3 mechanism line is explicitly excluded as the next ticket because its probe basket never overlaps the labeled hard tail.",
        ],
        "official_hardtail_manifest": {
            "path": str(args.manifest_json.resolve()),
            "status": manifest.get("status", ""),
            "eligible_entry_count": manifest.get("eligible_entry_count"),
            "hard_tail_entry_count": manifest.get("hard_tail_entry_count"),
            "hard_tail_entry_share": manifest.get("hard_tail_entry_share"),
            "tail_metric_name": manifest.get("tail_metric_name", ""),
            "threshold_rule": manifest.get("threshold_rule", {}),
        },
        "hard_tail_shape_summary": {
            "dominant_anchor_camera": dominant_anchor.get("promoted_anchor_camera", ""),
            "dominant_anchor_tail_count": dominant_anchor.get("tail_count"),
            "dominant_anchor_tail_enrichment_vs_baseline": dominant_anchor.get("tail_enrichment_vs_baseline"),
            "dominant_source_only_set": dominant_source_set.get("selected_source_only_set", []),
            "largest_hard_tail_segment": largest_segment,
        },
        "slot3_exclusion_reason": {
            "decision": slot3.get("decision", ""),
            "stop_reason": slot3.get("stop_reason", ""),
            "current_tail_problem_family_remains": slot3.get("current_tail_problem_family_remains", ""),
            "probe_gap": coverage.get("slot3_probe_gap", {}),
        },
        "execution_readiness": {
            "ready_for_manual_review": readiness_flags.get("ready_for_manual_review"),
            "ready_for_execution": readiness_flags.get("ready_for_execution"),
            "requires_new_manual_approval": readiness_flags.get("requires_new_manual_approval"),
            "do_not_auto_open_ticket": readiness_flags.get("do_not_auto_open_ticket"),
            "plumbing_smoke": readiness.get("plumbing_smoke", {}),
        },
        "single_ticket_execution_contract": {
            "arm_command": arm_command,
            "run_command": run_command,
            "gate_sequence": blueprint.get("gate_sequence", []),
            "same_night_second_candidate_forbidden": candidate_patch_plan.get(
                "same_night_second_candidate_forbidden", True
            ),
            "same_night_cousin_sweep_forbidden": candidate_patch_plan.get(
                "same_night_cousin_sweep_forbidden", True
            ),
            "return_to_guard_required": True,
            "cloud_must_remain_off": bool(seed.get("cloud_must_remain_off", True)),
        },
        "planning_sync": {
            "research_loop_state": research_status.get("state", ""),
            "current_priority_family": research_status.get("current_priority_family", ""),
            "preferred_first_family": research_status.get("preferred_first_family", ""),
            "seed_path": str(args.seed_json.resolve()),
            "blueprint_path": str(args.blueprint_json.resolve()),
            "research_status_path": str(args.research_status_json.resolve()),
            "candidate_patch_plan_path": str(args.candidate_patch_plan_json.resolve()),
        },
        "guard_clean_check": {
            "mode": cloud_runtime.get("mode", ""),
            "active_modal_app_count": cloud_runtime.get("active_modal_app_count"),
            "launcher_guard": cloud_runtime.get("launcher_guard", ""),
        },
        "evidence_refs": [
            str(args.manifest_json.resolve()),
            str(args.coverage_profile_json.resolve()),
            str(args.readiness_json.resolve()),
            str(args.slot3_decision_json.resolve()),
            str(args.seed_json.resolve()),
            str(args.blueprint_json.resolve()),
            str(args.research_status_json.resolve()),
            str(args.candidate_patch_plan_json.resolve()),
        ],
    }


def render_md(packet: dict) -> str:
    manifest = packet.get("official_hardtail_manifest", {})
    hard_tail = packet.get("hard_tail_shape_summary", {})
    slot3 = packet.get("slot3_exclusion_reason", {})
    readiness = packet.get("execution_readiness", {})
    contract = packet.get("single_ticket_execution_contract", {})
    planning = packet.get("planning_sync", {})
    guard = packet.get("guard_clean_check", {})
    segment = hard_tail.get("largest_hard_tail_segment", {}) or {}

    lines = [
        "# Residual Case Coverage Manual Approval Packet (2026-03-29)",
        "",
        "## Candidate",
        "",
        f"- family: `{packet.get('family', '')}`",
        f"- status: `{packet.get('status', '')}`",
        f"- first_candidate_shape: `{packet.get('first_candidate_shape', '')}`",
        f"- first_candidate_config: `{packet.get('first_candidate_config', '')}`",
        f"- promoted_local_lead_config: `{packet.get('promoted_local_lead_config', '')}`",
        "",
        "## Why Now",
        "",
    ]
    for item in packet.get("why_this_ticket_now", []):
        lines.append(f"- {item}")
    lines += [
        "",
        "## Official Hard Tail",
        "",
        f"- manifest_status: `{manifest.get('status', '')}`",
        f"- eligible_entry_count: `{manifest.get('eligible_entry_count', '')}`",
        f"- hard_tail_entry_count: `{manifest.get('hard_tail_entry_count', '')}`",
        f"- hard_tail_entry_share: `{fmt(manifest.get('hard_tail_entry_share'))}`",
        f"- tail_metric_name: `{manifest.get('tail_metric_name', '')}`",
        f"- dominant_anchor_camera: `{hard_tail.get('dominant_anchor_camera', '')}`",
        f"- dominant_anchor_tail_count: `{hard_tail.get('dominant_anchor_tail_count', '')}`",
        f"- dominant_anchor_tail_enrichment_vs_baseline: `{fmt(hard_tail.get('dominant_anchor_tail_enrichment_vs_baseline'))}`",
        f"- dominant_source_only_set: `{hard_tail.get('dominant_source_only_set', [])}`",
        f"- largest_hard_tail_segment: `frames {segment.get('frame_start', 'n/a')}-{segment.get('frame_end', 'n/a')} ({segment.get('entry_count', 'n/a')} entries)`",
        "",
        "## Why Not Slot_3",
        "",
        f"- decision: `{slot3.get('decision', '')}`",
        f"- stop_reason: `{slot3.get('stop_reason', '')}`",
        f"- current_tail_problem_family_remains: `{slot3.get('current_tail_problem_family_remains', '')}`",
        "",
        "## Execution Contract",
        "",
        f"- ready_for_manual_review: `{readiness.get('ready_for_manual_review', '')}`",
        f"- ready_for_execution: `{readiness.get('ready_for_execution', '')}`",
        f"- requires_new_manual_approval: `{readiness.get('requires_new_manual_approval', '')}`",
        f"- do_not_auto_open_ticket: `{readiness.get('do_not_auto_open_ticket', '')}`",
        f"- arm_command: `{contract.get('arm_command', '')}`",
        f"- run_command: `{contract.get('run_command', '')}`",
        f"- gate_sequence: `{contract.get('gate_sequence', [])}`",
        f"- same_night_second_candidate_forbidden: `{contract.get('same_night_second_candidate_forbidden', '')}`",
        f"- same_night_cousin_sweep_forbidden: `{contract.get('same_night_cousin_sweep_forbidden', '')}`",
        f"- return_to_guard_required: `{contract.get('return_to_guard_required', '')}`",
        f"- cloud_must_remain_off: `{contract.get('cloud_must_remain_off', '')}`",
        "",
        "## Planning Sync",
        "",
        f"- research_loop_state: `{planning.get('research_loop_state', '')}`",
        f"- current_priority_family: `{planning.get('current_priority_family', '')}`",
        f"- preferred_first_family: `{planning.get('preferred_first_family', '')}`",
        f"- seed_path: `{planning.get('seed_path', '')}`",
        f"- blueprint_path: `{planning.get('blueprint_path', '')}`",
        "",
        "## Guard",
        "",
        f"- mode: `{guard.get('mode', '')}`",
        f"- active_modal_app_count: `{guard.get('active_modal_app_count', '')}`",
        f"- launcher_guard: `{guard.get('launcher_guard', '')}`",
        "",
        "## Evidence",
        "",
    ]
    for ref in packet.get("evidence_refs", []):
        lines.append(f"- `{ref}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    packet = build_packet(args)
    write_json(args.output_json, packet)
    write_text(args.output_md, render_md(packet))
    print(f"[wrote] {args.output_json}")
    print(f"[wrote] {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
