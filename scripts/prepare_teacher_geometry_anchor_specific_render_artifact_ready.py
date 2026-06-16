import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
RESEARCH_ROOT = OUTPUT_ROOT / "zju_source_policy_research_loop"

FAMILY = "teacher_geometry_anchor_specific_render_artifact_audit"
FIRST_SHAPE = "stablelead_anchor_specific_render_artifact_layer_suppression_maskedhuman_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    checked_at = datetime.now().astimezone().isoformat()
    evidence = _read_json(RESEARCH_ROOT / f"{FAMILY}_evidence_summary.json")
    manual_packet = _read_json(RESEARCH_ROOT / f"{FAMILY}_manual_packet.json")
    research_status = _read_json(RESEARCH_ROOT / "research_loop_status.json")
    watch = _read_json(OUTPUT_ROOT / "zju_source_policy_research_watch" / "latest_watch_snapshot.json")
    allowlist = _read_json(RESEARCH_ROOT / "repo_process_allowlist.json")

    aggregate = evidence["aggregate"]
    remaining_failure_is_render_artifact = bool(
        evidence["correspondence_not_primary"]
        and aggregate["dominant_artifact_type"] in {
            "multilayer_residual",
            "multi_component_fragmentation",
            "primary_secondary_lobe_competition",
            "peak_rebound",
        }
    )
    worth_execution_ready_discussion = bool(
        remaining_failure_is_render_artifact
        and manual_packet["summary"]["best_control_accept"]
        and not manual_packet["summary"]["best_hero_accept"]
    )
    next_direction = (
        ""
        if worth_execution_ready_discussion
        else "teacher_geometry_anchor_specific_visible_component_fragmentation_audit"
    )
    dominant_mode = aggregate["dominant_artifact_type"]
    if dominant_mode == "multilayer_residual":
        specific_failure = "多层残留"
    elif dominant_mode == "multi_component_fragmentation":
        specific_failure = "多组件裂解"
    elif dominant_mode == "primary_secondary_lobe_competition":
        specific_failure = "主次 lobe 竞争"
    elif dominant_mode == "peak_rebound":
        specific_failure = "peak-only rebound"
    else:
        specific_failure = "混合 render artifact"

    payload = {
        "checked_at": checked_at,
        "family": FAMILY,
        "remaining_failure_mainly_inside_fg_render_artifact": remaining_failure_is_render_artifact,
        "artifact_primary_mode": dominant_mode,
        "artifact_primary_mode_zh_cn": specific_failure,
        "question_answers": {
            "is_remaining_failure_mainly_inside_fg_render_artifact": remaining_failure_is_render_artifact,
            "is_it_more_like_multilayer_residual": dominant_mode == "multilayer_residual",
            "is_it_more_like_multicomponent_fragmentation": dominant_mode == "multi_component_fragmentation",
            "is_it_more_like_primary_secondary_lobe_competition": dominant_mode == "primary_secondary_lobe_competition",
            "is_it_more_like_peak_only_rebound": dominant_mode == "peak_rebound",
        },
        "worth_execution_ready_family_discussion": worth_execution_ready_discussion,
        "execution_ready_now": False,
        "arm_allowed": False,
        "run_allowed": False,
        "cloud_allowed": False,
        "first_candidate_shape_if_promoted": FIRST_SHAPE if worth_execution_ready_discussion else "",
        "next_higher_level_honest_manual_direction": next_direction,
        "suggested_future_write_surface": [
            "render-artifact suppression / rehydrated composition logic around evaluator-side render path",
            "panel/metric instrumentation for multilayer vs duplicate-lobe tracking",
        ],
        "primary_risks": [
            "The residual artifact may still sit above correspondence and require render-composition changes rather than more correspondence tuning.",
            "Over-suppressing secondary lobes could hide valid visible human mass and regress masked quality.",
            "A local artifact suppressor may improve the anchor cases but still fail to generalize to hero/local20 without broader geometry changes.",
        ],
        "next_step": (
            "Keep research_loop_status at IDLE_GUARD, do not arm/run, and use this packet for manual execution-ready discussion."
            if worth_execution_ready_discussion
            else f"Do not promote this family yet; define a higher-level next manual direction: {next_direction}."
        ),
        "live_truth_guard": {
            "research_state": research_status["state"],
            "approved_problem_present": research_status["approved_problem_present"],
            "cloud_must_remain_off": research_status["cloud_must_remain_off"],
            "watch_current_review_packet": watch["research"]["summary"]["current_review_packet"],
            "allowlist_status": allowlist["status"],
        },
        "evidence_summary_json": str((RESEARCH_ROOT / f"{FAMILY}_evidence_summary.json").relative_to(REPO_ROOT)),
        "manual_packet_json": str((RESEARCH_ROOT / f"{FAMILY}_manual_packet.json").relative_to(REPO_ROOT)),
    }
    _write_json(RESEARCH_ROOT / f"{FAMILY}_readiness_check.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
