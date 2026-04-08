from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_teacher_geometry_rehydrated_render_autoloop as base  # noqa: E402
from scripts.visible_coverage_floor_mutation_bank import (  # noqa: E402
    stage_a_source_subset_mutations,
    stage_b_visible_floor_mutations,
    stage_c_label_mutations,
)


FAMILY = "teacher_geometry_visible_coverage_floor_audit"
FIRST_SHAPE = "stablelead_visiblefloor_rehydrated_relbaseline_maskedhuman_v1"
PRIOR_FAMILY = "teacher_geometry_rehydrated_render_audit"
OLD_AUTOLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_rehydrated_render"
OLD_PANELS_ROOT = OLD_AUTOLOOP_ROOT / "panels"
AUTLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_visible_coverage_floor"


def load_previous_artifacts() -> dict:
    old_config_paths = sorted((OLD_AUTOLOOP_ROOT / "configs").glob("proxy_config.iter*.json"))
    old_panel_paths = sorted(path for path in OLD_PANELS_ROOT.rglob("*.png"))
    return {
        "research_status": base.load_json(base.RESEARCH_STATUS_JSON),
        "task_plan": base.load_json(base.TASK_PLAN_JSON),
        "watch": base.load_json(base.WATCH_JSON),
        "allowlist": base.load_json(base.ALLOWLIST_JSON),
        "next_draft": base.load_json(base.RESEARCH_ROOT / "next_manual_problem_draft.teacher_geometry_visible_coverage_floor_audit.json"),
        "old_result": base.load_json(base.RESEARCH_ROOT / "teacher_geometry_rehydrated_render_audit_result.json"),
        "old_postmortem": base.load_json(OLD_AUTOLOOP_ROOT / "autoloop_final_postmortem.json"),
        "old_best_local_state": base.load_json(OLD_AUTOLOOP_ROOT / "best_local_state.json"),
        "old_mutation_history": base.load_json(OLD_AUTOLOOP_ROOT / "mutation_history.json"),
        "old_iteration_ledger_lines": [
            base.json.loads(line)
            for line in (OLD_AUTOLOOP_ROOT / "iteration_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "old_config_paths": [base.rel(path) for path in old_config_paths],
        "old_panel_paths": [base.rel(path) for path in old_panel_paths],
    }


def write_truth_repair_report(previous: dict) -> Path:
    report_path = base.RESEARCH_ROOT / "visible_coverage_floor_truth_repair_report.json"
    report = {
        "checked_at": base.now_iso(),
        "prior_failure_family": PRIOR_FAMILY,
        "current_research_state": previous["research_status"]["state"],
        "allowlist_status": previous["allowlist"]["status"],
        "current_family_must_be": FAMILY,
        "repair_statements": [
            "rehydrated-render line had real local progress on fragmentation, masked L1, masked SSIM, and off-body support.",
            "its visible-coverage gate was still structurally dishonest because human_erasure_penalty was driven by absolute visibility floors that the frozen baseline itself did not satisfy.",
            "current family must be teacher_geometry_visible_coverage_floor_audit and must rank candidates by relative visible retention from a null best-state, not by inherited best-state from the previous family.",
        ],
        "prior_best_local_state": previous["old_best_local_state"],
        "artifact_counts": {
            "old_iteration_count": len(previous["old_iteration_ledger_lines"]),
            "old_config_count": len(previous["old_config_paths"]),
            "old_panel_count": len(previous["old_panel_paths"]),
        },
        "repaired_truth": "teacher_geometry_rehydrated_render_audit found honest local progress, but its absolute visible-floor gate created a fake wall. The next honest move is a fresh family that preserves relative visible human coverage while still demanding lower fragmentation and lower peaks.",
    }
    base.write_json(report_path, report)
    return report_path


def write_packaging_files(report_path: Path) -> None:
    base.write_json(
        base.RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "autoloop_local_only",
            "ready_for_execution": False,
            "report_path": base.rel(report_path),
        },
    )
    base.write_json(
        base.RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "execution_mode": "evaluator_only_proxy_only_render_only",
            "cloud_must_remain_off": True,
            "same_family_retry_forbidden": True,
            "notes": "Only evaluator/proxy/render mutations are allowed this round; training, dataset, trainer, and cloud training entrypoints remain frozen.",
        },
    )
    base.write_json(
        base.RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "write_surface": [
                "scripts/evaluate_teacher_visual_lift_cases.py",
                "scripts/score_correspondence_progress.py",
                "scripts/visible_coverage_floor_mutation_bank.py",
                "scripts/run_teacher_geometry_visible_coverage_floor_autoloop.py",
            ],
        },
    )
    base.write_json(
        base.RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "autoloop_started",
            "reason": "Current honest next move is evaluator-only visible-coverage-floor auditing because the prior family already showed real local progress, but its erasure gate was structurally too absolute.",
        },
    )
    base.write_json(
        base.RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "status": "local_only_autoloop",
            "ready_for_execution": False,
            "cloud_gate_open": False,
            "training_code_frozen": True,
        },
    )
    base.write_json(
        base.RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.json",
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "decision": "HOLD_FOR_LOCAL_AUTOLOOP",
            "ready_for_execution": False,
            "cloud_gate_open": False,
        },
    )


def stage_failure_summary(compare: dict) -> str:
    failure_class = base.classify_failure(compare)
    if failure_class == "metric_truth_bug":
        return "relative visible-floor auditing still found a metric-truth bug; visible coverage / retained ratios remain inconsistent and must be repaired before another family is credible."
    if failure_class == "erasure_win":
        return "best local progress still came from rehydrated visibility that improved concentration only by under-retaining the visible human body relative to the frozen baseline."
    if failure_class == "fragmentation_win":
        return "relative visible coverage stayed honest, but fg fragmentation and multi-peak support still dominated the human body."
    if failure_class == "background_only_win":
        return "visible coverage stayed honest and outside-body support stayed controlled, but the body itself did not consolidate into a cleaner single-human rendering."
    return "bounded local visible-coverage-floor mutations produced no honest movement on the primary human-only objectives."


def smoke_stage_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["mean_delta_fg_connected_components"] <= -20.0
        and compare["mean_delta_fg_peak_count"] <= -2.0
        and compare["mean_delta_masked_l1"] <= -0.005
        and compare["mean_delta_masked_ssim"] >= 0.002
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.98
        and compare["mean_fg_visible_bbox_retained_ratio"] >= 0.95
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_human_erasure_penalty"] <= 0.05
    )


def stage_c_eligible(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.98
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_human_erasure_penalty"] <= 0.05
    )


def run_eval(manifest_path: Path, case_set: str, output_dir: Path, proxy_config_path: Path) -> dict:
    base.ensure_dir(output_dir)
    base.run_checked(
        [
            str(base.PYTHON_EXE if base.PYTHON_EXE.exists() else sys.executable),
            str(base.EVAL_SCRIPT),
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(base.TEACHER_CHECKPOINT),
            "--case-set",
            case_set,
            "--variants",
            ",".join(base.PROXY_VARIANTS),
            "--support-threshold",
            "0.25",
            "--bottom-band-ratio",
            "0.20",
            "--render-max-points",
            "300000",
            "--proxy-config-json",
            str(proxy_config_path),
        ]
    )
    payload = base.load_json(output_dir / "summary.json")
    payload["_root"] = output_dir
    return payload


def configure_runner() -> None:
    base.AUTLOOP_ROOT = AUTLOOP_ROOT
    base.RUNS_ROOT = AUTLOOP_ROOT / "runs"
    base.PANELS_ROOT = AUTLOOP_ROOT / "panels"
    base.CONFIG_ROOT = AUTLOOP_ROOT / "configs"
    base.FAMILY = FAMILY
    base.FIRST_SHAPE = FIRST_SHAPE
    base.PRIOR_FAMILY = PRIOR_FAMILY
    base.OLD_AUTOLOOP_ROOT = OLD_AUTOLOOP_ROOT
    base.OLD_PANELS_ROOT = OLD_PANELS_ROOT
    base.NEXT_DRAFT_BY_FAILURE = {
        "metric_truth_bug": "teacher_geometry_visible_mass_truth_audit",
        "erasure_win": "teacher_geometry_visible_mass_floor_audit",
        "fragmentation_win": "teacher_geometry_label_consistency_visible_audit",
        "background_only_win": "teacher_geometry_source_agreement_visible_audit",
        "no_movement": "teacher_geometry_rehydrated_consensus_blend_audit",
    }
    base.stage_a_source_subset_mutations = stage_a_source_subset_mutations
    base.stage_b_rehydrated_mutations = stage_b_visible_floor_mutations
    base.stage_c_label_mutations = stage_c_label_mutations
    base.load_previous_artifacts = load_previous_artifacts
    base.write_truth_repair_report = write_truth_repair_report
    base.write_packaging_files = write_packaging_files
    base.stage_failure_summary = stage_failure_summary
    base.smoke_stage_accept = smoke_stage_accept
    base.stage_c_eligible = stage_c_eligible
    base.run_eval = run_eval


def main() -> int:
    base.py_compile(
        [
            REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py",
            REPO_ROOT / "scripts" / "score_correspondence_progress.py",
            REPO_ROOT / "scripts" / "visible_coverage_floor_mutation_bank.py",
            Path(__file__),
        ]
    )
    configure_runner()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
