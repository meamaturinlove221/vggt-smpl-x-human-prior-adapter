from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_teacher_geometry_rehydrated_render_autoloop as base  # noqa: E402
from scripts.fragmentation_generalization_mutation_bank import (  # noqa: E402
    stage_a_seed_mutations,
    stage_b_generalization_mutations,
    stage_c_label_generalization_mutations,
)
from scripts.score_correspondence_progress import compare_variant  # noqa: E402


FAMILY = "teacher_geometry_fragmentation_generalization_audit"
FIRST_SHAPE = "stablelead_fragmentation_generalization_consensus_margin_softsmooth_v1"
PRIOR_FAMILY = "teacher_geometry_visible_coverage_floor_audit"
OLD_AUTOLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_visible_coverage_floor"
AUTLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_fragmentation_generalization"
RUNS_ROOT = AUTLOOP_ROOT / "runs"
PANELS_ROOT = AUTLOOP_ROOT / "panels"
CONFIG_ROOT = AUTLOOP_ROOT / "configs"
NEXT_FAMILY = "teacher_geometry_casewise_fragmentation_stability_audit"
PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
]
FIXED_SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"


def load_previous_artifacts() -> dict:
    old_config_paths = sorted((OLD_AUTOLOOP_ROOT / "configs").glob("proxy_config.iter*.json"))
    old_panel_paths = sorted(path for path in (OLD_AUTOLOOP_ROOT / "panels").rglob("*.png"))
    variant_rescue = base.load_json(OLD_AUTOLOOP_ROOT / "variant_rescue_report.json")
    local20_summary = base.load_json(Path(variant_rescue["local20_summary_json"]))
    return {
        "research_status": base.load_json(base.RESEARCH_STATUS_JSON),
        "task_plan": base.load_json(base.TASK_PLAN_JSON),
        "watch": base.load_json(base.WATCH_JSON),
        "allowlist": base.load_json(base.ALLOWLIST_JSON),
        "next_draft": base.load_json(
            base.RESEARCH_ROOT / "next_manual_problem_draft.teacher_geometry_fragmentation_generalization_audit.json"
        ),
        "old_result": base.load_json(base.RESEARCH_ROOT / "teacher_geometry_visible_coverage_floor_audit_result.json"),
        "old_postmortem": base.load_json(base.RESEARCH_ROOT / "teacher_geometry_visible_coverage_floor_audit_postmortem.json"),
        "old_best_local_state": base.load_json(OLD_AUTOLOOP_ROOT / "best_local_state.json"),
        "old_best_state": base.load_json(OLD_AUTOLOOP_ROOT / "best_state.json"),
        "old_variant_rescue": variant_rescue,
        "old_stage_a_shortlist": base.load_json(OLD_AUTOLOOP_ROOT / "stage_a_shortlist.json"),
        "old_mutation_history": base.load_json(OLD_AUTOLOOP_ROOT / "mutation_history.json"),
        "old_iteration_ledger_lines": [
            json.loads(line)
            for line in (OLD_AUTOLOOP_ROOT / "iteration_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "old_config_paths": [base.rel(path) for path in old_config_paths],
        "old_panel_paths": [base.rel(path) for path in old_panel_paths],
        "old_local20_summary": local20_summary,
    }


def write_truth_repair_report(previous: dict) -> Path:
    report_path = base.RESEARCH_ROOT / "fragmentation_generalization_truth_repair_report.json"
    report = {
        "checked_at": base.now_iso(),
        "prior_failure_family": PRIOR_FAMILY,
        "current_research_state": previous["research_status"]["state"],
        "allowlist_status": previous["allowlist"]["status"],
        "current_family_must_be": FAMILY,
        "repair_statements": [
            "visible-coverage-floor line repaired the fake erasure wall and rescued an honest consensus-margin candidate.",
            "that rescued candidate passed smoke and hero but failed local20 because fg fragmentation rebounded across the broader benchmark.",
            "current family must be teacher_geometry_fragmentation_generalization_audit, and it must rank candidates by benchmark-facing fragmentation stability rather than inherited single-iteration best-state alone.",
        ],
        "rescued_candidate": previous["old_variant_rescue"],
        "artifact_counts": {
            "old_iteration_count": len(previous["old_iteration_ledger_lines"]),
            "old_config_count": len(previous["old_config_paths"]),
            "old_panel_count": len(previous["old_panel_paths"]),
        },
        "repaired_truth": "We are not at the old fake erasure wall anymore. The next honest local bottleneck is fragmentation rebound from hero to local20, so the new family must optimize generalization stability rather than re-litigate coverage honesty.",
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
            "notes": "Only evaluator/proxy/render autoloop is allowed this round; training, dataset, trainer, and cloud training entrypoints remain frozen.",
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
                "scripts/fragmentation_generalization_mutation_bank.py",
                "scripts/run_teacher_geometry_fragmentation_generalization_autoloop.py",
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
            "reason": "Current honest next move is a benchmark-facing fragmentation-generalization autoloop because the prior family already rescued an honest hero candidate but local20 rebounded on fg fragmentation.",
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


def select_control_cases(previous: dict, *, limit: int = 6) -> list[dict]:
    summary = previous["old_local20_summary"]
    variant = previous["old_variant_rescue"]["rescued_variant"]
    baseline = {row["case_id"]: row for row in summary["rows"] if row["variant"] == "baseline_depth_unproject"}
    candidate = [row for row in summary["rows"] if row["variant"] == variant]
    ranked = []
    for row in candidate:
        base_row = baseline[row["case_id"]]
        support = row["support_metrics"]
        base_support = base_row["support_metrics"]
        ranked.append(
            {
                "row": row["case"],
                "delta_fg_connected_components": support["fg_connected_components"] - base_support["fg_connected_components"],
                "delta_fg_peak_count": support["fg_peak_count"] - base_support["fg_peak_count"],
            }
        )
    ranked.sort(key=lambda item: (-item["delta_fg_connected_components"], -item["delta_fg_peak_count"]))
    return [item["row"] for item in ranked[:limit]]


def _annotate(compare: dict) -> dict:
    compare = dict(compare)
    compare["smoke_stage_accept"] = smoke_accept(compare)
    compare["control_accept"] = control_accept(compare)
    compare["hero_accept"] = hero_accept(compare)
    compare["local20_accept"] = local20_accept(compare)
    compare["failure_class"] = classify_failure(compare)
    return compare


def smoke_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["mean_delta_fg_connected_components"] <= -10.0
        and compare["mean_delta_fg_peak_count"] <= -2.0
        and compare["mean_delta_masked_l1"] <= -0.005
        and compare["mean_delta_masked_ssim"] >= 0.002
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.98
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_human_erasure_penalty"] <= 0.05
    )


def control_accept(compare: dict) -> bool:
    return (
        smoke_accept(compare)
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.96
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.96
        and compare["mean_human_erasure_penalty"] <= 0.03
    )


def hero_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["improved_all_primary_count"] >= 3
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.58
        and compare["mean_human_erasure_penalty"] <= 0.03
    )


def local20_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["improved_all_primary_count"] >= 14
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.60
        and compare["mean_human_erasure_penalty"] <= 0.02
    )


def classify_failure(compare: dict) -> str:
    if compare.get("metric_truth_bug"):
        return "metric_truth_bug"
    if (
        compare["mean_human_erasure_penalty"] > 0.05
        or compare["mean_fg_visible_coverage_retained_ratio"] < 0.98
        or compare["mean_fg_visible_mass_retained_ratio"] < 0.95
        or compare["mean_largest_fg_visible_component_retained_ratio"] < 0.95
    ):
        return "coverage_regression"
    if compare["mean_delta_fg_connected_components"] >= 0.0 or compare["mean_delta_fg_peak_count"] >= 0.0:
        return "fragmentation_rebound"
    if compare["mean_delta_masked_l1"] > 0.0 or compare["mean_delta_masked_ssim"] < 0.0:
        return "quality_regression"
    if compare["mean_delta_off_body_support_ratio"] > 0.0 or compare["mean_delta_bg_bottom_support_ratio"] > 0.0:
        return "external_residue_rebound"
    return "no_movement"


def ranking_key(compare: dict) -> tuple:
    return (
        0 if compare["local20_accept"] else 1,
        0 if compare["hero_accept"] else 1,
        0 if compare["control_accept"] else 1,
        0 if compare["smoke_stage_accept"] else 1,
        0 if not compare["metric_truth_bug"] else 1,
        compare["mean_human_erasure_penalty"],
        -compare["mean_fg_visible_coverage_retained_ratio"],
        -compare["mean_fg_visible_mass_retained_ratio"],
        -compare["mean_largest_fg_visible_component_retained_ratio"],
        compare["mean_delta_fg_connected_components"],
        compare["mean_delta_fg_peak_count"],
        compare["mean_delta_masked_l1"],
        -compare["mean_delta_masked_ssim"],
        compare["mean_delta_off_body_support_ratio"],
        compare["mean_delta_bg_bottom_support_ratio"],
    )


def compare_stage(summary: dict, case_ids: list[str]) -> list[dict]:
    rows = []
    for variant in PROXY_VARIANTS:
        rows.append(_annotate(compare_variant(summary, variant, case_ids)))
    rows.sort(key=ranking_key)
    return rows


def stage_failure_summary(compare: dict) -> str:
    failure_class = classify_failure(compare)
    if failure_class == "metric_truth_bug":
        return "fragmentation-generalization line still found a metric truth bug; the evaluator is not yet honest enough to rank benchmark stability."
    if failure_class == "coverage_regression":
        return "the candidate only reduced fragmentation by giving back too much visible human coverage or visible mass."
    if failure_class == "quality_regression":
        return "fragmentation control improved but masked reconstruction quality regressed, so the apparent progress is not honest."
    if failure_class == "external_residue_rebound":
        return "fragmentation control improved locally, but off-body or bottom-band residue rebounded on the broader benchmark."
    if failure_class == "fragmentation_rebound":
        return "smoke/hero progress did not generalize; hard benchmark cases still rebound into fg fragmentation and multi-peak support."
    return "bounded fragmentation-generalization mutations produced no honest movement on the benchmark-facing bottleneck."


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
            ",".join(PROXY_VARIANTS),
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


def run_stage_eval(
    *,
    iter_idx: int,
    mutation: dict,
    manifest_path: Path,
    case_set: str,
    case_ids: list[str],
    gate_label: str,
) -> dict:
    iter_tag = f"iter{iter_idx:02d}"
    config_path = CONFIG_ROOT / f"proxy_config.{iter_tag}.json"
    base.write_json(config_path, mutation["proxy_config"])
    summary = run_eval(manifest_path, case_set, RUNS_ROOT / iter_tag / f"{gate_label}_eval", config_path)
    ranking = compare_stage(summary, case_ids)
    best_compare = ranking[0]
    panels = base.build_panels(summary, best_compare["variant"], iter_tag, case_ids[0])
    return {
        "checked_at": base.now_iso(),
        "iteration": iter_idx,
        "stage": mutation["stage"],
        "mutation": mutation,
        "gate_label": gate_label,
        "case_ids": case_ids,
        "best_variant": best_compare["variant"],
        "best_compare": best_compare,
        "ranking": ranking,
        "panels": panels,
        "summary_json": base.rel(Path(summary["_root"]) / "summary.json"),
        "summary_md": base.rel(Path(summary["_root"]) / "summary.md"),
    }


def main() -> int:
    base.AUTLOOP_ROOT = AUTLOOP_ROOT
    base.RUNS_ROOT = RUNS_ROOT
    base.PANELS_ROOT = PANELS_ROOT
    base.CONFIG_ROOT = CONFIG_ROOT
    base.FAMILY = FAMILY
    base.FIRST_SHAPE = FIRST_SHAPE
    base.PRIOR_FAMILY = PRIOR_FAMILY
    base.ensure_dir(AUTLOOP_ROOT)
    base.reset_dir(RUNS_ROOT)
    base.reset_dir(PANELS_ROOT)
    base.reset_dir(CONFIG_ROOT)
    for extra in [
        AUTLOOP_ROOT / "best_state.json",
        AUTLOOP_ROOT / "best_local_state.json",
        AUTLOOP_ROOT / "mutation_history.json",
        AUTLOOP_ROOT / "iteration_ledger.jsonl",
        AUTLOOP_ROOT / "autoloop_final_postmortem.json",
        AUTLOOP_ROOT / "stage_a_shortlist.json",
        AUTLOOP_ROOT / "control_cases_manifest.json",
    ]:
        if extra.exists():
            extra.unlink()

    previous = load_previous_artifacts()
    base.py_compile(
        [
            REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py",
            REPO_ROOT / "scripts" / "score_correspondence_progress.py",
            REPO_ROOT / "scripts" / "fragmentation_generalization_mutation_bank.py",
            Path(__file__),
        ]
    )

    truth_report_path = write_truth_repair_report(previous)
    loop_state_path = base.RESEARCH_ROOT / f"{FAMILY}_loop_state.json"
    write_packaging_files(truth_report_path)
    base.update_loop_state(
        loop_state_path,
        current_iteration=0,
        diagnosed_failure_mode="pending",
        chosen_stage="truth_repair",
        chosen_mutation="none",
        local_gate_status="pending",
        cloud_gate_open=False,
        artifact_paths=[base.rel(truth_report_path)],
    )

    manifest = base.load_json(base.BENCHMARK_MANIFEST)
    smoke_rows = [
        base.find_case(manifest, FIXED_SMOKE_CASE_ID),
        base.choose_second_smoke_anchor(manifest, FIXED_SMOKE_CASE_ID),
    ]
    smoke_case_ids = [base.case_id(row) for row in smoke_rows]
    hero_rows = base.select_manifest_rows(manifest, "hero_cases")
    hero_case_ids = [base.case_id(row) for row in hero_rows]
    control_rows = select_control_cases(previous, limit=6)
    control_case_ids = [base.case_id(row) for row in control_rows]
    benchmark_rows = base.select_manifest_rows(manifest, "benchmark_cases")
    benchmark_case_ids = [base.case_id(row) for row in benchmark_rows]
    source_count = len(smoke_rows[0]["source_cameras"])

    smoke_manifest_path = AUTLOOP_ROOT / "smoke_cases_manifest.json"
    control_manifest_path = AUTLOOP_ROOT / "control_cases_manifest.json"
    base.write_manifest(smoke_rows, smoke_manifest_path, key="cases")
    base.write_manifest(control_rows, control_manifest_path, key="cases")

    prior_subsets = []
    if previous["old_best_state"].get("mutation", {}).get("source_subset"):
        prior_subsets.append(list(previous["old_best_state"]["mutation"]["source_subset"]))
    if previous["old_best_local_state"].get("mutation", {}).get("source_subset"):
        prior_subsets.append(list(previous["old_best_local_state"]["mutation"]["source_subset"]))
    for row in previous["old_stage_a_shortlist"].get("rows", [])[:3]:
        if row.get("source_subset"):
            prior_subsets.append(list(row["source_subset"]))

    stage_a_mutations = stage_a_seed_mutations(source_count, prior_subsets=prior_subsets)
    all_reports: list[dict] = []
    report_paths: list[Path] = []
    mutation_history: list[str] = []
    best_state = None
    best_key = None
    iter_idx = 0

    def ingest(report: dict, *, decision_stage: str) -> None:
        nonlocal best_state, best_key
        all_reports.append(report)
        mutation_history.append(report["mutation"]["mutation_id"])
        base.append_jsonl(AUTLOOP_ROOT / "iteration_ledger.jsonl", report)
        base.write_json(AUTLOOP_ROOT / "mutation_history.json", {"checked_at": base.now_iso(), "mutations": mutation_history})
        current_key = ranking_key(report["best_compare"])
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_state = {
                "checked_at": base.now_iso(),
                "iteration": report["iteration"],
                "stage": report["stage"],
                "mutation": report["mutation"],
                "failure_class": report["best_compare"]["failure_class"],
                "compare": report["best_compare"],
                "panels": report["panels"],
                "summary_json": report["summary_json"],
            }
            base.write_json(AUTLOOP_ROOT / "best_state.json", best_state)

        report_path = base.RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{report['iteration']:02d}.json"
        decision_path = base.RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{report['iteration']:02d}.json"
        base.write_json(report_path, report)
        base.write_json(
            decision_path,
            {
                "checked_at": base.now_iso(),
                "iteration": report["iteration"],
                "stage": decision_stage,
                "chosen_mutation": report["mutation"]["mutation_id"],
                "smoke_pass": bool(report["best_compare"]["smoke_stage_accept"]),
                "control_pass": bool(report["best_compare"]["control_accept"]),
                "hero_pass": bool(report["best_compare"]["hero_accept"]),
                "local20_pass": bool(report["best_compare"]["local20_accept"]),
                "failure_class": report["best_compare"]["failure_class"],
            },
        )
        report_paths.extend([report_path, decision_path])
        base.update_loop_state(
            loop_state_path,
            current_iteration=report["iteration"],
            diagnosed_failure_mode=report["best_compare"]["failure_class"],
            chosen_stage=decision_stage,
            chosen_mutation=report["mutation"]["mutation_id"],
            local_gate_status=report["gate_label"],
            cloud_gate_open=False,
            artifact_paths=base.collect_artifact_paths(report_paths),
        )

    stage_a_reports = []
    for mutation in stage_a_mutations:
        iter_idx += 1
        report = run_stage_eval(
            iter_idx=iter_idx,
            mutation=mutation,
            manifest_path=smoke_manifest_path,
            case_set="cases",
            case_ids=smoke_case_ids,
            gate_label="smoke",
        )
        ingest(report, decision_stage="stage_a_smoke_seed")
        stage_a_reports.append(report)

    stage_a_sorted = sorted(stage_a_reports, key=lambda item: ranking_key(item["best_compare"]))
    shortlist = stage_a_sorted[:3]
    base.write_json(
        AUTLOOP_ROOT / "stage_a_shortlist.json",
        {
            "checked_at": base.now_iso(),
            "rows": [
                {
                    "iteration": item["iteration"],
                    "mutation_id": item["mutation"]["mutation_id"],
                    "source_subset": item["mutation"]["source_subset"],
                    "best_variant": item["best_compare"]["variant"],
                    "best_compare": item["best_compare"],
                }
                for item in shortlist
            ],
        },
    )

    smoke_pass_reports = [item for item in stage_a_reports if item["best_compare"]["smoke_stage_accept"]]
    stage_b_reports = []
    for short_item in shortlist:
        mutations = stage_b_generalization_mutations(
            short_item["mutation"]["source_subset"],
            prefix=short_item["mutation"]["mutation_id"],
            seed_config=short_item["mutation"]["proxy_config"],
        )
        for mutation in mutations:
            iter_idx += 1
            report = run_stage_eval(
                iter_idx=iter_idx,
                mutation=mutation,
                manifest_path=smoke_manifest_path,
                case_set="cases",
                case_ids=smoke_case_ids,
                gate_label="smoke",
            )
            ingest(report, decision_stage="stage_b_fragmentation_smoke")
            stage_b_reports.append(report)
            if report["best_compare"]["smoke_stage_accept"]:
                smoke_pass_reports.append(report)

    control_pass_reports = []
    for candidate in sorted(smoke_pass_reports, key=lambda item: ranking_key(item["best_compare"]))[:6]:
        iter_idx += 1
        mutation = candidate["mutation"]
        report = run_stage_eval(
            iter_idx=iter_idx,
            mutation=mutation,
            manifest_path=control_manifest_path,
            case_set="cases",
            case_ids=control_case_ids,
            gate_label="control6",
        )
        ingest(report, decision_stage="control_gate_6x1")
        if report["best_compare"]["control_accept"]:
            control_pass_reports.append(report)

    stage_c_reports = []
    for candidate in sorted(control_pass_reports, key=lambda item: ranking_key(item["best_compare"]))[:3]:
        mutations = stage_c_label_generalization_mutations(
            candidate["mutation"]["source_subset"],
            prefix=candidate["mutation"]["mutation_id"],
            seed_config=candidate["mutation"]["proxy_config"],
        )
        for mutation in mutations:
            iter_idx += 1
            report = run_stage_eval(
                iter_idx=iter_idx,
                mutation=mutation,
                manifest_path=smoke_manifest_path,
                case_set="cases",
                case_ids=smoke_case_ids,
                gate_label="smoke",
            )
            ingest(report, decision_stage="stage_c_label_generalization_smoke")
            stage_c_reports.append(report)
            if report["best_compare"]["smoke_stage_accept"]:
                iter_idx += 1
                control_report = run_stage_eval(
                    iter_idx=iter_idx,
                    mutation=mutation,
                    manifest_path=control_manifest_path,
                    case_set="cases",
                    case_ids=control_case_ids,
                    gate_label="control6",
                )
                ingest(control_report, decision_stage="control_gate_6x1")
                if control_report["best_compare"]["control_accept"]:
                    control_pass_reports.append(control_report)

    hero_pass_reports = []
    for candidate in sorted(control_pass_reports, key=lambda item: ranking_key(item["best_compare"]))[:4]:
        iter_idx += 1
        mutation = candidate["mutation"]
        report = run_stage_eval(
            iter_idx=iter_idx,
            mutation=mutation,
            manifest_path=base.BENCHMARK_MANIFEST,
            case_set="hero_cases",
            case_ids=hero_case_ids,
            gate_label="hero",
        )
        report["best_compare"] = _annotate(report["best_compare"])
        report["ranking"] = [report["best_compare"]]
        ingest(report, decision_stage="hero_gate_5x1")
        if report["best_compare"]["hero_accept"]:
            hero_pass_reports.append(report)

    local20_pass_report = None
    for candidate in sorted(hero_pass_reports, key=lambda item: ranking_key(item["best_compare"]))[:2]:
        iter_idx += 1
        mutation = candidate["mutation"]
        report = run_stage_eval(
            iter_idx=iter_idx,
            mutation=mutation,
            manifest_path=base.BENCHMARK_MANIFEST,
            case_set="benchmark_cases",
            case_ids=benchmark_case_ids,
            gate_label="local20",
        )
        report["best_compare"] = _annotate(report["best_compare"])
        report["ranking"] = [report["best_compare"]]
        ingest(report, decision_stage="local20_gate")
        if report["best_compare"]["local20_accept"]:
            local20_pass_report = report
            break

    if local20_pass_report is not None:
        pending_ticket_path = base.RESEARCH_ROOT / f"pending_cloud_ticket.{FAMILY}.json"
        cloud_ready_summary = AUTLOOP_ROOT / "cloud_ready_summary.md"
        base.write_json(
            pending_ticket_path,
            {
                "checked_at": base.now_iso(),
                "family": FAMILY,
                "shape": FIRST_SHAPE,
                "status": "local20_honest_pass",
                "cloud_gate_open": True,
                "best_variant": local20_pass_report["best_variant"],
                "best_mutation_id": local20_pass_report["mutation"]["mutation_id"],
                "local20_compare": local20_pass_report["best_compare"],
                "summary_json": local20_pass_report["summary_json"],
            },
        )
        base.write_text(
            cloud_ready_summary,
            "\n".join(
                [
                    f"# {FAMILY} cloud-ready summary",
                    "",
                    f"- shape: `{FIRST_SHAPE}`",
                    f"- best_variant: `{local20_pass_report['best_variant']}`",
                    f"- best_mutation: `{local20_pass_report['mutation']['mutation_id']}`",
                    "- local20 honest pass achieved; evaluator-only cloud validation may open exactly once.",
                ]
            ),
        )
        base.update_loop_state(
            loop_state_path,
            current_iteration=iter_idx,
            diagnosed_failure_mode="passed_local20_pending_cloud",
            chosen_stage="cloud_ready_pending",
            chosen_mutation=local20_pass_report["mutation"]["mutation_id"],
            local_gate_status="local20_pass",
            cloud_gate_open=True,
            artifact_paths=base.collect_artifact_paths(report_paths) + [base.rel(pending_ticket_path), base.rel(cloud_ready_summary)],
        )
        return 0

    best_compare = best_state["compare"] if best_state is not None else {
        "metric_truth_bug": False,
        "mean_human_erasure_penalty": 1.0,
        "mean_fg_visible_coverage_retained_ratio": 0.0,
        "mean_fg_visible_mass_retained_ratio": 0.0,
        "mean_largest_fg_visible_component_retained_ratio": 0.0,
        "mean_largest_fg_visible_component_ratio": 0.0,
        "mean_delta_fg_connected_components": 0.0,
        "mean_delta_fg_peak_count": 0.0,
        "mean_delta_masked_l1": 0.0,
        "mean_delta_masked_ssim": 0.0,
        "mean_delta_off_body_support_ratio": 0.0,
        "mean_delta_bg_bottom_support_ratio": 0.0,
    }
    failure_class = classify_failure(best_compare)
    final_postmortem = {
        "checked_at": base.now_iso(),
        "family": FAMILY,
        "shape": FIRST_SHAPE,
        "status": "dead_same_day",
        "gate_stage_reached": "autoloop_local_bounded",
        "summary_reason": stage_failure_summary(best_compare),
        "failure_class": failure_class,
        "best_local_state": best_state,
        "next_family": NEXT_FAMILY,
        "smoke_cases": smoke_case_ids,
        "control_cases": control_case_ids,
        "rescued_reference": previous["old_variant_rescue"],
        "stage_a_shortlist": base.load_json(AUTLOOP_ROOT / "stage_a_shortlist.json")["rows"],
    }
    base.write_json(AUTLOOP_ROOT / "autoloop_final_postmortem.json", final_postmortem)
    if best_state is not None:
        base.write_json(AUTLOOP_ROOT / "best_local_state.json", best_state)
    next_draft_path = base.RESEARCH_ROOT / f"next_manual_problem_draft.{NEXT_FAMILY}.json"
    base.write_json(
        next_draft_path,
        {
            "checked_at": base.now_iso(),
            "family": NEXT_FAMILY,
            "derived_from": FAMILY,
            "reason": final_postmortem["summary_reason"],
        },
    )
    result_path = base.RESEARCH_ROOT / f"{FAMILY}_result.json"
    postmortem_path = base.RESEARCH_ROOT / f"{FAMILY}_postmortem.json"
    base.write_json(
        result_path,
        {
            "checked_at": base.now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "autoloop_local_bounded",
            "summary_reason": final_postmortem["summary_reason"],
            "best_local_state": best_state,
        },
    )
    base.write_json(postmortem_path, final_postmortem)
    base.sync_failure_to_guard(final_postmortem, review_packet_rel=base.rel(AUTLOOP_ROOT / "autoloop_final_postmortem.json"))
    base.update_loop_state(
        loop_state_path,
        current_iteration=iter_idx,
        diagnosed_failure_mode=failure_class,
        chosen_stage="bounded_failure",
        chosen_mutation=best_state["mutation"]["mutation_id"] if best_state is not None else "none",
        local_gate_status="failed",
        cloud_gate_open=False,
        artifact_paths=base.collect_artifact_paths(report_paths) + [base.rel(result_path), base.rel(postmortem_path), base.rel(next_draft_path)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
