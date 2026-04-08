from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_teacher_geometry_rehydrated_render_autoloop as base  # noqa: E402
from scripts.casewise_source_poison_mutation_bank import (  # noqa: E402
    stage_a_source_poison_mutations,
    stage_b_rehydrated_mutations,
    stage_c_label_poison_mutations,
)
from scripts.score_correspondence_progress import compare_variant  # noqa: E402


FAMILY = "teacher_geometry_casewise_source_poison_audit"
FIRST_SHAPE = "stablelead_casewise_source_poison_consensus_margin_maskedhuman_v1"
PRIOR_FAMILY = "teacher_geometry_casewise_fragmentation_stability_audit"
OLD_AUTOLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_casewise_fragmentation_stability"
AUTLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_casewise_source_poison"
RUNS_ROOT = AUTLOOP_ROOT / "runs"
PANELS_ROOT = AUTLOOP_ROOT / "panels"
CONFIG_ROOT = AUTLOOP_ROOT / "configs"
NEXT_FAMILY_BY_FAILURE = {
    "metric_truth_bug": "teacher_geometry_metric_truth_reaudit",
    "coverage_regression": "teacher_geometry_visible_coverage_floor_audit",
    "fragmentation_rebound": "teacher_geometry_casewise_source_swap_audit",
    "quality_regression": "teacher_geometry_casewise_source_quality_tradeoff_audit",
    "external_residue_rebound": "teacher_geometry_casewise_external_residue_audit",
    "no_movement": "teacher_geometry_casewise_source_swap_audit",
}
PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
]
FIXED_SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"


def load_previous_artifacts() -> dict:
    stage_a_shortlist_path = OLD_AUTOLOOP_ROOT / "stage_a_shortlist.json"
    return {
        "research_status": base.load_json(base.RESEARCH_STATUS_JSON),
        "task_plan": base.load_json(base.TASK_PLAN_JSON),
        "watch": base.load_json(base.WATCH_JSON),
        "allowlist": base.load_json(base.ALLOWLIST_JSON),
        "old_result": base.load_json(base.RESEARCH_ROOT / f"{PRIOR_FAMILY}_result.json"),
        "old_postmortem": base.load_json(base.RESEARCH_ROOT / f"{PRIOR_FAMILY}_postmortem.json"),
        "old_best_state": base.load_json(OLD_AUTOLOOP_ROOT / "best_state.json"),
        "old_best_local_state": base.load_json(OLD_AUTOLOOP_ROOT / "best_local_state.json")
        if (OLD_AUTOLOOP_ROOT / "best_local_state.json").exists()
        else {},
        "old_stage_a_shortlist": base.load_json(stage_a_shortlist_path) if stage_a_shortlist_path.exists() else {"rows": []},
        "old_iteration_ledger_lines": [
            json.loads(line)
            for line in (OLD_AUTOLOOP_ROOT / "iteration_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    }


def write_truth_repair_report(previous: dict) -> Path:
    report_path = base.RESEARCH_ROOT / "casewise_source_poison_truth_repair_report.json"
    base.write_json(
        report_path,
        {
            "checked_at": base.now_iso(),
            "prior_failure_family": PRIOR_FAMILY,
            "current_research_state": previous["research_status"]["state"],
            "allowlist_status": previous["allowlist"]["status"],
            "current_family_must_be": FAMILY,
            "repair_statements": [
                "casewise-fragmentation already proved that subset pruning plus soft rehydration can reduce fragmentation on some anchors without opening cloud.",
                "that family still failed because B14-style smoke anchors rebounded on fragmentation even when visible coverage honesty was preserved.",
                "current family must restart from a fresh null baseline; the prior best-state may seed source subsets but may not define the new family ranking baseline.",
            ],
            "repaired_truth": "The next honest move is a casewise source-poison autoloop: identify per-case toxic source views that keep fragmentation alive even after global subset pruning.",
        },
    )
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
            "notes": "Only evaluator/proxy/render mutations are allowed this round. Training, dataset, trainer, and cloud training remain frozen.",
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
                "scripts/casewise_source_poison_mutation_bank.py",
                "scripts/run_teacher_geometry_casewise_source_poison_autoloop.py",
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
            "reason": "Current honest next move is casewise poison-source auditing because casewise fragmentation improvements did not generalize across both smoke anchors.",
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


def smoke_row_accept(row: dict) -> bool:
    return (
        (not row["metric_truth_bug"])
        and row["delta_fg_connected_components"] <= -5
        and row["delta_fg_peak_count"] <= -1
        and row["delta_masked_l1"] <= 0.0
        and row["delta_masked_ssim"] >= 0.0
        and row["delta_off_body_support_ratio"] <= 0.0
        and row["delta_bg_bottom_support_ratio"] <= 0.0
        and row["fg_visible_coverage_retained_ratio"] >= 0.98
        and row["fg_visible_mass_retained_ratio"] >= 0.95
        and row["largest_fg_visible_component_retained_ratio"] >= 0.95
        and row["largest_fg_visible_component_ratio"] >= 0.55
        and row["human_erasure_penalty"] <= 0.05
        and row["delta_source_id_switch_count_inside_fg"] <= 0.0
        and row["delta_source_top1_spatial_fragmentation"] <= 0.0
    )


def _annotate(compare: dict) -> dict:
    compare = dict(compare)
    honest_primary_count = int(sum(1 for row in compare["rows"] if smoke_row_accept(row)))
    compare["honest_primary_count"] = honest_primary_count
    compare["smoke_stage_accept"] = (
        honest_primary_count == compare["case_count"]
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.98
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.95
        and compare["mean_human_erasure_penalty"] <= 0.05
        and compare["mean_delta_source_id_switch_count_inside_fg"] <= 0.0
        and compare["mean_delta_source_top1_spatial_fragmentation"] <= 0.0
    )
    compare["control_accept"] = (
        honest_primary_count >= max(compare["case_count"] - 1, 1)
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.98
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.95
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.95
        and compare["mean_human_erasure_penalty"] <= 0.04
        and compare["mean_delta_source_id_switch_count_inside_fg"] <= 0.0
        and compare["mean_delta_source_top1_spatial_fragmentation"] <= 0.0
    )
    compare["hero_accept"] = (
        honest_primary_count >= 3
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.96
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.96
        and compare["mean_human_erasure_penalty"] <= 0.04
        and compare["mean_delta_source_id_switch_count_inside_fg"] <= 0.0
        and compare["mean_delta_source_top1_spatial_fragmentation"] <= 0.0
    )
    compare["local20_accept"] = (
        honest_primary_count >= 14
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.97
        and compare["mean_human_erasure_penalty"] <= 0.03
        and compare["mean_delta_source_id_switch_count_inside_fg"] <= 0.0
        and compare["mean_delta_source_top1_spatial_fragmentation"] <= 0.0
    )
    compare["failure_class"] = classify_failure(compare)
    return compare


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
    if (
        compare["mean_delta_fg_connected_components"] >= 0.0
        or compare["mean_delta_fg_peak_count"] >= 0.0
        or compare["mean_delta_source_id_switch_count_inside_fg"] > 0.0
        or compare["mean_delta_source_top1_spatial_fragmentation"] > 0.0
    ):
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
        compare["mean_delta_source_id_switch_count_inside_fg"],
        compare["mean_delta_source_top1_spatial_fragmentation"],
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
        return "casewise source-poison line still found a metric truth bug; the evaluator is not yet honest enough to rank poison-pruning progress."
    if failure_class == "coverage_regression":
        return "casewise source pruning improved some agreement metrics only by giving back too much visible human coverage or visible mass."
    if failure_class == "quality_regression":
        return "casewise poison pruning reduced fragmentation but regressed masked reconstruction quality, so the apparent progress is not honest."
    if failure_class == "external_residue_rebound":
        return "casewise poison pruning reduced fragmentation locally, but off-body or bottom-band residue rebounded on the broader set."
    if failure_class == "fragmentation_rebound":
        return "casewise poison pruning helped some anchors, but fragmentation still rebounded across the paired smoke/control cases."
    return "bounded casewise-source-poison mutations produced no honest movement on the smoke-facing stability bottleneck."


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
            REPO_ROOT / "scripts" / "casewise_source_poison_mutation_bank.py",
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
    control_rows = [
        smoke_rows[0],
        smoke_rows[1],
        *[
            row
            for row in base.select_manifest_rows(manifest, "hero_cases")
            if base.case_id(row) not in set(smoke_case_ids)
        ][:4],
    ]
    control_case_ids = [base.case_id(row) for row in control_rows]
    hero_rows = base.select_manifest_rows(manifest, "hero_cases")
    hero_case_ids = [base.case_id(row) for row in hero_rows]
    benchmark_rows = base.select_manifest_rows(manifest, "benchmark_cases")
    benchmark_case_ids = [base.case_id(row) for row in benchmark_rows]
    source_count = len(smoke_rows[0]["source_cameras"])

    smoke_manifest_path = AUTLOOP_ROOT / "smoke_cases_manifest.json"
    control_manifest_path = AUTLOOP_ROOT / "control_cases_manifest.json"
    base.write_manifest(smoke_rows, smoke_manifest_path, key="cases")
    base.write_manifest(control_rows, control_manifest_path, key="cases")

    prior_best_subset = previous["old_best_state"].get("mutation", {}).get("source_subset", [])
    stage_a_mutations = stage_a_source_poison_mutations(source_count=source_count, prior_best_subset=prior_best_subset)
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
        ingest(report, decision_stage="stage_a_casewise_poison_smoke")
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

    for short_item in shortlist:
        mutations = stage_b_rehydrated_mutations(
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
            ingest(report, decision_stage="stage_b_casewise_poison_rehydrated")

    stage_c_candidates = sorted(
        [item for item in all_reports if item["best_compare"]["smoke_stage_accept"]],
        key=lambda item: ranking_key(item["best_compare"]),
    )[:3]
    for base_item in stage_c_candidates:
        mutations = stage_c_label_poison_mutations(
            base_item["mutation"]["source_subset"],
            prefix=base_item["mutation"]["mutation_id"],
            seed_config=base_item["mutation"]["proxy_config"],
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
            ingest(report, decision_stage="stage_c_casewise_poison_label")

    smoke_pass_candidates = sorted(
        [item for item in all_reports if item["best_compare"]["smoke_stage_accept"]],
        key=lambda item: ranking_key(item["best_compare"]),
    )

    control_pass_candidates = []
    for item in smoke_pass_candidates[:3]:
        iter_idx += 1
        mutation = item["mutation"]
        config_path = CONFIG_ROOT / f"proxy_config.iter{iter_idx:02d}.json"
        base.write_json(config_path, mutation["proxy_config"])
        summary = run_eval(control_manifest_path, "cases", RUNS_ROOT / f"iter{iter_idx:02d}" / "control_eval", config_path)
        ranking = compare_stage(summary, control_case_ids)
        best_compare = ranking[0]
        panels = base.build_panels(summary, best_compare["variant"], f"iter{iter_idx:02d}", control_case_ids[0])
        report = {
            "checked_at": base.now_iso(),
            "iteration": iter_idx,
            "stage": "control_gate_6x1",
            "mutation": mutation,
            "gate_label": "control_6x1",
            "case_ids": control_case_ids,
            "best_variant": best_compare["variant"],
            "best_compare": best_compare,
            "ranking": ranking,
            "panels": panels,
            "summary_json": base.rel(Path(summary["_root"]) / "summary.json"),
            "summary_md": base.rel(Path(summary["_root"]) / "summary.md"),
        }
        ingest(report, decision_stage="control_gate_6x1")
        if best_compare["control_accept"]:
            control_pass_candidates.append(report)

    hero_pass_candidate = None
    for item in control_pass_candidates[:3]:
        iter_idx += 1
        mutation = item["mutation"]
        config_path = CONFIG_ROOT / f"proxy_config.iter{iter_idx:02d}.json"
        base.write_json(config_path, mutation["proxy_config"])
        summary = run_eval(base.BENCHMARK_MANIFEST, "hero_cases", RUNS_ROOT / f"iter{iter_idx:02d}" / "hero_eval", config_path)
        ranking = compare_stage(summary, hero_case_ids)
        best_compare = ranking[0]
        panels = base.build_panels(summary, best_compare["variant"], f"iter{iter_idx:02d}", hero_case_ids[0])
        report = {
            "checked_at": base.now_iso(),
            "iteration": iter_idx,
            "stage": "hero_gate_5x1",
            "mutation": mutation,
            "gate_label": "hero_5x1",
            "case_ids": hero_case_ids,
            "best_variant": best_compare["variant"],
            "best_compare": best_compare,
            "ranking": ranking,
            "panels": panels,
            "summary_json": base.rel(Path(summary["_root"]) / "summary.json"),
            "summary_md": base.rel(Path(summary["_root"]) / "summary.md"),
        }
        ingest(report, decision_stage="hero_gate_5x1")
        if best_compare["hero_accept"]:
            hero_pass_candidate = report
            break

    local20_pass_candidate = None
    if hero_pass_candidate is not None:
        iter_idx += 1
        mutation = hero_pass_candidate["mutation"]
        config_path = CONFIG_ROOT / f"proxy_config.iter{iter_idx:02d}.json"
        base.write_json(config_path, mutation["proxy_config"])
        summary = run_eval(base.BENCHMARK_MANIFEST, "benchmark_cases", RUNS_ROOT / f"iter{iter_idx:02d}" / "local20_eval", config_path)
        ranking = compare_stage(summary, benchmark_case_ids)
        best_compare = ranking[0]
        panels = base.build_panels(summary, best_compare["variant"], f"iter{iter_idx:02d}", benchmark_case_ids[0])
        report = {
            "checked_at": base.now_iso(),
            "iteration": iter_idx,
            "stage": "local20",
            "mutation": mutation,
            "gate_label": "local20",
            "case_ids": benchmark_case_ids,
            "best_variant": best_compare["variant"],
            "best_compare": best_compare,
            "ranking": ranking,
            "panels": panels,
            "summary_json": base.rel(Path(summary["_root"]) / "summary.json"),
            "summary_md": base.rel(Path(summary["_root"]) / "summary.md"),
        }
        ingest(report, decision_stage="local20")
        if best_compare["local20_accept"]:
            local20_pass_candidate = {
                "mutation": mutation,
                "variant": best_compare["variant"],
                "compare": best_compare,
                "panels": panels,
                "summary_json": report["summary_json"],
            }

    if local20_pass_candidate is not None:
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
                "best_variant": local20_pass_candidate["variant"],
                "best_mutation_id": local20_pass_candidate["mutation"]["mutation_id"],
                "local20_compare": local20_pass_candidate["compare"],
                "summary_json": local20_pass_candidate["summary_json"],
            },
        )
        base.write_text(
            cloud_ready_summary,
            "\n".join(
                [
                    f"# {FAMILY} cloud-ready summary",
                    "",
                    f"- shape: `{FIRST_SHAPE}`",
                    f"- best_variant: `{local20_pass_candidate['variant']}`",
                    f"- best_mutation: `{local20_pass_candidate['mutation']['mutation_id']}`",
                    "- local20 honest pass achieved; cloud may open exactly once with an evaluator-only benchmark.",
                ]
            ),
        )
        base.update_loop_state(
            loop_state_path,
            current_iteration=iter_idx,
            diagnosed_failure_mode="passed_local20_pending_cloud",
            chosen_stage="cloud_ready_pending",
            chosen_mutation=local20_pass_candidate["mutation"]["mutation_id"],
            local_gate_status="local20_pass",
            cloud_gate_open=True,
            artifact_paths=base.collect_artifact_paths(report_paths) + [base.rel(pending_ticket_path), base.rel(cloud_ready_summary)],
        )
        return 0

    best_compare = best_state["compare"] if best_state is not None else {
        "failure_class": "no_movement",
        "mean_delta_fg_connected_components": 0.0,
        "mean_delta_fg_peak_count": 0.0,
        "mean_delta_masked_l1": 0.0,
        "mean_delta_masked_ssim": 0.0,
        "mean_delta_off_body_support_ratio": 0.0,
        "mean_delta_bg_bottom_support_ratio": 0.0,
        "mean_fg_visible_coverage_retained_ratio": 0.0,
        "mean_fg_visible_mass_retained_ratio": 0.0,
        "mean_largest_fg_visible_component_retained_ratio": 0.0,
        "mean_human_erasure_penalty": 1.0,
        "metric_truth_bug": False,
        "variant": "none",
    }
    failure_class = classify_failure(best_compare)
    derived_next = NEXT_FAMILY_BY_FAILURE.get(failure_class, "teacher_geometry_casewise_source_swap_audit")
    final_postmortem = {
        "checked_at": base.now_iso(),
        "family": FAMILY,
        "shape": FIRST_SHAPE,
        "status": "dead_same_day",
        "gate_stage_reached": "autoloop_local_bounded",
        "summary_reason": stage_failure_summary(best_compare),
        "failure_class": failure_class,
        "best_local_state": best_state,
        "next_family": derived_next,
        "smoke_cases": smoke_case_ids,
        "control_cases": control_case_ids,
        "stage_a_shortlist": base.load_json(AUTLOOP_ROOT / "stage_a_shortlist.json")["rows"],
    }
    base.write_json(AUTLOOP_ROOT / "autoloop_final_postmortem.json", final_postmortem)
    if best_state is not None:
        base.write_json(AUTLOOP_ROOT / "best_local_state.json", best_state)
    next_draft_path = base.RESEARCH_ROOT / f"next_manual_problem_draft.{derived_next}.json"
    base.write_json(
        next_draft_path,
        {
            "checked_at": base.now_iso(),
            "family": derived_next,
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
