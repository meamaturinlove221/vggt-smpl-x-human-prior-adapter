from __future__ import annotations

import json
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rehydrated_render_mutation_bank import (  # noqa: E402
    stage_a_source_subset_mutations,
    stage_b_rehydrated_mutations,
    stage_c_label_mutations,
)
from scripts.score_correspondence_progress import (  # noqa: E402
    classify_failure,
    compare_variant,
    hero_accept,
    local20_accept,
    progress_key,
)


RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
WATCH_JSON = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
TASK_PLAN_JSON = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
TASK_PLAN_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.md"
SUMMARY_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "summary.md"
RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
FRONTIER_LEDGER_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "family_stop_reason.json"

AUTLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_rehydrated_render"
RUNS_ROOT = AUTLOOP_ROOT / "runs"
PANELS_ROOT = AUTLOOP_ROOT / "panels"
CONFIG_ROOT = AUTLOOP_ROOT / "configs"

FAMILY = "teacher_geometry_rehydrated_render_audit"
FIRST_SHAPE = "stablelead_rehydrated_render_visiblefloor_maskedhuman_v1"
PRIOR_FAMILY = "teacher_geometry_source_selection_audit"
OLD_AUTOLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_source_selection"
OLD_PANELS_ROOT = OLD_AUTOLOOP_ROOT / "panels"

EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"
SCORER_SCRIPT = REPO_ROOT / "scripts" / "score_correspondence_progress.py"
BENCHMARK_MANIFEST = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "benchmark_manifest.20260403.json"
TEACHER_CHECKPOINT = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "teacher_checkpoint" / "checkpoint.pt"
PYTHON_EXE = REPO_ROOT / ".venv5080" / "Scripts" / "python.exe"
PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
]
FIXED_SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"

NEXT_DRAFT_BY_FAILURE = {
    "metric_truth_bug": "teacher_geometry_visible_coverage_truth_audit",
    "erasure_win": "teacher_geometry_visible_coverage_floor_audit",
    "fragmentation_win": "teacher_geometry_label_consistency_audit",
    "background_only_win": "teacher_geometry_source_agreement_audit",
    "no_movement": "teacher_geometry_proxy_render_tradeoff_audit",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_checked(args: list[str]) -> str:
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def py_compile(paths: list[Path]) -> None:
    python_exe = str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable)
    for path in paths:
        run_checked([python_exe, "-m", "py_compile", str(path)])


def case_id(row: dict) -> str:
    return "{seq}_frame_{frame:06d}_{target}".format(
        seq=str(row["seq_name"]),
        frame=int(row["frame_id"]),
        target=str(row["target_camera"]),
    )


def find_case(manifest: dict, wanted: str) -> dict:
    for bucket in ("hero_cases", "benchmark_cases", "cases"):
        for row in manifest.get(bucket, []):
            if case_id(row) == wanted:
                return row
    raise KeyError(wanted)


def select_manifest_rows(manifest: dict, key: str) -> list[dict]:
    return list(manifest.get(key, []))


def choose_second_smoke_anchor(manifest: dict, fixed_case_id: str) -> dict:
    fixed = find_case(manifest, fixed_case_id)
    hero_rows = select_manifest_rows(manifest, "hero_cases")
    for row in hero_rows:
        row_id = case_id(row)
        if row_id != fixed_case_id and str(row["target_camera"]) != str(fixed["target_camera"]):
            return row
    for row in hero_rows:
        row_id = case_id(row)
        if row_id != fixed_case_id:
            return row
    raise RuntimeError("Could not auto-select a second smoke anchor.")


def write_manifest(rows: list[dict], out_path: Path, *, key: str = "cases") -> None:
    write_json(out_path, {"checked_at": now_iso(), key: rows})


def copy_panel_grid(label: str, items: list[tuple[Path, str]], out_path: Path, width: int = 340) -> None:
    ensure_dir(out_path.parent)
    font = ImageFont.load_default()
    loaded = []
    for path, title in items:
        image = Image.open(path).convert("RGB")
        image = image.resize((width, int(image.height * width / max(image.width, 1))), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, image.height + 28), (18, 18, 18))
        canvas.paste(image, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), title, fill=(240, 240, 240), font=font)
        loaded.append(canvas)
    total_w = sum(image.width for image in loaded)
    total_h = max(image.height for image in loaded) + 24
    panel = Image.new("RGB", (total_w, total_h), (12, 12, 12))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), label, fill=(255, 220, 120), font=font)
    x = 0
    for image in loaded:
        panel.paste(image, (x, 24))
        x += image.width
    panel.save(out_path)


def build_panels(summary: dict, variant: str, iter_tag: str, case_id_text: str) -> dict:
    baseline_rows = [row for row in summary["rows"] if row["variant"] == "baseline_depth_unproject" and row["case_id"] == case_id_text]
    candidate_rows = [row for row in summary["rows"] if row["variant"] == variant and row["case_id"] == case_id_text]
    if not baseline_rows or not candidate_rows:
        return {}
    base = baseline_rows[0]
    cand = candidate_rows[0]
    root = Path(summary["_root"])
    panel_dir = ensure_dir(PANELS_ROOT / iter_tag)
    outputs = {
        "target_baseline_geometry_candidate": panel_dir / f"target_baseline_geometry_candidate.{iter_tag}.png",
        "target_baseline_render_geometry_candidate": panel_dir / f"target_baseline_render_geometry_candidate.{iter_tag}.png",
        "candidate_support_triptych": panel_dir / f"candidate_support_triptych.{iter_tag}.png",
        "candidate_support_inside_outside": panel_dir / f"candidate_support_inside_outside.{iter_tag}.png",
        "candidate_source_dominance_triptych": panel_dir / f"candidate_source_dominance_triptych.{iter_tag}.png",
        "candidate_bg_nonblack_heatmap": panel_dir / f"candidate_bg_nonblack_heatmap.{iter_tag}.png",
        "baseline_vs_candidate_bottom_band_support": panel_dir / f"baseline_vs_candidate_bottom_band_support.{iter_tag}.png",
        "candidate_fg_coverage_panel": panel_dir / f"candidate_fg_coverage_panel.{iter_tag}.png",
        "candidate_source_label_map": panel_dir / f"candidate_source_label_map.{iter_tag}.png",
    }
    copy_panel_grid(
        f"{FAMILY} | {variant} | {case_id_text}",
        [
            (root / base["files"]["target_png"], "Target"),
            (root / base["files"]["depth_unproject_png"], "Frozen baseline geometry"),
            (root / cand["files"]["variant_png"], "Candidate"),
        ],
        outputs["target_baseline_geometry_candidate"],
    )
    copy_panel_grid(
        "Target / frozen baseline / candidate / fg coverage",
        [
            (root / base["files"]["target_png"], "Target"),
            (root / base["files"]["variant_png"], "Frozen baseline"),
            (root / cand["files"]["variant_png"], "Candidate"),
            (root / cand["files"]["fg_coverage_overlay_png"], "FG coverage"),
        ],
        outputs["target_baseline_render_geometry_candidate"],
        width=300,
    )
    copy_panel_grid(
        "Support triptych",
        [
            (root / cand["files"]["alpha_map_png"], "Alpha map"),
            (root / cand["files"]["support_inside_fg_png"], "Support inside fg"),
            (root / cand["files"]["support_overlay_on_fg_png"], "Support overlay on fg"),
        ],
        outputs["candidate_support_triptych"],
    )
    copy_panel_grid(
        "Support inside / outside / coverage",
        [
            (root / cand["files"]["fg_mask_png"], "FG mask"),
            (root / cand["files"]["support_inside_fg_png"], "Inside fg"),
            (root / cand["files"]["support_outside_fg_png"], "Outside fg"),
            (root / cand["files"]["fg_coverage_overlay_png"], "Coverage"),
        ],
        outputs["candidate_support_inside_outside"],
        width=280,
    )
    copy_panel_grid(
        "Source dominance",
        [
            (root / cand["files"]["source_top1_mass_png"], "Top1 mass"),
            (root / cand["files"]["source_top1_top2_margin_png"], "Top1-top2 margin"),
            (root / cand["files"]["source_label_smoothness_png"], "Label smoothness"),
        ],
        outputs["candidate_source_dominance_triptych"],
    )
    copy_panel_grid(
        "Background nonblack heatmap",
        [(root / cand["files"]["bg_nonblack_heatmap_png"], "BG nonblack heatmap")],
        outputs["candidate_bg_nonblack_heatmap"],
        width=420,
    )
    copy_panel_grid(
        "Bottom-band support: baseline vs candidate",
        [
            (root / base["files"]["bg_bottom_support_png"], "Baseline bottom band"),
            (root / cand["files"]["bg_bottom_support_png"], "Candidate bottom band"),
        ],
        outputs["baseline_vs_candidate_bottom_band_support"],
        width=420,
    )
    copy_panel_grid(
        "FG coverage panel",
        [
            (root / base["files"]["fg_coverage_overlay_png"], "Baseline coverage"),
            (root / cand["files"]["fg_coverage_overlay_png"], "Candidate coverage"),
        ],
        outputs["candidate_fg_coverage_panel"],
        width=420,
    )
    copy_panel_grid(
        "Source label map",
        [(root / cand["files"]["source_label_map_png"], "Source label map")],
        outputs["candidate_source_label_map"],
        width=420,
    )
    return {key: rel(path) for key, path in outputs.items()}


def run_eval(manifest_path: Path, case_set: str, output_dir: Path, proxy_config_path: Path) -> dict:
    ensure_dir(output_dir)
    run_checked(
        [
            str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable),
            str(EVAL_SCRIPT),
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(TEACHER_CHECKPOINT),
            "--case-set",
            case_set,
            "--variants",
            ",".join(PROXY_VARIANTS),
            "--support-threshold",
            "0.25",
            "--bottom-band-ratio",
            "0.20",
            "--proxy-config-json",
            str(proxy_config_path),
        ]
    )
    payload = load_json(output_dir / "summary.json")
    payload["_root"] = output_dir
    return payload


def smoke_stage_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["mean_delta_fg_connected_components"] <= -20.0
        and compare["mean_delta_fg_peak_count"] <= -2.0
        and compare["mean_delta_masked_l1"] <= -0.005
        and compare["mean_delta_masked_ssim"] >= 0.002
        and compare["mean_fg_visible_rgb_coverage_ratio"] >= 0.72
        and compare["mean_fg_retained_mass_ratio"] >= 0.85
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_human_erasure_penalty"] <= 0.08
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
    )


def stage_c_eligible(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and compare["mean_fg_visible_rgb_coverage_ratio"] >= 0.72
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_fg_retained_mass_ratio"] >= 0.85
    )


def compare_stage(summary: dict, case_ids: list[str]) -> list[dict]:
    rows = []
    for variant in PROXY_VARIANTS:
        compare = compare_variant(summary, variant, case_ids)
        compare["failure_class"] = classify_failure(compare)
        compare["hero_accept"] = hero_accept(compare)
        compare["local20_accept"] = local20_accept(compare)
        compare["smoke_stage_accept"] = smoke_stage_accept(compare)
        rows.append(compare)
    rows.sort(key=progress_key)
    return rows


def load_previous_artifacts() -> dict:
    old_config_paths = sorted((OLD_AUTOLOOP_ROOT / "configs").glob("proxy_config.iter*.json"))
    old_panel_paths = sorted(path for path in OLD_PANELS_ROOT.rglob("*.png"))
    return {
        "research_status": load_json(RESEARCH_STATUS_JSON),
        "task_plan": load_json(TASK_PLAN_JSON),
        "watch": load_json(WATCH_JSON),
        "allowlist": load_json(ALLOWLIST_JSON),
        "next_draft": load_json(RESEARCH_ROOT / "next_manual_problem_draft.teacher_geometry_rehydrated_render_audit.json"),
        "old_postmortem": load_json(OLD_AUTOLOOP_ROOT / "autoloop_final_postmortem.json"),
        "old_best_local_state": load_json(OLD_AUTOLOOP_ROOT / "best_local_state.json"),
        "old_source_subset_ranking": load_json(OLD_AUTOLOOP_ROOT / "source_subset_ranking.json"),
        "old_mutation_history": load_json(OLD_AUTOLOOP_ROOT / "mutation_history.json"),
        "old_iteration_ledger_lines": [json.loads(line) for line in (OLD_AUTOLOOP_ROOT / "iteration_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()],
        "old_config_paths": [rel(path) for path in old_config_paths],
        "old_panel_paths": [rel(path) for path in old_panel_paths],
    }


def write_truth_repair_report(previous: dict) -> Path:
    report_path = RESEARCH_ROOT / "rehydrated_render_truth_repair_report.json"
    report = {
        "checked_at": now_iso(),
        "prior_failure_family": PRIOR_FAMILY,
        "current_research_state": previous["research_status"]["state"],
        "allowlist_status": previous["allowlist"]["status"],
        "current_family_must_be": FAMILY,
        "repair_statements": [
            "source-selection line had real local progress but best-state ranking did not catch it honestly.",
            "fg_rehydrated_coverage_ratio was polluted by alpha floor / rehydrated visibility and cannot remain the main honest gate.",
            "current family must be teacher_geometry_rehydrated_render_audit.",
        ],
        "prior_best_local_state": previous["old_best_local_state"],
        "source_subset_progress_reference": previous["old_source_subset_ranking"]["rows"][:5],
        "artifact_counts": {
            "old_iteration_count": len(previous["old_iteration_ledger_lines"]),
            "old_config_count": len(previous["old_config_paths"]),
            "old_panel_count": len(previous["old_panel_paths"]),
        },
        "repaired_truth": "source-selection line had true local progress, but its best-state ordering missed that progress and its fg_rehydrated_coverage_ratio gate was polluted by alpha-floor-driven visibility rather than honest retained human coverage.",
    }
    write_json(report_path, report)
    return report_path


def write_packaging_files(report_path: Path) -> None:
    write_json(
        RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "autoloop_local_only",
            "ready_for_execution": False,
            "report_path": rel(report_path),
        },
    )
    write_json(
        RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "execution_mode": "evaluator_only_proxy_only_render_only",
            "cloud_must_remain_off": True,
            "same_family_retry_forbidden": True,
            "notes": "Only evaluator/proxy/render mutations are allowed this round; training, dataset, trainer, and cloud training entrypoints remain frozen.",
        },
    )
    write_json(
        RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "write_surface": [
                "scripts/evaluate_teacher_visual_lift_cases.py",
                "scripts/score_correspondence_progress.py",
                "scripts/rehydrated_render_mutation_bank.py",
                "scripts/run_teacher_geometry_rehydrated_render_autoloop.py",
            ],
        },
    )
    write_json(
        RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "autoloop_started",
            "reason": "Current honest next move is evaluator-only rehydrated-render auditing because source selection showed real progress but coverage honesty was still polluted.",
        },
    )
    write_json(
        RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "status": "local_only_autoloop",
            "ready_for_execution": False,
            "cloud_gate_open": False,
            "training_code_frozen": True,
        },
    )
    write_json(
        RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.json",
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "decision": "HOLD_FOR_LOCAL_AUTOLOOP",
            "ready_for_execution": False,
            "cloud_gate_open": False,
        },
    )


def update_loop_state(
    path: Path,
    *,
    current_iteration: int,
    diagnosed_failure_mode: str,
    chosen_stage: str,
    chosen_mutation: str,
    local_gate_status: str,
    cloud_gate_open: bool,
    artifact_paths: list[str],
) -> None:
    write_json(
        path,
        {
            "checked_at": now_iso(),
            "current_iteration": current_iteration,
            "prior_failure_family": PRIOR_FAMILY,
            "diagnosed_failure_mode": diagnosed_failure_mode,
            "chosen_stage": chosen_stage,
            "chosen_mutation": chosen_mutation,
            "chosen_candidate": FIRST_SHAPE,
            "local_gate_status": local_gate_status,
            "cloud_gate_open": cloud_gate_open,
            "cleanup_ok": True,
            "artifact_paths": artifact_paths,
        },
    )


def stage_failure_summary(compare: dict) -> str:
    failure_class = classify_failure(compare)
    if failure_class == "metric_truth_bug":
        return "metric truth bug remained after evaluator honesty repair; visible coverage / retained ratios are still inconsistent."
    if failure_class == "erasure_win":
        return "best local progress still came from erasure-biased rehydrated visibility rather than preserving honest human coverage."
    if failure_class == "fragmentation_win":
        return "rehydrated render preserved the body better, but fg fragmentation and multi-peak support still dominated."
    if failure_class == "background_only_win":
        return "background/support cleanup improved, but the body itself did not consolidate honestly into a better human rendering."
    return "bounded local rehydrated-render mutations produced no honest movement on the primary human-only objectives."


def sync_failure_to_guard(final_postmortem: dict, *, review_packet_rel: str) -> None:
    derived_next = final_postmortem["next_family"]
    research = load_json(RESEARCH_STATUS_JSON)
    research.update(
        {
            "checked_at": now_iso(),
            "state": "IDLE_GUARD",
            "reason": final_postmortem["summary_reason"],
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [],
            "current_priority_family": "",
            "current_priority_reason": final_postmortem["summary_reason"],
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": f"{FAMILY} exhausted its bounded local autoloop budget this round.",
            "next_requirement": f"Return to IDLE_GUARD. The next honest manual direction is {derived_next}.",
            "ready_for_execution": False,
            "manual_action_required": False,
            "manual_action_kind": "",
            "cloud_must_remain_off": True,
            "latest_formal_verdict": {
                "checked_at": now_iso(),
                "status": "dead_same_day",
                "family": FAMILY,
                "first_candidate_shape": FIRST_SHAPE,
                "gate_stage_reached": "autoloop_local_bounded",
                "reason": final_postmortem["summary_reason"],
            },
        }
    )
    write_json(RESEARCH_STATUS_JSON, research)

    task_plan = load_json(TASK_PLAN_JSON)
    task_plan.update(
        {
            "checked_at": now_iso(),
            "task_mode_status": "hard_blocker",
            "task_mode_focus": f"{FAMILY}_dead_same_day",
            "research_loop_mode": "IDLE_GUARD",
        }
    )
    task_plan["current_state_notes"] = [
        final_postmortem["summary_reason"],
        f"next_manual_problem_draft: {derived_next}",
    ]
    task_plan["summary_conclusion"] = [
        final_postmortem["summary_reason"],
        f"best_local_state: {rel(AUTLOOP_ROOT / 'best_state.json')}",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2))
    write_text(
        SUMMARY_MD,
        "\n".join(
            [
                "# ZJU Source-Policy Rawpool Status",
                "",
                *[f"- {line}" for line in task_plan["summary_conclusion"]],
            ]
        ),
    )

    watch = load_json(WATCH_JSON)
    watch["checked_at"] = now_iso()
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["watch_conclusion"] = final_postmortem["summary_reason"]
    watch["research"]["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "manual_action_required": False,
        "manual_action_kind": "",
        "ready_for_execution": False,
        "current_review_packet": review_packet_rel.replace("/", "\\"),
    }
    watch["research"]["research_status"] = research
    write_json(WATCH_JSON, watch)

    write_json(
        ALLOWLIST_JSON,
        {
            "checked_at": now_iso(),
            "status": "idle_empty_allowlist",
            "guard_track_must_continue": True,
            "notes": "No active approved research candidate is running.",
            "allowed_markers": [],
        },
    )
    if FRONTIER_LEDGER_JSON.exists():
        frontier = load_json(FRONTIER_LEDGER_JSON)
        frontier["latest_formal_verdict"] = research["latest_formal_verdict"]
        write_json(FRONTIER_LEDGER_JSON, frontier)
    if FAMILY_STOP_REASON_JSON.exists():
        family_stop = load_json(FAMILY_STOP_REASON_JSON)
        family_stop.setdefault("latest_family_outcomes", {})[FAMILY] = research["latest_formal_verdict"]
        write_json(FAMILY_STOP_REASON_JSON, family_stop)


def run_stage_eval(
    *,
    iter_idx: int,
    mutation: dict,
    manifest_path: Path,
    case_set: str,
    case_ids: list[str],
    gate_label: str,
) -> tuple[dict, list[dict], dict]:
    iter_tag = f"iter{iter_idx:02d}"
    iter_root = ensure_dir(RUNS_ROOT / iter_tag)
    config_path = CONFIG_ROOT / f"proxy_config.{iter_tag}.json"
    write_json(config_path, mutation["proxy_config"])
    summary = run_eval(manifest_path, case_set, iter_root / f"{gate_label}_eval", config_path)
    ranking = compare_stage(summary, case_ids)
    best_compare = ranking[0]
    panels = build_panels(summary, best_compare["variant"], iter_tag, case_ids[0])
    report = {
        "checked_at": now_iso(),
        "iteration": iter_idx,
        "stage": mutation["stage"],
        "mutation": mutation,
        "gate_label": gate_label,
        "case_ids": case_ids,
        "best_variant": best_compare["variant"],
        "best_compare": best_compare,
        "ranking": ranking,
        "panels": panels,
        "summary_json": rel(Path(summary["_root"]) / "summary.json"),
        "summary_md": rel(Path(summary["_root"]) / "summary.md"),
    }
    return report, ranking, panels


def collect_artifact_paths(report_paths: list[Path]) -> list[str]:
    out = [rel(path) for path in report_paths if path.exists()]
    for extra in [AUTLOOP_ROOT / "best_state.json", AUTLOOP_ROOT / "mutation_history.json", AUTLOOP_ROOT / "iteration_ledger.jsonl"]:
        if extra.exists():
            out.append(rel(extra))
    return sorted(set(out))


def main() -> int:
    ensure_dir(AUTLOOP_ROOT)
    reset_dir(RUNS_ROOT)
    reset_dir(PANELS_ROOT)
    reset_dir(CONFIG_ROOT)
    for extra in [AUTLOOP_ROOT / "best_state.json", AUTLOOP_ROOT / "best_local_state.json", AUTLOOP_ROOT / "mutation_history.json", AUTLOOP_ROOT / "iteration_ledger.jsonl", AUTLOOP_ROOT / "autoloop_final_postmortem.json", AUTLOOP_ROOT / "stage_a_shortlist.json"]:
        if extra.exists():
            extra.unlink()
    previous = load_previous_artifacts()
    py_compile([EVAL_SCRIPT, SCORER_SCRIPT, Path(__file__), REPO_ROOT / "scripts" / "rehydrated_render_mutation_bank.py"])

    truth_report_path = write_truth_repair_report(previous)
    loop_state_path = RESEARCH_ROOT / f"{FAMILY}_loop_state.json"
    write_packaging_files(truth_report_path)
    update_loop_state(
        loop_state_path,
        current_iteration=0,
        diagnosed_failure_mode="pending",
        chosen_stage="truth_repair",
        chosen_mutation="none",
        local_gate_status="pending",
        cloud_gate_open=False,
        artifact_paths=[rel(truth_report_path)],
    )

    manifest = load_json(BENCHMARK_MANIFEST)
    smoke_row_a = find_case(manifest, FIXED_SMOKE_CASE_ID)
    smoke_row_b = choose_second_smoke_anchor(manifest, FIXED_SMOKE_CASE_ID)
    smoke_rows = [smoke_row_a, smoke_row_b]
    smoke_case_ids = [case_id(row) for row in smoke_rows]
    hero_rows = select_manifest_rows(manifest, "hero_cases")
    hero_case_ids = [case_id(row) for row in hero_rows]
    benchmark_rows = select_manifest_rows(manifest, "benchmark_cases")
    benchmark_case_ids = [case_id(row) for row in benchmark_rows]
    source_count = len(smoke_row_a["source_cameras"])

    smoke_manifest_path = AUTLOOP_ROOT / "smoke_cases_manifest.json"
    write_manifest(smoke_rows, smoke_manifest_path, key="cases")

    stage_a_mutations = stage_a_source_subset_mutations(source_count)
    all_results: list[dict] = []
    report_paths: list[Path] = []
    best_state = None
    best_key = None
    mutation_history: list[str] = []
    iter_idx = 0

    def ingest_result(report: dict) -> None:
        nonlocal best_state, best_key
        all_results.append(report)
        mutation_history.append(report["mutation"]["mutation_id"])
        append_jsonl(AUTLOOP_ROOT / "iteration_ledger.jsonl", report)
        write_json(AUTLOOP_ROOT / "mutation_history.json", {"checked_at": now_iso(), "mutations": mutation_history})
        current_key = progress_key(report["best_compare"])
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_state = {
                "checked_at": now_iso(),
                "iteration": report["iteration"],
                "stage": report["stage"],
                "mutation": report["mutation"],
                "failure_class": report["best_compare"]["failure_class"],
                "compare": report["best_compare"],
                "panels": report["panels"],
                "summary_json": report["summary_json"],
            }
            write_json(AUTLOOP_ROOT / "best_state.json", best_state)

    stage_a_reports = []
    for mutation in stage_a_mutations:
        iter_idx += 1
        report, ranking, panels = run_stage_eval(
            iter_idx=iter_idx,
            mutation=mutation,
            manifest_path=smoke_manifest_path,
            case_set="cases",
            case_ids=smoke_case_ids,
            gate_label="smoke",
        )
        report_path = RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{iter_idx:02d}.json"
        decision_path = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{iter_idx:02d}.json"
        write_json(report_path, report)
        write_json(
            decision_path,
            {
                "checked_at": now_iso(),
                "iteration": iter_idx,
                "stage": "A",
                "chosen_mutation": mutation["mutation_id"],
                "smoke_pass": bool(report["best_compare"]["smoke_stage_accept"]),
                "failure_class": report["best_compare"]["failure_class"],
                "next_stage": "stage_b_shortlist",
            },
        )
        report_paths.extend([report_path, decision_path])
        update_loop_state(
            loop_state_path,
            current_iteration=iter_idx,
            diagnosed_failure_mode=report["best_compare"]["failure_class"],
            chosen_stage="stage_a_source_subset_smoke",
            chosen_mutation=mutation["mutation_id"],
            local_gate_status="smoke_running",
            cloud_gate_open=False,
            artifact_paths=collect_artifact_paths(report_paths),
        )
        ingest_result(report)
        stage_a_reports.append(report)

    stage_a_sorted = sorted(stage_a_reports, key=lambda item: progress_key(item["best_compare"]))
    shortlist = stage_a_sorted[:3]
    shortlist_rows = [
        {
            "iteration": item["iteration"],
            "mutation_id": item["mutation"]["mutation_id"],
            "source_subset": item["mutation"]["source_subset"],
            "best_variant": item["best_compare"]["variant"],
            "best_compare": item["best_compare"],
        }
        for item in shortlist
    ]
    write_json(AUTLOOP_ROOT / "stage_a_shortlist.json", {"checked_at": now_iso(), "rows": shortlist_rows})

    stage_b_reports = []
    for short_item in shortlist:
        mutations = stage_b_rehydrated_mutations(short_item["mutation"]["source_subset"], prefix=short_item["mutation"]["mutation_id"])
        for mutation in mutations:
            iter_idx += 1
            report, ranking, panels = run_stage_eval(
                iter_idx=iter_idx,
                mutation=mutation,
                manifest_path=smoke_manifest_path,
                case_set="cases",
                case_ids=smoke_case_ids,
                gate_label="smoke",
            )
            report_path = RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{iter_idx:02d}.json"
            decision_path = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{iter_idx:02d}.json"
            write_json(report_path, report)
            write_json(
                decision_path,
                {
                    "checked_at": now_iso(),
                    "iteration": iter_idx,
                    "stage": "B",
                    "chosen_mutation": mutation["mutation_id"],
                    "smoke_pass": bool(report["best_compare"]["smoke_stage_accept"]),
                    "stage_c_eligible": bool(stage_c_eligible(report["best_compare"])),
                    "failure_class": report["best_compare"]["failure_class"],
                    "next_stage": "stage_c_label_consistency" if stage_c_eligible(report["best_compare"]) else "continue",
                },
            )
            report_paths.extend([report_path, decision_path])
            update_loop_state(
                loop_state_path,
                current_iteration=iter_idx,
                diagnosed_failure_mode=report["best_compare"]["failure_class"],
                chosen_stage="stage_b_rehydrated_render_tuning",
                chosen_mutation=mutation["mutation_id"],
                local_gate_status="smoke_running",
                cloud_gate_open=False,
                artifact_paths=collect_artifact_paths(report_paths),
            )
            ingest_result(report)
            stage_b_reports.append(report)

    stage_c_candidates = sorted(
        [item for item in stage_b_reports if stage_c_eligible(item["best_compare"])],
        key=lambda item: progress_key(item["best_compare"]),
    )[:3]
    stage_c_reports = []
    for base_item in stage_c_candidates:
        mutations = stage_c_label_mutations(
            base_item["mutation"]["source_subset"],
            prefix=base_item["mutation"]["mutation_id"],
            seed_config=base_item["mutation"]["proxy_config"],
        )
        for mutation in mutations:
            iter_idx += 1
            report, ranking, panels = run_stage_eval(
                iter_idx=iter_idx,
                mutation=mutation,
                manifest_path=smoke_manifest_path,
                case_set="cases",
                case_ids=smoke_case_ids,
                gate_label="smoke",
            )
            report_path = RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{iter_idx:02d}.json"
            decision_path = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{iter_idx:02d}.json"
            write_json(report_path, report)
            write_json(
                decision_path,
                {
                    "checked_at": now_iso(),
                    "iteration": iter_idx,
                    "stage": "C",
                    "chosen_mutation": mutation["mutation_id"],
                    "smoke_pass": bool(report["best_compare"]["smoke_stage_accept"]),
                    "failure_class": report["best_compare"]["failure_class"],
                    "next_stage": "hero_gate_5x1" if report["best_compare"]["smoke_stage_accept"] else "continue",
                },
            )
            report_paths.extend([report_path, decision_path])
            update_loop_state(
                loop_state_path,
                current_iteration=iter_idx,
                diagnosed_failure_mode=report["best_compare"]["failure_class"],
                chosen_stage="stage_c_label_consistency",
                chosen_mutation=mutation["mutation_id"],
                local_gate_status="smoke_running",
                cloud_gate_open=False,
                artifact_paths=collect_artifact_paths(report_paths),
            )
            ingest_result(report)
            stage_c_reports.append(report)

    smoke_pass_candidates = sorted(
        [item for item in (stage_a_reports + stage_b_reports + stage_c_reports) if item["best_compare"]["smoke_stage_accept"]],
        key=lambda item: progress_key(item["best_compare"]),
    )

    hero_pass_candidate = None
    if smoke_pass_candidates:
        for item in smoke_pass_candidates[:3]:
            iter_idx += 1
            mutation = item["mutation"]
            config_path = CONFIG_ROOT / f"proxy_config.iter{iter_idx:02d}.json"
            write_json(config_path, mutation["proxy_config"])
            summary = run_eval(BENCHMARK_MANIFEST, "hero_cases", RUNS_ROOT / f"iter{iter_idx:02d}" / "hero_eval", config_path)
            compare = compare_variant(summary, item["best_compare"]["variant"], hero_case_ids)
            compare["failure_class"] = classify_failure(compare)
            compare["hero_accept"] = hero_accept(compare)
            compare["local20_accept"] = local20_accept(compare)
            compare["smoke_stage_accept"] = smoke_stage_accept(compare)
            panels = build_panels(summary, item["best_compare"]["variant"], f"iter{iter_idx:02d}", hero_case_ids[0])
            report = {
                "checked_at": now_iso(),
                "iteration": iter_idx,
                "stage": "hero_gate_5x1",
                "mutation": mutation,
                "gate_label": "hero_5x1",
                "case_ids": hero_case_ids,
                "best_variant": item["best_compare"]["variant"],
                "best_compare": compare,
                "ranking": [compare],
                "panels": panels,
                "summary_json": rel(Path(summary["_root"]) / "summary.json"),
                "summary_md": rel(Path(summary["_root"]) / "summary.md"),
            }
            report_path = RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{iter_idx:02d}.json"
            decision_path = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{iter_idx:02d}.json"
            write_json(report_path, report)
            write_json(
                decision_path,
                {
                    "checked_at": now_iso(),
                    "iteration": iter_idx,
                    "stage": "hero_gate_5x1",
                    "chosen_mutation": mutation["mutation_id"],
                    "hero_pass": bool(compare["hero_accept"]),
                    "failure_class": compare["failure_class"],
                    "next_stage": "local20" if compare["hero_accept"] else "stop_or_next_smoke_pass",
                },
            )
            report_paths.extend([report_path, decision_path])
            update_loop_state(
                loop_state_path,
                current_iteration=iter_idx,
                diagnosed_failure_mode=compare["failure_class"],
                chosen_stage="hero_gate_5x1",
                chosen_mutation=mutation["mutation_id"],
                local_gate_status="hero_running",
                cloud_gate_open=False,
                artifact_paths=collect_artifact_paths(report_paths),
            )
            ingest_result(report)
            if compare["hero_accept"]:
                hero_pass_candidate = {"mutation": mutation, "variant": item["best_compare"]["variant"], "compare": compare, "panels": panels}
                break

    local20_pass_candidate = None
    if hero_pass_candidate is not None:
        iter_idx += 1
        mutation = hero_pass_candidate["mutation"]
        config_path = CONFIG_ROOT / f"proxy_config.iter{iter_idx:02d}.json"
        write_json(config_path, mutation["proxy_config"])
        summary = run_eval(BENCHMARK_MANIFEST, "benchmark_cases", RUNS_ROOT / f"iter{iter_idx:02d}" / "local20_eval", config_path)
        compare = compare_variant(summary, hero_pass_candidate["variant"], benchmark_case_ids)
        compare["failure_class"] = classify_failure(compare)
        compare["hero_accept"] = hero_accept(compare)
        compare["local20_accept"] = local20_accept(compare)
        compare["smoke_stage_accept"] = smoke_stage_accept(compare)
        panels = build_panels(summary, hero_pass_candidate["variant"], f"iter{iter_idx:02d}", benchmark_case_ids[0])
        report = {
            "checked_at": now_iso(),
            "iteration": iter_idx,
            "stage": "local20",
            "mutation": mutation,
            "gate_label": "local20",
            "case_ids": benchmark_case_ids,
            "best_variant": hero_pass_candidate["variant"],
            "best_compare": compare,
            "ranking": [compare],
            "panels": panels,
            "summary_json": rel(Path(summary["_root"]) / "summary.json"),
            "summary_md": rel(Path(summary["_root"]) / "summary.md"),
        }
        report_path = RESEARCH_ROOT / f"{FAMILY}_iteration_report.iter{iter_idx:02d}.json"
        decision_path = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.iter{iter_idx:02d}.json"
        write_json(report_path, report)
        write_json(
            decision_path,
            {
                "checked_at": now_iso(),
                "iteration": iter_idx,
                "stage": "local20",
                "chosen_mutation": mutation["mutation_id"],
                "local20_pass": bool(compare["local20_accept"]),
                "failure_class": compare["failure_class"],
                "next_stage": "cloud_ticket" if compare["local20_accept"] else "stop_fail",
            },
        )
        report_paths.extend([report_path, decision_path])
        update_loop_state(
            loop_state_path,
            current_iteration=iter_idx,
            diagnosed_failure_mode=compare["failure_class"],
            chosen_stage="local20",
            chosen_mutation=mutation["mutation_id"],
            local_gate_status="local20_running",
            cloud_gate_open=False,
            artifact_paths=collect_artifact_paths(report_paths),
        )
        ingest_result(report)
        if compare["local20_accept"]:
            local20_pass_candidate = {
                "mutation": mutation,
                "variant": hero_pass_candidate["variant"],
                "compare": compare,
                "panels": panels,
                "summary_json": report["summary_json"],
            }

    if local20_pass_candidate is not None:
        pending_ticket_path = RESEARCH_ROOT / f"pending_cloud_ticket.{FAMILY}.json"
        cloud_ready_summary = AUTLOOP_ROOT / "cloud_ready_summary.md"
        write_json(
            pending_ticket_path,
            {
                "checked_at": now_iso(),
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
        write_text(
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
        update_loop_state(
            loop_state_path,
            current_iteration=iter_idx,
            diagnosed_failure_mode="passed_local20_pending_cloud",
            chosen_stage="cloud_ready_pending",
            chosen_mutation=local20_pass_candidate["mutation"]["mutation_id"],
            local_gate_status="local20_pass",
            cloud_gate_open=True,
            artifact_paths=collect_artifact_paths(report_paths) + [rel(pending_ticket_path), rel(cloud_ready_summary)],
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
        "mean_fg_visible_rgb_coverage_ratio": 0.0,
        "mean_fg_retained_mass_ratio": 0.0,
        "mean_largest_fg_visible_component_ratio": 0.0,
        "mean_human_erasure_penalty": 1.0,
        "metric_truth_bug": False,
        "variant": "none",
    }
    failure_class = classify_failure(best_compare)
    derived_next = NEXT_DRAFT_BY_FAILURE.get(failure_class, "teacher_geometry_visible_coverage_floor_audit")
    final_postmortem = {
        "checked_at": now_iso(),
        "family": FAMILY,
        "shape": FIRST_SHAPE,
        "status": "dead_same_day",
        "gate_stage_reached": "autoloop_local_bounded",
        "summary_reason": stage_failure_summary(best_compare),
        "failure_class": failure_class,
        "best_local_state": best_state,
        "next_family": derived_next,
        "smoke_cases": smoke_case_ids,
        "stage_a_shortlist": shortlist_rows,
    }
    write_json(AUTLOOP_ROOT / "autoloop_final_postmortem.json", final_postmortem)
    if best_state is not None:
        write_json(AUTLOOP_ROOT / "best_local_state.json", best_state)
    next_draft_path = RESEARCH_ROOT / f"next_manual_problem_draft.{derived_next}.json"
    write_json(
        next_draft_path,
        {
            "checked_at": now_iso(),
            "family": derived_next,
            "derived_from": FAMILY,
            "reason": final_postmortem["summary_reason"],
        },
    )
    result_path = RESEARCH_ROOT / f"{FAMILY}_result.json"
    postmortem_path = RESEARCH_ROOT / f"{FAMILY}_postmortem.json"
    write_json(
        result_path,
        {
            "checked_at": now_iso(),
            "family": FAMILY,
            "shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "autoloop_local_bounded",
            "summary_reason": final_postmortem["summary_reason"],
            "best_local_state": best_state,
        },
    )
    write_json(postmortem_path, final_postmortem)
    sync_failure_to_guard(final_postmortem, review_packet_rel=rel(AUTLOOP_ROOT / "autoloop_final_postmortem.json"))
    update_loop_state(
        loop_state_path,
        current_iteration=iter_idx,
        diagnosed_failure_mode=failure_class,
        chosen_stage="bounded_failure",
        chosen_mutation=best_state["mutation"]["mutation_id"] if best_state is not None else "none",
        local_gate_status="failed",
        cloud_gate_open=False,
        artifact_paths=collect_artifact_paths(report_paths) + [rel(result_path), rel(postmortem_path), rel(next_draft_path)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
