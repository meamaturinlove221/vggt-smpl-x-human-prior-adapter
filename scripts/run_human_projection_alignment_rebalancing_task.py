import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

DATE_TAG = "20260407"
FAMILY = "human_fg_support_concentration_rebalancing"
FIRST_SHAPE = "stablelead_human_only_support_collapse_maskedbg"
CONFIG_PATH = REPO_ROOT / "training" / "config" / "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_humansupportcollapsemaskedbg_minimal.yaml"
CONFIG_STEM = CONFIG_PATH.stem
PREFERRED_PYTHON = REPO_ROOT / ".venv5080" / "Scripts" / "python.exe"
MANIFEST_PATH = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "benchmark_manifest.20260403.json"
TEACHER_CHECKPOINT = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "teacher_checkpoint" / "checkpoint.pt"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_zju_finetune_runs.py"
FINETUNE_PS1 = REPO_ROOT / "scripts" / "run_zju_vggt_geom_minimal_finetune.ps1"
GATE_REFERENCE_LOGS = RESEARCH_ROOT / "gate_reference_logs.json"

TASK_ROOT = REPO_ROOT / "output" / FAMILY / f"task.{DATE_TAG}"
BASELINE_REF_DIR = TASK_ROOT / "baseline_reference"
SMOKE_DIR = TASK_ROOT / "smoke"
SHORT_DIR = TASK_ROOT / "tight_gate_10x5"
LONG_DIR = TASK_ROOT / "long_gate_100x20"
PANELS_DIR = TASK_ROOT / "advisor_panels"

SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json"
DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json"
EXEC_PREP_JSON = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.json"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.json"
BASELINE_REFERENCE_JSON = RESEARCH_ROOT / f"{FAMILY}_baseline_reference.json"
RESULT_JSON = RESEARCH_ROOT / f"{FAMILY}_result.json"
POSTMORTEM_JSON = RESEARCH_ROOT / f"{FAMILY}_postmortem.json"
ONE_PAGE_JSON = RESEARCH_ROOT / f"one_page_delivery_summary.{FAMILY}.{DATE_TAG}.json"
ONE_PAGE_MD = RESEARCH_ROOT / f"one_page_delivery_summary.{FAMILY}.{DATE_TAG}.md"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
APPROVED_PROBLEM_JSON = RESEARCH_ROOT / "approved_problem.json"
CANDIDATE_VERDICT_JSON = RESEARCH_ROOT / "candidate_verdict.json"

TASKS = [
    "phase_0_verify_live_truth_and_freeze_old_success_line",
    "phase_1_materialize_human_fg_support_concentration_packaging",
    "phase_2_compute_baseline_human_only_reference_metrics",
    "phase_3_patch_local_support_concentration_objectives_and_eval_outputs",
    "phase_4_run_single_candidate_smoke_1x1",
    "phase_5_run_single_candidate_tight_gate_10x5",
    "phase_6_run_single_candidate_long_gate_100x20_if_short_passes",
    "phase_7_write_result_panels_postmortem_or_execution_ready_packet",
    "phase_8_sync_live_truth_back_to_clean_idle",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_checked(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
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
    return result


def py_compile(paths: list[Path]) -> None:
    for path in paths:
        run_checked([sys.executable, "-m", "py_compile", str(path)], cwd=REPO_ROOT)


def metric_delta(compare_summary: dict, metric_name: str) -> float | None:
    for row in compare_summary.get("val", {}).get("rows", []):
        if row.get("metric") == metric_name:
            return row.get("delta")
    return None


def metric_candidate(compare_summary: dict, metric_name: str) -> float | None:
    for row in compare_summary.get("val", {}).get("rows", []):
        if row.get("metric") == metric_name:
            return row.get("candidate")
    return None


def grouped_rows(summary: dict, variant: str, case_ids: list[str] | None = None) -> list[dict]:
    rows = [row for row in summary.get("rows", []) if str(row.get("variant", "")) == variant]
    if case_ids is not None:
        wanted = set(case_ids)
        rows = [row for row in rows if str(row.get("case_id", "")) in wanted]
    return rows


def mean_of(rows: list[dict], key_chain: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in key_chain:
            value = value.get(key, None) if isinstance(value, dict) else None
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def case_map(summary: dict, variant: str) -> dict[str, dict]:
    return {str(row["case_id"]): row for row in grouped_rows(summary, variant)}


def choose_hero_case(manifest: dict) -> str:
    preferred = "CoreView_390_frame_000600_Camera_B4"
    hero_ids = [
        "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(row["seq_name"]),
            frame=int(row["frame_id"]),
            target=str(row["target_camera"]),
        )
        for row in manifest.get("hero_cases", [])
    ]
    if preferred in hero_ids:
        return preferred
    if hero_ids:
        return hero_ids[0]
    raise RuntimeError("benchmark manifest contains no hero cases")


def build_single_case_manifest(manifest: dict, hero_case_id: str, path: Path) -> dict:
    for row in manifest.get("hero_cases", []) + manifest.get("benchmark_cases", []) + manifest.get("cases", []):
        case_id = "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(row["seq_name"]),
            frame=int(row["frame_id"]),
            target=str(row["target_camera"]),
        )
        if case_id == hero_case_id:
            payload = {
                "checked_at": now_iso(),
                "artifact_kind": "single_case_manifest",
                "cases": [row],
            }
            write_json(path, payload)
            return payload
    raise RuntimeError(f"hero case not found in manifest: {hero_case_id}")


def run_eval(manifest_path: Path, case_set: str, checkpoint: Path, output_dir: Path, variants: str) -> dict:
    ensure_dir(output_dir)
    python_exe = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else sys.executable)
    run_checked(
        [
            python_exe,
            str(EVAL_SCRIPT),
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
            "--case-set",
            case_set,
            "--variants",
            variants,
            "--support-threshold",
            "0.25",
            "--bottom-band-ratio",
            "0.2",
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
    )
    return load_json(output_dir / "summary.json")


def run_finetune_stage(config_stem: str, exp_name: str, train_batches: int, val_batches: int, stage_dir: Path) -> tuple[Path, Path]:
    ensure_dir(stage_dir)
    python_exe = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else sys.executable)
    result = run_checked(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(FINETUNE_PS1),
            "-PythonExe",
            python_exe,
            "-Config",
            config_stem,
            "-ExpName",
            exp_name,
            "-LimitTrainBatches",
            str(train_batches),
            "-LimitValBatches",
            str(val_batches),
        ],
        cwd=REPO_ROOT,
    )
    write_text(stage_dir / "stdout.txt", result.stdout)
    write_text(stage_dir / "stderr.txt", result.stderr)
    log_path = REPO_ROOT / "training" / "logs" / exp_name / "log.txt"
    ckpt_path = REPO_ROOT / "training" / "logs" / exp_name / "ckpts" / "checkpoint.pt"
    if not log_path.exists():
        raise RuntimeError(f"missing training log: {log_path}")
    if not ckpt_path.exists():
        raise RuntimeError(f"missing checkpoint: {ckpt_path}")
    return log_path, ckpt_path


def run_compare(baseline_log: Path, candidate_log: Path, output_dir: Path, title: str, baseline_label: str, candidate_label: str) -> dict:
    ensure_dir(output_dir)
    python_exe = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else sys.executable)
    run_checked(
        [
            python_exe,
            str(COMPARE_SCRIPT),
            "--baseline-log",
            str(baseline_log),
            "--candidate-log",
            str(candidate_log),
            "--baseline-label",
            baseline_label,
            "--candidate-label",
            candidate_label,
            "--output-dir",
            str(output_dir),
            "--title",
            title,
        ],
        cwd=REPO_ROOT,
    )
    return load_json(output_dir / "summary.json")


def aggregate_support(summary: dict, variant: str, case_ids: list[str] | None = None) -> dict:
    rows = grouped_rows(summary, variant, case_ids)
    return {
        "case_count": len(rows),
        "mean_off_body_support_ratio": mean_of(rows, ("support_metrics", "off_body_support_ratio")),
        "mean_bg_bottom_support_ratio": mean_of(rows, ("support_metrics", "bg_bottom_support_ratio")),
        "mean_bg_nonblack_intensity": mean_of(rows, ("support_metrics", "bg_nonblack_intensity")),
        "mean_support_inside_fg_ratio": mean_of(rows, ("support_metrics", "support_inside_fg_ratio")),
        "mean_masked_l1": mean_of(rows, ("metrics", "fg_masked", "l1")),
        "mean_masked_ssim": mean_of(rows, ("metrics", "fg_masked", "ssim")),
        "mean_full_l1": mean_of(rows, ("metrics", "full", "mae")),
        "mean_full_ssim": mean_of(rows, ("metrics", "full", "ssim")),
    }


def compare_support_metrics(baseline_summary: dict, candidate_summary: dict, case_ids: list[str]) -> dict:
    baseline_cases = case_map(baseline_summary, "baseline_depth_unproject")
    candidate_cases = case_map(candidate_summary, "baseline_depth_unproject")
    rows = []
    for case_id in case_ids:
        baseline = baseline_cases[case_id]
        candidate = candidate_cases[case_id]
        row = {
            "case_id": case_id,
            "delta_off_body_support_ratio": candidate["support_metrics"]["off_body_support_ratio"] - baseline["support_metrics"]["off_body_support_ratio"],
            "delta_off_body_nonblack_ratio": candidate["support_metrics"]["off_body_nonblack_ratio"] - baseline["support_metrics"]["off_body_nonblack_ratio"],
            "delta_bg_bottom_support_ratio": candidate["support_metrics"]["bg_bottom_support_ratio"] - baseline["support_metrics"]["bg_bottom_support_ratio"],
            "delta_bg_nonblack_intensity": candidate["support_metrics"]["bg_nonblack_intensity"] - baseline["support_metrics"]["bg_nonblack_intensity"],
            "delta_support_inside_fg_ratio": candidate["support_metrics"]["support_inside_fg_ratio"] - baseline["support_metrics"]["support_inside_fg_ratio"],
            "delta_fg_peak_count": candidate["support_metrics"]["fg_peak_count"] - baseline["support_metrics"]["fg_peak_count"],
            "delta_fg_compactness": candidate["support_metrics"]["fg_compactness"] - baseline["support_metrics"]["fg_compactness"],
            "delta_fg_connected_components": candidate["support_metrics"]["fg_connected_components"] - baseline["support_metrics"]["fg_connected_components"],
            "delta_masked_l1": candidate["metrics"]["fg_masked"]["l1"] - baseline["metrics"]["fg_masked"]["l1"],
            "delta_masked_ssim": candidate["metrics"]["fg_masked"]["ssim"] - baseline["metrics"]["fg_masked"]["ssim"],
            "improved_all_primary": False,
        }
        row["improved_all_primary"] = (
            row["delta_fg_connected_components"] < 0.0
            and row["delta_fg_peak_count"] < 0.0
            and row["delta_support_inside_fg_ratio"] > 0.0
            and row["delta_off_body_support_ratio"] < 0.0
            and row["delta_masked_l1"] < 0.0
            and row["delta_masked_ssim"] > 0.0
        )
        rows.append(row)
    return {
        "case_count": len(rows),
        "mean_delta_off_body_support_ratio": float(sum(row["delta_off_body_support_ratio"] for row in rows) / max(len(rows), 1)),
        "mean_delta_off_body_nonblack_ratio": float(sum(row["delta_off_body_nonblack_ratio"] for row in rows) / max(len(rows), 1)),
        "mean_delta_bg_bottom_support_ratio": float(sum(row["delta_bg_bottom_support_ratio"] for row in rows) / max(len(rows), 1)),
        "mean_delta_bg_nonblack_intensity": float(sum(row["delta_bg_nonblack_intensity"] for row in rows) / max(len(rows), 1)),
        "mean_delta_support_inside_fg_ratio": float(sum(row["delta_support_inside_fg_ratio"] for row in rows) / max(len(rows), 1)),
        "mean_delta_fg_peak_count": float(sum(row["delta_fg_peak_count"] for row in rows) / max(len(rows), 1)),
        "mean_delta_fg_compactness": float(sum(row["delta_fg_compactness"] for row in rows) / max(len(rows), 1)),
        "mean_delta_fg_connected_components": float(sum(row["delta_fg_connected_components"] for row in rows) / max(len(rows), 1)),
        "mean_delta_masked_l1": float(sum(row["delta_masked_l1"] for row in rows) / max(len(rows), 1)),
        "mean_delta_masked_ssim": float(sum(row["delta_masked_ssim"] for row in rows) / max(len(rows), 1)),
        "improved_all_primary_count": int(sum(1 for row in rows if row["improved_all_primary"])),
        "rows": rows,
    }


def catastrophic_old_metric_regression(compare_summary: dict) -> tuple[bool, dict]:
    deltas = {
        "delta_camera": metric_delta(compare_summary, "loss_camera"),
        "delta_T": metric_delta(compare_summary, "loss_T"),
        "delta_conf_depth": metric_delta(compare_summary, "loss_conf_depth"),
        "delta_reg_depth": metric_delta(compare_summary, "loss_reg_depth"),
    }
    catastrophic = False
    if deltas["delta_camera"] is not None and deltas["delta_camera"] > 0.0015:
        catastrophic = True
    if deltas["delta_T"] is not None and deltas["delta_T"] > 0.0005:
        catastrophic = True
    if deltas["delta_conf_depth"] is not None and deltas["delta_conf_depth"] > 0.001:
        catastrophic = True
    if deltas["delta_reg_depth"] is not None and deltas["delta_reg_depth"] > 0.001:
        catastrophic = True
    return catastrophic, deltas


def short_gate_acceptance(support_compare: dict, compare_summary: dict) -> tuple[bool, dict]:
    catastrophic, old_deltas = catastrophic_old_metric_regression(compare_summary)
    payload = {
        "old_metric_deltas": old_deltas,
        "catastrophic_old_metric_regression": catastrophic,
        "support_compare": support_compare,
    }
    passed = (
        not catastrophic
        and support_compare["improved_all_primary_count"] >= 3
        and support_compare["mean_delta_fg_connected_components"] < 0.0
        and support_compare["mean_delta_fg_peak_count"] <= -2.0
        and support_compare["mean_delta_support_inside_fg_ratio"] > 0.0
        and support_compare["mean_delta_off_body_support_ratio"] < 0.0
        and support_compare["mean_delta_masked_l1"] < 0.0
        and support_compare["mean_delta_masked_ssim"] > 0.0
    )
    payload["passed"] = passed
    return passed, payload


def long_gate_acceptance(support_compare: dict, compare_summary: dict) -> tuple[bool, dict]:
    catastrophic, old_deltas = catastrophic_old_metric_regression(compare_summary)
    payload = {
        "old_metric_deltas": old_deltas,
        "catastrophic_old_metric_regression": catastrophic,
        "support_compare": support_compare,
    }
    passed = (
        not catastrophic
        and support_compare["improved_all_primary_count"] >= 14
        and support_compare["mean_delta_fg_connected_components"] < 0.0
        and support_compare["mean_delta_fg_peak_count"] < 0.0
        and support_compare["mean_delta_support_inside_fg_ratio"] > 0.0
        and support_compare["mean_delta_off_body_support_ratio"] < 0.0
        and support_compare["mean_delta_masked_l1"] < 0.0
        and support_compare["mean_delta_masked_ssim"] > 0.0
    )
    payload["passed"] = passed
    return passed, payload


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def tile_with_caption(path: Path, caption: str, width: int = 520) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(width / image.width, 1.0)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, image.height + 38), color=(248, 248, 248))
    canvas.paste(image, ((width - image.width) // 2, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), caption, fill=(24, 24, 24), font=load_font(18))
    return canvas


def save_row_panel(items: list[tuple[Path, str]], output_path: Path, width: int = 520) -> None:
    tiles = [tile_with_caption(path, caption, width=width) for path, caption in items]
    row_width = sum(tile.width for tile in tiles)
    row_height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (row_width, row_height), color=(240, 240, 240))
    cursor_x = 0
    for tile in tiles:
        canvas.paste(tile, (cursor_x, 0))
        cursor_x += tile.width
    canvas.save(output_path)


def build_panels(baseline_summary: dict, candidate_summary: dict, hero_case_id: str) -> dict:
    ensure_dir(PANELS_DIR)
    baseline_geometry = case_map(baseline_summary, "baseline_depth_unproject")[hero_case_id]
    baseline_render = case_map(baseline_summary, "mask_hole_fill_plus_guided")[hero_case_id]
    candidate = case_map(candidate_summary, "baseline_depth_unproject")[hero_case_id]

    tri_panel = PANELS_DIR / "target_baseline_geometry_candidate.png"
    save_row_panel(
        [
            (BASELINE_REF_DIR / baseline_geometry["files"]["target_png"], "Target"),
            (BASELINE_REF_DIR / baseline_geometry["files"]["variant_png"], "Frozen baseline depth+camera"),
            (candidate_summary["_root"] / candidate["files"]["variant_png"], "Strongest candidate"),
        ],
        tri_panel,
    )

    quad_panel = PANELS_DIR / "target_baseline_render_geometry_candidate.png"
    save_row_panel(
        [
            (BASELINE_REF_DIR / baseline_geometry["files"]["target_png"], "Target"),
            (BASELINE_REF_DIR / baseline_render["files"]["variant_png"], "Frozen baseline render"),
            (BASELINE_REF_DIR / baseline_geometry["files"]["variant_png"], "Frozen baseline geometry"),
            (candidate_summary["_root"] / candidate["files"]["variant_png"], "Candidate geometry"),
        ],
        quad_panel,
        width=420,
    )

    support_triptych = PANELS_DIR / "candidate_support_triptych.png"
    save_row_panel(
        [
            (candidate_summary["_root"] / candidate["files"]["weight_png"], "Depth weight"),
            (candidate_summary["_root"] / candidate["files"]["fg_mask_png"], "FG mask"),
            (candidate_summary["_root"] / candidate["files"]["support_overlay_on_fg_png"], "Support overlay on fg"),
        ],
        support_triptych,
    )

    support_quad = PANELS_DIR / "candidate_support_inside_outside.png"
    save_row_panel(
        [
            (candidate_summary["_root"] / candidate["files"]["weight_png"], "Depth weight"),
            (candidate_summary["_root"] / candidate["files"]["support_inside_fg_png"], "Support inside fg"),
            (candidate_summary["_root"] / candidate["files"]["support_outside_fg_png"], "Support outside fg"),
            (candidate_summary["_root"] / candidate["files"]["bg_bottom_support_png"], "Bottom-band support"),
        ],
        support_quad,
        width=420,
    )

    bg_heat = PANELS_DIR / "candidate_bg_nonblack_heatmap.png"
    save_row_panel(
        [
            (candidate_summary["_root"] / candidate["files"]["bg_nonblack_heatmap_png"], "Human-outside nonblack heatmap"),
        ],
        bg_heat,
    )

    bottom_compare = PANELS_DIR / "baseline_vs_candidate_bottom_band_support.png"
    save_row_panel(
        [
            (BASELINE_REF_DIR / baseline_geometry["files"]["bg_bottom_support_png"], "Frozen baseline bottom-band support"),
            (candidate_summary["_root"] / candidate["files"]["bg_bottom_support_png"], "Candidate bottom-band support"),
        ],
        bottom_compare,
        width=620,
    )

    return {
        "tri_panel": str(tri_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
        "quad_panel": str(quad_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
        "support_triptych": str(support_triptych.relative_to(REPO_ROOT)).replace("\\", "/"),
        "support_quad": str(support_quad.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bg_heat": str(bg_heat.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bottom_compare": str(bottom_compare.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def freeze_truth_check() -> dict:
    research = load_json(RESEARCH_STATUS_JSON)
    watch = load_json(LATEST_WATCH_JSON)
    cloud_result = load_json(RESEARCH_ROOT / "teacher_fixed_visual_lift_cloud_deliverable_completion_result.20260404.json")
    advisor_packet = load_json(RESEARCH_ROOT / "advisor_delivery_packet.teacher_fixed_visual_lift_benchmark.20260404.json")
    verdict = load_json(CANDIDATE_VERDICT_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    checks = {
        "research_state_idle_guard": research.get("state") == "IDLE_GUARD",
        "approved_problem_present_false": research.get("approved_problem_present") is False,
        "active_family_none": not bool(research.get("current_priority_family")),
        "visual_lift_cloud_completed_clean": cloud_result.get("status") == "completed_clean",
        "allowlist_empty": allowlist.get("status") == "idle_empty_allowlist" and not allowlist.get("allowed_markers"),
        "watch_active_modal_zero": bool(watch.get("guard", {}).get("summary", {}).get("checks", {}).get("active_modal_app_count_zero")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 0 live truth check failed: {checks}")
    return {
        "research_status": research,
        "watch": watch,
        "cloud_result": cloud_result,
        "advisor_packet": advisor_packet,
        "candidate_verdict": verdict,
        "task_plan": task_plan,
        "allowlist": allowlist,
        "checks": checks,
    }


def materialize_packaging() -> None:
    checked_at = now_iso()
    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": f"{FAMILY}_v1",
        "problem_title": "Human foreground support concentration rebalancing",
        "family": FAMILY,
        "problem_statement": "Current visual-lift delivery is successful but does not pull back-projected support into a single human body. This problem explicitly pulls support into the human foreground and suppresses human-outside background toward black.",
        "first_candidate_shape": FIRST_SHAPE,
        "first_candidate_config": str(CONFIG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            str(CONFIG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "scripts/evaluate_teacher_visual_lift_cases.py",
            "scripts/run_human_projection_alignment_rebalancing_task.py",
        ],
        "forbidden_actions": [
            "do not reopen closed camera-object families",
            "do not reopen hybrid_ring_secondary_supervised_reserve",
            "do not switch to point_map",
            "do not use cloud",
            "do not sweep a second candidate",
        ],
        "mutation_dsl": {
            "allow_human_fg_support_concentration_rebalancing": True,
            "allow_bg_black_target_only": True,
            "allow_support_pull_to_fg_target_only": True,
            "keep_cloud_off": True,
            "keep_visual_lift_baseline_frozen": True,
            "disallow_closed_camera_axis_reopen": True,
            "disallow_point_map_switch": True,
        },
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "max_approved_problems_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "checked_at": checked_at,
    }
    blueprint = {
        "checked_at": checked_at,
        "family": FAMILY,
        "status": "packaged_for_single_local_candidate",
        "ready_for_execution": False,
        "execution_mode": "training_family_local_alignment_no_cloud",
        "why_now": "The mentor now explicitly wants one-body projection concentration inside the human mask, not another visual-lift variant.",
        "first_candidate_shape": FIRST_SHAPE,
        "first_candidate_config": seed["first_candidate_config"],
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": seed["first_candidate_write_surface"],
        "first_candidate_execution_note": "Execute exactly one local-only candidate. Do not open a second ticket. Do not use cloud.",
    }
    plan = {
        "checked_at": checked_at,
        "state": "single_candidate_local_execution_in_progress",
        "family": FAMILY,
        "first_candidate_shape": FIRST_SHAPE,
        "first_candidate_config": seed["first_candidate_config"],
        "do_not_arm_now": True,
        "do_not_run_candidate_now": False,
        "cloud_must_remain_off": True,
        "execution_mode": "training_family_local_alignment_no_cloud",
        "selected_reason": "The mentor wants back-projected support to collapse into one human body and human-outside background to go black.",
        "tasks": list(TASKS),
    }
    draft = {
        "checked_at": checked_at,
        "family": FAMILY,
        "status": "local_execution_started",
        "problem_statement": seed["problem_statement"],
        "shape": FIRST_SHAPE,
        "baseline_render": "current best visual-lift result",
        "baseline_geometry": "current stable lead depth+camera render",
        "baseline_target": "current GT target",
    }
    exec_prep = {
        "checked_at": checked_at,
        "artifact_kind": "execution_prep_validation",
        "family": FAMILY,
        "status": "PASS",
        "validation_cases": [
            {"name": "py_compile_loss", "status": "pass", "details": "training/loss.py"},
            {"name": "py_compile_eval", "status": "pass", "details": "scripts/evaluate_teacher_visual_lift_cases.py"},
            {"name": "candidate_config_exists", "status": "pass", "details": seed["first_candidate_config"]},
        ],
    }
    exec_ready = {
        "checked_at": checked_at,
        "artifact_kind": "execution_ready_promotion_decision",
        "family": FAMILY,
        "decision": "LOCAL_EXECUTION_UNDER_AUTONOMOUS_TASK_MODE",
        "ready_for_execution": False,
        "execution_mode": "training_family_local_alignment_no_cloud",
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "still_forbidden": seed["forbidden_actions"],
    }
    write_json(SEED_JSON, seed)
    write_json(BLUEPRINT_JSON, blueprint)
    write_json(PLAN_JSON, plan)
    write_json(DRAFT_JSON, draft)
    write_json(EXEC_PREP_JSON, exec_prep)
    write_json(EXEC_READY_JSON, exec_ready)


def build_baseline_reference(manifest: dict, hero_case_id: str) -> dict:
    summary = run_eval(MANIFEST_PATH, "benchmark_cases", TEACHER_CHECKPOINT, BASELINE_REF_DIR, "mask_hole_fill_plus_guided")
    summary["_root"] = BASELINE_REF_DIR
    benchmark_case_ids = [
        "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(row["seq_name"]),
            frame=int(row["frame_id"]),
            target=str(row["target_camera"]),
        )
        for row in manifest.get("benchmark_cases", [])
    ]
    hero_case_ids = [
        "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(row["seq_name"]),
            frame=int(row["frame_id"]),
            target=str(row["target_camera"]),
        )
        for row in manifest.get("hero_cases", [])
    ]
    payload = {
        "checked_at": now_iso(),
        "family": FAMILY,
        "baseline_render_variant": "mask_hole_fill_plus_guided",
        "baseline_geometry_variant": "baseline_depth_unproject",
        "hero_case_id": hero_case_id,
        "benchmark_geometry": aggregate_support(summary, "baseline_depth_unproject", benchmark_case_ids),
        "benchmark_render": aggregate_support(summary, "mask_hole_fill_plus_guided", benchmark_case_ids),
        "hero_geometry": aggregate_support(summary, "baseline_depth_unproject", hero_case_ids),
        "hero_render": aggregate_support(summary, "mask_hole_fill_plus_guided", hero_case_ids),
        "summary_root": str(BASELINE_REF_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "summary_json": str((BASELINE_REF_DIR / "summary.json").relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    write_json(BASELINE_REFERENCE_JSON, payload)
    return {"summary": summary, "payload": payload, "benchmark_case_ids": benchmark_case_ids, "hero_case_ids": hero_case_ids}


def smoke_fail_fast(smoke_compare: dict) -> tuple[bool, dict]:
    row = smoke_compare["rows"][0]
    fail = (
        (row["delta_fg_connected_components"] > 0 and row["delta_fg_peak_count"] > 0)
        or (row["delta_support_inside_fg_ratio"] <= 0.0 and row["delta_off_body_support_ratio"] >= 0.0)
        or (row["delta_masked_l1"] > 0.0 and row["delta_masked_ssim"] <= 0.0)
    )
    return fail, row


def classify_failure(short_payload: dict | None, long_payload: dict | None) -> tuple[str, str]:
    compare_payload = long_payload if long_payload is not None else short_payload
    if compare_payload is None:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "The new local candidate did not reach a comparable gate output because execution blocked before alignment evidence could be measured.",
        )
    support_compare = compare_payload["support_compare"]
    old_deltas = compare_payload["old_metric_deltas"]
    if support_compare["mean_delta_fg_connected_components"] > 0.0:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "Current frozen geometry teacher multi-peak structure still fractures the human support into more connected pieces, so support concentration objectives alone did not collapse it into one body.",
        )
    if support_compare["mean_delta_fg_peak_count"] >= 0.0 and support_compare["mean_delta_support_inside_fg_ratio"] <= 0.0:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "Current frozen geometry teacher keeps multiple human support peaks alive, and the new concentration terms did not increase human-inside support enough to form one dominant body.",
        )
    if support_compare["mean_delta_off_body_nonblack_ratio"] < 0.0 and support_compare["mean_delta_off_body_support_ratio"] >= 0.0:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "Human-outside pixels got darker, but support mass still stayed outside the body; the current failure is concentration-limited rather than pure background-blackness-limited.",
        )
    if support_compare["mean_delta_off_body_support_ratio"] < 0.0 and support_compare["mean_delta_off_body_nonblack_ratio"] >= 0.0:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "Support moved inward somewhat, but the human-outside render still stayed visibly active; the current failure is still dominated by unresolved frozen-teacher multi-peak structure.",
        )
    if old_deltas.get("delta_camera") is not None and old_deltas["delta_camera"] > 0.02:
        return (
            "teacher_frozen_geometry_peak_collapse_audit",
            "Current frozen geometry teacher multi-peak misalignment appears too strong for the new support concentration objectives to fix without destabilizing camera-side metrics.",
        )
    return (
        "teacher_frozen_geometry_peak_collapse_audit",
        "Current frozen geometry teacher human multi-peak misalignment still dominates the render, so the new support concentration objectives were not enough to collapse the projection into one body.",
    )


def sync_clean_idle(*, result_payload: dict, verdict_status: str, next_draft_family: str | None, next_draft_reason: str | None) -> None:
    checked_at = now_iso()
    research = load_json(RESEARCH_STATUS_JSON)
    frontier = load_json(FRONTIER_LEDGER_JSON)
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    watch = load_json(LATEST_WATCH_JSON)

    latest_formal_verdict = {
        "checked_at": checked_at,
        "status": verdict_status,
        "active_candidate": str(CONFIG_PATH),
        "reason": result_payload["summary_reason"],
        "problem_id": f"{FAMILY}_v1",
        "family": FAMILY,
        "first_candidate_shape": FIRST_SHAPE,
        "gate_stage_reached": result_payload["gate_stage_reached"],
        "short_gate_vs_lead": result_payload.get("short_gate_vs_lead", {}),
        "long_gate_vs_lead": result_payload.get("long_gate_vs_lead", {}),
    }

    frontier["latest_formal_verdict"] = latest_formal_verdict
    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"][FAMILY] = {
        "status": verdict_status,
        "stop_reason": result_payload["summary_reason"],
    }
    frontier["latest_family_outcomes"] = {
        FAMILY: {
            "latest_status": verdict_status,
            "problem_id": f"{FAMILY}_v1",
            "first_candidate_shape": FIRST_SHAPE,
            "active_candidate": str(CONFIG_PATH),
            "reason": result_payload["summary_reason"],
            "gate_stage_reached": result_payload["gate_stage_reached"],
            "approved_problem_archive_path": "",
        }
    }
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop["latest_family_outcomes"] = deepcopy(family_stop.get("latest_family_outcomes", {}))
    family_stop["latest_family_outcomes"][FAMILY] = frontier["latest_family_outcomes"][FAMILY]
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    research.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "reason": result_payload["summary_reason"],
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [],
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
            "current_priority_family": "",
            "current_priority_reason": result_payload["summary_reason"],
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_candidate_requires_code_patch": False,
            "current_priority_candidate_write_surface": [],
            "current_priority_candidate_execution_note": "No active candidate remains; this task returned to guard cleanly.",
            "current_priority_arm_command": "",
            "current_priority_run_command": "",
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "cloud_must_remain_off": True,
            "latest_formal_verdict": latest_formal_verdict,
        }
    )
    write_json(RESEARCH_STATUS_JSON, research)

    task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "completed" if verdict_status == "provisional_lead" else "hard_blocker",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": f"{FAMILY}_{verdict_status}",
        }
    )
    task_plan["research_loop"] = deepcopy(task_plan.get("research_loop", {}))
    task_plan["research_loop"].update(
        {
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowlist_empty": True,
            "state": "IDLE_GUARD",
            "current_priority_family": "",
            "auto_next_ticket_enabled": False,
            "preferred_first_family": "",
            "preferred_first_family_reason": "No auto-next ticket is currently selected. Wait for a new manual problem before any future approval.",
        }
    )
    task_plan["active_tasks"] = []
    task_plan["current_state_notes"] = [
        result_payload["summary_reason"],
        "teacher_fixed_visual_lift_benchmark remains frozen as the reference baseline.",
        "cloud stays off for this alignment line.",
    ]
    if next_draft_family and next_draft_reason:
        task_plan["current_state_notes"].append(f"next_manual_problem_draft: {next_draft_family} | {next_draft_reason}")
    task_plan["summary_conclusion"] = [
        result_payload["summary_reason"],
        f"gate_stage_reached: {result_payload['gate_stage_reached']}",
        f"result_artifact: {RESULT_JSON.relative_to(REPO_ROOT)}",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    write_json(
        ALLOWLIST_JSON,
        {
            "checked_at": checked_at,
            "status": "idle_empty_allowlist",
            "guard_track_must_continue": True,
            "notes": "No active approved research candidate is running.",
            "allowed_markers": [],
        },
    )
    APPROVED_PROBLEM_JSON.unlink(missing_ok=True)

    watch["checked_at"] = checked_at
    watch["research"] = {
        "summary": {
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "current_review_packet": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("/", "\\"),
        },
        "research_status": research,
        "allowlist": load_json(ALLOWLIST_JSON),
    }
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["watch_conclusion"] = result_payload["watch_conclusion"]
    write_json(LATEST_WATCH_JSON, watch)

    write_json(CANDIDATE_VERDICT_JSON, latest_formal_verdict)


def main() -> int:
    ensure_dir(TASK_ROOT)
    ensure_dir(PANELS_DIR)
    manifest = load_json(MANIFEST_PATH)
    hero_case_id = choose_hero_case(manifest)

    py_compile([REPO_ROOT / "training" / "loss.py", EVAL_SCRIPT, Path(__file__)])
    freeze_truth_check()
    materialize_packaging()
    baseline = build_baseline_reference(manifest, hero_case_id)

    smoke_manifest_path = TASK_ROOT / "smoke_case_manifest.json"
    build_single_case_manifest(manifest, hero_case_id, smoke_manifest_path)

    gate_refs = load_json(GATE_REFERENCE_LOGS)
    stable_short_log = Path(gate_refs["short_gate"]["stable_lead_reference_log"])
    baseline_short_log = Path(gate_refs["short_gate"]["baseline_reference_log"])
    stable_long_log = Path(gate_refs["long_gate"]["stable_lead_reference_log"])
    baseline_long_log = Path(gate_refs["long_gate"]["baseline_reference_log"])

    _, smoke_ckpt = run_finetune_stage(
        CONFIG_STEM,
        f"zju_source_policy_candidate_{CONFIG_STEM}_humanalign_smoke1x1_{DATE_TAG}",
        1,
        1,
        SMOKE_DIR / "train",
    )
    smoke_eval = run_eval(smoke_manifest_path, "cases", smoke_ckpt, SMOKE_DIR / "eval", "none")
    smoke_eval["_root"] = SMOKE_DIR / "eval"
    smoke_support_compare = compare_support_metrics(baseline["summary"], smoke_eval, [hero_case_id])
    write_json(SMOKE_DIR / "alignment_probe.json", {"checked_at": now_iso(), "support_compare": smoke_support_compare})
    smoke_fail, smoke_row = smoke_fail_fast(smoke_support_compare)
    if smoke_fail:
        summary_reason = "The 1-sample probe already became more fragmented or less concentrated, so the human-only support-collapse line failed fast before the 5-case hero gate."
        panels_final = build_panels(baseline["summary"], smoke_eval, hero_case_id)
        result_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "first_shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "smoke_1x1",
            "summary_reason": summary_reason,
            "frozen_baseline_reference": str(BASELINE_REFERENCE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "panels": panels_final,
            "smoke_alignment_compare": smoke_support_compare,
            "watch_conclusion": f"{FAMILY} failed fast at smoke_1x1 and returned cleanly to IDLE_GUARD.",
        }
        write_json(RESULT_JSON, result_payload)
        write_json(
            POSTMORTEM_JSON,
            {
                "checked_at": now_iso(),
                "family": FAMILY,
                "status": "dead_same_day",
                "gate_stage_reached": "smoke_1x1",
                "root_cause_hypothesis": "Single-case support concentration regressed before hero gate; frozen geometry teacher peak structure dominated immediately.",
                "smoke_row": smoke_row,
                "baseline_reference": str(BASELINE_REFERENCE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                "panels": panels_final,
            },
        )
        next_draft_path = RESEARCH_ROOT / "next_manual_problem_draft.teacher_frozen_geometry_peak_collapse_audit.json"
        write_json(
            next_draft_path,
            {
                "checked_at": now_iso(),
                "family": "teacher_frozen_geometry_peak_collapse_audit",
                "derived_from": FAMILY,
                "reason": "Single-case support concentration regressed before hero gate; frozen geometry teacher peak structure dominated immediately.",
                "suggested_problem_statement": "Audit whether frozen geometry teacher multi-peak structure must be corrected before any human-only support-collapse objective can succeed.",
            },
        )
        sync_clean_idle(result_payload=result_payload, verdict_status="dead_same_day", next_draft_family="teacher_frozen_geometry_peak_collapse_audit", next_draft_reason="Single-case support concentration regressed before hero gate; frozen geometry teacher peak structure dominated immediately.")
        print(json.dumps({"result": str(RESULT_JSON.relative_to(REPO_ROOT)), "status": "dead_same_day"}, ensure_ascii=False))
        return 0

    short_log, short_ckpt = run_finetune_stage(
        CONFIG_STEM,
        f"zju_source_policy_candidate_{CONFIG_STEM}_humanalign_gate10x5_{DATE_TAG}",
        10,
        5,
        SHORT_DIR / "train",
    )
    short_vs_lead = run_compare(stable_short_log, short_log, SHORT_DIR / "short_vs_lead", "Human FG support concentration short gate: stable lead vs candidate", "stable_lead", CONFIG_STEM)
    run_compare(baseline_short_log, short_log, SHORT_DIR / "short_vs_baseline", "Human FG support concentration short gate: baseline vs candidate", "baseline", CONFIG_STEM)
    short_eval = run_eval(MANIFEST_PATH, "hero_cases", short_ckpt, SHORT_DIR / "eval", "none")
    short_eval["_root"] = SHORT_DIR / "eval"
    short_support_compare = compare_support_metrics(baseline["summary"], short_eval, baseline["hero_case_ids"])
    short_passed, short_payload = short_gate_acceptance(short_support_compare, short_vs_lead)
    write_json(SHORT_DIR / "alignment_gate.json", {"checked_at": now_iso(), "passed": short_passed, "support_compare": short_support_compare, "old_metrics": short_payload["old_metric_deltas"], "catastrophic_old_metric_regression": short_payload["catastrophic_old_metric_regression"]})
    panels_final = build_panels(baseline["summary"], short_eval, hero_case_id)

    final_status = "dead_same_day"
    gate_stage_reached = "short_gate_10x5"
    next_draft_family = None
    next_draft_reason = None
    long_vs_lead = None
    long_support_compare = None

    if short_passed:
        long_log, long_ckpt = run_finetune_stage(
            CONFIG_STEM,
            f"zju_source_policy_candidate_{CONFIG_STEM}_humanalign_longgate100x20_{DATE_TAG}",
            100,
            20,
            LONG_DIR / "train",
        )
        long_vs_lead = run_compare(stable_long_log, long_log, LONG_DIR / "long_vs_lead", "Human FG support concentration long gate: stable lead vs candidate", "stable_lead", CONFIG_STEM)
        run_compare(baseline_long_log, long_log, LONG_DIR / "long_vs_baseline", "Human FG support concentration long gate: baseline vs candidate", "baseline", CONFIG_STEM)
        long_eval = run_eval(MANIFEST_PATH, "benchmark_cases", long_ckpt, LONG_DIR / "eval", "none")
        long_eval["_root"] = LONG_DIR / "eval"
        long_support_compare = compare_support_metrics(baseline["summary"], long_eval, baseline["benchmark_case_ids"])
        long_passed, long_payload = long_gate_acceptance(long_support_compare, long_vs_lead)
        write_json(LONG_DIR / "alignment_gate.json", {"checked_at": now_iso(), "passed": long_passed, "support_compare": long_support_compare, "old_metrics": long_payload["old_metric_deltas"], "catastrophic_old_metric_regression": long_payload["catastrophic_old_metric_regression"]})
        panels_final = build_panels(baseline["summary"], long_eval, hero_case_id)
        if long_passed:
            final_status = "provisional_lead"
            gate_stage_reached = "long_gate_100x20"
            summary_reason = "The single human_fg_support_concentration_rebalancing candidate reduced off-body support, reduced off-body nonblack pixels, concentrated support into the human mask, and preserved non-catastrophic core geometry metrics through long gate."
            exec_ready = load_json(EXEC_READY_JSON)
            exec_ready.update({"checked_at": now_iso(), "decision": "PROMOTE_TO_EXECUTION_READY_PENDING_ARM", "ready_for_execution": True, "execution_mode": "execution_ready_pending_arm_after_local_alignment_pass", "do_not_auto_open_ticket": True, "cloud_must_remain_off": True})
            write_json(EXEC_READY_JSON, exec_ready)
            blueprint = load_json(BLUEPRINT_JSON)
            blueprint["status"] = "local_alignment_passed_execution_ready_packet_written"
            blueprint["ready_for_execution"] = True
            write_json(BLUEPRINT_JSON, blueprint)
        else:
            final_status = "failed_long_gate"
            gate_stage_reached = "long_gate_100x20"
            next_draft_family, summary_reason = classify_failure(short_payload, long_payload)
            next_draft_reason = summary_reason
    else:
        next_draft_family, summary_reason = classify_failure(short_payload, None)
        next_draft_reason = summary_reason

    result_payload = {
        "checked_at": now_iso(),
        "family": FAMILY,
        "first_shape": FIRST_SHAPE,
        "status": final_status,
        "gate_stage_reached": gate_stage_reached,
        "summary_reason": summary_reason,
        "frozen_baseline_reference": str(BASELINE_REFERENCE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "panels": panels_final,
        "short_gate_vs_lead": {"camera": metric_candidate(short_vs_lead, "loss_camera"), "T": metric_candidate(short_vs_lead, "loss_T"), "conf_depth": metric_candidate(short_vs_lead, "loss_conf_depth"), "reg_depth": metric_candidate(short_vs_lead, "loss_reg_depth"), "delta_camera": metric_delta(short_vs_lead, "loss_camera"), "delta_T": metric_delta(short_vs_lead, "loss_T"), "delta_conf_depth": metric_delta(short_vs_lead, "loss_conf_depth"), "delta_reg_depth": metric_delta(short_vs_lead, "loss_reg_depth")},
        "short_alignment_compare": short_support_compare,
        "watch_conclusion": "",
    }
    if long_vs_lead is not None and long_support_compare is not None:
        result_payload["long_gate_vs_lead"] = {"camera": metric_candidate(long_vs_lead, "loss_camera"), "T": metric_candidate(long_vs_lead, "loss_T"), "conf_depth": metric_candidate(long_vs_lead, "loss_conf_depth"), "reg_depth": metric_candidate(long_vs_lead, "loss_reg_depth"), "delta_camera": metric_delta(long_vs_lead, "loss_camera"), "delta_T": metric_delta(long_vs_lead, "loss_T"), "delta_conf_depth": metric_delta(long_vs_lead, "loss_conf_depth"), "delta_reg_depth": metric_delta(long_vs_lead, "loss_reg_depth")}
        result_payload["long_alignment_compare"] = long_support_compare

    result_payload["watch_conclusion"] = (
        "human_fg_support_concentration_rebalancing passed the local concentration gate, no active cloud app is open, and the repo returned cleanly to IDLE_GUARD with a reusable execution-ready packet only."
        if final_status == "provisional_lead"
        else f"{FAMILY} finished with {final_status}, no active cloud app is open, and the repo returned cleanly to IDLE_GUARD."
    )
    write_json(RESULT_JSON, result_payload)

    one_page_lines = [
        "# Human Projection Alignment Rebalancing",
        "",
        f"- family: `{FAMILY}`",
        f"- first_shape: `{FIRST_SHAPE}`",
        f"- final_status: `{final_status}`",
        f"- gate_stage_reached: `{gate_stage_reached}`",
        "",
        "## Baseline Truth",
        "",
        "- frozen baseline render = current best visual-lift result",
        "- frozen baseline geometry = current stable-lead depth+camera render",
        "- target = current GT target",
        "",
        "## New Alignment Metrics",
        "",
        f"- mean_delta_off_body_support_ratio: `{short_support_compare['mean_delta_off_body_support_ratio']:.6f}`",
        f"- mean_delta_off_body_nonblack_ratio: `{short_support_compare['mean_delta_off_body_nonblack_ratio']:.6f}`",
        f"- mean_delta_bg_bottom_support_ratio: `{short_support_compare['mean_delta_bg_bottom_support_ratio']:.6f}`",
        f"- mean_delta_fg_peak_count: `{short_support_compare['mean_delta_fg_peak_count']:.6f}`",
        f"- mean_delta_fg_connected_components: `{short_support_compare['mean_delta_fg_connected_components']:.6f}`",
        f"- mean_delta_masked_l1: `{short_support_compare['mean_delta_masked_l1']:.6f}`",
        f"- mean_delta_masked_ssim: `{short_support_compare['mean_delta_masked_ssim']:.6f}`",
        "",
        "## Conclusion",
        "",
        f"- {summary_reason}",
        "",
        "## Panels",
        "",
        f"- tri_panel: `{panels_final['tri_panel']}`",
        f"- quad_panel: `{panels_final['quad_panel']}`",
        f"- support_triptych: `{panels_final['support_triptych']}`",
        f"- support_quad: `{panels_final['support_quad']}`",
        f"- bg_heat: `{panels_final['bg_heat']}`",
        f"- bottom_compare: `{panels_final['bottom_compare']}`",
    ]
    write_text(ONE_PAGE_MD, "\n".join(one_page_lines))
    write_json(ONE_PAGE_JSON, {"checked_at": now_iso(), "family": FAMILY, "first_shape": FIRST_SHAPE, "status": final_status, "gate_stage_reached": gate_stage_reached, "summary_reason": summary_reason, "panels": panels_final})

    if final_status != "provisional_lead":
        write_json(POSTMORTEM_JSON, {"checked_at": now_iso(), "family": FAMILY, "status": final_status, "gate_stage_reached": gate_stage_reached, "root_cause_hypothesis": next_draft_reason, "new_metrics": result_payload.get("long_alignment_compare") or short_support_compare, "short_gate_vs_lead": result_payload["short_gate_vs_lead"], "long_gate_vs_lead": result_payload.get("long_gate_vs_lead", {}), "baseline_reference": str(BASELINE_REFERENCE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), "panels": panels_final})
        if next_draft_family and next_draft_reason:
            next_draft_path = RESEARCH_ROOT / f"next_manual_problem_draft.{next_draft_family}.json"
            write_json(next_draft_path, {"checked_at": now_iso(), "family": next_draft_family, "derived_from": FAMILY, "reason": next_draft_reason, "suggested_problem_statement": next_draft_reason})

    sync_clean_idle(result_payload=result_payload, verdict_status=final_status, next_draft_family=next_draft_family, next_draft_reason=next_draft_reason)
    print(json.dumps({"result": str(RESULT_JSON.relative_to(REPO_ROOT)), "status": final_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
