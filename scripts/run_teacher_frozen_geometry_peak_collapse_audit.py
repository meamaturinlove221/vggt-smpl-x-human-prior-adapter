import json
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
ITERATION_INDEX = 1
ITER_TAG = f"iter{ITERATION_INDEX:02d}"
FAMILY = "teacher_frozen_geometry_peak_collapse_audit"
FIRST_SHAPE = "stablelead_teacherpeakcollapse_source_dominance_softtop1maskedbg"
NEXT_FAILURE_FAMILY = "teacher_geometry_multiview_correspondence_audit"
PREFERRED_PYTHON = REPO_ROOT / ".venv5080" / "Scripts" / "python.exe"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"
MANIFEST_PATH = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "benchmark_manifest.20260403.json"
TEACHER_CHECKPOINT = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "teacher_checkpoint" / "checkpoint.pt"

TASK_ROOT = REPO_ROOT / "output" / FAMILY / f"task.{DATE_TAG}"
SMOKE_DIR = TASK_ROOT / "smoke"
HERO_DIR = TASK_ROOT / "hero"
PANELS_DIR = TASK_ROOT / "advisor_panels"

LOOP_STATE_JSON = RESEARCH_ROOT / f"{FAMILY}_loop_state.json"
ITER_REPORT_JSON = RESEARCH_ROOT / f"{FAMILY}_iteration_report.{ITER_TAG}.json"
ITER_DECISION_JSON = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.{ITER_TAG}.json"
PROXY_SWEEP_JSON = RESEARCH_ROOT / f"teacher_peak_collapse_proxy_sweep.{ITER_TAG}.json"
PROXY_RANKING_JSON = RESEARCH_ROOT / f"teacher_peak_collapse_proxy_ranking.{ITER_TAG}.json"
PROXY_BEST_JSON = RESEARCH_ROOT / f"teacher_peak_collapse_proxy_best.{ITER_TAG}.json"
RESULT_JSON = RESEARCH_ROOT / f"{FAMILY}_result.json"
POSTMORTEM_JSON = RESEARCH_ROOT / f"{FAMILY}_postmortem.json"
NEXT_DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{NEXT_FAILURE_FAMILY}.json"

SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json"
DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json"
EXEC_PREP_JSON = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.json"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.json"

PREV_RESULT_JSON = RESEARCH_ROOT / "human_fg_support_concentration_rebalancing_result.json"
PREV_POSTMORTEM_JSON = RESEARCH_ROOT / "human_fg_support_concentration_rebalancing_postmortem.json"
PREV_BASELINE_REFERENCE_JSON = RESEARCH_ROOT / "human_fg_support_concentration_rebalancing_baseline_reference.json"
PREV_NEXT_DRAFT_JSON = RESEARCH_ROOT / "next_manual_problem_draft.teacher_frozen_geometry_peak_collapse_audit.json"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
APPROVED_PROBLEM_JSON = RESEARCH_ROOT / "approved_problem.json"
CANDIDATE_VERDICT_JSON = RESEARCH_ROOT / "candidate_verdict.json"

PROXY_VARIANTS = [
    "soft_top1_inside_fg",
    "soft_top1_margin_inside_fg",
    "soft_top1_margin_plus_bottom_suppress",
    "soft_top1_margin_plus_fg_lcc_proxy",
]

SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"


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


def py_compile(paths: list[Path]) -> None:
    import subprocess

    python_exe = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else sys.executable)
    for path in paths:
        result = subprocess.run(
            [python_exe, "-m", "py_compile", str(path)],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"py_compile failed for {path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


def run_checked(args: list[str], cwd: Path | None = None) -> str:
    import subprocess

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
    return result.stdout


def load_required_truth() -> dict:
    required_image_paths = [
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "target_baseline_geometry_candidate.png",
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "target_baseline_render_geometry_candidate.png",
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "candidate_support_triptych.png",
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "candidate_support_inside_outside.png",
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "baseline_vs_candidate_bottom_band_support.png",
        REPO_ROOT / "output" / "human_fg_support_concentration_rebalancing" / "task.20260407" / "advisor_panels" / "candidate_bg_nonblack_heatmap.png",
    ]
    for path in required_image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required audit image: {path}")

    payload = {
        "prev_result": load_json(PREV_RESULT_JSON),
        "prev_postmortem": load_json(PREV_POSTMORTEM_JSON),
        "prev_baseline_reference": load_json(PREV_BASELINE_REFERENCE_JSON),
        "prev_next_draft": load_json(PREV_NEXT_DRAFT_JSON),
        "research_status": load_json(RESEARCH_STATUS_JSON),
        "task_plan": load_json(TASK_PLAN_JSON),
        "watch": load_json(LATEST_WATCH_JSON),
        "allowlist": load_json(ALLOWLIST_JSON),
        "required_images": [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in required_image_paths],
    }
    research = payload["research_status"]
    watch = payload["watch"]
    allowlist = payload["allowlist"]
    checks = {
        "state_idle_guard": research.get("state") == "IDLE_GUARD",
        "approved_problem_present_false": research.get("approved_problem_present") is False,
        "allowlist_empty": allowlist.get("status") == "idle_empty_allowlist" and not allowlist.get("allowed_markers"),
        "active_modal_zero": len(watch.get("modal_apps", [])) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 0 truth guard failed: {checks}")
    payload["checks"] = checks
    return payload


def write_loop_state(*, current_iteration: int, diagnosed_failure_mode: str, chosen_mutation: str, chosen_proxy: str, local_gate_status: str, cloud_gate_open: bool, artifact_paths: list[str], cleanup_ok: bool) -> dict:
    payload = {
        "checked_at": now_iso(),
        "current_iteration": int(current_iteration),
        "diagnosed_failure_mode": diagnosed_failure_mode,
        "chosen_mutation": chosen_mutation,
        "chosen_proxy": chosen_proxy,
        "local_gate_status": local_gate_status,
        "cloud_gate_open": bool(cloud_gate_open),
        "artifact_paths": artifact_paths,
        "cleanup_ok": bool(cleanup_ok),
    }
    write_json(LOOP_STATE_JSON, payload)
    return payload


def materialize_packaging() -> None:
    checked_at = now_iso()
    write_json(
        SEED_JSON,
        {
            "checked_at": checked_at,
            "problem_id": f"{FAMILY}_v1",
            "family": FAMILY,
            "problem_statement": "Current human-only support concentration showed that human-outside suppression works, but frozen geometry teacher multi-peak support still dominates. This audit first validates teacher-side peak collapse proxies before allowing any training-side change.",
            "first_candidate_shape": FIRST_SHAPE,
            "execution_mode": "offline_proxy_then_minimal_local_training",
            "allowed_write_surface": [
                "scripts/evaluate_teacher_visual_lift_cases.py",
                "scripts/run_teacher_frozen_geometry_peak_collapse_audit.py",
            ],
            "forbidden_actions": [
                "no cousin sweep",
                "no second ticket",
                "no point_map pivot",
                "no old scalar camera family reopen",
                "no cloud before local 20-case pass",
            ],
        },
    )


def build_single_case_manifest(manifest: dict, case_id: str, path: Path) -> None:
    for row in manifest.get("hero_cases", []) + manifest.get("benchmark_cases", []) + manifest.get("cases", []):
        candidate_id = "{seq}_frame_{frame:06d}_{target}".format(
            seq=str(row["seq_name"]),
            frame=int(row["frame_id"]),
            target=str(row["target_camera"]),
        )
        if candidate_id == case_id:
            write_json(path, {"checked_at": now_iso(), "artifact_kind": "single_case_manifest", "cases": [row]})
            return
    raise RuntimeError(f"case not found in manifest: {case_id}")


def run_eval(manifest_path: Path, case_set: str, output_dir: Path, variants: list[str]) -> dict:
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
            str(TEACHER_CHECKPOINT),
            "--case-set",
            case_set,
            "--variants",
            ",".join(variants) if variants else "none",
            "--support-threshold",
            "0.25",
            "--bottom-band-ratio",
            "0.2",
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
    )
    payload = load_json(output_dir / "summary.json")
    payload["_root"] = output_dir
    return payload


def rows_by_variant(summary: dict, variant: str) -> dict[str, dict]:
    return {str(row["case_id"]): row for row in summary.get("rows", []) if str(row.get("variant")) == variant}


def compare_variant(summary: dict, variant: str, case_ids: list[str]) -> dict:
    baseline = rows_by_variant(summary, "baseline_depth_unproject")
    candidate = rows_by_variant(summary, variant)
    rows = []
    for case_id in case_ids:
        base_row = baseline[case_id]
        cand_row = candidate[case_id]
        row = {
            "case_id": case_id,
            "delta_fg_connected_components": cand_row["support_metrics"]["fg_connected_components"] - base_row["support_metrics"]["fg_connected_components"],
            "delta_fg_peak_count": cand_row["support_metrics"]["fg_peak_count"] - base_row["support_metrics"]["fg_peak_count"],
            "delta_support_inside_fg_ratio": cand_row["support_metrics"]["support_inside_fg_ratio"] - base_row["support_metrics"]["support_inside_fg_ratio"],
            "delta_off_body_support_ratio": cand_row["support_metrics"]["off_body_support_ratio"] - base_row["support_metrics"]["off_body_support_ratio"],
            "delta_off_body_nonblack_ratio": cand_row["support_metrics"]["off_body_nonblack_ratio"] - base_row["support_metrics"]["off_body_nonblack_ratio"],
            "delta_bg_bottom_support_ratio": cand_row["support_metrics"]["bg_bottom_support_ratio"] - base_row["support_metrics"]["bg_bottom_support_ratio"],
            "delta_masked_l1": cand_row["metrics"]["fg_masked"]["l1"] - base_row["metrics"]["fg_masked"]["l1"],
            "delta_masked_ssim": cand_row["metrics"]["fg_masked"]["ssim"] - base_row["metrics"]["fg_masked"]["ssim"],
            "delta_fg_largest_component_ratio": cand_row["support_metrics"]["fg_largest_component_ratio"] - base_row["support_metrics"]["fg_largest_component_ratio"],
            "delta_source_entropy_inside_fg": cand_row["support_metrics"]["source_entropy_inside_fg"] - base_row["support_metrics"]["source_entropy_inside_fg"],
            "delta_source_top1_mass_ratio_inside_fg": cand_row["support_metrics"]["source_top1_mass_ratio_inside_fg"] - base_row["support_metrics"]["source_top1_mass_ratio_inside_fg"],
            "delta_source_top1_top2_margin_inside_fg": cand_row["support_metrics"]["source_top1_top2_margin_inside_fg"] - base_row["support_metrics"]["source_top1_top2_margin_inside_fg"],
        }
        row["smoke_pass"] = (
            row["delta_fg_connected_components"] < 0
            and row["delta_fg_peak_count"] <= -1
            and row["delta_support_inside_fg_ratio"] > 0
            and row["delta_off_body_support_ratio"] < 0
            and row["delta_masked_l1"] < 0
            and row["delta_masked_ssim"] > 0
            and row["delta_source_entropy_inside_fg"] < 0
            and row["delta_source_top1_top2_margin_inside_fg"] > 0
        )
        row["hero_primary_pass"] = (
            row["delta_fg_connected_components"] < 0
            and row["delta_fg_peak_count"] < 0
            and row["delta_support_inside_fg_ratio"] > 0
            and row["delta_off_body_support_ratio"] < 0
            and row["delta_masked_l1"] < 0
            and row["delta_masked_ssim"] > 0
            and row["delta_fg_largest_component_ratio"] > 0
            and row["delta_source_entropy_inside_fg"] < 0
            and row["delta_source_top1_top2_margin_inside_fg"] > 0
        )
        rows.append(row)
    count = max(len(rows), 1)
    return {
        "variant": variant,
        "case_count": len(rows),
        "improved_all_primary_count": int(sum(1 for row in rows if row["hero_primary_pass"])),
        "mean_delta_fg_connected_components": float(sum(row["delta_fg_connected_components"] for row in rows) / count),
        "mean_delta_fg_peak_count": float(sum(row["delta_fg_peak_count"] for row in rows) / count),
        "mean_delta_support_inside_fg_ratio": float(sum(row["delta_support_inside_fg_ratio"] for row in rows) / count),
        "mean_delta_off_body_support_ratio": float(sum(row["delta_off_body_support_ratio"] for row in rows) / count),
        "mean_delta_off_body_nonblack_ratio": float(sum(row["delta_off_body_nonblack_ratio"] for row in rows) / count),
        "mean_delta_bg_bottom_support_ratio": float(sum(row["delta_bg_bottom_support_ratio"] for row in rows) / count),
        "mean_delta_masked_l1": float(sum(row["delta_masked_l1"] for row in rows) / count),
        "mean_delta_masked_ssim": float(sum(row["delta_masked_ssim"] for row in rows) / count),
        "mean_fg_largest_component_ratio_delta": float(sum(row["delta_fg_largest_component_ratio"] for row in rows) / count),
        "mean_source_entropy_inside_fg_delta": float(sum(row["delta_source_entropy_inside_fg"] for row in rows) / count),
        "mean_source_top1_mass_ratio_inside_fg_delta": float(sum(row["delta_source_top1_mass_ratio_inside_fg"] for row in rows) / count),
        "mean_source_top1_top2_margin_inside_fg_delta": float(sum(row["delta_source_top1_top2_margin_inside_fg"] for row in rows) / count),
        "rows": rows,
    }


def smoke_pass(compare: dict) -> bool:
    return compare["case_count"] == 1 and bool(compare["rows"][0]["smoke_pass"])


def hero_pass(compare: dict) -> bool:
    return (
        compare["improved_all_primary_count"] >= 3
        and compare["mean_delta_fg_connected_components"] < 0
        and compare["mean_delta_fg_peak_count"] < 0
        and compare["mean_delta_support_inside_fg_ratio"] > 0
        and compare["mean_delta_off_body_support_ratio"] < 0
        and compare["mean_delta_masked_l1"] < 0
        and compare["mean_delta_masked_ssim"] > 0
        and compare["mean_fg_largest_component_ratio_delta"] > 0
        and compare["mean_source_entropy_inside_fg_delta"] < 0
        and compare["mean_source_top1_top2_margin_inside_fg_delta"] > 0
    )


def proxy_rank_key(compare: dict) -> tuple:
    return (
        0 if smoke_pass(compare) else 1,
        compare["mean_delta_fg_connected_components"],
        compare["mean_delta_fg_peak_count"],
        -compare["mean_delta_support_inside_fg_ratio"],
        compare["mean_delta_off_body_support_ratio"],
        compare["mean_delta_masked_l1"],
        -compare["mean_delta_masked_ssim"],
        -compare["mean_fg_largest_component_ratio_delta"],
        compare["mean_source_entropy_inside_fg_delta"],
        -compare["mean_source_top1_top2_margin_inside_fg_delta"],
    )


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def tile_with_caption(path: Path, caption: str, width: int = 420) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(width / image.width, 1.0)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, image.height + 38), color=(248, 248, 248))
    canvas.paste(image, ((width - image.width) // 2, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), caption, fill=(24, 24, 24), font=load_font(18))
    return canvas


def save_row_panel(items: list[tuple[Path, str]], output_path: Path, width: int = 420) -> None:
    tiles = [tile_with_caption(path, caption, width=width) for path, caption in items]
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), color=(240, 240, 240))
    cursor_x = 0
    for tile in tiles:
        canvas.paste(tile, (cursor_x, 0))
        cursor_x += tile.width
    canvas.save(output_path)


def build_panels(baseline_summary: dict, candidate_summary: dict, variant: str, case_id: str, suffix: str) -> dict:
    ensure_dir(PANELS_DIR)
    baseline_geometry = rows_by_variant(baseline_summary, "baseline_depth_unproject")[case_id]
    baseline_render = rows_by_variant(baseline_summary, "mask_hole_fill_plus_guided")[case_id]
    candidate = rows_by_variant(candidate_summary, variant)[case_id]

    tri_panel = PANELS_DIR / f"target_baseline_geometry_candidate.{suffix}.png"
    save_row_panel(
        [
            (Path(baseline_summary["_root"]) / baseline_geometry["files"]["target_png"], "Target"),
            (Path(baseline_summary["_root"]) / baseline_geometry["files"]["variant_png"], "Frozen baseline geometry"),
            (Path(candidate_summary["_root"]) / candidate["files"]["variant_png"], variant),
        ],
        tri_panel,
    )
    quad_panel = PANELS_DIR / f"target_baseline_render_geometry_candidate.{suffix}.png"
    save_row_panel(
        [
            (Path(baseline_summary["_root"]) / baseline_geometry["files"]["target_png"], "Target"),
            (Path(baseline_summary["_root"]) / baseline_render["files"]["variant_png"], "Frozen baseline render"),
            (Path(baseline_summary["_root"]) / baseline_geometry["files"]["variant_png"], "Frozen baseline geometry"),
            (Path(candidate_summary["_root"]) / candidate["files"]["variant_png"], variant),
        ],
        quad_panel,
        width=360,
    )
    support_triptych = PANELS_DIR / f"candidate_support_triptych.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate["files"]["weight_png"], "Depth weight"),
            (Path(candidate_summary["_root"]) / candidate["files"]["support_overlay_on_fg_png"], "Support overlay on fg"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_top1_mass_png"], "Source top1 mass"),
        ],
        support_triptych,
    )
    support_quad = PANELS_DIR / f"candidate_support_inside_outside.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate["files"]["support_inside_fg_png"], "Support inside fg"),
            (Path(candidate_summary["_root"]) / candidate["files"]["support_outside_fg_png"], "Support outside fg"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_top1_top2_margin_png"], "Top1-top2 margin"),
            (Path(candidate_summary["_root"]) / candidate["files"]["bg_bottom_support_png"], "Bottom-band support"),
        ],
        support_quad,
        width=360,
    )
    source_triptych = PANELS_DIR / f"candidate_source_dominance_triptych.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate["files"]["source_top1_mass_png"], "Source top1 mass"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_top1_top2_margin_png"], "Top1-top2 margin"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_entropy_png"], "Source entropy"),
        ],
        source_triptych,
    )
    bg_heat = PANELS_DIR / f"candidate_bg_nonblack_heatmap.{suffix}.png"
    save_row_panel(
        [(Path(candidate_summary["_root"]) / candidate["files"]["bg_nonblack_heatmap_png"], "Human-outside nonblack heatmap")],
        bg_heat,
    )
    bottom_compare = PANELS_DIR / f"baseline_vs_candidate_bottom_band_support.{suffix}.png"
    save_row_panel(
        [
            (Path(baseline_summary["_root"]) / baseline_geometry["files"]["bg_bottom_support_png"], "Frozen baseline bottom-band support"),
            (Path(candidate_summary["_root"]) / candidate["files"]["bg_bottom_support_png"], "Candidate bottom-band support"),
        ],
        bottom_compare,
        width=520,
    )
    return {
        "tri_panel": str(tri_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
        "quad_panel": str(quad_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
        "support_triptych": str(support_triptych.relative_to(REPO_ROOT)).replace("\\", "/"),
        "support_inside_outside": str(support_quad.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_dominance_triptych": str(source_triptych.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bg_heat": str(bg_heat.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bottom_compare": str(bottom_compare.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def sync_failure(*, result_payload: dict, postmortem_payload: dict, next_reason: str) -> None:
    checked_at = now_iso()
    research = load_json(RESEARCH_STATUS_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    frontier = load_json(FRONTIER_LEDGER_JSON)
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    watch = load_json(LATEST_WATCH_JSON)
    latest_formal_verdict = {
        "checked_at": checked_at,
        "status": "dead_same_day",
        "active_candidate": FIRST_SHAPE,
        "reason": result_payload["summary_reason"],
        "problem_id": f"{FAMILY}_v1",
        "family": FAMILY,
        "first_candidate_shape": FIRST_SHAPE,
        "gate_stage_reached": result_payload["gate_stage_reached"],
        "short_gate_vs_lead": {},
        "long_gate_vs_lead": {},
    }
    research.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "reason": result_payload["summary_reason"],
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [],
            "current_priority_family": "",
            "current_priority_reason": result_payload["summary_reason"],
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
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
    frontier["latest_formal_verdict"] = latest_formal_verdict
    frontier["latest_family_outcomes"] = deepcopy(frontier.get("latest_family_outcomes", {}))
    frontier["latest_family_outcomes"][FAMILY] = {
        "latest_status": "dead_same_day",
        "problem_id": f"{FAMILY}_v1",
        "first_candidate_shape": FIRST_SHAPE,
        "active_candidate": FIRST_SHAPE,
        "reason": result_payload["summary_reason"],
        "gate_stage_reached": result_payload["gate_stage_reached"],
        "approved_problem_archive_path": "",
    }
    write_json(FRONTIER_LEDGER_JSON, frontier)
    family_stop["latest_family_outcomes"] = deepcopy(family_stop.get("latest_family_outcomes", {}))
    family_stop["latest_family_outcomes"][FAMILY] = frontier["latest_family_outcomes"][FAMILY]
    write_json(FAMILY_STOP_REASON_JSON, family_stop)
    task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "hard_blocker",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": f"{FAMILY}_dead_same_day",
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
        }
    )
    task_plan["active_tasks"] = []
    task_plan["current_state_notes"] = [
        result_payload["summary_reason"],
        f"next_manual_problem_draft: {NEXT_FAILURE_FAMILY} | {next_reason}",
    ]
    task_plan["summary_conclusion"] = [
        result_payload["summary_reason"],
        f"gate_stage_reached: {result_payload['gate_stage_reached']}",
        f"result_artifact: {RESULT_JSON.relative_to(REPO_ROOT)}",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2))
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]))
    write_json(ALLOWLIST_JSON, {"checked_at": checked_at, "status": "idle_empty_allowlist", "guard_track_must_continue": True, "notes": "No active approved research candidate is running.", "allowed_markers": []})
    APPROVED_PROBLEM_JSON.unlink(missing_ok=True)
    write_json(CANDIDATE_VERDICT_JSON, latest_formal_verdict)
    watch["checked_at"] = checked_at
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["watch_conclusion"] = result_payload["watch_conclusion"]
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
    write_json(LATEST_WATCH_JSON, watch)
    write_json(RESULT_JSON, result_payload)
    write_json(POSTMORTEM_JSON, postmortem_payload)
    write_json(
        NEXT_DRAFT_JSON,
        {
            "checked_at": checked_at,
            "family": NEXT_FAILURE_FAMILY,
            "derived_from": FAMILY,
            "reason": next_reason,
            "suggested_problem_statement": "Audit whether frozen geometry teacher multi-view correspondence is too ambiguous for local source-dominance collapse losses to fix downstream.",
        },
    )


def main() -> int:
    ensure_dir(TASK_ROOT)
    ensure_dir(PANELS_DIR)
    py_compile([EVAL_SCRIPT, Path(__file__)])
    truth = load_required_truth()
    materialize_packaging()

    baseline_root = REPO_ROOT / truth["prev_baseline_reference"]["summary_root"]
    baseline_summary = load_json(baseline_root / "summary.json")
    baseline_summary["_root"] = baseline_root
    artifact_paths = [
        str(PREV_RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PREV_POSTMORTEM_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PREV_BASELINE_REFERENCE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PREV_NEXT_DRAFT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]
    write_loop_state(
        current_iteration=ITERATION_INDEX,
        diagnosed_failure_mode="human-outside suppression improved while frozen teacher multi-peak support still fragmented the body",
        chosen_mutation="offline_proxy_only",
        chosen_proxy="",
        local_gate_status="proxy_smoke_pending",
        cloud_gate_open=False,
        artifact_paths=artifact_paths,
        cleanup_ok=True,
    )

    manifest = load_json(MANIFEST_PATH)
    smoke_manifest_path = TASK_ROOT / "smoke_case_manifest.json"
    build_single_case_manifest(manifest, SMOKE_CASE_ID, smoke_manifest_path)

    smoke_summary = run_eval(smoke_manifest_path, "cases", SMOKE_DIR / "eval", PROXY_VARIANTS)
    smoke_compares = [compare_variant(smoke_summary, variant, [SMOKE_CASE_ID]) for variant in PROXY_VARIANTS]
    smoke_compares.sort(key=proxy_rank_key)
    smoke_passers = [row for row in smoke_compares if smoke_pass(row)]
    best_smoke = smoke_passers[0] if smoke_passers else smoke_compares[0]
    write_json(PROXY_SWEEP_JSON, {"checked_at": now_iso(), "stage": "smoke", "case_id": SMOKE_CASE_ID, "variants": smoke_compares})
    write_json(PROXY_RANKING_JSON, {"checked_at": now_iso(), "stage": "smoke", "ranking": smoke_compares})
    write_json(PROXY_BEST_JSON, {"checked_at": now_iso(), "stage": "smoke", "best": best_smoke, "smoke_pass": smoke_pass(best_smoke)})

    if not smoke_passers:
        panels = build_panels(baseline_summary, smoke_summary, best_smoke["variant"], SMOKE_CASE_ID, ITER_TAG)
        result_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "first_shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_smoke_1x1",
            "summary_reason": "All four bounded offline peak-collapse proxies failed the 1-sample smoke gate, so frozen teacher baseline support could not be collapsed into a single human body before training.",
            "best_proxy": best_smoke,
            "proxy_ranking_artifact": str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "panels": panels,
            "watch_conclusion": f"{FAMILY} exhausted all four offline proxies at smoke_1x1 and returned cleanly to IDLE_GUARD.",
        }
        postmortem_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_smoke_1x1",
            "root_cause_hypothesis": "Frozen teacher baseline could not collapse to a single human body even under bounded offline source-dominance proxies.",
            "proxy_ranking": smoke_compares,
            "best_proxy": best_smoke,
            "panels": panels,
        }
        next_reason = "Frozen teacher baseline failed all four offline peak-collapse proxies at smoke, so the next honest question is whether multiview correspondence itself is too ambiguous upstream."
        write_json(ITER_REPORT_JSON, {"checked_at": now_iso(), "iteration": ITERATION_INDEX, "stage": "proxy_smoke", "status": "dead_same_day", "diagnosed_failure_mode": "offline proxies cannot collapse frozen teacher multi-peak support on the smoke case", "best_proxy": best_smoke})
        write_json(ITER_DECISION_JSON, {"checked_at": now_iso(), "iteration": ITERATION_INDEX, "decision": "STOP_AT_FAILURE_ENDPOINT_B", "chosen_proxy": best_smoke["variant"], "next_manual_problem_draft": NEXT_FAILURE_FAMILY})
        write_loop_state(
            current_iteration=ITERATION_INDEX,
            diagnosed_failure_mode="all bounded offline proxies failed smoke on frozen teacher geometry",
            chosen_mutation="proxy_only",
            chosen_proxy=best_smoke["variant"],
            local_gate_status="dead_same_day_proxy_smoke_fail",
            cloud_gate_open=False,
            artifact_paths=artifact_paths + [str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/")],
            cleanup_ok=True,
        )
        sync_failure(result_payload=result_payload, postmortem_payload=postmortem_payload, next_reason=next_reason)
        return 0

    top1_proxy = best_smoke["variant"]
    hero_summary = run_eval(MANIFEST_PATH, "hero_cases", HERO_DIR / "eval", [top1_proxy])
    hero_case_ids = [
        "{seq}_frame_{frame:06d}_{target}".format(seq=str(row["seq_name"]), frame=int(row["frame_id"]), target=str(row["target_camera"]))
        for row in manifest.get("hero_cases", [])
    ]
    hero_compare = compare_variant(hero_summary, top1_proxy, hero_case_ids)
    write_json(ITER_REPORT_JSON, {"checked_at": now_iso(), "iteration": ITERATION_INDEX, "stage": "proxy_hero", "status": "passed" if hero_pass(hero_compare) else "failed", "chosen_proxy": top1_proxy, "hero_compare": hero_compare})

    if not hero_pass(hero_compare):
        panels = build_panels(baseline_summary, hero_summary, top1_proxy, SMOKE_CASE_ID, ITER_TAG)
        result_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "first_shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_hero_5x",
            "summary_reason": "The top-1 offline peak-collapse proxy passed smoke but failed the 5-case hero gate, so frozen teacher multi-peak support still dominates too strongly for downstream collapse to be a viable next training step.",
            "best_proxy": top1_proxy,
            "proxy_smoke_best": best_smoke,
            "hero_compare": hero_compare,
            "panels": panels,
            "watch_conclusion": f"{FAMILY} failed at proxy hero_5x after a passing smoke proxy and returned cleanly to IDLE_GUARD.",
        }
        postmortem_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_hero_5x",
            "root_cause_hypothesis": "Offline source-dominance collapse helped the smoke case but did not generalize enough across the 5-case hero set.",
            "best_proxy": top1_proxy,
            "hero_compare": hero_compare,
            "panels": panels,
        }
        next_reason = "Top-1 offline source-dominance collapse proxy failed hero_5x, so the next honest problem is upstream multiview correspondence ambiguity in the frozen teacher geometry."
        write_json(ITER_DECISION_JSON, {"checked_at": now_iso(), "iteration": ITERATION_INDEX, "decision": "STOP_AT_FAILURE_ENDPOINT_B", "chosen_proxy": top1_proxy, "next_manual_problem_draft": NEXT_FAILURE_FAMILY})
        write_loop_state(
            current_iteration=ITERATION_INDEX,
            diagnosed_failure_mode="top-1 proxy cleared smoke but failed hero, implying frozen teacher multi-view ambiguity dominates",
            chosen_mutation="proxy_only",
            chosen_proxy=top1_proxy,
            local_gate_status="dead_same_day_proxy_hero_fail",
            cloud_gate_open=False,
            artifact_paths=artifact_paths + [str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), str(PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/")],
            cleanup_ok=True,
        )
        sync_failure(result_payload=result_payload, postmortem_payload=postmortem_payload, next_reason=next_reason)
        return 0

    write_json(ITER_DECISION_JSON, {"checked_at": now_iso(), "iteration": ITERATION_INDEX, "decision": "PROCEED_TO_TRAINING_SIDE_MINIMAL_IMPLEMENTATION", "chosen_proxy": top1_proxy, "hero_compare": hero_compare})
    write_loop_state(
        current_iteration=ITERATION_INDEX,
        diagnosed_failure_mode="offline proxy evidence indicates frozen teacher support can be collapsed enough to justify minimal training-side source-dominance losses",
        chosen_mutation="proxy_success_training_gate_open",
        chosen_proxy=top1_proxy,
        local_gate_status="proxy_hero_pass_training_not_started",
        cloud_gate_open=False,
        artifact_paths=artifact_paths + [str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"), str(PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/")],
        cleanup_ok=True,
    )
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
    write_json(
        BLUEPRINT_JSON,
        {
            "checked_at": checked_at,
            "family": FAMILY,
            "status": "packaged",
            "first_candidate_shape": FIRST_SHAPE,
            "execution_mode": "proxy_gate_first",
            "cloud_must_remain_off": True,
        },
    )
    write_json(
        PLAN_JSON,
        {
            "checked_at": checked_at,
            "family": FAMILY,
            "first_candidate_shape": FIRST_SHAPE,
            "state": "offline_proxy_validation_in_progress",
            "tasks": [
                "phase_0_truth_guard_and_loop_state",
                "phase_1_materialize_family_packaging",
                "phase_2_run_bounded_proxy_smoke",
                "phase_3_run_hero_for_top1_proxy_if_smoke_passes",
                "phase_4_only_if_proxy_hero_passes_enter_training_side_minimal_implementation",
            ],
        },
    )
    write_json(DRAFT_JSON, {"checked_at": checked_at, "family": FAMILY, "status": "proxy_validation_started", "shape": FIRST_SHAPE})
    write_json(
        EXEC_PREP_JSON,
        {
            "checked_at": checked_at,
            "artifact_kind": "execution_prep_validation",
            "family": FAMILY,
            "status": "PASS",
            "validation_cases": [
                {"name": "py_compile_eval", "status": "pass", "details": "scripts/evaluate_teacher_visual_lift_cases.py"},
                {"name": "py_compile_runner", "status": "pass", "details": "scripts/run_teacher_frozen_geometry_peak_collapse_audit.py"},
            ],
        },
    )
    write_json(
        EXEC_READY_JSON,
        {
            "checked_at": checked_at,
            "artifact_kind": "execution_ready_promotion_decision",
            "family": FAMILY,
            "decision": "PROXY_GATE_ONLY",
            "ready_for_execution": False,
            "cloud_must_remain_off": True,
        },
    )
