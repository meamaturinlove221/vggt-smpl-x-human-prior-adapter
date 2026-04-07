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
FAMILY = "teacher_geometry_multiview_correspondence_audit"
FIRST_SHAPE = "stablelead_teachercorr_consensus_medoid_maskedhuman_v1"
PRIOR_FAILURE_FAMILY = "teacher_frozen_geometry_peak_collapse_audit"
NEXT_FAILURE_FAMILY = "teacher_geometry_source_selection_audit"
PREFERRED_PYTHON = REPO_ROOT / ".venv5080" / "Scripts" / "python.exe"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"
MANIFEST_PATH = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "benchmark_manifest.20260403.json"
BASELINE_REFERENCE_JSON = RESEARCH_ROOT / "human_fg_support_concentration_rebalancing_baseline_reference.json"
TEACHER_CHECKPOINT = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "teacher_checkpoint" / "checkpoint.pt"

TASK_ROOT = REPO_ROOT / "output" / FAMILY / f"task.{DATE_TAG}"
SMOKE_DIR = TASK_ROOT / "smoke"
HERO_DIR = TASK_ROOT / "hero"
PANELS_DIR = TASK_ROOT / "advisor_panels"

LOOP_STATE_JSON = RESEARCH_ROOT / f"{FAMILY}_loop_state.json"
ITER_REPORT_JSON = RESEARCH_ROOT / f"{FAMILY}_iteration_report.{ITER_TAG}.json"
ITER_DECISION_JSON = RESEARCH_ROOT / f"{FAMILY}_iteration_decision.{ITER_TAG}.json"
PROXY_SWEEP_JSON = RESEARCH_ROOT / f"teacher_geometry_correspondence_proxy_sweep.{ITER_TAG}.json"
PROXY_RANKING_JSON = RESEARCH_ROOT / f"teacher_geometry_correspondence_proxy_ranking.{ITER_TAG}.json"
PROXY_BEST_JSON = RESEARCH_ROOT / f"teacher_geometry_correspondence_proxy_best.{ITER_TAG}.json"
RESULT_JSON = RESEARCH_ROOT / f"{FAMILY}_result.json"
POSTMORTEM_JSON = RESEARCH_ROOT / f"{FAMILY}_postmortem.json"
TRUTH_REPAIR_JSON = RESEARCH_ROOT / f"truth_repair_report.{FAMILY}.json"
NEXT_DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{NEXT_FAILURE_FAMILY}.json"

SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json"
DRAFT_JSON = RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json"
EXEC_PREP_JSON = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.json"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.json"

PRIOR_RESULT_JSON = RESEARCH_ROOT / "teacher_frozen_geometry_peak_collapse_audit_result.json"
PRIOR_POSTMORTEM_JSON = RESEARCH_ROOT / "teacher_frozen_geometry_peak_collapse_audit_postmortem.json"
PRIOR_LOOP_STATE_JSON = RESEARCH_ROOT / "teacher_frozen_geometry_peak_collapse_audit_loop_state.json"
PRIOR_PROXY_BEST_JSON = RESEARCH_ROOT / "teacher_peak_collapse_proxy_best.iter01.json"
PRIOR_PROXY_RANKING_JSON = RESEARCH_ROOT / "teacher_peak_collapse_proxy_ranking.iter01.json"
PRIOR_PROXY_SWEEP_JSON = RESEARCH_ROOT / "teacher_peak_collapse_proxy_sweep.iter01.json"
PRIOR_ITER_REPORT_JSON = RESEARCH_ROOT / "teacher_frozen_geometry_peak_collapse_audit_iteration_report.iter01.json"
PRIOR_ITER_DECISION_JSON = RESEARCH_ROOT / "teacher_frozen_geometry_peak_collapse_audit_iteration_decision.iter01.json"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
APPROVED_PROBLEM_JSON = RESEARCH_ROOT / "approved_problem.json"

SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"
PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
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


def load_required_inputs() -> dict:
    required_image_paths = [
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "target_baseline_geometry_candidate.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "target_baseline_render_geometry_candidate.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "candidate_support_triptych.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "candidate_support_inside_outside.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "candidate_source_dominance_triptych.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "candidate_bg_nonblack_heatmap.iter01.png",
        REPO_ROOT / "output" / PRIOR_FAILURE_FAMILY / "task.20260407" / "advisor_panels" / "baseline_vs_candidate_bottom_band_support.iter01.png",
    ]
    for path in required_image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required audit image: {path}")

    payload = {
        "prior_result": load_json(PRIOR_RESULT_JSON),
        "prior_postmortem": load_json(PRIOR_POSTMORTEM_JSON),
        "prior_loop_state": load_json(PRIOR_LOOP_STATE_JSON),
        "prior_proxy_best": load_json(PRIOR_PROXY_BEST_JSON),
        "prior_proxy_ranking": load_json(PRIOR_PROXY_RANKING_JSON),
        "prior_proxy_sweep": load_json(PRIOR_PROXY_SWEEP_JSON),
        "prior_iteration_report": load_json(PRIOR_ITER_REPORT_JSON),
        "prior_iteration_decision": load_json(PRIOR_ITER_DECISION_JSON),
        "research_status": load_json(RESEARCH_STATUS_JSON),
        "watch": load_json(LATEST_WATCH_JSON),
        "task_plan": load_json(TASK_PLAN_JSON),
        "allowlist": load_json(ALLOWLIST_JSON),
        "required_images": [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in required_image_paths],
    }
    checks = {
        "state_idle_guard": payload["research_status"].get("state") == "IDLE_GUARD",
        "approved_problem_present_false": payload["research_status"].get("approved_problem_present") is False,
        "allowlist_empty": payload["allowlist"].get("status") == "idle_empty_allowlist" and not payload["allowlist"].get("allowed_markers"),
        "active_modal_zero": len(payload["watch"].get("modal_apps", [])) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 0 truth guard failed: {checks}")
    payload["checks"] = checks
    return payload


def truth_repair() -> dict:
    checked_at = now_iso()
    research = load_json(RESEARCH_STATUS_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    watch = load_json(LATEST_WATCH_JSON)

    formal_verdict = deepcopy(research.get("latest_formal_verdict", {}))
    formal_verdict["family"] = PRIOR_FAILURE_FAMILY
    formal_verdict["status"] = "dead_same_day"
    formal_verdict["gate_stage_reached"] = "proxy_smoke_1x1"

    research.update(
        {
            "checked_at": checked_at,
            "state": "IDLE_GUARD",
            "approved_problem_present": False,
            "approved_problem_ready": False,
            "allowed_families": [],
            "current_priority_family": "",
            "current_priority_candidate_shape": "",
            "current_priority_candidate_config": "",
            "current_priority_reason": (
                f"{PRIOR_FAILURE_FAMILY} is formally closed at proxy_smoke_1x1; "
                f"the next honest manual direction is {FAMILY}."
            ),
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": (
                f"{PRIOR_FAILURE_FAMILY} already consumed its bounded proxy family budget at proxy_smoke_1x1; "
                "do not retry the same family."
            ),
            "manual_action_required": False,
            "manual_action_kind": "",
            "ready_for_execution": False,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "cloud_must_remain_off": True,
            "next_requirement": (
                f"Start a fresh manual problem for {FAMILY}. Keep cloud off until a bounded local 20-case pass opens a later cloud decision."
            ),
            "latest_formal_verdict": formal_verdict,
        }
    )
    write_json(RESEARCH_STATUS_JSON, research)

    task_plan.update(
        {
            "checked_at": checked_at,
            "task_mode_status": "in_progress",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": f"{FAMILY}_packaging_and_proxy_validation",
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
    task_plan["current_state_notes"] = [
        f"formal verdict remains {PRIOR_FAILURE_FAMILY} dead_same_day at proxy_smoke_1x1",
        f"next manual direction: {FAMILY}",
    ]
    task_plan["summary_conclusion"] = [
        f"truth repaired to {FAMILY} start boundary",
        "current active family: none",
        "cloud gate remains off",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2))
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]))

    watch["checked_at"] = checked_at
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["research"] = deepcopy(watch.get("research", {}))
    watch["research"]["summary"] = {
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "manual_action_required": False,
        "manual_action_kind": "",
        "ready_for_execution": False,
        "current_review_packet": str(DRAFT_JSON.relative_to(REPO_ROOT)).replace("/", "\\"),
    }
    watch["research"]["research_status"] = research
    watch["research"]["allowlist"] = load_json(ALLOWLIST_JSON)
    watch["watch_conclusion"] = (
        f"{PRIOR_FAILURE_FAMILY} remains formally closed; {FAMILY} is now the next manual direction with cloud off."
    )
    write_json(LATEST_WATCH_JSON, watch)

    payload = {
        "checked_at": checked_at,
        "prior_failure_family": PRIOR_FAILURE_FAMILY,
        "formal_verdict_family": formal_verdict["family"],
        "formal_verdict_status": formal_verdict["status"],
        "formal_verdict_gate": formal_verdict["gate_stage_reached"],
        "current_family": "",
        "current_priority_family": "",
        "next_manual_direction": FAMILY,
        "allowlist_empty": True,
        "cloud_gate_open": False,
        "repaired_fields": [
            "research_loop_status.same_family_retry_reason",
            "research_loop_status.current_priority_reason",
            "latest_watch_snapshot.research.summary.current_review_packet",
            "task_plan.current_state_notes",
            "task_plan.summary_conclusion",
        ],
    }
    write_json(TRUTH_REPAIR_JSON, payload)
    return payload


def materialize_packaging() -> None:
    checked_at = now_iso()
    payloads = {
        SEED_JSON: {
            "checked_at": checked_at,
            "problem_id": f"{FAMILY}_v1",
            "family": FAMILY,
            "problem_statement": (
                "Current frozen teacher geometry still shows upstream multiview correspondence ambiguity inside the human fg region. "
                "This family audits whether consensus-preserving source agreement proxies can collapse the teacher support toward a single human without sacrificing coverage."
            ),
            "first_candidate_shape": FIRST_SHAPE,
            "execution_mode": "offline_correspondence_proxy_then_minimal_training_if_local_passes",
            "cloud_must_remain_off": True,
            "allowed_write_surface": [
                "scripts/evaluate_teacher_visual_lift_cases.py",
                "scripts/run_teacher_geometry_multiview_correspondence_audit.py",
            ],
            "forbidden_actions": [
                "no cousin sweep",
                "no second ticket",
                "no cloud before local 20-case pass",
                "no dataset/trainer changes in phase 1",
            ],
        },
        BLUEPRINT_JSON: {
            "checked_at": checked_at,
            "family": FAMILY,
            "first_candidate_shape": FIRST_SHAPE,
            "status": "packaged",
            "execution_mode": "correspondence_proxy_first",
            "cloud_must_remain_off": True,
        },
        PLAN_JSON: {
            "checked_at": checked_at,
            "family": FAMILY,
            "first_candidate_shape": FIRST_SHAPE,
            "state": "offline_correspondence_proxy_validation",
            "tasks": [
                "phase_0_truth_guard_and_loop_state",
                "phase_1_materialize_family_packaging",
                "phase_2_run_bounded_correspondence_proxy_smoke",
                "phase_3_run_hero_for_top1_proxy_if_smoke_passes",
                "phase_4_only_if_proxy_hero_passes_enter_training_side_minimal_implementation",
            ],
        },
        DRAFT_JSON: {
            "checked_at": checked_at,
            "family": FAMILY,
            "status": "proxy_validation_started",
            "shape": FIRST_SHAPE,
            "problem_statement": (
                "Audit whether frozen teacher geometry has upstream correspondence ambiguity that forces downstream support collapse to choose between human coverage and single-human concentration."
            ),
        },
        EXEC_PREP_JSON: {
            "checked_at": checked_at,
            "artifact_kind": "execution_prep_validation",
            "family": FAMILY,
            "status": "PASS",
            "validation_cases": [
                {"name": "py_compile_eval", "status": "pass", "details": "scripts/evaluate_teacher_visual_lift_cases.py"},
                {"name": "py_compile_runner", "status": "pass", "details": "scripts/run_teacher_geometry_multiview_correspondence_audit.py"},
            ],
        },
        EXEC_READY_JSON: {
            "checked_at": checked_at,
            "artifact_kind": "execution_ready_promotion_decision",
            "family": FAMILY,
            "decision": "PROXY_GATE_ONLY",
            "ready_for_execution": False,
            "cloud_must_remain_off": True,
            "reason": "Do not open training or cloud until correspondence proxy smoke and hero pass.",
        },
    }
    for path, payload in payloads.items():
        write_json(path, payload)


def write_loop_state(
    *,
    current_iteration: int,
    diagnosed_failure_mode: str,
    chosen_stage: str,
    chosen_mutation: str,
    chosen_candidate: str,
    local_gate_status: str,
    cloud_gate_open: bool,
    cleanup_ok: bool,
    artifact_paths: list[str],
) -> dict:
    payload = {
        "checked_at": now_iso(),
        "current_iteration": int(current_iteration),
        "prior_failure_family": PRIOR_FAILURE_FAMILY,
        "diagnosed_failure_mode": diagnosed_failure_mode,
        "chosen_stage": chosen_stage,
        "chosen_mutation": chosen_mutation,
        "chosen_candidate": chosen_candidate,
        "local_gate_status": local_gate_status,
        "cloud_gate_open": bool(cloud_gate_open),
        "cleanup_ok": bool(cleanup_ok),
        "artifact_paths": artifact_paths,
    }
    write_json(LOOP_STATE_JSON, payload)
    return payload


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
        ],
        cwd=REPO_ROOT,
    )
    payload = load_json(output_dir / "summary.json")
    payload["_root"] = output_dir
    return payload


def rows_by_variant(summary: dict, variant: str) -> dict[str, dict]:
    return {str(row["case_id"]): row for row in summary.get("rows", []) if str(row.get("variant")) == variant}


def compare_variant(summary: dict, variant: str, case_ids: list[str], *, smoke_mode: bool) -> dict:
    baseline = rows_by_variant(summary, "baseline_depth_unproject")
    candidate = rows_by_variant(summary, variant)
    rows = []
    for case_id in case_ids:
        base_row = baseline[case_id]
        cand_row = candidate[case_id]
        support_base = base_row["support_metrics"]
        support_cand = cand_row["support_metrics"]
        row = {
            "case_id": case_id,
            "delta_fg_connected_components": support_cand["fg_connected_components"] - support_base["fg_connected_components"],
            "delta_fg_peak_count": support_cand["fg_peak_count"] - support_base["fg_peak_count"],
            "delta_masked_l1": cand_row["metrics"]["fg_masked"]["l1"] - base_row["metrics"]["fg_masked"]["l1"],
            "delta_masked_ssim": cand_row["metrics"]["fg_masked"]["ssim"] - base_row["metrics"]["fg_masked"]["ssim"],
            "delta_support_inside_fg_ratio": support_cand["support_inside_fg_ratio"] - support_base["support_inside_fg_ratio"],
            "delta_off_body_support_ratio": support_cand["off_body_support_ratio"] - support_base["off_body_support_ratio"],
            "delta_off_body_nonblack_ratio": support_cand["off_body_nonblack_ratio"] - support_base["off_body_nonblack_ratio"],
            "delta_bg_bottom_support_ratio": support_cand["bg_bottom_support_ratio"] - support_base["bg_bottom_support_ratio"],
            "delta_source_entropy_inside_fg": support_cand["source_entropy_inside_fg"] - support_base["source_entropy_inside_fg"],
            "delta_source_top1_top2_margin_inside_fg": support_cand["source_top1_top2_margin_inside_fg"] - support_base["source_top1_top2_margin_inside_fg"],
            "delta_correspondence_consensus_ratio_inside_fg": support_cand["correspondence_consensus_ratio_inside_fg"] - support_base["correspondence_consensus_ratio_inside_fg"],
            "delta_source_label_smoothness_inside_fg": support_cand["source_label_smoothness_inside_fg"] - support_base["source_label_smoothness_inside_fg"],
            "delta_fg_largest_component_ratio": support_cand["fg_largest_component_ratio"] - support_base["fg_largest_component_ratio"],
            "delta_source_id_switch_count_inside_fg": support_cand["source_id_switch_count_inside_fg"] - support_base["source_id_switch_count_inside_fg"],
            "delta_source_top1_spatial_fragmentation": support_cand["source_top1_spatial_fragmentation"] - support_base["source_top1_spatial_fragmentation"],
            "delta_fg_bbox_cover_ratio": support_cand["fg_bbox_cover_ratio"] - support_base["fg_bbox_cover_ratio"],
            "fg_mask_coverage_ratio": support_cand["fg_mask_coverage_ratio"],
            "fg_retained_area_ratio": support_cand["fg_retained_area_ratio"],
            "fg_retained_support_area_ratio": support_cand["fg_retained_support_area_ratio"],
            "fg_retained_mass_ratio": support_cand["fg_retained_mass_ratio"],
            "human_erasure_penalty": support_cand["human_erasure_penalty"],
        }
        coverage_pass = row["fg_mask_coverage_ratio"] >= (0.70 if smoke_mode else 0.75)
        retained_area_pass = row["fg_retained_area_ratio"] >= (0.70 if smoke_mode else 0.75)
        retained_support_pass = row["fg_retained_support_area_ratio"] >= (0.60 if smoke_mode else 0.65)
        retained_mass_pass = row["fg_retained_mass_ratio"] >= (0.60 if smoke_mode else 0.65)
        anti_erasure_pass = row["human_erasure_penalty"] <= (0.02 if smoke_mode else 0.01)
        row["primary_pass"] = (
            row["delta_fg_connected_components"] < 0
            and row["delta_fg_peak_count"] < 0
            and row["delta_masked_l1"] <= 0
            and row["delta_masked_ssim"] >= 0
            and row["delta_support_inside_fg_ratio"] > 0
            and row["delta_off_body_support_ratio"] <= 0
            and row["delta_off_body_nonblack_ratio"] <= 0
            and row["delta_source_entropy_inside_fg"] < 0
            and row["delta_source_top1_top2_margin_inside_fg"] > 0
            and row["delta_source_label_smoothness_inside_fg"] > 0
            and row["delta_correspondence_consensus_ratio_inside_fg"] > 0
            and row["delta_source_id_switch_count_inside_fg"] < 0
            and row["delta_source_top1_spatial_fragmentation"] < 0
            and coverage_pass
            and retained_area_pass
            and retained_support_pass
            and retained_mass_pass
            and anti_erasure_pass
        )
        if smoke_mode:
            row["primary_pass"] = row["primary_pass"] and row["delta_fg_peak_count"] <= -1
        rows.append(row)

    count = max(len(rows), 1)
    return {
        "variant": variant,
        "case_count": len(rows),
        "improved_all_primary_count": int(sum(1 for row in rows if row["primary_pass"])),
        "mean_delta_fg_connected_components": float(sum(row["delta_fg_connected_components"] for row in rows) / count),
        "mean_delta_fg_peak_count": float(sum(row["delta_fg_peak_count"] for row in rows) / count),
        "mean_delta_masked_l1": float(sum(row["delta_masked_l1"] for row in rows) / count),
        "mean_delta_masked_ssim": float(sum(row["delta_masked_ssim"] for row in rows) / count),
        "mean_delta_support_inside_fg_ratio": float(sum(row["delta_support_inside_fg_ratio"] for row in rows) / count),
        "mean_delta_off_body_support_ratio": float(sum(row["delta_off_body_support_ratio"] for row in rows) / count),
        "mean_delta_off_body_nonblack_ratio": float(sum(row["delta_off_body_nonblack_ratio"] for row in rows) / count),
        "mean_delta_bg_bottom_support_ratio": float(sum(row["delta_bg_bottom_support_ratio"] for row in rows) / count),
        "mean_delta_source_entropy_inside_fg": float(sum(row["delta_source_entropy_inside_fg"] for row in rows) / count),
        "mean_delta_source_top1_top2_margin_inside_fg": float(sum(row["delta_source_top1_top2_margin_inside_fg"] for row in rows) / count),
        "mean_delta_correspondence_consensus_ratio_inside_fg": float(sum(row["delta_correspondence_consensus_ratio_inside_fg"] for row in rows) / count),
        "mean_delta_source_label_smoothness_inside_fg": float(sum(row["delta_source_label_smoothness_inside_fg"] for row in rows) / count),
        "mean_delta_source_id_switch_count_inside_fg": float(sum(row["delta_source_id_switch_count_inside_fg"] for row in rows) / count),
        "mean_delta_source_top1_spatial_fragmentation": float(sum(row["delta_source_top1_spatial_fragmentation"] for row in rows) / count),
        "mean_fg_largest_component_ratio_delta": float(sum(row["delta_fg_largest_component_ratio"] for row in rows) / count),
        "mean_fg_mask_coverage_ratio": float(sum(row["fg_mask_coverage_ratio"] for row in rows) / count),
        "mean_fg_retained_area_ratio": float(sum(row["fg_retained_area_ratio"] for row in rows) / count),
        "mean_fg_retained_support_area_ratio": float(sum(row["fg_retained_support_area_ratio"] for row in rows) / count),
        "mean_fg_retained_mass_ratio": float(sum(row["fg_retained_mass_ratio"] for row in rows) / count),
        "mean_human_erasure_penalty": float(sum(row["human_erasure_penalty"] for row in rows) / count),
        "mean_delta_fg_bbox_cover_ratio": float(sum(row["delta_fg_bbox_cover_ratio"] for row in rows) / count),
        "rows": rows,
    }


def smoke_pass(compare: dict) -> bool:
    return compare["case_count"] == 1 and bool(compare["rows"][0]["primary_pass"])


def hero_pass(compare: dict) -> bool:
    return (
        compare["improved_all_primary_count"] >= 3
        and compare["mean_delta_fg_connected_components"] < 0
        and compare["mean_delta_fg_peak_count"] < 0
        and compare["mean_delta_masked_l1"] <= 0
        and compare["mean_delta_masked_ssim"] >= 0
        and compare["mean_fg_mask_coverage_ratio"] >= 0.75
        and compare["mean_fg_retained_area_ratio"] >= 0.75
        and compare["mean_fg_retained_support_area_ratio"] >= 0.65
        and compare["mean_fg_retained_mass_ratio"] >= 0.65
        and compare["mean_human_erasure_penalty"] <= 0.01
        and compare["mean_delta_correspondence_consensus_ratio_inside_fg"] > 0
        and compare["mean_delta_source_label_smoothness_inside_fg"] > 0
        and compare["mean_delta_source_top1_top2_margin_inside_fg"] > 0
        and compare["mean_delta_source_id_switch_count_inside_fg"] < 0
        and compare["mean_delta_source_top1_spatial_fragmentation"] < 0
        and compare["mean_delta_support_inside_fg_ratio"] > 0
        and compare["mean_delta_off_body_support_ratio"] <= 0
    )


def proxy_rank_key(compare: dict) -> tuple:
    return (
        0 if smoke_pass(compare) else 1,
        -compare["improved_all_primary_count"],
        compare["mean_delta_fg_connected_components"],
        compare["mean_delta_fg_peak_count"],
        compare["mean_delta_masked_l1"],
        -compare["mean_delta_masked_ssim"],
        -compare["mean_fg_mask_coverage_ratio"],
        -compare["mean_fg_retained_area_ratio"],
        -compare["mean_fg_retained_support_area_ratio"],
        -compare["mean_fg_retained_mass_ratio"],
        compare["mean_human_erasure_penalty"],
        -compare["mean_delta_correspondence_consensus_ratio_inside_fg"],
        -compare["mean_delta_source_label_smoothness_inside_fg"],
        compare["mean_delta_source_id_switch_count_inside_fg"],
        compare["mean_delta_source_top1_spatial_fragmentation"],
        compare["mean_delta_off_body_support_ratio"],
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
    candidate_baseline_geometry = rows_by_variant(candidate_summary, "baseline_depth_unproject")[case_id]
    baseline_geometry = candidate_baseline_geometry
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
            (Path(candidate_summary["_root"]) / candidate["files"]["correspondence_consensus_png"], "Consensus inside fg"),
        ],
        support_triptych,
    )
    support_quad = PANELS_DIR / f"candidate_support_inside_outside.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate["files"]["support_inside_fg_png"], "Support inside fg"),
            (Path(candidate_summary["_root"]) / candidate["files"]["support_outside_fg_png"], "Support outside fg"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_label_smoothness_png"], "Source label smoothness"),
            (Path(candidate_summary["_root"]) / candidate["files"]["bg_bottom_support_png"], "Bottom-band support"),
        ],
        support_quad,
        width=360,
    )
    source_triptych = PANELS_DIR / f"candidate_source_dominance_triptych.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate["files"]["source_medoid_support_png"], "Source medoid support"),
            (Path(candidate_summary["_root"]) / candidate["files"]["correspondence_consensus_png"], "Correspondence consensus"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_label_smoothness_png"], "Source label smoothness"),
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
            (Path(candidate_summary["_root"]) / candidate_baseline_geometry["files"]["bg_bottom_support_png"], "Frozen baseline bottom-band support"),
            (Path(candidate_summary["_root"]) / candidate["files"]["bg_bottom_support_png"], "Candidate bottom-band support"),
        ],
        bottom_compare,
        width=520,
    )
    coverage_panel = PANELS_DIR / f"candidate_fg_coverage_panel.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate_baseline_geometry["files"]["fg_coverage_overlay_png"], "Frozen baseline fg coverage"),
            (Path(candidate_summary["_root"]) / candidate["files"]["fg_coverage_overlay_png"], "Candidate fg coverage"),
        ],
        coverage_panel,
        width=520,
    )
    source_label_panel = PANELS_DIR / f"candidate_source_label_map.{suffix}.png"
    save_row_panel(
        [
            (Path(candidate_summary["_root"]) / candidate_baseline_geometry["files"]["source_label_map_png"], "Frozen baseline source labels"),
            (Path(candidate_summary["_root"]) / candidate["files"]["source_label_map_png"], "Candidate source labels"),
        ],
        source_label_panel,
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
        "fg_coverage_panel": str(coverage_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_label_panel": str(source_label_panel.relative_to(REPO_ROOT)).replace("\\", "/"),
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
            "same_family_retry_forbidden": True,
            "same_family_retry_reason": f"{FAMILY} already exhausted its bounded offline proxy family budget this round.",
            "next_requirement": f"Return to IDLE_GUARD. The next honest move is {NEXT_FAILURE_FAMILY}.",
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
            "suggested_problem_statement": "Audit whether source selection, rather than local source-dominance collapse, is the upstream bottleneck in the frozen teacher geometry.",
        },
    )


def main() -> int:
    ensure_dir(TASK_ROOT)
    ensure_dir(PANELS_DIR)
    load_required_inputs()
    py_compile([EVAL_SCRIPT, Path(__file__)])
    truth_repair()
    materialize_packaging()

    baseline_reference = load_json(BASELINE_REFERENCE_JSON)
    baseline_summary_root = REPO_ROOT / baseline_reference["summary_root"]
    baseline_summary = load_json(baseline_summary_root / "summary.json")
    baseline_summary["_root"] = baseline_summary_root

    artifact_paths = [
        str(PRIOR_RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_POSTMORTEM_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_LOOP_STATE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_ITER_REPORT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(PRIOR_ITER_DECISION_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(TRUTH_REPAIR_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]
    write_loop_state(
        current_iteration=ITERATION_INDEX,
        diagnosed_failure_mode=(
            "teacher_frozen_geometry_peak_collapse_audit proved that stronger collapse without correspondence agreement either keeps the body fragmented or deletes most of it"
        ),
        chosen_stage="offline_correspondence_proxy",
        chosen_mutation="proxy_only",
        chosen_candidate=FIRST_SHAPE,
        local_gate_status="proxy_smoke_pending",
        cloud_gate_open=False,
        cleanup_ok=True,
        artifact_paths=artifact_paths,
    )

    manifest = load_json(MANIFEST_PATH)
    smoke_manifest_path = TASK_ROOT / "smoke_case_manifest.json"
    build_single_case_manifest(manifest, SMOKE_CASE_ID, smoke_manifest_path)

    smoke_summary = run_eval(smoke_manifest_path, "cases", SMOKE_DIR / "eval", PROXY_VARIANTS)
    smoke_compares = [compare_variant(smoke_summary, variant, [SMOKE_CASE_ID], smoke_mode=True) for variant in PROXY_VARIANTS]
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
            "summary_reason": "All four bounded correspondence proxies failed the 1-sample smoke gate, so frozen teacher geometry could not improve single-human concentration without losing either coverage or masked quality.",
            "best_proxy": best_smoke,
            "proxy_ranking_artifact": str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "panels": panels,
            "watch_conclusion": f"{FAMILY} exhausted all four offline correspondence proxies at smoke_1x1 and returned cleanly to IDLE_GUARD.",
        }
        postmortem_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_smoke_1x1",
            "root_cause_hypothesis": "Frozen teacher geometry retains upstream multiview correspondence ambiguity: agreement-preserving collapse cannot improve concentration without paying too much in coverage or masked quality.",
            "proxy_ranking": smoke_compares,
            "best_proxy": best_smoke,
            "panels": panels,
        }
        next_reason = (
            "Frozen teacher geometry failed all four bounded correspondence proxies at smoke, so the next honest question is whether source selection itself is ambiguous upstream."
        )
        write_json(
            ITER_REPORT_JSON,
            {
                "checked_at": now_iso(),
                "iteration": ITERATION_INDEX,
                "stage": "proxy_smoke",
                "status": "dead_same_day",
                "diagnosed_failure_mode": "offline correspondence proxies cannot improve concentration while preserving masked quality and human coverage on the smoke case",
                "best_proxy": best_smoke,
            },
        )
        write_json(
            ITER_DECISION_JSON,
            {
                "checked_at": now_iso(),
                "iteration": ITERATION_INDEX,
                "decision": "STOP_AT_FAILURE_ENDPOINT_B",
                "chosen_proxy": best_smoke["variant"],
                "next_manual_problem_draft": NEXT_FAILURE_FAMILY,
            },
        )
        write_loop_state(
            current_iteration=ITERATION_INDEX,
            diagnosed_failure_mode="all bounded offline correspondence proxies failed smoke on frozen teacher geometry",
            chosen_stage="offline_correspondence_proxy",
            chosen_mutation="proxy_only",
            chosen_candidate=best_smoke["variant"],
            local_gate_status="dead_same_day_proxy_smoke_fail",
            cloud_gate_open=False,
            cleanup_ok=True,
            artifact_paths=artifact_paths
            + [
                str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                str(PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            ],
        )
        sync_failure(result_payload=result_payload, postmortem_payload=postmortem_payload, next_reason=next_reason)
        return 0

    top1_proxy = smoke_passers[0]["variant"]
    hero_summary = run_eval(MANIFEST_PATH, "hero_cases", HERO_DIR / "eval", [top1_proxy])
    hero_case_ids = [
        "{seq}_frame_{frame:06d}_{target}".format(seq=str(row["seq_name"]), frame=int(row["frame_id"]), target=str(row["target_camera"]))
        for row in manifest.get("hero_cases", [])
    ]
    hero_compare = compare_variant(hero_summary, top1_proxy, hero_case_ids, smoke_mode=False)
    write_json(
        ITER_REPORT_JSON,
        {
            "checked_at": now_iso(),
            "iteration": ITERATION_INDEX,
            "stage": "proxy_hero",
            "status": "passed" if hero_pass(hero_compare) else "failed",
            "chosen_proxy": top1_proxy,
            "hero_compare": hero_compare,
        },
    )
    if not hero_pass(hero_compare):
        panels = build_panels(baseline_summary, hero_summary, top1_proxy, SMOKE_CASE_ID, ITER_TAG)
        result_payload = {
            "checked_at": now_iso(),
            "family": FAMILY,
            "first_shape": FIRST_SHAPE,
            "status": "dead_same_day",
            "gate_stage_reached": "proxy_hero_5x",
            "summary_reason": "The top-1 correspondence proxy cleared smoke but failed the 5-case hero gate, so frozen teacher multiview disagreement still dominates too strongly for a local downstream fix to be the next honest move.",
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
            "root_cause_hypothesis": "Agreement-preserving correspondence collapse helped the smoke case but did not generalize across the 5-case hero set without sacrificing some core objective.",
            "best_proxy": top1_proxy,
            "hero_compare": hero_compare,
            "panels": panels,
        }
        next_reason = "The best agreement-preserving correspondence proxy failed hero_5x, so the next honest problem is upstream source selection ambiguity."
        write_json(
            ITER_DECISION_JSON,
            {
                "checked_at": now_iso(),
                "iteration": ITERATION_INDEX,
                "decision": "STOP_AT_FAILURE_ENDPOINT_B",
                "chosen_proxy": top1_proxy,
                "next_manual_problem_draft": NEXT_FAILURE_FAMILY,
            },
        )
        write_loop_state(
            current_iteration=ITERATION_INDEX,
            diagnosed_failure_mode="top-1 correspondence proxy cleared smoke but failed hero, implying frozen teacher multiview ambiguity dominates",
            chosen_stage="offline_correspondence_proxy",
            chosen_mutation="proxy_only",
            chosen_candidate=top1_proxy,
            local_gate_status="dead_same_day_proxy_hero_fail",
            cloud_gate_open=False,
            cleanup_ok=True,
            artifact_paths=artifact_paths
            + [
                str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                str(PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            ],
        )
        sync_failure(result_payload=result_payload, postmortem_payload=postmortem_payload, next_reason=next_reason)
        return 0

    write_json(
        ITER_DECISION_JSON,
        {
            "checked_at": now_iso(),
            "iteration": ITERATION_INDEX,
            "decision": "PROCEED_TO_TRAINING_SIDE_MINIMAL_IMPLEMENTATION",
            "chosen_proxy": top1_proxy,
            "hero_compare": hero_compare,
        },
    )
    write_loop_state(
        current_iteration=ITERATION_INDEX,
        diagnosed_failure_mode="offline correspondence proxy evidence indicates a minimal training-side consensus implementation is justified",
        chosen_stage="training_side_pending",
        chosen_mutation="proxy_hero_pass",
        chosen_candidate=top1_proxy,
        local_gate_status="proxy_hero_pass_training_pending",
        cloud_gate_open=False,
        cleanup_ok=True,
        artifact_paths=artifact_paths
        + [
            str(PROXY_SWEEP_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            str(PROXY_RANKING_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            str(PROXY_BEST_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        ],
    )
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
