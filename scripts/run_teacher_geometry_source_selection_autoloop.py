import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.correspondence_mutation_bank import build_mutation_candidates
from scripts.read_latest_audit_state import collect_latest_audit_state
from scripts.score_correspondence_progress import (
    classify_failure,
    compare_variant,
    effective_progress,
    hero_accept,
    local20_accept,
    progress_key,
    smoke_accept,
)

RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
WATCH_JSON = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
TASK_PLAN_JSON = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
TASK_PLAN_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.md"
SUMMARY_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "summary.md"
RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"

AUTLOOP_ROOT = REPO_ROOT / "output" / "autoloop_teacher_geometry_source_selection"
RUNS_ROOT = AUTLOOP_ROOT / "runs"
PANELS_ROOT = AUTLOOP_ROOT / "panels"
CONFIG_ROOT = AUTLOOP_ROOT / "configs"

FAMILY = "teacher_geometry_source_selection_audit"
FIRST_SHAPE = "stablelead_sourcesel_consensus_rehydrated_maskedhuman_v1"
PRIOR_FAMILY = "teacher_geometry_multiview_correspondence_audit"
NEXT_FAMILY_BY_FAILURE = {
    "erasure_win": "teacher_geometry_rehydrated_render_audit",
    "fragmentation_win": "teacher_geometry_source_label_consistency_audit",
    "background_only_win": "teacher_geometry_poison_source_pruning_audit",
    "no_movement": "teacher_geometry_source_selection_audit_exhausted",
}
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"
BENCHMARK_MANIFEST = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "benchmark_manifest.20260403.json"
TEACHER_CHECKPOINT = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / "teacher_checkpoint" / "checkpoint.pt"
PYTHON_EXE = REPO_ROOT / ".venv5080" / "Scripts" / "python.exe"
PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
]
SMOKE_CASE_ID = "CoreView_390_frame_000600_Camera_B4"
MAX_ROUNDS = 24
MAX_ACCEPTED_MUTATIONS = 8
MAX_SOURCE_SELECTION_TRIALS = 40


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    for path in paths:
        run_checked([str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable), "-m", "py_compile", str(path)])


def case_id(row: dict) -> str:
    return "{seq}_frame_{frame:06d}_{target}".format(
        seq=str(row["seq_name"]),
        frame=int(row["frame_id"]),
        target=str(row["target_camera"]),
    )


def select_manifest_rows(manifest: dict, key: str) -> list[dict]:
    return list(manifest.get(key, []))


def find_case(manifest: dict, wanted: str) -> dict:
    for bucket in ("hero_cases", "benchmark_cases", "cases"):
        for row in manifest.get(bucket, []):
            if case_id(row) == wanted:
                return row
    raise KeyError(wanted)


def write_case_manifest(row: dict, out_path: Path) -> None:
    write_json(out_path, {"checked_at": now_iso(), "cases": [row]})


def init_best_compare(prior_best: dict) -> dict:
    return {
        "variant": prior_best.get("variant", "baseline"),
        "improved_all_primary_count": int(prior_best.get("improved_all_primary_count", 0)),
        "mean_delta_fg_connected_components": float(prior_best.get("mean_delta_fg_connected_components", 999.0)),
        "mean_delta_fg_peak_count": float(prior_best.get("mean_delta_fg_peak_count", 999.0)),
        "mean_delta_masked_l1": float(prior_best.get("mean_delta_masked_l1", 999.0)),
        "mean_delta_masked_ssim": float(prior_best.get("mean_delta_masked_ssim", -999.0)),
        "mean_delta_off_body_support_ratio": float(prior_best.get("mean_delta_off_body_support_ratio", 999.0)),
        "mean_delta_bg_bottom_support_ratio": float(prior_best.get("mean_delta_bg_bottom_support_ratio", 999.0)),
        "mean_fg_rehydrated_coverage_ratio": float(prior_best.get("mean_fg_mask_coverage_ratio", 0.0)),
        "mean_fg_retained_mass_ratio": float(prior_best.get("mean_fg_retained_mass_ratio", 0.0)),
        "mean_human_erasure_penalty": float(prior_best.get("mean_human_erasure_penalty", 999.0)),
    }


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


def compare_stage(summary: dict, case_ids: list[str]) -> list[dict]:
    rows = [compare_variant(summary, variant, case_ids) for variant in PROXY_VARIANTS]
    rows.sort(key=progress_key)
    return rows


def main() -> int:
    ensure_dir(AUTLOOP_ROOT)
    ensure_dir(RUNS_ROOT)
    ensure_dir(CONFIG_ROOT)
    state = collect_latest_audit_state()
    py_compile([EVAL_SCRIPT, Path(__file__)])

    research = state["research_status"]
    truth_repair_report = {
        "checked_at": now_iso(),
        "prior_failure_family": PRIOR_FAMILY,
        "prior_failure_status": state["correspondence_result"]["status"],
        "research_state": research["state"],
        "allowlist_status": state["allowlist"]["status"],
        "next_family": FAMILY,
        "repaired_truth": "peak-collapse failed because the best offline proxy improved concentration mainly by erasing most of the human body, not by solving multiview correspondence.",
    }
    write_json(RESEARCH_ROOT / f"truth_repair_report.{FAMILY}.json", truth_repair_report)
    loop_state_path = RESEARCH_ROOT / f"{FAMILY}_loop_state.json"
    write_json(
        loop_state_path,
        {
            "checked_at": now_iso(),
            "current_iteration": 0,
            "prior_failure_family": PRIOR_FAMILY,
            "diagnosed_failure_mode": "erasure_win",
            "chosen_stage": "autoloop_boot",
            "chosen_mutation": "none",
            "chosen_candidate": FIRST_SHAPE,
            "local_gate_status": "pending",
            "cloud_gate_open": False,
            "cleanup_ok": True,
            "artifact_paths": [rel(loop_state_path)],
        },
    )

    manifest = load_json(BENCHMARK_MANIFEST)
    smoke_row = find_case(manifest, SMOKE_CASE_ID)
    hero_rows = select_manifest_rows(manifest, "hero_cases")
    benchmark_rows = select_manifest_rows(manifest, "benchmark_cases")
    source_count = len(smoke_row["source_cameras"])

    write_json(RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.json", {"checked_at": now_iso(), "family": FAMILY, "shape": FIRST_SHAPE, "ready_for_execution": False})
    write_json(RESEARCH_ROOT / f"family_blueprint.{FAMILY}.json", {"checked_at": now_iso(), "family": FAMILY, "shape": FIRST_SHAPE, "status": "autoloop_local_only"})
    write_json(RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.json", {"checked_at": now_iso(), "family": FAMILY, "write_surface": ["scripts/evaluate_teacher_visual_lift_cases.py", "scripts/run_teacher_geometry_source_selection_autoloop.py"]})
    write_json(RESEARCH_ROOT / f"next_manual_problem_draft.{FAMILY}.json", {"checked_at": now_iso(), "family": FAMILY, "status": "autoloop_started"})

    prior_best = state["proxy_best"].get("best", state["correspondence_result"]["best_proxy"])
    best_compare = init_best_compare(prior_best)
    best_state = {"checked_at": now_iso(), "failure_class": classify_failure(best_compare), "compare": best_compare}
    best_state_key = progress_key(best_compare)
    write_json(AUTLOOP_ROOT / "best_state.json", best_state)
    write_json(AUTLOOP_ROOT / "mutation_history.json", {"checked_at": now_iso(), "mutations": []})

    tried_ids: set[str] = set()
    accepted_mutations = 0
    source_selection_trials = 0
    final_result = None
    best_subset = list(range(source_count))
    failure_class = classify_failure(best_compare)
    write_json(AUTLOOP_ROOT / "failure_classification.json", {"checked_at": now_iso(), "failure_class": failure_class, "based_on": PRIOR_FAMILY})

    for round_idx in range(1, MAX_ROUNDS + 1):
        mutations = build_mutation_candidates(
            failure_class=failure_class,
            source_count=source_count,
            tried_ids=tried_ids,
            best_subset=best_subset,
        )
        if not mutations or source_selection_trials >= MAX_SOURCE_SELECTION_TRIALS or accepted_mutations >= MAX_ACCEPTED_MUTATIONS:
            break
        mutation = mutations[0]
        tried_ids.add(mutation["mutation_id"])
        if mutation["family"] in {"source_selection", "source_pruning"}:
            source_selection_trials += 1
        iter_tag = f"iter{round_idx:02d}"
        iter_root = ensure_dir(RUNS_ROOT / iter_tag)
        proxy_config_path = CONFIG_ROOT / f"proxy_config.{iter_tag}.json"
        write_json(proxy_config_path, mutation["proxy_config"])

        smoke_manifest_path = iter_root / "smoke_case_manifest.json"
        write_case_manifest(smoke_row, smoke_manifest_path)
        smoke_summary = run_eval(smoke_manifest_path, "cases", iter_root / "smoke_eval", proxy_config_path)
        smoke_ranking = compare_stage(smoke_summary, [SMOKE_CASE_ID])
        smoke_best = smoke_ranking[0]
        smoke_passers = [row for row in smoke_ranking if row.get("rows") and smoke_accept(row["rows"][0])]
        panels = build_panels(smoke_summary, smoke_best["variant"], iter_tag, SMOKE_CASE_ID)

        hero_result = None
        local20_result = None
        if smoke_passers:
            top_variant = smoke_passers[0]["variant"]
            hero_manifest_path = iter_root / "hero_manifest.json"
            write_json(hero_manifest_path, {"checked_at": now_iso(), "hero_cases": hero_rows})
            hero_summary = run_eval(BENCHMARK_MANIFEST, "hero_cases", iter_root / "hero_eval", proxy_config_path)
            hero_result = compare_variant(hero_summary, top_variant, [case_id(row) for row in hero_rows])
            if hero_accept(hero_result):
                local20_summary = run_eval(BENCHMARK_MANIFEST, "benchmark_cases", iter_root / "local20_eval", proxy_config_path)
                local20_result = compare_variant(local20_summary, top_variant, [case_id(row) for row in benchmark_rows])

        compare_for_progress = local20_result or hero_result or smoke_best
        improvement = effective_progress(compare_for_progress, best_compare)
        compare_key = progress_key(compare_for_progress)
        if compare_key < best_state_key:
            best_state_key = compare_key
            best_state = {
                "checked_at": now_iso(),
                "round": round_idx,
                "mutation": mutation,
                "failure_class": classify_failure(compare_for_progress),
                "compare": compare_for_progress,
                "panels": panels,
            }
            write_json(AUTLOOP_ROOT / "best_state.json", best_state)
        if improvement:
            best_compare = compare_for_progress
            best_subset = mutation["proxy_config"].get("source_subset") or best_subset
            accepted_mutations += 1
        failure_class = classify_failure(compare_for_progress)
        append_jsonl(
            AUTLOOP_ROOT / "iteration_ledger.jsonl",
            {
                "checked_at": now_iso(),
                "iteration": round_idx,
                "mutation_id": mutation["mutation_id"],
                "family": mutation["family"],
                "failure_class": failure_class,
                "smoke_best": smoke_best,
                "hero_result": hero_result,
                "local20_result": local20_result,
                "improvement": improvement,
                "panels": panels,
            },
        )
        write_json(RESEARCH_ROOT / f"{FAMILY}_iteration_report.{iter_tag}.json", {"checked_at": now_iso(), "iteration": round_idx, "mutation": mutation, "smoke_ranking": smoke_ranking, "hero_result": hero_result, "local20_result": local20_result, "improvement": improvement, "panels": panels})
        write_json(RESEARCH_ROOT / f"{FAMILY}_iteration_decision.{iter_tag}.json", {"checked_at": now_iso(), "iteration": round_idx, "chosen_mutation": mutation["mutation_id"], "failure_class": failure_class, "next_stage": "continue" if local20_result is None else "local_pass_ready"})
        write_json(AUTLOOP_ROOT / "mutation_history.json", {"checked_at": now_iso(), "mutations": sorted(tried_ids)})

        if local20_result and local20_accept(local20_result):
            final_result = {
                "status": "local_pass",
                "iteration": round_idx,
                "mutation": mutation,
                "best_variant": smoke_passers[0]["variant"],
                "smoke": smoke_best,
                "hero": hero_result,
                "local20": local20_result,
                "panels": panels,
            }
            break

    source_rank_rows = []
    for line in (AUTLOOP_ROOT / "iteration_ledger.jsonl").read_text(encoding="utf-8").splitlines() if (AUTLOOP_ROOT / "iteration_ledger.jsonl").exists() else []:
        item = json.loads(line)
        if item["family"] in {"source_selection", "source_pruning"}:
            source_rank_rows.append({"mutation_id": item["mutation_id"], "score": progress_key(item["smoke_best"]), "smoke_best": item["smoke_best"]})
    write_json(AUTLOOP_ROOT / "source_subset_ranking.json", {"checked_at": now_iso(), "rows": source_rank_rows})
    poison_rows = []
    for item in source_rank_rows:
        mid = item["mutation_id"]
        if "drop_" in mid:
            poison_rows.append({"mutation_id": mid, "smoke_best": item["smoke_best"]})
    write_json(AUTLOOP_ROOT / "poison_source_report.json", {"checked_at": now_iso(), "rows": poison_rows})

    if final_result:
        write_json(AUTLOOP_ROOT / "pending_cloud_ticket.json", {"checked_at": now_iso(), "family": FAMILY, "shape": FIRST_SHAPE, "best_variant": final_result["best_variant"], "local20": final_result["local20"], "panels": final_result["panels"]})
        write_text(AUTLOOP_ROOT / "cloud_ready_summary.md", "\n".join([
            f"# {FAMILY} cloud-ready summary",
            "",
            f"- status: local_pass",
            f"- best_variant: `{final_result['best_variant']}`",
            f"- iteration: `{final_result['iteration']}`",
            "- cloud remains off; this is only a pending cloud-ready ticket.",
        ]))
        return 0

    derived_next = NEXT_FAMILY_BY_FAILURE.get(best_state["failure_class"], "teacher_geometry_source_selection_audit_exhausted")
    final_postmortem = {
        "checked_at": now_iso(),
        "family": FAMILY,
        "status": "dead_same_day",
        "summary_reason": "Bounded local autoloop exhausted its source-selection / rehydration / smoothing mutation budget without producing an honest local pass that preserved human coverage while reducing fg fragmentation.",
        "best_local_state": best_state,
        "next_family": derived_next,
    }
    write_json(AUTLOOP_ROOT / "autoloop_final_postmortem.json", final_postmortem)
    write_json(AUTLOOP_ROOT / "best_local_state.json", best_state)
    write_json(RESEARCH_ROOT / f"next_manual_problem_draft.{derived_next}.json", {"checked_at": now_iso(), "family": derived_next, "derived_from": FAMILY, "reason": final_postmortem["summary_reason"]})

    research = load_json(RESEARCH_STATUS_JSON)
    research.update({
        "checked_at": now_iso(),
        "state": "IDLE_GUARD",
        "approved_problem_present": False,
        "approved_problem_ready": False,
        "allowed_families": [],
        "current_priority_family": "",
        "current_priority_reason": final_postmortem["summary_reason"],
        "same_family_retry_reason": f"{FAMILY} exhausted its bounded local autoloop budget this round.",
        "next_requirement": f"Return to IDLE_GUARD. The next honest manual direction is {derived_next}.",
        "cloud_must_remain_off": True,
        "latest_formal_verdict": {"checked_at": now_iso(), "status": "dead_same_day", "family": FAMILY, "first_candidate_shape": FIRST_SHAPE, "gate_stage_reached": "autoloop_local_bounded", "reason": final_postmortem["summary_reason"]},
    })
    write_json(RESEARCH_STATUS_JSON, research)
    task_plan = load_json(TASK_PLAN_JSON)
    task_plan.update({"checked_at": now_iso(), "task_mode_status": "hard_blocker", "task_mode_focus": f"{FAMILY}_dead_same_day", "research_loop_mode": "IDLE_GUARD"})
    task_plan["current_state_notes"] = [final_postmortem["summary_reason"], f"next_manual_problem_draft: {derived_next}"]
    task_plan["summary_conclusion"] = [final_postmortem["summary_reason"], f"best_local_state: {rel(AUTLOOP_ROOT / 'best_local_state.json')}"]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2))
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]))
    watch = load_json(WATCH_JSON)
    watch["checked_at"] = now_iso()
    watch["modal_apps"] = []
    watch["research_runtime_processes"] = []
    watch["watch_conclusion"] = final_postmortem["summary_reason"]
    watch["research"]["summary"] = {"state": "IDLE_GUARD", "approved_problem_present": False, "approved_problem_ready": False, "manual_action_required": False, "manual_action_kind": "", "ready_for_execution": False, "current_review_packet": rel(AUTLOOP_ROOT / "autoloop_final_postmortem.json").replace("/", "\\")}
    watch["research"]["research_status"] = research
    write_json(WATCH_JSON, watch)
    write_json(RESEARCH_ROOT / f"{FAMILY}_result.json", {"checked_at": now_iso(), "family": FAMILY, "status": "dead_same_day", "gate_stage_reached": "autoloop_local_bounded", "summary_reason": final_postmortem["summary_reason"], "best_local_state": best_state})
    write_json(RESEARCH_ROOT / f"{FAMILY}_postmortem.json", final_postmortem)
    write_json(ALLOWLIST_JSON, {"checked_at": now_iso(), "status": "idle_empty_allowlist", "guard_track_must_continue": True, "notes": "No active approved research candidate is running.", "allowed_markers": []})
    if FRONTIER_LEDGER_JSON.exists():
        frontier = load_json(FRONTIER_LEDGER_JSON)
        frontier["latest_formal_verdict"] = research["latest_formal_verdict"]
        write_json(FRONTIER_LEDGER_JSON, frontier)
    if FAMILY_STOP_REASON_JSON.exists():
        family_stop = load_json(FAMILY_STOP_REASON_JSON)
        family_stop.setdefault("latest_family_outcomes", {})[FAMILY] = research["latest_formal_verdict"]
        write_json(FAMILY_STOP_REASON_JSON, family_stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
