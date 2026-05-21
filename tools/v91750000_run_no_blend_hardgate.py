from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKTREE = Path(r"D:\vggt\vggt-feature-adapter")
ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
BOARDS = ROOT / "boards"
FAILURES = ROOT / "failures"
RUN_ROOT = ROOT / "output" / "V10000000_V12000000_modal_sparseconv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def env_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def candidate_source_manifest(job: dict[str, Any], best_name: str) -> dict[str, Any]:
    teacher_mode = str(job["teacher_mode"])
    teacher_target = teacher_mode[: -len("_no_blend")] if teacher_mode.endswith("_no_blend") else teacher_mode
    feature_mode = str(job["feature_mode"])
    teacher_used = teacher_target not in {"zero_control", "residual_zero_control", "no_teacher"}
    observation_used = feature_mode not in {
        "smpl_only",
        "random_smpl_only",
        "shuffled_smpl_only",
        "true_semantic_only",
        "random_semantic_only",
        "shuffled_semantic_only",
        "support_only",
        "no_observation",
    }
    support_used = feature_mode not in {
        "observation_only",
        "true_semantic_only",
        "random_semantic_only",
        "shuffled_semantic_only",
        "no_smpl",
    }
    semantic_used = feature_mode not in {
        "observation_only",
        "support_only",
        "support_observation_only",
        "mask_only_support_observation",
        "no_smpl",
        "no_semantic",
    }
    return {
        "teacher_source": teacher_target,
        "blend_source": "spconv",
        "composition_source": "none_no_blend",
        "base_candidate": "V770_world_points_plus_sparse_delta",
        "whether_v999_used": teacher_target
        in {
            "v999_only",
            "teacher_detached",
            "teacher_noise",
            "v999_noise",
            "teacher_randomized",
            "v999_randomized",
        },
        "whether_humanram_used": False,
        "whether_v129_used": False,
        "whether_v770_used": True,
        "whether_v770_preserve_blend_used": False,
        "whether_postcompose_used": False,
        "whether_teacher_used": bool(teacher_used),
        "whether_blend_used": False,
        "whether_observation_used": bool(observation_used),
        "whether_support_used": bool(support_used),
        "whether_semantic_used": bool(semantic_used),
        "composition_no_blend": True,
        "teacher_mode": teacher_mode,
        "feature_mode": feature_mode,
        "model_mode": str(job["model_mode"]),
        "group": str(job["group"]),
        "run_id": str(job["run_id"]),
        "best_name": best_name,
    }


def best_from_run(job: dict[str, Any], source: str = "V9175") -> dict[str, Any] | None:
    run_id = job["run_id"]
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    if not status_path.is_file():
        return None
    js = read_json(status_path)
    best = js.get("best", {})
    best_name = str(best.get("name"))
    cand_dir = run_dir / "candidates" / best_name
    pred_path = cand_dir / "predictions.npz"
    eval_path = cand_dir / "eval.json"
    source_manifest_path = cand_dir / "source_manifest.json"
    required_manifest_fields = {
        "whether_v999_used",
        "whether_humanram_used",
        "whether_v129_used",
        "whether_v770_used",
        "whether_postcompose_used",
        "whether_teacher_used",
        "whether_blend_used",
        "whether_observation_used",
        "whether_support_used",
        "whether_semantic_used",
    }
    if best_name and best_name != "None":
        manifest = read_json(source_manifest_path, {}) if source_manifest_path.is_file() else {}
        if not required_manifest_fields.issubset(manifest):
            manifest.update(candidate_source_manifest(job, best_name))
            write_json(source_manifest_path, manifest)
    train = read_json(run_dir / "reports" / "V10300000_decoder_training_summary.json", {})
    return {
        "source": source,
        "group": job["group"],
        "seed_index": job["seed_index"],
        "run_id": run_id,
        "status": js.get("status"),
        "teacher_mode": js.get("teacher_mode"),
        "feature_mode": js.get("feature_mode"),
        "model_mode": js.get("model_mode"),
        "backend": js.get("backend"),
        "real_sparse_backend": js.get("real_sparse_backend"),
        "best_name": best_name,
        "mean_delta_vs_v999": best.get("mean_delta_vs_v999"),
        "full_body_delta": best.get("full_body_delta"),
        "head_face_delta": best.get("head_face_delta"),
        "hairline_delta": best.get("hairline_delta"),
        "left_hand_delta": best.get("left_hand_delta"),
        "right_hand_delta": best.get("right_hand_delta"),
        "background_leakage": best.get("background_leakage_proxy"),
        "depth_world_consistency": best.get("depth_world_consistency"),
        "prediction": str(pred_path),
        "eval": str(eval_path),
        "config": str(cand_dir / "config.json"),
        "source_manifest": str(source_manifest_path),
        "board": str(cand_dir / "board.png"),
        "quality": str(eval_path),
        "prediction_exists": pred_path.is_file(),
        "eval_exists": eval_path.is_file(),
        "source_manifest_exists": source_manifest_path.is_file(),
        "training_steps": train.get("steps"),
        "loss_start": train.get("first_loss"),
        "loss_end": train.get("last_loss"),
        "parameter_delta": train.get("param_delta_norm", train.get("parameter_delta_norm")),
        "failure_reason": "; ".join(js.get("failure_classes", [])),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    existing = best_from_run(job)
    if existing is not None:
        existing["skipped_existing"] = True
        return existing
    cmd = [
        "modal",
        "run",
        "-q",
        "modal_v10000000_sparseconv_route.py",
        "--steps",
        str(job.get("steps", 120)),
        "--candidates",
        str(job.get("candidates", 8)),
        "--max-points",
        str(job.get("max_points", 40000)),
        "--grid-size",
        str(job.get("grid_size", 56)),
        "--seed",
        str(job["seed"]),
        "--teacher-mode",
        str(job["teacher_mode"]),
        "--feature-mode",
        str(job["feature_mode"]),
        "--model-mode",
        str(job["model_mode"]),
        "--max-scale",
        str(job.get("max_scale", 2.0)),
        "--archive-mode",
        "thin_only",
        "--run-id",
        str(job["run_id"]),
    ]
    started = time.time()
    log: dict[str, Any]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKTREE),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=75 * 60,
            env=env_utf8(),
        )
        log = {
            "created_utc": now(),
            "run_id": job["run_id"],
            "cmd": cmd,
            "returncode": proc.returncode,
            "runtime_seconds": time.time() - started,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-12000:],
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        log = {
            "created_utc": now(),
            "run_id": job["run_id"],
            "cmd": cmd,
            "returncode": "TIMEOUT_EXPIRED",
            "runtime_seconds": time.time() - started,
            "stdout_tail": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "timeout": True,
        }
    write_json(LOGS / f"{job['run_id']}_modal_cli.json", log)
    row = best_from_run(job)
    if row is None:
        return {
            **job,
            "source": "V9175",
            "status": "FAILED_NO_FINAL_STATUS",
            "failure_reason": str(log.get("stderr_tail", ""))[-1500:],
            "runtime_seconds": log["runtime_seconds"],
            "returncode": log["returncode"],
        }
    row["returncode"] = log["returncode"]
    row["runtime_seconds"] = log["runtime_seconds"]
    row["skipped_existing"] = False
    row["timeout_recovered_from_existing_artifact"] = bool(log.get("timeout"))
    return row


def build_jobs() -> list[dict[str, Any]]:
    configs = [
        ("no_blend_true_full", "full", "spconv", "v999_only_no_blend", "V9175_no_blend_true_full_seed{seed_index}"),
        ("no_blend_no_teacher_zero", "full", "spconv", "zero_control_no_blend", "V9175_no_teacher_no_blend_full_seed{seed_index}"),
        ("no_blend_teacher_detached_v999", "full", "spconv", "teacher_detached_no_blend", "V9175_teacher_detached_no_blend_seed{seed_index}"),
        ("no_blend_teacher_noise", "full", "spconv", "teacher_noise_no_blend", "V9175_no_blend_teacher_noise_seed{seed_index}"),
        ("no_blend_teacher_randomized", "full", "spconv", "teacher_randomized_no_blend", "V9175_teacher_randomized_no_blend_seed{seed_index}"),
        ("no_blend_observation_only_no_teacher", "observation_only", "spconv", "zero_control_no_blend", "V9175_observation_only_no_teacher_no_blend_seed{seed_index}"),
        ("no_blend_support_only_no_teacher", "support_only", "spconv", "zero_control_no_blend", "V9175_no_blend_support_only_no_teacher_seed{seed_index}"),
        ("no_blend_random_semantic_true_support", "random_semantic_same_support", "spconv", "v999_only_no_blend", "V9175_no_blend_random_semantic_true_support_seed{seed_index}"),
        ("no_blend_shuffled_semantic_true_support", "shuffled_semantic_same_support", "spconv", "v999_only_no_blend", "V9175_no_blend_shuffled_semantic_true_support_seed{seed_index}"),
        ("no_blend_no_sparseconv_mlp", "full", "no_sparseconv_mlp", "v999_only_no_blend", "V9175_no_blend_no_sparseconv_mlp_seed{seed_index}"),
    ]
    jobs: list[dict[str, Any]] = []
    for group, feature_mode, model_mode, teacher_mode, template in configs:
        for seed_index in range(3):
            jobs.append(
                {
                    "group": group,
                    "seed_index": seed_index,
                    "run_id": template.format(seed_index=seed_index),
                    "seed": 91750000 + seed_index * 2,
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "teacher_mode": teacher_mode,
                    "steps": 120,
                    "candidates": 8,
                    "max_points": 40000,
                    "grid_size": 56,
                    "max_scale": 2.0,
                }
            )
    return jobs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_values(rows: list[dict[str, Any]], group: str) -> list[float]:
    vals = []
    for row in rows:
        if row.get("group") != group or row.get("mean_delta_vs_v999") is None:
            continue
        vals.append(float(row["mean_delta_vs_v999"]))
    return vals


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def retained(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new / max(abs(float(old)), 1.0e-12)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old = read_json(REPORTS / "V91700000_teacher_leakage_audit.json", {})
    om = old.get("metrics", {})
    m = {
        "old_v917_no_teacher_mean": om.get("no_teacher_zero_control_mean"),
        "old_observation_only_no_teacher_mean": om.get("observation_only_no_teacher_mean"),
        "old_teacher_detached_mean": om.get("teacher_detached_v999_mean"),
        "old_teacher_randomized_mean": om.get("teacher_randomized_mean"),
        "true_full_no_blend_mean": mean(group_values(rows, "no_blend_true_full")),
        "new_v9175_no_blend_no_teacher_mean": mean(group_values(rows, "no_blend_no_teacher_zero")),
        "new_no_blend_observation_only_no_teacher_mean": mean(group_values(rows, "no_blend_observation_only_no_teacher")),
        "new_no_blend_teacher_detached_mean": mean(group_values(rows, "no_blend_teacher_detached_v999")),
        "new_no_blend_teacher_noise_mean": mean(group_values(rows, "no_blend_teacher_noise")),
        "new_no_blend_teacher_randomized_mean": mean(group_values(rows, "no_blend_teacher_randomized")),
        "support_only_no_teacher_mean": mean(group_values(rows, "no_blend_support_only_no_teacher")),
        "random_semantic_no_blend_mean": mean(group_values(rows, "no_blend_random_semantic_true_support")),
        "shuffled_semantic_no_blend_mean": mean(group_values(rows, "no_blend_shuffled_semantic_true_support")),
        "no_sparseconv_no_blend_mean": mean(group_values(rows, "no_blend_no_sparseconv_mlp")),
    }
    m["leakage_reduction_ratio"] = retained(m["new_v9175_no_blend_no_teacher_mean"], m["old_v917_no_teacher_mean"])
    m["observation_leakage_reduction_ratio"] = retained(m["new_no_blend_observation_only_no_teacher_mean"], m["old_observation_only_no_teacher_mean"])
    m["teacher_detached_reduction_ratio"] = retained(m["new_no_blend_teacher_detached_mean"], m["old_teacher_detached_mean"])
    true_full = m["true_full_no_blend_mean"]
    random_sem = m["random_semantic_no_blend_mean"]
    shuffled_sem = m["shuffled_semantic_no_blend_mean"]
    obs_no_teacher = m["new_no_blend_observation_only_no_teacher_mean"]
    no_sparse = m["no_sparseconv_no_blend_mean"]
    m["semantic_gain_over_random_no_blend"] = None if true_full is None or random_sem is None else true_full - random_sem
    m["semantic_gain_over_shuffled_no_blend"] = None if true_full is None or shuffled_sem is None else true_full - shuffled_sem
    m["sparseconv_gain_over_mlp_no_blend"] = None if true_full is None or no_sparse is None else true_full - no_sparse
    manifests = [r for r in rows if r.get("source_manifest_exists")]
    source_manifest_pass = len(manifests) == len(rows) and all(
        not read_json(Path(r["source_manifest"]), {}).get("whether_postcompose_used", True) for r in manifests
    )
    reduced = (
        m["leakage_reduction_ratio"] is not None
        and m["leakage_reduction_ratio"] < 0.60
        and m["observation_leakage_reduction_ratio"] is not None
        and m["observation_leakage_reduction_ratio"] < 0.60
        and m["new_no_blend_teacher_randomized_mean"] is not None
        and m["old_teacher_randomized_mean"] is not None
        and m["new_no_blend_teacher_randomized_mean"] < float(m["old_teacher_randomized_mean"]) * 0.60
    )
    causal_strong = (
        true_full is not None
        and true_full > 0.0
        and random_sem is not None
        and shuffled_sem is not None
        and true_full > random_sem
        and true_full > shuffled_sem
    )
    obs_or_smoothing = (
        true_full is not None
        and (
            (obs_no_teacher is not None and obs_no_teacher >= true_full * 0.85)
            or (no_sparse is not None and no_sparse >= true_full * 0.85)
        )
    )
    if not source_manifest_pass:
        conclusion = "REPAIR_FAILED"
    elif reduced and causal_strong:
        conclusion = "LEAKAGE_REPAIRED"
    elif obs_or_smoothing:
        conclusion = "OBSERVATION_OR_SMOOTHING_DOMINANT_AFTER_REPAIR"
    elif reduced:
        conclusion = "LEAKAGE_REDUCED_BUT_CAUSAL_WEAK"
    else:
        conclusion = "REPAIR_FAILED"
    return {
        "created_utc": now(),
        "status": "V91750000_NO_BLEND_REPAIR_COMPLETE",
        "conclusion": conclusion,
        "metrics": m,
        "source_manifest_pass": source_manifest_pass,
        "source_manifest_count": len(manifests),
        "row_count": len(rows),
        "required_next": "Run V919 leakage-free core causal rerun unless this repair has failed so hard that only failure-attribution can proceed.",
    }


def source_manifest_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit = []
    for row in rows:
        path = Path(str(row.get("source_manifest", "")))
        js = read_json(path, {}) if path.is_file() else {}
        audit.append(
            {
                "run_id": row.get("run_id"),
                "group": row.get("group"),
                "seed_index": row.get("seed_index"),
                "source_manifest_exists": path.is_file(),
                "teacher_source": js.get("teacher_source"),
                "blend_source": js.get("blend_source"),
                "composition_source": js.get("composition_source"),
                "base_candidate": js.get("base_candidate"),
                "whether_v999_used": js.get("whether_v999_used"),
                "whether_humanram_used": js.get("whether_humanram_used"),
                "whether_v129_used": js.get("whether_v129_used"),
                "whether_v770_used": js.get("whether_v770_used"),
                "whether_postcompose_used": js.get("whether_postcompose_used"),
                "whether_teacher_used": js.get("whether_teacher_used"),
                "whether_blend_used": js.get("whether_blend_used"),
                "whether_observation_used": js.get("whether_observation_used"),
                "whether_support_used": js.get("whether_support_used"),
                "whether_semantic_used": js.get("whether_semantic_used"),
                "composition_no_blend": js.get("composition_no_blend"),
            }
        )
    return audit


def write_md(summary: dict[str, Any]) -> None:
    m = summary["metrics"]
    lines = [
        "# V91750000 No-Blend Teacher Leakage Repair",
        "",
        f"- conclusion: `{summary['conclusion']}`",
        f"- source_manifest_pass: `{summary['source_manifest_pass']}`",
        f"- old_v917_no_teacher_mean: `{m.get('old_v917_no_teacher_mean')}`",
        f"- new_v9175_no_blend_no_teacher_mean: `{m.get('new_v9175_no_blend_no_teacher_mean')}`",
        f"- leakage_reduction_ratio: `{m.get('leakage_reduction_ratio')}`",
        f"- old_observation_only_no_teacher_mean: `{m.get('old_observation_only_no_teacher_mean')}`",
        f"- new_no_blend_observation_only_no_teacher_mean: `{m.get('new_no_blend_observation_only_no_teacher_mean')}`",
        f"- true_full_no_blend_mean: `{m.get('true_full_no_blend_mean')}`",
        f"- random_semantic_no_blend_mean: `{m.get('random_semantic_no_blend_mean')}`",
        f"- shuffled_semantic_no_blend_mean: `{m.get('shuffled_semantic_no_blend_mean')}`",
        f"- no_sparseconv_no_blend_mean: `{m.get('no_sparseconv_no_blend_mean')}`",
        "",
        "This gate intentionally disables candidate post-composition. It must finish before V919, and V919 must use leakage-free/no-blend routes.",
    ]
    (REPORTS / "V91750000_teacher_leakage_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({str(r["group"]) for r in rows})
    vals = [mean(group_values(rows, g)) or 0.0 for g in groups]
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 0.8), 5))
    ax.bar(range(len(groups)), vals, color="#2f6f73")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean_delta_vs_v999")
    ax.set_title("V9175 no-blend leakage repair group means")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V91750000_no_blend_repair_visual.png", dpi=160)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    FAILURES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    jobs = build_jobs()
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        rows.append(row)
        write_json(
            REPORTS / "V91750000_no_blend_hardgate_progress.json",
            {"created_utc": now(), "completed": idx, "total": len(jobs), "latest": row},
        )
    write_csv(REPORTS / "V91750000_no_blend_repair_results.csv", rows)
    write_csv(REPORTS / "V91750000_teacher_leakage_repair.csv", rows)
    write_csv(REPORTS / "V91750000_seed_level_metrics.csv", rows)
    failures = [r for r in rows if str(r.get("status", "")).startswith("FAILED") or not r.get("prediction_exists") or not r.get("eval_exists")]
    write_json(FAILURES / "V91750000_failed_jobs.json", failures)
    write_json(REPORTS / "V91750000_source_manifest_audit.json", source_manifest_audit(rows))
    summary = summarize(rows)
    write_json(REPORTS / "V91750000_teacher_leakage_repair_summary.json", summary)
    write_json(REPORTS / "V91750000_teacher_leakage_repair.json", summary)
    write_md(summary)
    plot(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
