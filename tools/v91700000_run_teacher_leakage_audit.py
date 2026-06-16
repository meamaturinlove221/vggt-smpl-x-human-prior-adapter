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


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def best_from_run(run_id: str, group: str, seed_index: int, source: str = "V917") -> dict[str, Any] | None:
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    if not status_path.is_file():
        return None
    js = read_json(status_path)
    best = js.get("best", {})
    best_name = str(best.get("name"))
    pred_path = run_dir / "candidates" / best_name / "predictions.npz"
    train = read_json(run_dir / "reports" / "V10300000_decoder_training_summary.json", {})
    return {
        "source": source,
        "group": group,
        "seed_index": seed_index,
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
        "eval": str(run_dir / "candidates" / best_name / "eval.json"),
        "training_steps": train.get("steps"),
        "loss_start": train.get("first_loss"),
        "loss_end": train.get("last_loss"),
        "fit_drop": train.get("fit_drop"),
        "failure_reason": "; ".join(js.get("failure_classes", [])),
        "prediction_exists": pred_path.is_file(),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    run_id = job["run_id"]
    existing = best_from_run(run_id, job["group"], job["seed_index"])
    if existing is not None:
        existing["skipped_existing"] = True
        return existing
    cmd = [
        "modal",
        "run",
        "-q",
        "modal_v10000000_sparseconv_route.py",
        "--steps",
        str(job["steps"]),
        "--candidates",
        str(job["candidates"]),
        "--max-points",
        str(job["max_points"]),
        "--grid-size",
        str(job["grid_size"]),
        "--seed",
        str(job["seed"]),
        "--teacher-mode",
        job["teacher_mode"],
        "--feature-mode",
        job["feature_mode"],
        "--model-mode",
        job["model_mode"],
        "--max-scale",
        str(job["max_scale"]),
        "--archive-mode",
        "thin_only",
        "--run-id",
        run_id,
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(WORKTREE),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60 * 60,
        env=os_environ_utf8(),
    )
    log = {
        "created_utc": now(),
        "run_id": run_id,
        "cmd": cmd,
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }
    write_json(LOGS / f"{run_id}_modal_cli.json", log)
    row = best_from_run(run_id, job["group"], job["seed_index"])
    if row is None:
        return {**job, "status": "FAILED_NO_FINAL_STATUS", "failure_reason": log["stderr_tail"][-1000:]}
    row["returncode"] = proc.returncode
    row["runtime_seconds"] = log["runtime_seconds"]
    row["skipped_existing"] = False
    return row


def build_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reused = []
    for i in range(3):
        for run_id, group in [
            (f"V910_true_smpl_full_seed{i}", "v999_teacher_full_reuse"),
            (f"V915_true_observation_seed{i}", "observation_only_reuse"),
        ]:
            row = best_from_run(run_id, group, i, "reuse")
            if row is not None:
                reused.append(row)
    configs = [
        ("no_teacher_zero_control", "full", "spconv", "zero_control"),
        ("teacher_detached_v999", "full", "spconv", "teacher_detached"),
        ("teacher_noise", "full", "spconv", "teacher_noise"),
        ("teacher_randomized", "full", "spconv", "teacher_randomized"),
        ("observation_only_no_teacher", "observation_only", "spconv", "zero_control"),
    ]
    jobs: list[dict[str, Any]] = []
    for group, feature_mode, model_mode, teacher_mode in configs:
        for seed_index in range(3):
            jobs.append(
                {
                    "group": group,
                    "seed_index": seed_index,
                    "run_id": f"V917_{group}_seed{seed_index}",
                    "seed": 91700000 + seed_index * 2,
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "teacher_mode": teacher_mode,
                    "steps": 120,
                    "candidates": 16,
                    "max_points": 40000,
                    "grid_size": 56,
                    "max_scale": 2.0,
                }
            )
    return reused, jobs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows: list[dict[str, Any]], group: str) -> float | None:
    vals = [float(r["mean_delta_vs_v999"]) for r in rows if r.get("group") == group and r.get("mean_delta_vs_v999") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = {
        "v999_teacher_full_mean": group_mean(rows, "v999_teacher_full_reuse"),
        "observation_only_mean": group_mean(rows, "observation_only_reuse"),
        "no_teacher_zero_control_mean": group_mean(rows, "no_teacher_zero_control"),
        "teacher_detached_v999_mean": group_mean(rows, "teacher_detached_v999"),
        "teacher_noise_mean": group_mean(rows, "teacher_noise"),
        "teacher_randomized_mean": group_mean(rows, "teacher_randomized"),
        "observation_only_no_teacher_mean": group_mean(rows, "observation_only_no_teacher"),
    }
    base = m["v999_teacher_full_mean"] or 0.0
    no_teacher = m["no_teacher_zero_control_mean"] or 0.0
    obs = m["observation_only_mean"] or 0.0
    detached = m["teacher_detached_v999_mean"] or 0.0
    randomized = m["teacher_randomized_mean"] or 0.0
    m.update(
        {
            "teacher_detached_minus_v999_reuse": detached - base,
            "no_teacher_retained_fraction": no_teacher / max(abs(base), 1.0e-12),
            "observation_only_minus_v999_reuse": obs - base,
            "teacher_randomized_retained_fraction": randomized / max(abs(base), 1.0e-12),
        }
    )
    if no_teacher >= base * 0.65:
        conclusion = "BLENDING_OR_COMPOSITION_LEAKAGE_CONFIRMED"
    elif obs >= base * 0.90:
        conclusion = "OBSERVATION_DOMINANT_TEACHER_LEAKAGE_WEAK"
    elif randomized >= base * 0.75:
        conclusion = "TEACHER_TARGET_RANDOMIZATION_STILL_POSITIVE"
    elif detached > base * 0.90 and no_teacher < base * 0.50:
        conclusion = "V999_TEACHER_TARGET_IMPORTANT"
    else:
        conclusion = "TEACHER_LEAKAGE_INCONCLUSIVE"
    return {
        "created_utc": now(),
        "conclusion": conclusion,
        "metrics": m,
        "required_next": "Do not claim SMPL semantic causality; use limitation-first report and route through V940 only as feature-injection engineering, not causal proof.",
    }


def write_md(summary: dict[str, Any]) -> None:
    m = summary["metrics"]
    lines = [
        "# V91700000 Teacher Leakage Audit",
        "",
        f"- conclusion: `{summary['conclusion']}`",
        f"- v999_teacher_full_mean: `{m.get('v999_teacher_full_mean')}`",
        f"- observation_only_mean: `{m.get('observation_only_mean')}`",
        f"- no_teacher_zero_control_mean: `{m.get('no_teacher_zero_control_mean')}`",
        f"- teacher_detached_v999_mean: `{m.get('teacher_detached_v999_mean')}`",
        f"- teacher_noise_mean: `{m.get('teacher_noise_mean')}`",
        f"- teacher_randomized_mean: `{m.get('teacher_randomized_mean')}`",
        f"- observation_only_no_teacher_mean: `{m.get('observation_only_no_teacher_mean')}`",
        "",
        "This audit is mandatory because V915 classified the route as OBSERVATION_OR_TEACHER_DOMINANT.",
    ]
    (REPORTS / "V91700000_teacher_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    reused, jobs = build_jobs()
    rows = list(reused)
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        rows.append(row)
        write_json(
            REPORTS / "V91700000_teacher_leakage_progress.json",
            {"created_utc": now(), "completed_new_jobs": idx, "total_new_jobs": len(jobs), "latest": row},
        )
    write_csv(REPORTS / "V91700000_teacher_leakage_audit.csv", rows)
    summary = summarize(rows)
    write_json(REPORTS / "V91700000_teacher_leakage_audit.json", summary)
    write_md(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
