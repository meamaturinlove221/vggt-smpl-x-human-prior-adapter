from __future__ import annotations

import csv
import json
import subprocess
import time
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


def env_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def best_from_run(run_id: str, group: str, seed_index: int) -> dict[str, Any] | None:
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    if not status_path.is_file():
        return None
    js = read_json(status_path)
    best = js.get("best", {})
    name = str(best.get("name"))
    return {
        "group": group,
        "seed_index": seed_index,
        "run_id": run_id,
        "status": js.get("status"),
        "teacher_mode": js.get("teacher_mode"),
        "feature_mode": js.get("feature_mode"),
        "model_mode": js.get("model_mode"),
        "backend": js.get("backend"),
        "real_sparse_backend": js.get("real_sparse_backend"),
        "best_name": name,
        "mean_delta_vs_v999": best.get("mean_delta_vs_v999"),
        "full_body_delta": best.get("full_body_delta"),
        "head_face_delta": best.get("head_face_delta"),
        "hairline_delta": best.get("hairline_delta"),
        "left_hand_delta": best.get("left_hand_delta"),
        "right_hand_delta": best.get("right_hand_delta"),
        "background_leakage": best.get("background_leakage_proxy"),
        "depth_world_consistency": best.get("depth_world_consistency"),
        "prediction": str(run_dir / "candidates" / name / "predictions.npz"),
        "failure_reason": "; ".join(js.get("failure_classes", [])),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    existing = best_from_run(job["run_id"], job["group"], job["seed_index"])
    if existing is not None:
        existing["skipped_existing"] = True
        return existing
    cmd = [
        "modal",
        "run",
        "-q",
        "modal_v10000000_sparseconv_route.py",
        "--steps",
        "120",
        "--candidates",
        "16",
        "--max-points",
        "40000",
        "--grid-size",
        "56",
        "--seed",
        str(job["seed"]),
        "--teacher-mode",
        job["teacher_mode"],
        "--feature-mode",
        job["feature_mode"],
        "--model-mode",
        job["model_mode"],
        "--max-scale",
        "2.0",
        "--archive-mode",
        "thin_only",
        "--run-id",
        job["run_id"],
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKTREE), text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60 * 60, env=env_utf8())
    write_json(
        LOGS / f"{job['run_id']}_modal_cli.json",
        {
            "created_utc": now(),
            "run_id": job["run_id"],
            "cmd": cmd,
            "returncode": proc.returncode,
            "runtime_seconds": time.time() - started,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        },
    )
    row = best_from_run(job["run_id"], job["group"], job["seed_index"])
    if row is None:
        return {**job, "status": "FAILED_NO_FINAL_STATUS", "failure_reason": proc.stderr[-1000:]}
    row["returncode"] = proc.returncode
    row["runtime_seconds"] = time.time() - started
    row["skipped_existing"] = False
    return row


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


def main() -> None:
    configs = [
        ("no_teacher_no_blend_full", "full", "spconv", "zero_control_no_blend"),
        ("observation_only_no_teacher_no_blend", "observation_only", "spconv", "zero_control_no_blend"),
        ("teacher_detached_no_blend", "full", "spconv", "teacher_detached_no_blend"),
        ("teacher_randomized_no_blend", "full", "spconv", "teacher_randomized_no_blend"),
    ]
    rows: list[dict[str, Any]] = []
    jobs = []
    for group, feature_mode, model_mode, teacher_mode in configs:
        for seed_index in range(3):
            jobs.append(
                {
                    "group": group,
                    "seed_index": seed_index,
                    "run_id": f"V9175_{group}_seed{seed_index}",
                    "seed": 91750000 + seed_index * 2,
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "teacher_mode": teacher_mode,
                }
            )
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        rows.append(row)
        write_json(REPORTS / "V91750000_teacher_leakage_repair_progress.json", {"created_utc": now(), "completed": idx, "total": len(jobs), "latest": row})
    write_csv(REPORTS / "V91750000_teacher_leakage_repair.csv", rows)
    metrics = {
        "no_teacher_no_blend_full_mean": group_mean(rows, "no_teacher_no_blend_full"),
        "observation_only_no_teacher_no_blend_mean": group_mean(rows, "observation_only_no_teacher_no_blend"),
        "teacher_detached_no_blend_mean": group_mean(rows, "teacher_detached_no_blend"),
        "teacher_randomized_no_blend_mean": group_mean(rows, "teacher_randomized_no_blend"),
    }
    original = read_json(REPORTS / "V91700000_teacher_leakage_audit.json", {})
    original_no_teacher = original.get("metrics", {}).get("no_teacher_zero_control_mean")
    repaired_no_teacher = metrics["no_teacher_no_blend_full_mean"]
    leakage_reduced = bool(original_no_teacher is not None and repaired_no_teacher is not None and repaired_no_teacher < float(original_no_teacher) * 0.60)
    summary = {
        "created_utc": now(),
        "status": "V91750000_TEACHER_LEAKAGE_REPAIR_COMPLETE",
        "metrics": metrics,
        "original_no_teacher_zero_control_mean": original_no_teacher,
        "leakage_reduced": leakage_reduced,
        "conclusion": "COMPOSITION_LEAKAGE_REDUCED" if leakage_reduced else "COMPOSITION_LEAKAGE_REMAINS",
    }
    write_json(REPORTS / "V91750000_teacher_leakage_repair.json", summary)
    (REPORTS / "V91750000_teacher_leakage_repair.md").write_text(
        "\n".join(
            [
                "# V91750000 Teacher Leakage Repair",
                "",
                f"- conclusion: `{summary['conclusion']}`",
                f"- original_no_teacher_zero_control_mean: `{original_no_teacher}`",
                f"- repaired_no_teacher_no_blend_full_mean: `{repaired_no_teacher}`",
                f"- observation_only_no_teacher_no_blend_mean: `{metrics['observation_only_no_teacher_no_blend_mean']}`",
                f"- teacher_detached_no_blend_mean: `{metrics['teacher_detached_no_blend_mean']}`",
                f"- teacher_randomized_no_blend_mean: `{metrics['teacher_randomized_no_blend_mean']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
