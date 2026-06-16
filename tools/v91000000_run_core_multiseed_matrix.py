from __future__ import annotations

import csv
import json
import subprocess
import sys
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


def best_from_run(run_id: str) -> dict[str, Any] | None:
    status = RUN_ROOT / run_id / "reports" / "V12000000_final_status.json"
    if not status.is_file():
        return None
    js = read_json(status)
    best = js.get("best", {})
    return {
        "run_id": run_id,
        "status": js.get("status"),
        "teacher_mode": js.get("teacher_mode"),
        "feature_mode": js.get("feature_mode"),
        "model_mode": js.get("model_mode", "spconv"),
        "seed": js.get("seed"),
        "real_sparse_backend": js.get("real_sparse_backend"),
        "best_name": best.get("name"),
        "mean_delta_vs_v999": best.get("mean_delta_vs_v999"),
        "full_body_delta": best.get("full_body_delta"),
        "head_face_delta": best.get("head_face_delta"),
        "hairline_delta": best.get("hairline_delta"),
        "left_hand_delta": best.get("left_hand_delta"),
        "right_hand_delta": best.get("right_hand_delta"),
        "prediction": str(RUN_ROOT / run_id / "candidates" / str(best.get("name")) / "predictions.npz"),
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    run_id = job["run_id"]
    existing = best_from_run(run_id)
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
        "v999_only",
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
    LOGS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(WORKTREE),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60 * 60,
    )
    log_payload = {
        "created_utc": now(),
        "run_id": run_id,
        "cmd": cmd,
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }
    write_json(LOGS / f"{run_id}_modal_cli.json", log_payload)
    row = best_from_run(run_id)
    if row is None:
        return {**job, "status": "FAILED_NO_FINAL_STATUS", "returncode": proc.returncode, "runtime_seconds": log_payload["runtime_seconds"]}
    row["returncode"] = proc.returncode
    row["runtime_seconds"] = log_payload["runtime_seconds"]
    row["skipped_existing"] = False
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    out = []
    for group, vals in sorted(groups.items()):
        nums = [float(v.get("mean_delta_vs_v999") or 0.0) for v in vals]
        out.append(
            {
                "group": group,
                "seed_count": len(vals),
                "mean_delta_mean": sum(nums) / max(1, len(nums)),
                "mean_delta_min": min(nums) if nums else 0.0,
                "mean_delta_max": max(nums) if nums else 0.0,
                "all_success": all(v.get("status") in {"V12000000_REVIEW_READY_NOT_PROMOTED", "V12000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"} for v in vals),
            }
        )
    csv_path = REPORTS / "V91000000_core_multiseed_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_utc": now(),
        "status": "V91000000_CORE_MULTI_SEED_MATRIX_COMPLETE" if all(r["seed_count"] >= 5 for r in out) else "V91000000_CORE_MULTI_SEED_MATRIX_PARTIAL",
        "groups": out,
        "matrix_csv": str(csv_path),
    }
    write_json(REPORTS / "V91000000_core_multiseed_matrix_summary.json", summary)
    return summary


def main() -> None:
    seeds = [91000000, 91000001, 91000002, 91000003, 91000004]
    configs = [
        ("true_smpl_full", "full", "spconv", 2.0),
        ("random_smpl_full", "random_smpl_full", "spconv", 2.0),
        ("shuffled_smpl_full", "shuffled_smpl_full", "spconv", 2.0),
        ("no_sparseconv_mlp", "full", "mlp", 2.0),
    ]
    jobs: list[dict[str, Any]] = []
    for group, feature_mode, model_mode, max_scale in configs:
        for i, seed in enumerate(seeds):
            jobs.append(
                {
                    "group": group,
                    "run_id": f"V910_{group}_seed{i}",
                    "seed": seed + i,
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "max_scale": max_scale,
                    "steps": 180,
                    "candidates": 24,
                    "max_points": 40000,
                    "grid_size": 56,
                }
            )
    rows = []
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        row["group"] = job["group"]
        rows.append(row)
        write_json(REPORTS / "V91000000_core_multiseed_matrix_progress.json", {"created_utc": now(), "completed": idx, "total": len(jobs), "latest": row})
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
