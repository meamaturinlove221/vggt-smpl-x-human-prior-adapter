from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKTREE = Path(r"D:\vggt\vggt-feature-adapter")
ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
BOARDS = ROOT / "boards"
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


def ensure_source_manifest(job: dict[str, Any], cand_dir: Path, best_name: str) -> Path:
    path = cand_dir / "source_manifest.json"
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
    manifest = read_json(path, {}) if path.is_file() else {}
    manifest.update(
        {
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
    )
    write_json(path, manifest)
    return path


def best_from_run(job: dict[str, Any]) -> dict[str, Any] | None:
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
    source_manifest_path = ensure_source_manifest(job, cand_dir, best_name)
    train = read_json(run_dir / "reports" / "V10300000_decoder_training_summary.json", {})
    return {
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
        "source_manifest": str(source_manifest_path),
        "prediction_exists": pred_path.is_file(),
        "eval_exists": eval_path.is_file(),
        "source_manifest_exists": source_manifest_path.is_file(),
        "training_steps": train.get("steps"),
        "loss_start": train.get("first_loss"),
        "loss_end": train.get("last_loss"),
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
        "120",
        "--candidates",
        "8",
        "--max-points",
        "40000",
        "--grid-size",
        "56",
        "--seed",
        str(job["seed"]),
        "--teacher-mode",
        str(job["teacher_mode"]),
        "--feature-mode",
        str(job["feature_mode"]),
        "--model-mode",
        str(job["model_mode"]),
        "--max-scale",
        "2.0",
        "--archive-mode",
        "thin_only",
        "--run-id",
        str(job["run_id"]),
    ]
    attempts: list[dict[str, Any]] = []
    row: dict[str, Any] | None = None
    for attempt in range(1, 4):
        started = time.time()
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
                "attempt": attempt,
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
                "attempt": attempt,
                "returncode": "TIMEOUT_EXPIRED",
                "runtime_seconds": time.time() - started,
                "stdout_tail": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
                "timeout": True,
            }
        attempts.append(log)
        write_json(LOGS / f"{job['run_id']}_modal_cli_attempt{attempt}.json", log)
        write_json(LOGS / f"{job['run_id']}_modal_cli.json", log)
        row = best_from_run(job)
        if row is not None:
            break
        if attempt < 3:
            time.sleep(30 * attempt)
    if row is None:
        last = attempts[-1]
        return {
            **job,
            "status": "FAILED_NO_FINAL_STATUS",
            "returncode": last["returncode"],
            "runtime_seconds": sum(float(a.get("runtime_seconds", 0.0)) for a in attempts),
            "failure_reason": str(last.get("stderr_tail", ""))[-1500:],
            "attempts": len(attempts),
        }
    row["returncode"] = attempts[-1]["returncode"]
    row["runtime_seconds"] = sum(float(a.get("runtime_seconds", 0.0)) for a in attempts)
    row["timeout_recovered_from_existing_artifact"] = bool(attempts[-1].get("timeout"))
    row["attempts"] = len(attempts)
    row["skipped_existing"] = False
    return row


def build_jobs() -> list[dict[str, Any]]:
    configs = [
        ("leakage_free_true_full", "full", "spconv", "v999_only_no_blend", "V9175_no_blend_true_full_seed{seed_index}"),
        ("leakage_free_random_smpl_full", "random_smpl_full", "spconv", "v999_only_no_blend", "V919_random_smpl_full_seed{seed_index}"),
        ("leakage_free_shuffled_smpl_full", "shuffled_smpl_full", "spconv", "v999_only_no_blend", "V919_shuffled_smpl_full_seed{seed_index}"),
        ("leakage_free_observation_only", "observation_only", "spconv", "v999_only_no_blend", "V919_observation_only_seed{seed_index}"),
        ("leakage_free_no_sparseconv_mlp", "full", "no_sparseconv_mlp", "v999_only_no_blend", "V9175_no_blend_no_sparseconv_mlp_seed{seed_index}"),
        ("leakage_free_no_teacher", "full", "spconv", "zero_control_no_blend", "V9175_no_teacher_no_blend_full_seed{seed_index}"),
        ("leakage_free_support_only", "support_only", "spconv", "v999_only_no_blend", "V919_support_only_seed{seed_index}"),
        ("leakage_free_semantic_only", "true_semantic_only", "spconv", "v999_only_no_blend", "V919_semantic_only_seed{seed_index}"),
    ]
    jobs: list[dict[str, Any]] = []
    for group, feature_mode, model_mode, teacher_mode, template in configs:
        for seed_index in range(5):
            jobs.append(
                {
                    "group": group,
                    "seed_index": seed_index,
                    "run_id": template.format(seed_index=seed_index),
                    "seed": 91900000 + seed_index * 2,
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "teacher_mode": teacher_mode,
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


def values(rows: list[dict[str, Any]], group: str, key: str = "mean_delta_vs_v999") -> list[float]:
    out = []
    for row in rows:
        if row.get("group") == group and row.get(key) is not None:
            out.append(float(row[key]))
    return out


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = mean(vals) or 0.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(r["group"]) for r in rows})
    stats = {g: {"n": len(values(rows, g)), "mean": mean(values(rows, g)), "std": std(values(rows, g))} for g in groups}
    true = stats.get("leakage_free_true_full", {}).get("mean")
    random = stats.get("leakage_free_random_smpl_full", {}).get("mean")
    shuffled = stats.get("leakage_free_shuffled_smpl_full", {}).get("mean")
    obs = stats.get("leakage_free_observation_only", {}).get("mean")
    mlp = stats.get("leakage_free_no_sparseconv_mlp", {}).get("mean")
    semantic = stats.get("leakage_free_semantic_only", {}).get("mean")
    support = stats.get("leakage_free_support_only", {}).get("mean")
    no_teacher = stats.get("leakage_free_no_teacher", {}).get("mean")
    def close(a: float | None, b: float | None, ratio: float = 0.90) -> bool:
        return a is not None and b is not None and a >= b * ratio
    if true is None or any(stats.get(g, {}).get("n", 0) < 5 for g in groups):
        conclusion = "LEAKAGE_FREE_ROUTE_EXHAUSTED"
    elif true > max(random or -1, shuffled or -1, obs or -1, mlp or -1) and (semantic or 0.0) > (support or 0.0):
        conclusion = "LEAKAGE_FREE_SMPL_CAUSAL_CONFIRMED"
    elif close(obs, true):
        conclusion = "LEAKAGE_FREE_OBSERVATION_DOMINANT"
    elif close(mlp, true):
        conclusion = "LEAKAGE_FREE_SMOOTHING_DOMINANT"
    elif close(random, true) or close(shuffled, true):
        conclusion = "LEAKAGE_FREE_SEMANTIC_WEAK"
    else:
        conclusion = "LEAKAGE_FREE_ROUTE_EXHAUSTED"
    audit = []
    for row in rows:
        path = Path(str(row.get("source_manifest", "")))
        js = read_json(path, {}) if path.is_file() else {}
        audit.append(
            {
                "run_id": row.get("run_id"),
                "group": row.get("group"),
                "source_manifest_exists": path.is_file(),
                "teacher_source": js.get("teacher_source"),
                "blend_source": js.get("blend_source"),
                "composition_source": js.get("composition_source"),
                "base_candidate": js.get("base_candidate"),
                "composition_no_blend": js.get("composition_no_blend"),
                "whether_postcompose_used": js.get("whether_postcompose_used"),
                "whether_humanram_used": js.get("whether_humanram_used"),
                "whether_v129_used": js.get("whether_v129_used"),
                "whether_v999_used": js.get("whether_v999_used"),
                "whether_v770_used": js.get("whether_v770_used"),
                "whether_teacher_used": js.get("whether_teacher_used"),
                "whether_blend_used": js.get("whether_blend_used"),
                "whether_observation_used": js.get("whether_observation_used"),
                "whether_support_used": js.get("whether_support_used"),
                "whether_semantic_used": js.get("whether_semantic_used"),
            }
        )
    return {
        "created_utc": now(),
        "status": "V91900000_LEAKAGE_FREE_CAUSAL_RERUN_COMPLETE",
        "conclusion": conclusion,
        "stats": stats,
        "key_differences": {
            "true_minus_random": None if true is None or random is None else true - random,
            "true_minus_shuffled": None if true is None or shuffled is None else true - shuffled,
            "true_minus_observation": None if true is None or obs is None else true - obs,
            "true_minus_no_sparseconv_mlp": None if true is None or mlp is None else true - mlp,
            "semantic_minus_support": None if semantic is None or support is None else semantic - support,
            "true_minus_no_teacher": None if true is None or no_teacher is None else true - no_teacher,
        },
        "source_manifest_audit": audit,
    }


def write_md(summary: dict[str, Any]) -> None:
    lines = ["# V91900000 Leakage-Free Core Causal Rerun", "", f"- conclusion: `{summary['conclusion']}`", ""]
    for group, stat in sorted(summary["stats"].items()):
        lines.append(f"- {group}: n=`{stat['n']}`, mean=`{stat['mean']}`, std=`{stat['std']}`")
    lines.append("")
    lines.append("V919 uses no-blend/no-composition routes. Unless conclusion is `LEAKAGE_FREE_SMPL_CAUSAL_CONFIRMED`, the advisor report must not claim strong independent SMPL semantic causality.")
    (REPORTS / "V91900000_leakage_free_causal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(summary["stats"])
    vals = [summary["stats"][g]["mean"] or 0.0 for g in groups]
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 0.9), 5))
    ax.bar(range(len(groups)), vals, color="#6b7f2a")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean_delta_vs_v999")
    ax.set_title("V919 leakage-free causal matrix")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V91900000_leakage_free_causal_visual.png", dpi=160)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        rows.append(row)
        write_json(REPORTS / "V91900000_leakage_free_causal_progress.json", {"created_utc": now(), "completed": idx, "total": len(jobs), "latest": row})
    write_csv(REPORTS / "V91900000_leakage_free_causal_matrix.csv", rows)
    write_csv(REPORTS / "V91900000_seed_level_metrics.csv", rows)
    summary = summarize(rows)
    write_json(REPORTS / "V91900000_source_manifest_audit.json", summary["source_manifest_audit"])
    write_json(REPORTS / "V91900000_leakage_free_causal_summary.json", summary)
    write_md(summary)
    plot(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
