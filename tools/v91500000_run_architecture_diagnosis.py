from __future__ import annotations

import csv
import json
import shutil
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


def best_from_run(run_id: str, group: str, seed_index: int, source: str = "V915") -> dict[str, Any] | None:
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    if not status_path.is_file():
        return None
    js = read_json(status_path)
    best = js.get("best", {})
    best_name = str(best.get("name"))
    pred_path = run_dir / "candidates" / best_name / "predictions.npz"
    if not pred_path.is_file():
        recover_best_candidate(run_id, best_name)
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
        "array_equal_v770": best.get("array_equal_v770"),
        "array_equal_v999": best.get("array_equal_v999"),
        "prediction": str(pred_path),
        "eval": str(run_dir / "candidates" / best_name / "eval.json"),
        "training_steps": train.get("steps"),
        "loss_start": train.get("first_loss"),
        "loss_end": train.get("last_loss"),
        "fit_drop": train.get("fit_drop"),
        "failure_reason": "; ".join(js.get("failure_classes", [])),
    }


def recover_best_candidate(run_id: str, best_name: str) -> None:
    dest = RUN_ROOT / run_id / "candidates" / best_name
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"v10000000_outputs/{run_id}/candidates/{best_name}"
    env = dict(**os_environ_utf8())
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", "vggt-sparseconv-output", remote, str(dest)],
        cwd=str(WORKTREE),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    nested = dest / best_name
    if nested.is_dir():
        for p in nested.iterdir():
            shutil.move(str(p), str(dest / p.name))
        nested.rmdir()
    if proc.returncode != 0 and not (dest / "predictions.npz").is_file():
        raise RuntimeError(f"Failed to recover {run_id}/{best_name}: {proc.stderr[-2000:]}")


def os_environ_utf8() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


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
    LOGS.mkdir(parents=True, exist_ok=True)
    last_log = None
    for attempt in range(1, 3):
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
        last_log = {
            "created_utc": now(),
            "run_id": run_id,
            "attempt": attempt,
            "cmd": cmd,
            "returncode": proc.returncode,
            "runtime_seconds": time.time() - started,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
        write_json(LOGS / f"{run_id}_modal_cli_attempt{attempt}.json", last_log)
        row = best_from_run(run_id, job["group"], job["seed_index"])
        if row is not None and Path(row["prediction"]).is_file():
            row["returncode"] = proc.returncode
            row["runtime_seconds"] = last_log["runtime_seconds"]
            row["attempt"] = attempt
            row["skipped_existing"] = False
            return row
    return {**job, "status": "FAILED_NO_FINAL_STATUS_OR_PREDICTION", "failure_reason": (last_log or {}).get("stderr_tail", "")[-1000:]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def feature_taxonomy() -> dict[str, Any]:
    return {
        "created_utc": now(),
        "support_features": [
            "voxel occupancy",
            "voxel center xyz / canonical bin",
            "smplx_depth",
            "smplx_visibility",
            "signed_boundary",
            "semantic_foreground",
            "region masks: head_face/hairline/left_hand/right_hand/phone_exclusion",
            "synthetic region support stack",
        ],
        "smpl_semantic_features": [
            "canonical xyz",
            "posed xyz",
            "SMPL normal",
            "vertex_id_sin/cos",
            "macro_part_scaled as body-part/skinning proxy",
        ],
        "vggt_observation_features": ["VGGT world point", "VGGT normal", "confidence"],
        "teacher_composition_features": [
            "V999 teacher residual",
            "V770 preserve base",
            "optional V129 guarded teacher",
            "local_gain and candidate blend coefficient",
        ],
        "diagnostic_note": "V915 adds feature modes to perturb semantic, support, and observation blocks independently.",
    }


def reuse_v910(group: str, alias: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(5):
        run_id = f"V910_{group}_seed{i}"
        row = best_from_run(run_id, alias, i, source="V910_reuse")
        if row is not None:
            rows.append(row)
    return rows[:3]


def build_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reused: list[dict[str, Any]] = []
    reused += reuse_v910("true_smpl_full", "true_support_true_semantic_true_observation")
    reused += reuse_v910("true_smpl_full", "sparseconv_true")
    reused += reuse_v910("no_sparseconv_mlp", "no_sparseconv_mlp")
    jobs: list[dict[str, Any]] = []
    configs = [
        ("true_support_random_semantic_true_observation", "random_semantic_same_support", "spconv", "v999_only"),
        ("random_support_true_semantic_true_observation", "random_support_true_semantic", "spconv", "v999_only"),
        ("mask_only_support_no_semantic_true_observation", "mask_only_support_observation", "spconv", "v999_only"),
        ("true_semantic_only", "true_semantic_only", "spconv", "v999_only"),
        ("random_semantic_only", "random_semantic_only", "spconv", "v999_only"),
        ("shuffled_semantic_only", "shuffled_semantic_only", "spconv", "v999_only"),
        ("shuffled_body_part", "shuffled_bodypart_labels", "spconv", "v999_only"),
        ("shuffled_canonical_xyz", "shuffled_canonical_xyz", "spconv", "v999_only"),
        ("shuffled_skinning", "shuffled_skinning", "spconv", "v999_only"),
        ("true_observation", "observation_only", "spconv", "v999_only"),
        ("no_observation", "no_observation", "spconv", "v999_only"),
        ("shuffled_observation", "shuffled_observation", "spconv", "v999_only"),
        ("random_observation", "random_observation", "spconv", "v999_only"),
        ("no_voxel_diffusion_direct_residual", "full", "direct_residual", "v999_only"),
        ("support_only_mlp", "support_only", "mlp", "v999_only"),
        ("teacher_guarded_v129_full", "full", "spconv", "guarded_v129"),
    ]
    for group, feature_mode, model_mode, teacher_mode in configs:
        for seed_index in range(3):
            jobs.append(
                {
                    "group": group,
                    "seed_index": seed_index,
                    "run_id": f"V915_{group}_seed{seed_index}",
                    "seed": 91500000 + seed_index * 2,
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


def group_mean(rows: list[dict[str, Any]], group: str) -> float | None:
    vals = [float(r["mean_delta_vs_v999"]) for r in rows if r.get("group") == group and r.get("mean_delta_vs_v999") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "true_full_mean": group_mean(rows, "true_support_true_semantic_true_observation"),
        "random_semantic_same_support_mean": group_mean(rows, "true_support_random_semantic_true_observation"),
        "true_semantic_random_support_mean": group_mean(rows, "random_support_true_semantic_true_observation"),
        "support_only_mean": group_mean(rows, "support_only_mlp"),
        "observation_only_mean": group_mean(rows, "true_observation"),
        "no_sparseconv_mlp_mean": group_mean(rows, "no_sparseconv_mlp"),
        "sparseconv_true_mean": group_mean(rows, "sparseconv_true"),
        "teacher_guarded_v129_full_mean": group_mean(rows, "teacher_guarded_v129_full"),
    }
    tf = metrics["true_full_mean"] or 0.0
    rand_sem = metrics["random_semantic_same_support_mean"] or 0.0
    rand_support = metrics["true_semantic_random_support_mean"] or 0.0
    support_only = metrics["support_only_mean"] or 0.0
    obs_only = metrics["observation_only_mean"] or 0.0
    mlp = metrics["no_sparseconv_mlp_mean"] or 0.0
    teacher = metrics["teacher_guarded_v129_full_mean"] or tf
    metrics.update(
        {
            "sparseconv_gain_over_mlp": tf - mlp,
            "semantic_gain_over_random": tf - rand_sem,
            "support_gain_over_random_support": tf - rand_support,
            "teacher_blending_gain": teacher - tf,
        }
    )
    tolerance = max(5.0e-5, abs(tf) * 0.15)
    if tf > rand_sem + tolerance and tf > rand_support + tolerance:
        conclusion = "SMPL_SEMANTIC_CAUSAL_CONFIRMED"
    elif abs(rand_sem - tf) <= tolerance or rand_sem > tf:
        conclusion = "SUPPORT_DOMINANT"
    elif obs_only >= tf - tolerance or teacher > tf + tolerance:
        conclusion = "OBSERVATION_OR_TEACHER_DOMINANT"
    elif rand_sem > 0 and mlp < tf - tolerance:
        conclusion = "SPARSECONV_SMOOTHING_DOMINANT"
    else:
        conclusion = "CAUSAL_UNRESOLVED"
    return {"created_utc": now(), "metrics": metrics, "conclusion": conclusion}


def write_attribution(summary: dict[str, Any]) -> None:
    m = summary["metrics"]
    lines = [
        "# V91500000 Support Vs Semantic Attribution",
        "",
        f"- conclusion: `{summary['conclusion']}`",
        f"- true_full_mean: `{m.get('true_full_mean')}`",
        f"- random_semantic_same_support_mean: `{m.get('random_semantic_same_support_mean')}`",
        f"- true_semantic_random_support_mean: `{m.get('true_semantic_random_support_mean')}`",
        f"- support_only_mean: `{m.get('support_only_mean')}`",
        f"- observation_only_mean: `{m.get('observation_only_mean')}`",
        f"- no_sparseconv_mlp_mean: `{m.get('no_sparseconv_mlp_mean')}`",
        f"- sparseconv_gain_over_mlp: `{m.get('sparseconv_gain_over_mlp')}`",
        f"- semantic_gain_over_random: `{m.get('semantic_gain_over_random')}`",
        f"- support_gain_over_random_support: `{m.get('support_gain_over_random_support')}`",
        f"- teacher_blending_gain: `{m.get('teacher_blending_gain')}`",
        "",
        "Do not claim SMPL semantic encoding is fully proven unless the conclusion is `SMPL_SEMANTIC_CAUSAL_CONFIRMED`.",
    ]
    if summary["conclusion"] == "SUPPORT_DOMINANT":
        lines.append("Current attribution points to support/occupancy/foreground structure dominating over semantic identity channels.")
    elif summary["conclusion"] == "OBSERVATION_OR_TEACHER_DOMINANT":
        lines.append("Current attribution points to observation or teacher/blending leakage dominating the gain.")
    elif summary["conclusion"] == "SPARSECONV_SMOOTHING_DOMINANT":
        lines.append("Current attribution points to sparse/diffusion smoothing contributing strongly even under randomized semantic features.")
    (REPORTS / "V91500000_support_vs_semantic_attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "V91500000_feature_taxonomy.json", feature_taxonomy())
    reused, jobs = build_jobs()
    rows = list(reused)
    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['run_id']} {job['group']}", flush=True)
        row = run_one(job)
        rows.append(row)
        write_json(
            REPORTS / "V91500000_architecture_diagnosis_progress.json",
            {"created_utc": now(), "completed_new_jobs": idx, "total_new_jobs": len(jobs), "latest": row},
        )
    write_csv(REPORTS / "V91500000_architecture_diagnosis.csv", rows)
    summary = summarize(rows)
    write_json(REPORTS / "V91500000_architecture_diagnosis_summary.json", summary)
    write_attribution(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
