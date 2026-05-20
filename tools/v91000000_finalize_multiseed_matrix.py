from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
BOARDS = ROOT / "boards"
ARCHIVE = ROOT / "archive"
RUN_ROOT = ROOT / "output" / "V10000000_V12000000_modal_sparseconv"
COMPACT_PRED = ROOT / "output" / "V91000000_multiseed_predictions"
MODAL_JOB_LOGS = LOGS / "V91000000_modal_jobs"

GROUPS = ["true_smpl_full", "random_smpl_full", "shuffled_smpl_full", "no_sparseconv_mlp"]
REGION_KEYS = [
    "mean_delta_vs_v999",
    "full_body_delta",
    "head_face_delta",
    "hairline_delta",
    "left_hand_delta",
    "right_hand_delta",
]


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pred(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        pts = np.asarray(z["world_points"] if "world_points" in z else z["points"], dtype=np.float32)
        depth = np.asarray(z["depth"] if "depth" in z else pts[..., 2], dtype=np.float32)
        normal = np.asarray(z["normal"] if "normal" in z else np.zeros_like(pts), dtype=np.float32)
    return {"world_points": pts, "depth": depth, "normal": normal}


def run_id_parts(run_id: str) -> tuple[str, int]:
    prefix, seed_s = run_id.rsplit("_seed", 1)
    return prefix.removeprefix("V910_"), int(seed_s)


def best_row_for_run(run_id: str) -> dict[str, Any]:
    group, seed_index = run_id_parts(run_id)
    run_dir = RUN_ROOT / run_id
    status_path = run_dir / "reports" / "V12000000_final_status.json"
    status = read_json(status_path)
    best = status.get("best", {})
    train = read_json(run_dir / "reports" / "V10300000_decoder_training_summary.json", {})
    env = read_json(run_dir / "reports" / "V10040000_modal_env_matrix.json", {})
    log_path = LOGS / f"{run_id}_modal_cli.json"
    cli = read_json(log_path, {})
    pred_path = run_dir / "candidates" / str(best.get("name")) / "predictions.npz"
    checkpoint_path = run_dir / "V10300000_sparseconv_checkpoint.pt"
    parameter_delta_proxy = None
    if checkpoint_path.is_file():
        parameter_delta_proxy = checkpoint_path.stat().st_size
    return {
        "group": group,
        "seed_index": seed_index,
        "run_id": run_id,
        "modal_app_id": None,
        "modal_job_id": run_id,
        "modal_id_note": "Modal historical app id unavailable from captured quiet CLI tail; run_id and CLI command are preserved.",
        "backend": status.get("backend"),
        "model_mode": status.get("model_mode"),
        "feature_mode": status.get("feature_mode"),
        "teacher_mode": status.get("teacher_mode"),
        "gpu_type": env.get("cuda_name") or env.get("gpu_spec"),
        "training_steps": train.get("steps"),
        "runtime_seconds": cli.get("runtime_seconds") or train.get("runtime_seconds"),
        "loss_start": train.get("first_loss"),
        "loss_end": train.get("last_loss"),
        "fit_drop": train.get("fit_drop"),
        "parameter_delta": parameter_delta_proxy,
        "parameter_delta_note": "checkpoint size proxy; initial parameter snapshot was not stored by V910 runner",
        "prediction_path": str(pred_path),
        "eval_path": str(run_dir / "candidates" / str(best.get("name")) / "eval.json"),
        "mean_delta_vs_v999": best.get("mean_delta_vs_v999"),
        "full_body_delta": best.get("full_body_delta"),
        "head_face_delta": best.get("head_face_delta"),
        "hairline_delta": best.get("hairline_delta"),
        "left_hand_delta": best.get("left_hand_delta"),
        "right_hand_delta": best.get("right_hand_delta"),
        "background_leakage": best.get("background_leakage_proxy"),
        "depth_world_consistency": best.get("depth_world_consistency"),
        "normal_consistency": None,
        "point_quality_proxy": (
            float(best.get("full_body_delta", 0.0))
            + float(best.get("head_face_delta", 0.0))
            + float(best.get("left_hand_delta", 0.0))
            + float(best.get("right_hand_delta", 0.0))
        )
        / 4.0,
        "array_equal_v770": best.get("array_equal_v770"),
        "array_equal_v999": best.get("array_equal_v999"),
        "returncode": cli.get("returncode"),
        "status": status.get("status"),
        "failure_reason": "; ".join(status.get("failure_classes", [])),
        "real_sparse_backend": status.get("real_sparse_backend"),
        "cli_log_path": str(log_path),
        "thin_bundle_size": status.get("bundles", {}).get("thin", {}).get("size"),
        "thin_bundle_sha256": status.get("bundles", {}).get("thin", {}).get("sha256"),
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUN_ROOT.glob("V910_*_seed*/reports/V12000000_final_status.json")):
        rows.append(best_row_for_run(path.parents[1].name))
    rows.sort(key=lambda r: (GROUPS.index(r["group"]) if r["group"] in GROUPS else 99, int(r["seed_index"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def paired_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[row["group"]][int(row["seed_index"])] = row
    rng = np.random.default_rng(91000000)
    stats: dict[str, Any] = {"groups": {}, "paired": {}, "bootstrap_resamples": 1000}
    for group, seed_rows in by_group.items():
        vals = [float(seed_rows[i]["mean_delta_vs_v999"]) for i in sorted(seed_rows)]
        m, s = mean_std(vals)
        stats["groups"][group] = {
            "seed_count": len(vals),
            "mean": m,
            "std": s,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "failure_rate": float(np.mean([bool(seed_rows[i].get("failure_reason")) for i in sorted(seed_rows)])),
        }
    true_rows = by_group.get("true_smpl_full", {})
    for other in ["random_smpl_full", "shuffled_smpl_full", "no_sparseconv_mlp"]:
        other_rows = by_group.get(other, {})
        common = sorted(set(true_rows) & set(other_rows))
        comp: dict[str, Any] = {"seed_count": len(common), "regions": {}}
        for key in REGION_KEYS:
            diff = np.asarray([float(true_rows[i][key]) - float(other_rows[i][key]) for i in common], dtype=np.float64)
            if diff.size == 0:
                continue
            boot = np.asarray([diff[rng.integers(0, diff.size, size=diff.size)].mean() for _ in range(1000)])
            comp["regions"][key] = {
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std(ddof=1) if diff.size > 1 else 0.0),
                "effect_size": float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if diff.size > 1 else 0.0,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            }
        stats["paired"][f"true_minus_{other}"] = comp
    tf = stats["groups"].get("true_smpl_full", {})
    random_g = stats["groups"].get("random_smpl_full", {})
    shuffled_g = stats["groups"].get("shuffled_smpl_full", {})
    mlp_g = stats["groups"].get("no_sparseconv_mlp", {})
    if (
        tf.get("mean", -math.inf) > random_g.get("mean", math.inf)
        and tf.get("mean", -math.inf) > shuffled_g.get("mean", math.inf)
        and tf.get("mean", -math.inf) > mlp_g.get("mean", math.inf)
        and stats["paired"]["true_minus_random_smpl_full"]["regions"]["mean_delta_vs_v999"]["bootstrap_ci_low"] > 0
        and stats["paired"]["true_minus_shuffled_smpl_full"]["regions"]["mean_delta_vs_v999"]["bootstrap_ci_low"] > 0
    ):
        classification = "CAUSAL_STRONG"
    elif (
        tf.get("mean", -math.inf) >= max(random_g.get("mean", -math.inf), shuffled_g.get("mean", -math.inf), mlp_g.get("mean", -math.inf))
        and mlp_g.get("mean", math.inf) < tf.get("mean", -math.inf)
    ):
        classification = "CAUSAL_WEAK_BUT_POSITIVE"
    else:
        classification = "CAUSAL_NOT_CONFIRMED"
    stats["classification"] = classification
    stats["classification_reason"] = (
        "random/shuffled/no-SparseConv controls are close to or exceed true_smpl_full"
        if classification == "CAUSAL_NOT_CONFIRMED"
        else "true_smpl_full remains above controls but with overlapping/weak margins"
    )
    return stats


def baseline_paths(rows: list[dict[str, Any]]) -> dict[str, Path]:
    schema = read_json(REPORTS / "V8110000_schema_report.json")
    best_mlp = max((r for r in rows if r["group"] == "no_sparseconv_mlp"), key=lambda r: float(r["mean_delta_vs_v999"]))
    return {
        "V770": Path(schema["inputs"]["V770"]),
        "V999": ROOT / "output" / "V9400000_V9990000_longrun_feature_adapter" / "V9800000_candidates" / "cand_129_triplane_only_w080" / "predictions.npz",
        "V500_best_proxy": ROOT / "output" / "V31000000_token_injection" / "gated_add_proxy" / "predictions.npz",
        "V260_best_proxy": ROOT / "output" / "V23500000_compositions" / "comp_007_full_no_v129_w1p00" / "predictions.npz",
        "no_sparseconv_mlp_best": Path(best_mlp["prediction_path"]),
    }


def unique_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {name: load_pred(path) for name, path in baseline_paths(rows).items() if path.is_file()}
    checks: list[dict[str, Any]] = []
    for row in rows:
        pred_path = Path(row["prediction_path"])
        pred = load_pred(pred_path)
        for name, base in baselines.items():
            wp_diff = pred["world_points"] - base["world_points"]
            l2 = np.linalg.norm(wp_diff, axis=-1)
            n_diff = pred["normal"] - base["normal"]
            checks.append(
                {
                    "run_id": row["run_id"],
                    "group": row["group"],
                    "seed_index": row["seed_index"],
                    "baseline": name,
                    "array_equal": bool(np.array_equal(pred["world_points"], base["world_points"])),
                    "mean_l2_diff": float(l2.mean()),
                    "max_l2_diff": float(l2.max()),
                    "changed_pixels": int((l2 > 1.0e-7).sum()),
                    "depth_mean_abs_diff": float(np.abs(pred["depth"] - base["depth"]).mean()),
                    "normal_mean_l2_diff": float(np.linalg.norm(n_diff, axis=-1).mean()),
                }
            )
    return checks


def copy_modal_logs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    MODAL_JOB_LOGS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        src = Path(row["cli_log_path"])
        dst = MODAL_JOB_LOGS / f"{row['run_id']}.log"
        if src.is_file():
            payload = read_json(src, {})
            text = [
                f"run_id: {row['run_id']}",
                f"group: {row['group']}",
                f"seed_index: {row['seed_index']}",
                f"cmd: {' '.join(payload.get('cmd', []))}",
                f"returncode: {payload.get('returncode')}",
                f"runtime_seconds: {payload.get('runtime_seconds')}",
                "",
                "stdout_tail:",
                str(payload.get("stdout_tail", "")),
                "",
                "stderr_tail:",
                str(payload.get("stderr_tail", "")),
            ]
            dst.write_text("\n".join(text), encoding="utf-8")
        manifest.append(
            {
                "run_id": row["run_id"],
                "group": row["group"],
                "seed_index": row["seed_index"],
                "modal_app_id": row["modal_app_id"],
                "modal_job_id": row["modal_job_id"],
                "cli_log": str(dst),
                "prediction_path": row["prediction_path"],
                "eval_path": row["eval_path"],
                "runtime_seconds": row["runtime_seconds"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
            }
        )
    return manifest


def compact_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    COMPACT_PRED.mkdir(parents=True, exist_ok=True)
    selected = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        best = max(group_rows, key=lambda r: float(r["mean_delta_vs_v999"]))
        selected.append(best)
        dst_dir = COMPACT_PRED / best["run_id"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["prediction_path"], dst_dir / "predictions.npz")
        eval_path = Path(best["eval_path"])
        if eval_path.is_file():
            shutil.copy2(eval_path, dst_dir / "eval.json")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    bundle = ARCHIVE / "V91000000_compact_top_predictions_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for p in COMPACT_PRED.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(COMPACT_PRED.parent).as_posix())
    omitted = []
    selected_paths = {Path(r["prediction_path"]).resolve() for r in selected}
    for row in rows:
        p = Path(row["prediction_path"])
        if p.resolve() not in selected_paths:
            omitted.append({"run_id": row["run_id"], "path": str(p), "size": p.stat().st_size, "sha256": sha256(p)})
    return {
        "selected_run_ids": [r["run_id"] for r in selected],
        "bundle": str(bundle),
        "bundle_size": bundle.stat().st_size,
        "bundle_sha256": sha256(bundle),
        "bundle_zip_test": zipfile.ZipFile(bundle).testzip() or "clean",
        "omitted_large_predictions": omitted,
    }


def make_board(stats: dict[str, Any]) -> None:
    BOARDS.mkdir(parents=True, exist_ok=True)
    groups = GROUPS
    means = [stats["groups"][g]["mean"] for g in groups]
    stds = [stats["groups"][g]["std"] for g in groups]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(groups, means, yerr=stds, capsize=4, color=["#3268a8", "#b85c38", "#9a6fb0", "#5c7a45"])
    ax.set_ylabel("mean_delta_vs_v999")
    ax.set_title(f"V910 causal matrix: {stats['classification']}")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(BOARDS / "V91000000_causal_matrix_visual.png", dpi=160)
    plt.close(fig)


def write_summary_md(stats: dict[str, Any], compact: dict[str, Any]) -> None:
    lines = [
        "# V91000000 Core Multi-Seed Matrix Summary",
        "",
        f"- created_utc: `{now()}`",
        f"- classification: `{stats['classification']}`",
        f"- reason: {stats['classification_reason']}",
        f"- bootstrap_resamples: `{stats['bootstrap_resamples']}`",
        "",
        "## Group Means",
    ]
    for group in GROUPS:
        g = stats["groups"][group]
        lines.append(f"- `{group}`: mean={g['mean']:.9f}, std={g['std']:.9f}, seeds={g['seed_count']}, failure_rate={g['failure_rate']:.3f}")
    lines += [
        "",
        "## Key Paired Differences",
    ]
    for key, comp in stats["paired"].items():
        r = comp["regions"]["mean_delta_vs_v999"]
        lines.append(
            f"- `{key}`: mean_diff={r['mean_diff']:.9f}, "
            f"CI95=[{r['bootstrap_ci_low']:.9f}, {r['bootstrap_ci_high']:.9f}], effect={r['effect_size']:.3f}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "V910 does not confirm a clean SMPL semantic causal claim by itself because random/shuffled SMPL controls remain positive and their means are close to or above the true-SMPL group. V915 architecture diagnosis is mandatory before any paper-grade claim.",
        "",
        "## Compact Upload",
        "",
        f"- compact_bundle: `{compact['bundle']}`",
        f"- compact_bundle_size: `{compact['bundle_size']}`",
        f"- compact_bundle_sha256: `{compact['bundle_sha256']}`",
    ]
    (REPORTS / "V91000000_core_multiseed_matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 V910 rows, found {len(rows)}")
    write_csv(REPORTS / "V91000000_core_multiseed_matrix_results.csv", rows)
    write_csv(REPORTS / "V91000000_seed_level_metrics.csv", rows)
    stats = paired_stats(rows)
    write_json(REPORTS / "V91000000_core_multiseed_matrix_statistics.json", stats)
    checks = unique_checks(rows)
    write_csv(REPORTS / "V91000000_unique_prediction_checks.csv", checks)
    write_json(REPORTS / "V91000000_unique_prediction_checks.json", checks)
    manifest = copy_modal_logs(rows)
    write_json(REPORTS / "V91000000_modal_job_manifest.json", manifest)
    failures = [
        {
            "run_id": r["run_id"],
            "group": r["group"],
            "seed_index": r["seed_index"],
            "returncode": r["returncode"],
            "status": r["status"],
            "failure_reason": r["failure_reason"],
        }
        for r in rows
        if r.get("returncode") not in {None, "", 0, "0"} or r.get("failure_reason")
    ]
    write_json(ROOT / "failures" / "V91000000_failed_jobs.json", failures)
    compact = compact_predictions(rows)
    write_json(REPORTS / "V91000000_compact_upload_plan.json", compact)
    make_board(stats)
    write_summary_md(stats, compact)
    write_json(
        REPORTS / "V91000000_core_multiseed_matrix_progress.json",
        {
            "created_utc": now(),
            "completed": 20,
            "total": 20,
            "classification": stats["classification"],
            "results_csv": str(REPORTS / "V91000000_core_multiseed_matrix_results.csv"),
            "summary_md": str(REPORTS / "V91000000_core_multiseed_matrix_summary.md"),
            "latest": rows[-1],
        },
    )
    print(json.dumps({"status": "V910_FINALIZED", "classification": stats["classification"], "compact": compact}, indent=2))


if __name__ == "__main__":
    main()
