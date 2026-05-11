from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BFUS_QUERY_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_BFUS_LABELS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D9_colmap_depth_sdf_labels_hybrid6_layer23/"
    "b_fus3d_colmap_depth_sdf_labels_arrays.npz"
)
DEFAULT_BHAND_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_hand5_label_evidence_learnability_probe_hybrid6/"
    "b_hand_label_evidence_learnability_arrays.npz"
)
DEFAULT_BHAND_DEPTH = Path(
    "output/surface_research_preflight_local/"
    "B_hand6_colmap_depth_evidence_probe_hybrid12/"
    "b_hand_colmap_depth_evidence_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_joint_surface_hand_contract_probe_hybrid6"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_joint_surface_hand_contract_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "joint_contract_probe_only": True,
    "fixed_smoke_not_tuning_loop": True,
    "not_decoder": True,
    "not_surface_or_hand_training": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_vggt_training": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_checkpoint": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only joint surface-token/hand-token contract probe. It checks whether "
            "global/family B-Fus3D context can add signal to B-hand ROI evidence for weak "
            "hand risk labels. It does not train, render, write predictions, export a "
            "teacher/candidate, or write strict registry state."
        )
    )
    parser.add_argument("--bfus-query-cache", type=Path, default=DEFAULT_BFUS_QUERY_CACHE)
    parser.add_argument("--bfus-labels", type=Path, default=DEFAULT_BFUS_LABELS)
    parser.add_argument("--bhand-arrays", type=Path, default=DEFAULT_BHAND_ARRAYS)
    parser.add_argument("--bhand-depth", type=Path, default=DEFAULT_BHAND_DEPTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def scalar_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return {"count": int(arr.size), "finite": int(finite.sum())}
    vals = arr[finite]
    return {
        "count": int(arr.size),
        "finite": int(finite.sum()),
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)),
    }


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sigma = np.nanstd(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    out = (x - mu) / sigma
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def ridge_loo(x: np.ndarray, y: np.ndarray, *, ridge: float) -> dict[str, Any]:
    x = standardize(x)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    classes = np.asarray(sorted(set(int(v) for v in y.tolist())), dtype=np.int64)
    preds = []
    for idx in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[idx] = False
        xt = x[train]
        yt = y[train]
        design = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float64)], axis=1)
        test = np.concatenate([x[idx : idx + 1], np.ones((1, 1), dtype=np.float64)], axis=1)
        targets = np.stack([(yt == cls).astype(np.float64) for cls in classes], axis=1)
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=np.float64)
        gram[-1, -1] -= float(ridge)
        try:
            weights = np.linalg.solve(gram, design.T @ targets)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(gram) @ design.T @ targets
        scores = test @ weights
        preds.append(int(classes[int(np.argmax(scores[0]))]))
    return metrics(y, np.asarray(preds, dtype=np.int64), name="ridge_leave_one_out")


def prototype_loo(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = standardize(x)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    classes = np.asarray(sorted(set(int(v) for v in y.tolist())), dtype=np.int64)
    preds = []
    for idx in range(len(y)):
        train = np.ones(len(y), dtype=bool)
        train[idx] = False
        centers = []
        for cls in classes:
            cls_x = x[train & (y == cls)]
            if cls_x.size == 0:
                centers.append(np.zeros(x.shape[1], dtype=np.float64))
            else:
                centers.append(cls_x.mean(axis=0))
        centers_arr = np.stack(centers, axis=0)
        dist = ((centers_arr - x[idx][None, :]) ** 2).mean(axis=1)
        preds.append(int(classes[int(np.argmin(dist))]))
    return metrics(y, np.asarray(preds, dtype=np.int64), name="prototype_leave_one_out")


def majority_baseline(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    values, counts = np.unique(y, return_counts=True)
    majority = int(values[int(np.argmax(counts))])
    return metrics(y, np.full_like(y, majority), name="majority")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, *, name: str) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_metrics = {}
    recalls = []
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        class_metrics[str(cls)] = {"support": support, "precision": precision, "recall": recall}
        if support:
            recalls.append(recall)
    return {
        "name": name,
        "count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "class_metrics": class_metrics,
    }


def family_context(query_cache: np.lib.npyio.NpzFile, labels: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    families = np.asarray(query_cache["query_families"]).astype(str)
    support = np.asarray(query_cache["support"], dtype=np.float32)
    query_label = np.asarray(labels["query_label"], dtype=np.int64)
    query_valid = np.asarray(labels["query_valid_count"], dtype=np.float32)
    query_margin = np.asarray(labels["query_center_margin"], dtype=np.float32)
    best_offset = np.asarray(labels["query_best_shell_offset"], dtype=np.float32)
    shell_valid = np.asarray(labels["query_shell_curve_valid"], dtype=bool)
    feature_blocks = []
    names = []
    summary = {}
    for family in ["full_body", "face_core", "hairline", "left_hand", "right_hand"]:
        mask = families == family
        valid = mask & (query_label >= 0)
        if not np.any(mask):
            vals = np.zeros(9, dtype=np.float32)
        else:
            fam_labels = query_label[valid] if np.any(valid) else np.asarray([], dtype=np.int64)
            vals = np.asarray(
                [
                    float(np.mean(mask)),
                    float(np.mean(support[mask] >= 1)),
                    float(np.mean(support[mask] >= 2)),
                    float(np.mean(support[mask] >= 3)),
                    float(np.mean(query_valid[mask])),
                    float(np.mean(query_label[valid] == 0)) if np.any(valid) else 0.0,
                    float(np.mean(query_label[valid] == 1)) if np.any(valid) else 0.0,
                    float(np.mean(query_label[valid] == 2)) if np.any(valid) else 0.0,
                    float(np.mean(shell_valid[mask])),
                    float(np.nanmean(query_margin[mask])),
                    float(np.nanmean(best_offset[mask])),
                ],
                dtype=np.float32,
            )
            summary[family] = {
                "query_count": int(np.sum(mask)),
                "valid_label_count": int(np.sum(valid)),
                "support_mean": float(np.mean(support[mask])),
                "label_counts": {str(int(k)): int(np.sum(fam_labels == k)) for k in np.unique(fam_labels)} if fam_labels.size else {},
                "shell_curve_valid_ratio": float(np.mean(shell_valid[mask])),
                "center_margin_stats": scalar_stats(query_margin[mask]),
            }
        feature_blocks.append(vals)
        names.extend(
            [
                f"{family}_query_fraction",
                f"{family}_support_ge1",
                f"{family}_support_ge2",
                f"{family}_support_ge3",
                f"{family}_valid_count_mean",
                f"{family}_label_front_frac",
                f"{family}_label_surface_frac",
                f"{family}_label_behind_frac",
                f"{family}_shell_curve_valid_frac",
                f"{family}_center_margin_mean",
                f"{family}_best_offset_mean",
            ]
        )
    context = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    return context, names, summary


def run_task(name: str, x: np.ndarray, y: np.ndarray, *, ridge: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    counts = {str(int(k)): int(np.sum(y == k)) for k in np.unique(y)}
    if len(y) < 4 or len(counts) < 2:
        return {"task": name, "status": "skipped_insufficient_labels", "label_counts": counts}
    ridge_m = ridge_loo(x, y, ridge=ridge)
    proto_m = prototype_loo(x, y)
    maj_m = majority_baseline(y)
    return {
        "task": name,
        "status": "ok_leave_one_out_fixed",
        "label_counts": counts,
        "ridge": ridge_m,
        "prototype": proto_m,
        "majority": maj_m,
        "ridge_gain_over_majority": float(ridge_m["accuracy"] - maj_m["accuracy"]),
        "prototype_gain_over_majority": float(proto_m["accuracy"] - maj_m["accuracy"]),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Joint Surface/Hand Contract Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a one-shot research-only contract probe. It concatenates a global",
        "B-Fus3D family context vector onto B-hand ROI evidence and checks whether",
        "the joint interface improves weak hand risk readouts. It does not render,",
        "train, export teacher/candidate, write predictions, or touch the strict registry.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud_train_infer_export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "```",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(summary["inputs"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Context",
        "",
        "```json",
        json.dumps(summary["bfus_family_context_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Results",
        "",
        "```json",
        json.dumps(summary["results"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(summary["decision"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Stop Rules",
        "",
        "- If combined features do not beat hand-only controls, freeze this lightweight interface smoke.",
        "- If gains depend on absolute x/view-id leakage, freeze this lightweight interface smoke.",
        "- No Open3D hand/fullbody visual pass is implied by this probe.",
        "- No cloud/formal training/export is allowed from this result.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    bfus_cache = np.load(args.bfus_query_cache, allow_pickle=True)
    bfus_labels = np.load(args.bfus_labels, allow_pickle=True)
    bhand = np.load(args.bhand_arrays, allow_pickle=True)
    bhand_depth = np.load(args.bhand_depth, allow_pickle=True)

    context, context_names, context_summary = family_context(bfus_cache, bfus_labels)
    n = int(bhand["features_all"].shape[0])
    repeated_context = np.repeat(context[None, :], n, axis=0)
    hand_all = np.asarray(bhand["features_all"], dtype=np.float32)
    hand_no_x = np.asarray(bhand["features_no_absolute_x"], dtype=np.float32)
    hand_depth_no_direct = np.asarray(bhand["features_depth_no_direct"], dtype=np.float32)
    depth_ratio = np.asarray(bhand_depth["depth_valid_ratio"], dtype=np.float32).reshape(-1, 1)
    depth_present = np.asarray(bhand_depth["depth_present"], dtype=np.float32).reshape(-1, 1)
    if depth_ratio.shape[0] != n:
        raise ValueError(f"B-hand depth rows {depth_ratio.shape[0]} do not match B-hand evidence rows {n}")
    depth_ratio = np.nan_to_num(depth_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    hand_depth_evidence = np.concatenate([hand_depth_no_direct, depth_ratio, depth_present], axis=1)
    combined = np.concatenate([hand_depth_evidence, repeated_context], axis=1)
    combined_no_x = np.concatenate([hand_no_x, depth_ratio, depth_present, repeated_context], axis=1)
    surface_context_only = np.concatenate([repeated_context, depth_ratio, depth_present], axis=1)

    y_depth = np.asarray(bhand["depth_risk_labels"], dtype=np.int64)
    y_connection = np.asarray(bhand["connection_risk_labels"], dtype=np.int64)
    results = {
        "depth_risk_hand_only_no_direct": run_task(
            "depth_risk_hand_only_no_direct", hand_depth_no_direct, y_depth, ridge=float(args.ridge)
        ),
        "depth_risk_hand_plus_colmap_depth": run_task(
            "depth_risk_hand_plus_colmap_depth", hand_depth_evidence, y_depth, ridge=float(args.ridge)
        ),
        "depth_risk_surface_context_only": run_task(
            "depth_risk_surface_context_only", surface_context_only, y_depth, ridge=float(args.ridge)
        ),
        "depth_risk_combined": run_task("depth_risk_combined", combined, y_depth, ridge=float(args.ridge)),
        "depth_risk_combined_no_absolute_x": run_task(
            "depth_risk_combined_no_absolute_x", combined_no_x, y_depth, ridge=float(args.ridge)
        ),
        "connection_risk_hand_only": run_task(
            "connection_risk_hand_only", np.asarray(bhand["features_connection_no_direct"], dtype=np.float32), y_connection, ridge=float(args.ridge)
        ),
        "connection_risk_combined": run_task("connection_risk_combined", combined, y_connection, ridge=float(args.ridge)),
    }

    hand_base = results["depth_risk_hand_only_no_direct"].get("ridge", {}).get("accuracy", 0.0)
    hand_depth = results["depth_risk_hand_plus_colmap_depth"].get("ridge", {}).get("accuracy", 0.0)
    combined_acc = results["depth_risk_combined"].get("ridge", {}).get("accuracy", 0.0)
    combined_no_x_acc = results["depth_risk_combined_no_absolute_x"].get("ridge", {}).get("accuracy", 0.0)
    surface_only_acc = results["depth_risk_surface_context_only"].get("ridge", {}).get("accuracy", 0.0)
    combined_beats_hand = bool(combined_acc > max(hand_base, hand_depth) + 1e-6)
    no_x_survives = bool(combined_no_x_acc >= max(hand_base, 0.0) + 1e-6)
    decision = {
        "status": "research_joint_contract_probe_no_pass",
        "combined_beats_hand_only": combined_beats_hand,
        "combined_no_absolute_x_survives": no_x_survives,
        "surface_context_only_accuracy": float(surface_only_acc),
        "hand_only_depth_risk_accuracy": float(hand_base),
        "hand_plus_colmap_depth_accuracy": float(hand_depth),
        "combined_accuracy": float(combined_acc),
        "combined_no_absolute_x_accuracy": float(combined_no_x_acc),
        "interpretation": (
            "joint_interface_has_additional_fixed_readout_signal_but_still_no_decoder_or_visual_pass"
            if combined_beats_hand and no_x_survives
            else "no_stable_additional_joint_signal_beyond_existing_hand_controls"
        ),
        "next_allowed_action": (
            "If a future backend is built, use this only as an interface contract; do not train a decoder from this probe alone."
        ),
        "blocked_actions": [
            "do_not_train_surface_or_hand_decoder_from_this_probe",
            "do_not_claim_hand_or_fullbody_gate_pass",
            "do_not_export_teacher_or_candidate",
            "do_not_unblock_formal_cloud",
            "do_not_tune_steps_hidden_thresholds_or_viewsets",
        ],
    }

    arrays_path = output_dir / "b_joint_surface_hand_contract_arrays.npz"
    np.savez_compressed(
        arrays_path,
        bfus_context=context.astype(np.float32),
        bfus_context_names=np.asarray(context_names),
        hand_depth_evidence=hand_depth_evidence.astype(np.float32),
        surface_context_only=surface_context_only.astype(np.float32),
        combined=combined.astype(np.float32),
        combined_no_absolute_x=combined_no_x.astype(np.float32),
        depth_risk_labels=y_depth,
        connection_risk_labels=y_connection,
    )

    summary = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "status": "research_only_joint_contract_probe_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "bfus_query_cache": str(args.bfus_query_cache.expanduser().resolve()),
            "bfus_labels": str(args.bfus_labels.expanduser().resolve()),
            "bhand_arrays": str(args.bhand_arrays.expanduser().resolve()),
            "bhand_depth": str(args.bhand_depth.expanduser().resolve()),
        },
        "bfus_family_context_summary": context_summary,
        "feature_shapes": {
            "hand_depth_evidence": list(hand_depth_evidence.shape),
            "surface_context_only": list(surface_context_only.shape),
            "combined": list(combined.shape),
            "combined_no_absolute_x": list(combined_no_x.shape),
        },
        "results": results,
        "decision": decision,
        "outputs": {
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_joint_surface_hand_contract_summary.json"),
            "summary_md": str(output_dir / "b_joint_surface_hand_contract_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    summary_path = output_dir / "b_joint_surface_hand_contract_summary.json"
    md_path = output_dir / "b_joint_surface_hand_contract_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
