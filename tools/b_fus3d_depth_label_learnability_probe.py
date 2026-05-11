from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QUERY_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_LABELS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D9_colmap_depth_sdf_labels_hybrid6_layer23/"
    "b_fus3d_colmap_depth_sdf_labels_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D10_depth_label_learnability_probe_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_depth_label_learnability_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "label_learnability_probe_only": True,
    "not_sdf_decoder": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train_vggt": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_checkpoint": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D10 learnability probe. It fits a fixed ridge/"
            "prototype classifier from B-Fus3D query token evidence to B-Fus3D9 "
            "COLMAP-depth weak labels. This tests whether the token evidence can "
            "predict front/surface/behind labels at all. It is not a decoder, not "
            "VGGT training, not a teacher, not a candidate, and writes no pass state."
        )
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--feature-dim", type=int, default=96)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--min-support", type=int, default=2)
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
    return value


def load_inputs(query_cache: Path, labels: Path) -> dict[str, np.ndarray]:
    with np.load(query_cache.expanduser().resolve(), allow_pickle=False) as payload:
        features = np.asarray(payload["mean_features"], dtype=np.float32)
        variances = np.asarray(payload["variance_features"], dtype=np.float32)
        support = np.asarray(payload["support"], dtype=np.int32)
        families = np.asarray(payload["query_families"]).astype(str)
        positions = np.asarray(payload["query_positions"], dtype=np.float32)
    with np.load(labels.expanduser().resolve(), allow_pickle=False) as payload:
        y = np.asarray(payload["query_label"], dtype=np.int64)
        near = np.asarray(payload["query_near_count"], dtype=np.int32)
        residual = np.asarray(payload["query_residual"], dtype=np.float32)
    if features.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label query count mismatch: {features.shape[0]} vs {y.shape[0]}")
    return {
        "features": features,
        "variances": variances,
        "support": support,
        "families": families,
        "positions": positions,
        "labels": y,
        "near_count": near,
        "residual": residual,
    }


def normalize_features(features: np.ndarray, variances: np.ndarray, support: np.ndarray) -> np.ndarray:
    x = np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            np.log1p(np.asarray(variances, dtype=np.float32)),
            support[:, None].astype(np.float32) / 6.0,
        ],
        axis=1,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / np.clip(std, 1e-6, None)).astype(np.float32)


def pca_project(x: np.ndarray, dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    if x.shape[1] <= dim:
        return x.astype(np.float32), {"method": "identity", "feature_dim": int(x.shape[1])}
    u, s, vt = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    basis = vt[: int(dim)].T
    projected = x.astype(np.float64) @ basis
    total = float(np.sum(s**2))
    kept = float(np.sum(s[: int(dim)] ** 2))
    return projected.astype(np.float32), {
        "method": "svd_pca",
        "input_dim": int(x.shape[1]),
        "feature_dim": int(dim),
        "explained_variance_ratio": kept / total if total > 0.0 else 0.0,
    }


def split_indices(y: np.ndarray, families: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_rows: list[int] = []
    test_rows: list[int] = []
    for family in sorted(set(families.tolist())):
        for label in sorted(set(y.tolist())):
            rows = np.flatnonzero((families == family) & (y == label))
            if rows.size == 0:
                continue
            rng.shuffle(rows)
            if rows.size == 1:
                train_rows.extend(rows.tolist())
                continue
            n_test = max(1, int(round(rows.size * 0.35)))
            test_rows.extend(rows[:n_test].tolist())
            train_rows.extend(rows[n_test:].tolist())
    return np.asarray(sorted(train_rows), dtype=np.int64), np.asarray(sorted(test_rows), dtype=np.int64)


def fit_ridge_classifier(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    labels = sorted(set(int(v) for v in y.tolist()))
    y_onehot = np.zeros((x.shape[0], len(labels)), dtype=np.float64)
    label_to_col = {label: idx for idx, label in enumerate(labels)}
    for row, label in enumerate(y.tolist()):
        y_onehot[row, label_to_col[int(label)]] = 1.0
    design = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y_onehot)
    return weights.astype(np.float64), np.asarray(labels, dtype=np.int64)


def predict_ridge(x: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    design = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    scores = design @ weights
    return labels[np.argmax(scores, axis=1)]


def predict_prototype(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    labels = np.asarray(sorted(set(int(v) for v in y_train.tolist())), dtype=np.int64)
    centers = []
    for label in labels:
        centers.append(x_train[y_train == label].mean(axis=0))
    centers_arr = np.stack(centers, axis=0).astype(np.float32)
    distances = ((x_eval[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(distances, axis=1)]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(set([int(v) for v in y_true.tolist()] + [int(v) for v in y_pred.tolist()]))
    correct = y_true == y_pred
    rows: dict[str, Any] = {}
    for label in labels:
        mask = y_true == label
        pred_mask = y_pred == label
        tp = int((mask & pred_mask).sum())
        fp = int((~mask & pred_mask).sum())
        fn = int((mask & ~pred_mask).sum())
        rows[str(label)] = {
            "support": int(mask.sum()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
        }
    return {
        "count": int(y_true.shape[0]),
        "accuracy": float(correct.mean()) if correct.size else 0.0,
        "class_metrics": rows,
    }


def grouped_metrics(y: np.ndarray, pred: np.ndarray, families: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted(set(families[rows].tolist())):
        mask = families[rows] == family
        out[family] = metrics(y[rows][mask], pred[mask])
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D10 Depth-Label Learnability Probe",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        "This local probe tests whether frozen VGGT query-token evidence can predict",
        "A5 per-view COLMAP-depth weak labels. It is not VGGT training, not a surface",
        "decoder, not a teacher, not a candidate, and not a pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Results",
        "",
        "```json",
        json.dumps(summary["results"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Results",
        "",
        "```json",
        json.dumps(summary["family_results"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(args.query_cache, args.labels)
    label_mask = (data["labels"] >= 0) & (data["support"] >= int(args.min_support))
    # Labels: 0=front, 1=surface, 2=behind. Unknown/ambiguous labels are intentionally excluded.
    class_mask = label_mask & np.isin(data["labels"], [0, 1, 2])
    selected = np.flatnonzero(class_mask)
    x_raw = normalize_features(data["features"][selected], data["variances"][selected], data["support"][selected])
    x, projection = pca_project(x_raw, int(args.feature_dim))
    y = data["labels"][selected].astype(np.int64)
    families = data["families"][selected]
    train_local, test_local = split_indices(y, families, int(args.seed))
    weights, class_labels = fit_ridge_classifier(x[train_local], y[train_local], float(args.ridge))
    ridge_train = predict_ridge(x[train_local], weights, class_labels)
    ridge_test = predict_ridge(x[test_local], weights, class_labels)
    proto_test = predict_prototype(x[train_local], y[train_local], x[test_local])
    majority = np.bincount(y[train_local].clip(min=0), minlength=3).argmax()
    majority_test = np.full_like(y[test_local], int(majority))

    results = {
        "eligible_queries": int(selected.shape[0]),
        "train_queries": int(train_local.shape[0]),
        "test_queries": int(test_local.shape[0]),
        "label_counts": {str(label): int((y == label).sum()) for label in sorted(set(y.tolist()))},
        "projection": projection,
        "ridge_train": metrics(y[train_local], ridge_train),
        "ridge_test": metrics(y[test_local], ridge_test),
        "prototype_test": metrics(y[test_local], proto_test),
        "majority_test": metrics(y[test_local], majority_test),
    }
    family_results = {
        "ridge_test_by_family": grouped_metrics(y, ridge_test, families, test_local),
        "prototype_test_by_family": grouped_metrics(y, proto_test, families, test_local),
    }
    ridge_beats_majority = results["ridge_test"]["accuracy"] >= results["majority_test"]["accuracy"] + 0.10
    surface_recall = results["ridge_test"]["class_metrics"].get("1", {}).get("recall", 0.0)
    if ridge_beats_majority and surface_recall >= 0.35:
        decision = (
            "Frozen VGGT query-token evidence can predict A5 per-view depth labels above "
            "a majority baseline in this fixed probe. This supports a later research-only "
            "B-Fus3D query decoder using COLMAP depth labels as weak local supervision, "
            "while keeping A5 teacher-negative and strict gates red."
        )
    else:
        decision = (
            "The fixed learnability probe does not show enough predictive signal over "
            "the majority baseline. Do not spend cycles on a learned B-Fus3D decoder "
            "with these labels until stronger supervision or better token/ROI evidence exists."
        )

    arrays_path = out_dir / "b_fus3d_depth_label_learnability_arrays.npz"
    np.savez_compressed(
        arrays_path,
        selected_query_indices=selected.astype(np.int64),
        train_indices=selected[train_local].astype(np.int64),
        test_indices=selected[test_local].astype(np.int64),
        labels=y.astype(np.int64),
        ridge_test_pred=ridge_test.astype(np.int64),
        prototype_test_pred=proto_test.astype(np.int64),
        families=families,
    )
    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d10_depth_label_learnability_probe",
        "truthful_status": "research_only_learnability_probe_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "query_cache": str(args.query_cache.expanduser().resolve()),
            "labels": str(args.labels.expanduser().resolve()),
        },
        "configuration": {
            "seed": int(args.seed),
            "feature_dim": int(args.feature_dim),
            "ridge": float(args.ridge),
            "min_support": int(args.min_support),
        },
        "results": results,
        "family_results": family_results,
        "decision": decision,
        "outputs": {
            "arrays_npz": str(arrays_path),
            "summary_json": str(out_dir / "b_fus3d_depth_label_learnability_summary.json"),
            "summary_md": str(out_dir / "b_fus3d_depth_label_learnability_summary.md"),
            "status_report": str(args.status_report.expanduser().resolve()),
        },
    }
    summary = json_ready(summary)
    (out_dir / "b_fus3d_depth_label_learnability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "b_fus3d_depth_label_learnability_summary.md", summary)
    write_markdown(args.status_report.expanduser().resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
