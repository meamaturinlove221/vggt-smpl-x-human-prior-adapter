from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_HAND_EVIDENCE_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_hand0_evidence_cache_60v_humancrop_hybrid6/"
    "b_hand_evidence_cache.json"
)
DEFAULT_HAND_TOKEN_SMOKE = Path(
    "output/surface_research_preflight_local/"
    "B_hand1_token_backend_smoke_hybrid6/"
    "b_hand_token_backend_smoke_summary.json"
)
DEFAULT_CONNECTED_PRECHECK = Path(
    "output/surface_research_preflight_local/"
    "B_hand4_connected_mesh_precheck_hybrid6/"
    "b_hand_connected_mesh_precheck_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_hand5_label_evidence_learnability_probe_hybrid6"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_hand_label_evidence_learnability_status.md")

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
    "label_evidence_learnability_probe_only": True,
    "fixed_smoke_not_tuning_loop": True,
    "not_hand_decoder": True,
    "not_mano_or_smplx_success_claim": True,
    "not_smplx_scaffold_success": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_vggt_training": True,
    "no_inference_export": True,
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

SIDE_LABELS = {"left": 0, "right": 1}
SIDE_NAMES = {0: "left", 1: "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-hand5 learnability probe. It consumes frozen "
            "B-hand ROI/token evidence and tests whether fixed evidence vectors "
            "can predict weak left/right labels or weak connection/depth risk "
            "labels. It does not train VGGT, export predictions, export a "
            "teacher/candidate, write a strict pass, or call cloud."
        )
    )
    parser.add_argument("--hand-evidence-cache", type=Path, default=DEFAULT_HAND_EVIDENCE_CACHE)
    parser.add_argument("--hand-token-smoke", type=Path, default=DEFAULT_HAND_TOKEN_SMOKE)
    parser.add_argument("--connected-precheck", type=Path, default=DEFAULT_CONNECTED_PRECHECK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--min-risk-roi-pixels", type=int, default=64)
    parser.add_argument("--min-risk-visible-ratio", type=float, default=0.25)
    parser.add_argument("--max-depth-range", type=float, default=0.75)
    parser.add_argument("--max-pca-thin-ratio", type=float, default=0.35)
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


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(resolved)
        return {"available": False, "path": str(resolved), "reason": "missing"}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload in {resolved}")
    payload["_resolved_path"] = str(resolved)
    return payload


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-8)


def log1p_clip(value: Any) -> float:
    return float(np.log1p(max(0.0, as_float(value))))


def numeric_camera_id(value: Any) -> float:
    text = str(value)
    try:
        return float(int(text))
    except ValueError:
        total = sum(ord(ch) for ch in text)
        return float(total % 1000)


def vector_stats(values: Any, prefix: str) -> tuple[list[str], list[float]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return (
            [f"{prefix}_mean_x", f"{prefix}_mean_y", f"{prefix}_mean_z", f"{prefix}_std_x", f"{prefix}_std_y", f"{prefix}_std_z"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.shape[0] == 0:
        mean = np.zeros(3, dtype=np.float32)
        std = np.zeros(3, dtype=np.float32)
    else:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
    return (
        [f"{prefix}_mean_x", f"{prefix}_mean_y", f"{prefix}_mean_z", f"{prefix}_std_x", f"{prefix}_std_y", f"{prefix}_std_z"],
        [float(v) for v in np.concatenate([mean, std], axis=0).tolist()],
    )


def add_feature(names: list[str], values: list[float], name: str, value: Any) -> None:
    names.append(name)
    values.append(as_float(value))


def add_vector_features(names: list[str], values: list[float], feature_names: list[str], feature_values: list[float]) -> None:
    names.extend(feature_names)
    values.extend(as_float(v) for v in feature_values)


def token_preview_stats(hook: dict[str, Any]) -> tuple[float, float]:
    tokens = np.asarray(hook.get("token_ids_preview", []), dtype=np.float32)
    if tokens.size == 0:
        return 0.0, 0.0
    grid = hook.get("patch_grid_hw") if isinstance(hook.get("patch_grid_hw"), list) else [1, 1]
    grid_area = max(1.0, as_float(grid[0], 1.0) * as_float(grid[1], 1.0))
    return float(tokens.mean() / grid_area), float(tokens.std() / grid_area)


def feature_row(row: dict[str, Any], view_payload: dict[str, Any], max_view_index: int, max_camera_id: float) -> tuple[list[str], list[float]]:
    crop = row.get("crop_metadata") if isinstance(row.get("crop_metadata"), dict) else {}
    prior = row.get("smplx_prior") if isinstance(row.get("smplx_prior"), dict) else {}
    spatial = prior.get("spatial") if isinstance(prior.get("spatial"), dict) else {}
    support = row.get("prediction_support") if isinstance(row.get("prediction_support"), dict) else {}
    rays = row.get("camera_rays") if isinstance(row.get("camera_rays"), dict) else {}
    hook = row.get("vggt_token_hook") if isinstance(row.get("vggt_token_hook"), dict) else {}

    image_hw = crop.get("image_hw") if isinstance(crop.get("image_hw"), list) else [1, 1]
    image_h = max(1.0, as_float(image_hw[0], 1.0))
    image_w = max(1.0, as_float(image_hw[1], 1.0))
    bbox = row.get("bbox_xyxy") if isinstance(row.get("bbox_xyxy"), list) and len(row.get("bbox_xyxy")) == 4 else [0, 0, 0, 0]
    x0, y0, x1, y1 = [as_float(v) for v in bbox]
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    body_bbox = view_payload.get("body_bbox_xyxy") if isinstance(view_payload.get("body_bbox_xyxy"), list) else [0, 0, image_w, image_h]
    bx0, by0, bx1, by1 = [as_float(v) for v in body_bbox]
    body_w = max(1.0, bx1 - bx0)
    body_h = max(1.0, by1 - by0)
    body_cx = 0.5 * (bx0 + bx1)
    body_cy = 0.5 * (by0 + by1)

    component = crop.get("component_stats") if isinstance(crop.get("component_stats"), dict) else {}
    patch_grid = hook.get("patch_grid_hw") if isinstance(hook.get("patch_grid_hw"), list) and len(hook.get("patch_grid_hw")) == 2 else [1, 1]
    grid_h = max(1.0, as_float(patch_grid[0], 1.0))
    grid_w = max(1.0, as_float(patch_grid[1], 1.0))
    patch_range = hook.get("patch_range_xyxy") if isinstance(hook.get("patch_range_xyxy"), list) and len(hook.get("patch_range_xyxy")) == 4 else [0, 0, 0, 0]
    px0, py0, px1, py1 = [as_float(v) for v in patch_range]
    patch_count = as_float(hook.get("estimated_patch_token_count", 0))
    token_mean, token_std = token_preview_stats(hook)

    names: list[str] = []
    values: list[float] = []
    add_feature(names, values, "view_index_norm", safe_div(as_float(row.get("view_index")), max(1, max_view_index)))
    add_feature(names, values, "camera_id_norm", safe_div(numeric_camera_id(row.get("camera_id", 0)), max_camera_id))
    add_feature(names, values, "bbox_x0_norm", x0 / image_w)
    add_feature(names, values, "bbox_y0_norm", y0 / image_h)
    add_feature(names, values, "bbox_x1_norm", x1 / image_w)
    add_feature(names, values, "bbox_y1_norm", y1 / image_h)
    add_feature(names, values, "bbox_cx_norm", cx / image_w)
    add_feature(names, values, "bbox_cy_norm", cy / image_h)
    add_feature(names, values, "bbox_w_norm", bw / image_w)
    add_feature(names, values, "bbox_h_norm", bh / image_h)
    add_feature(names, values, "bbox_area_norm", safe_div(bw * bh, image_w * image_h))
    add_feature(names, values, "bbox_aspect_w_over_h", safe_div(bw, bh))
    add_feature(names, values, "bbox_log_area", log1p_clip(bw * bh))
    add_feature(names, values, "body_rel_cx_norm", safe_div(cx - body_cx, body_w))
    add_feature(names, values, "body_rel_cy_norm", safe_div(cy - body_cy, body_h))
    add_feature(names, values, "body_rel_w_norm", bw / body_w)
    add_feature(names, values, "body_rel_h_norm", bh / body_h)
    add_feature(names, values, "body_bbox_w_norm", body_w / image_w)
    add_feature(names, values, "body_bbox_h_norm", body_h / image_h)
    add_feature(names, values, "roi_log_pixels", log1p_clip(crop.get("roi_pixels", 0)))
    add_feature(names, values, "roi_density_in_bbox", crop.get("roi_density_in_bbox", 0.0))
    add_feature(names, values, "roi_body_overlap_ratio", crop.get("body_overlap_ratio", 0.0))
    add_feature(names, values, "roi_component_count", component.get("components", 0))
    add_feature(names, values, "roi_largest_component_ratio", component.get("largest_component_ratio", 0.0))
    add_feature(names, values, "smplx_prior_available", 1.0 if prior.get("available") else 0.0)
    add_feature(names, values, "smplx_visible_log_pixels", log1p_clip(prior.get("visible_pixels", 0)))
    add_feature(names, values, "smplx_visible_ratio_in_roi", prior.get("visible_ratio_in_roi", 0.0))
    add_feature(names, values, "smplx_spatial_log_points", log1p_clip(spatial.get("points", 0)))
    extent = spatial.get("extent_xyz") if isinstance(spatial.get("extent_xyz"), list) and len(spatial.get("extent_xyz")) >= 3 else [0.0, 0.0, 0.0]
    add_feature(names, values, "smplx_extent_x", extent[0])
    add_feature(names, values, "smplx_extent_y", extent[1])
    add_feature(names, values, "smplx_extent_z", extent[2])
    add_feature(names, values, "smplx_max_extent", spatial.get("max_extent", 0.0))
    add_feature(names, values, "smplx_depth_range", spatial.get("depth_range", 0.0))
    add_feature(names, values, "smplx_pca_mid_ratio", spatial.get("pca_mid_ratio", 0.0))
    add_feature(names, values, "smplx_pca_thin_ratio", spatial.get("pca_thin_ratio", 0.0))
    add_feature(names, values, "prediction_support_available", 1.0 if support.get("available") else 0.0)
    add_feature(names, values, "prediction_support_log_pixels", log1p_clip(support.get("support_pixels", 0)))
    add_feature(names, values, "camera_rays_available", 1.0 if rays.get("available") else 0.0)
    camera_names, camera_values = vector_stats(rays.get("camera_ray_unit_samples", []), "camera_ray")
    world_names, world_values = vector_stats(rays.get("world_ray_unit_samples", []), "world_ray")
    add_vector_features(names, values, camera_names, camera_values)
    add_vector_features(names, values, world_names, world_values)
    add_feature(names, values, "patch_grid_h_norm", grid_h / image_h)
    add_feature(names, values, "patch_grid_w_norm", grid_w / image_w)
    add_feature(names, values, "patch_px0_norm", px0 / grid_w)
    add_feature(names, values, "patch_py0_norm", py0 / grid_h)
    add_feature(names, values, "patch_px1_norm", px1 / grid_w)
    add_feature(names, values, "patch_py1_norm", py1 / grid_h)
    add_feature(names, values, "patch_cx_norm", safe_div(0.5 * (px0 + px1), grid_w))
    add_feature(names, values, "patch_cy_norm", safe_div(0.5 * (py0 + py1), grid_h))
    add_feature(names, values, "patch_w_norm", safe_div(px1 - px0, grid_w))
    add_feature(names, values, "patch_h_norm", safe_div(py1 - py0, grid_h))
    add_feature(names, values, "patch_token_density", safe_div(patch_count, grid_h * grid_w))
    add_feature(names, values, "token_preview_mean_norm", token_mean)
    add_feature(names, values, "token_preview_std_norm", token_std)
    return names, values


def weak_connection_risk(row: dict[str, Any], *, min_roi_pixels: int, min_visible_ratio: float) -> bool:
    crop = row.get("crop_metadata") if isinstance(row.get("crop_metadata"), dict) else {}
    prior = row.get("smplx_prior") if isinstance(row.get("smplx_prior"), dict) else {}
    component = crop.get("component_stats") if isinstance(crop.get("component_stats"), dict) else {}
    roi_pixels = as_int(crop.get("roi_pixels", 0))
    visible_ratio = as_float(prior.get("visible_ratio_in_roi", 0.0))
    body_overlap = as_float(crop.get("body_overlap_ratio", 0.0))
    components = as_int(component.get("components", 0))
    largest_ratio = as_float(component.get("largest_component_ratio", 0.0))
    return bool(
        roi_pixels < int(min_roi_pixels)
        or visible_ratio < float(min_visible_ratio)
        or body_overlap < 0.001
        or components > 2
        or largest_ratio < 0.75
    )


def weak_depth_risk(row: dict[str, Any], *, max_depth_range: float, max_pca_thin_ratio: float) -> bool:
    prior = row.get("smplx_prior") if isinstance(row.get("smplx_prior"), dict) else {}
    spatial = prior.get("spatial") if isinstance(prior.get("spatial"), dict) else {}
    visible_ratio = as_float(prior.get("visible_ratio_in_roi", 0.0))
    points = as_int(spatial.get("points", 0))
    depth_range = as_float(spatial.get("depth_range", 0.0))
    max_extent = as_float(spatial.get("max_extent", 0.0))
    pca_thin = as_float(spatial.get("pca_thin_ratio", 0.0))
    return bool(
        depth_range >= float(max_depth_range)
        or max_extent >= float(max_depth_range)
        or pca_thin >= float(max_pca_thin_ratio)
        or (points <= 0 and visible_ratio < 0.1)
    )


def build_dataset(cache: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    per_view = cache.get("per_view") if isinstance(cache.get("per_view"), dict) else {}
    rows = cache.get("hand_crop_metadata") if isinstance(cache.get("hand_crop_metadata"), list) else []
    max_view_index = max([as_int(v) for v in cache.get("view_indices", [])] + [as_int(row.get("view_index", 0)) for row in rows] + [1])
    camera_values = [numeric_camera_id(row.get("camera_id", 0)) for row in rows]
    max_camera_id = max([1.0] + [max(1.0, value) for value in camera_values])

    feature_names: list[str] | None = None
    feature_values: list[list[float]] = []
    side_labels: list[int] = []
    connection_labels: list[int] = []
    depth_labels: list[int] = []
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        side = str(row.get("side", ""))
        if side not in SIDE_LABELS:
            continue
        view_key = str(row.get("view_index"))
        view_payload = per_view.get(view_key) if isinstance(per_view.get(view_key), dict) else {}
        names, values = feature_row(row, view_payload, max_view_index, max_camera_id)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Feature name mismatch while building B-hand5 dataset")
        feature_values.append(values)
        side_labels.append(SIDE_LABELS[side])
        conn_risk = weak_connection_risk(
            row,
            min_roi_pixels=int(args.min_risk_roi_pixels),
            min_visible_ratio=float(args.min_risk_visible_ratio),
        )
        depth_risk = weak_depth_risk(
            row,
            max_depth_range=float(args.max_depth_range),
            max_pca_thin_ratio=float(args.max_pca_thin_ratio),
        )
        connection_labels.append(int(conn_risk))
        depth_labels.append(int(depth_risk))
        records.append(
            {
                "row_index": int(row_idx),
                "view_index": as_int(row.get("view_index", 0)),
                "camera_id": str(row.get("camera_id", "")),
                "side_weak_label": side,
                "bbox_xyxy": row.get("bbox_xyxy"),
                "roi_pixels": as_int((row.get("crop_metadata") or {}).get("roi_pixels", 0)),
                "smplx_visible_ratio_in_roi": as_float((row.get("smplx_prior") or {}).get("visible_ratio_in_roi", 0.0)),
                "smplx_depth_range": as_float(((row.get("smplx_prior") or {}).get("spatial") or {}).get("depth_range", 0.0)),
                "weak_connection_risk": bool(conn_risk),
                "weak_depth_risk": bool(depth_risk),
            }
        )
    if not feature_values:
        raise ValueError("No left/right hand ROI rows found in evidence cache")
    return {
        "features": np.asarray(feature_values, dtype=np.float32),
        "feature_names": feature_names or [],
        "side_labels": np.asarray(side_labels, dtype=np.int64),
        "connection_labels": np.asarray(connection_labels, dtype=np.int64),
        "depth_labels": np.asarray(depth_labels, dtype=np.int64),
        "records": records,
    }


def no_absolute_x_indices(feature_names: list[str]) -> np.ndarray:
    blocked = {
        "bbox_x0_norm",
        "bbox_x1_norm",
        "bbox_cx_norm",
        "body_rel_cx_norm",
        "patch_px0_norm",
        "patch_px1_norm",
        "patch_cx_norm",
        "token_preview_mean_norm",
        "token_preview_std_norm",
        "camera_ray_mean_x",
        "world_ray_mean_x",
    }
    return np.asarray([idx for idx, name in enumerate(feature_names) if name not in blocked], dtype=np.int64)


def normalize_train_eval(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (
        ((x_train - mean) / std).astype(np.float64),
        ((x_eval - mean) / std).astype(np.float64),
    )


def majority_label(y_train: np.ndarray) -> int:
    labels, counts = np.unique(y_train.astype(np.int64), return_counts=True)
    max_count = counts.max()
    return int(labels[counts == max_count].min())


def fit_ridge_classifier(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(set(int(v) for v in y.tolist())), dtype=np.int64)
    y_onehot = np.zeros((x.shape[0], labels.shape[0]), dtype=np.float64)
    label_to_col = {int(label): idx for idx, label in enumerate(labels.tolist())}
    for row, label in enumerate(y.tolist()):
        y_onehot[row, label_to_col[int(label)]] = 1.0
    design = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y_onehot)
    return weights, labels


def predict_ridge(x: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    design = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    scores = design @ weights
    return labels[np.argmax(scores, axis=1)]


def predict_prototype(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    labels = np.asarray(sorted(set(int(v) for v in y_train.tolist())), dtype=np.int64)
    centers = []
    for label in labels:
        centers.append(x_train[y_train == label].mean(axis=0))
    centers_arr = np.stack(centers, axis=0).astype(np.float64)
    distances = ((x_eval[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(distances, axis=1)]


def leave_one_out_predictions(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    n = int(y.shape[0])
    labels = sorted(set(int(v) for v in y.tolist()))
    if n < 3 or len(labels) < 2:
        fallback = np.full(n, int(labels[0]) if labels else 0, dtype=np.int64)
        return {
            "status": "not_enough_classes",
            "ridge": fallback.copy(),
            "prototype": fallback.copy(),
            "majority": fallback.copy(),
        }
    ridge_pred = np.zeros(n, dtype=np.int64)
    proto_pred = np.zeros(n, dtype=np.int64)
    majority_pred = np.zeros(n, dtype=np.int64)
    for heldout in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[heldout] = False
        x_train, x_eval = normalize_train_eval(x[train_mask], x[heldout : heldout + 1])
        y_train = y[train_mask]
        maj = majority_label(y_train)
        majority_pred[heldout] = maj
        if np.unique(y_train).shape[0] < 2:
            ridge_pred[heldout] = maj
            proto_pred[heldout] = maj
            continue
        weights, class_labels = fit_ridge_classifier(x_train, y_train, ridge)
        ridge_pred[heldout] = int(predict_ridge(x_eval, weights, class_labels)[0])
        proto_pred[heldout] = int(predict_prototype(x_train, y_train, x_eval)[0])
    return {
        "status": "ok_leave_one_out_fixed",
        "ridge": ridge_pred,
        "prototype": proto_pred,
        "majority": majority_pred,
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(set([int(v) for v in y_true.tolist()] + [int(v) for v in y_pred.tolist()]))
    correct = y_true == y_pred
    class_rows: dict[str, Any] = {}
    recalls: list[float] = []
    for label in labels:
        truth = y_true == label
        pred = y_pred == label
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        recall = float(tp / max(tp + fn, 1))
        recalls.append(recall)
        class_rows[str(label)] = {
            "support": int(truth.sum()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": recall,
        }
    return {
        "count": int(y_true.shape[0]),
        "accuracy": float(correct.mean()) if correct.size else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "class_metrics": class_rows,
    }


def evaluate_task(name: str, y: np.ndarray, x: np.ndarray, ridge: float) -> dict[str, Any]:
    predictions = leave_one_out_predictions(x, y, ridge)
    ridge_metrics = metrics(y, predictions["ridge"])
    proto_metrics = metrics(y, predictions["prototype"])
    majority_metrics = metrics(y, predictions["majority"])
    labels, counts = np.unique(y, return_counts=True)
    ridge_gain = ridge_metrics["accuracy"] - majority_metrics["accuracy"]
    proto_gain = proto_metrics["accuracy"] - majority_metrics["accuracy"]
    signal = bool(
        predictions["status"] == "ok_leave_one_out_fixed"
        and len(labels) >= 2
        and int(counts.min()) >= 2
        and ridge_metrics["accuracy"] >= majority_metrics["accuracy"] + 0.10
        and ridge_metrics["balanced_accuracy"] >= 0.60
    )
    return {
        "task": name,
        "label_counts": {str(int(label)): int(count) for label, count in zip(labels.tolist(), counts.tolist())},
        "evaluation": predictions["status"],
        "ridge_leave_one_out": ridge_metrics,
        "prototype_leave_one_out": proto_metrics,
        "majority_leave_one_out": majority_metrics,
        "ridge_accuracy_gain_over_majority": float(ridge_gain),
        "prototype_accuracy_gain_over_majority": float(proto_gain),
        "learnability_signal": signal,
        "_ridge_pred": predictions["ridge"],
        "_prototype_pred": predictions["prototype"],
        "_majority_pred": predictions["majority"],
    }


def strip_private_arrays(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def upstream_readout(token_smoke: dict[str, Any], connected_precheck: dict[str, Any]) -> dict[str, Any]:
    sides = token_smoke.get("sides") if isinstance(token_smoke.get("sides"), dict) else {}
    connected_available = bool(connected_precheck.get("_resolved_path"))
    return {
        "token_smoke": {
            "available": bool(token_smoke.get("_resolved_path")),
            "status": token_smoke.get("status"),
            "left_support_views": ((sides.get("left") or {}).get("support_views") if isinstance(sides.get("left"), dict) else None),
            "right_support_views": ((sides.get("right") or {}).get("support_views") if isinstance(sides.get("right"), dict) else None),
            "left_unique_aggregator_tokens": ((sides.get("left") or {}).get("unique_aggregator_tokens") if isinstance(sides.get("left"), dict) else None),
            "right_unique_aggregator_tokens": ((sides.get("right") or {}).get("unique_aggregator_tokens") if isinstance(sides.get("right"), dict) else None),
        },
        "connected_precheck": {
            "available": connected_available,
            "truthful_status": connected_precheck.get("truthful_status"),
            "gate_decision": connected_precheck.get("gate_decision"),
            "pass": connected_precheck.get("pass"),
            "strict_candidate_passes": connected_precheck.get("strict_candidate_passes", 0),
            "strict_teacher_passes": connected_precheck.get("strict_teacher_passes", 0),
            "connected_proxy_built": connected_precheck.get("connected_proxy_built"),
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Hand5 Label/Evidence Learnability Probe",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        "This local probe tests whether frozen B-hand ROI/token evidence can predict",
        "weak left/right side labels and weak connection/depth risk labels. It is not",
        "a hand decoder, not SMPL-X scaffold success, not a teacher, not a candidate,",
        "not a strict pass, and not a cloud unblocker.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        f"teacher_export = {summary['teacher_export']}",
        f"candidate_export = {summary['candidate_export']}",
        "```",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(summary["inputs"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Label Definitions",
        "",
        "```json",
        json.dumps(summary["label_definitions"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Results",
        "",
        "```json",
        json.dumps(summary["results"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Upstream Gate Readout",
        "",
        "```json",
        json.dumps(summary["upstream_gate_readout"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision_text(results: dict[str, Any], upstream: dict[str, Any]) -> str:
    side_signal = bool(results["side_weak_label_all_roi_token_evidence"].get("learnability_signal"))
    side_no_x_signal = bool(results["side_weak_label_no_absolute_x_control"].get("learnability_signal"))
    conn_signal = bool(results["weak_connection_risk_all_roi_token_evidence"].get("learnability_signal"))
    depth_signal = bool(results["weak_depth_risk_all_roi_token_evidence"].get("learnability_signal"))
    connected_pass = bool((upstream.get("connected_precheck") or {}).get("pass"))
    if side_signal and (conn_signal or depth_signal):
        first = (
            "Frozen B-hand ROI/token evidence shows a fixed-probe signal for weak side labels "
            "and at least one weak risk label."
        )
    elif side_signal:
        first = (
            "Frozen B-hand ROI/token evidence shows a fixed-probe signal for weak side labels, "
            "but the weak connection/depth risk labels are not reliably learnable in this tiny smoke."
        )
    else:
        first = (
            "This fixed probe does not show enough learnability signal for weak side labels or risk labels."
        )
    control = (
        " The no-absolute-x side control also shows signal, so the side result is not only a raw x-coordinate readout."
        if side_no_x_signal
        else " The no-absolute-x side control is weaker, so the side result should be treated as ROI/token-position sanity only."
    )
    gate = (
        " Upstream connected precheck is passing."
        if connected_pass
        else " Upstream B-hand3/B-hand4 hand gates remain failed; SMPL-X/connected scaffold evidence stays weak diagnostic evidence only."
    )
    return (
        first
        + control
        + gate
        + " Do not write predictions, teacher/candidate artifacts, strict pass state, or cloud exports from B-hand5."
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_json(args.hand_evidence_cache)
    token_smoke = load_json(args.hand_token_smoke, required=False)
    connected_precheck = load_json(args.connected_precheck, required=False)
    dataset = build_dataset(cache, args)
    x_all = np.nan_to_num(dataset["features"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    no_x_idx = no_absolute_x_indices(dataset["feature_names"])
    x_no_x = x_all[:, no_x_idx]

    side_all = evaluate_task("side_weak_label_all_roi_token_evidence", dataset["side_labels"], x_all, float(args.ridge))
    side_no_x = evaluate_task("side_weak_label_no_absolute_x_control", dataset["side_labels"], x_no_x, float(args.ridge))
    conn_all = evaluate_task("weak_connection_risk_all_roi_token_evidence", dataset["connection_labels"], x_all, float(args.ridge))
    depth_all = evaluate_task("weak_depth_risk_all_roi_token_evidence", dataset["depth_labels"], x_all, float(args.ridge))
    results = {
        "roi_examples": int(x_all.shape[0]),
        "feature_dim_all_roi_token_evidence": int(x_all.shape[1]),
        "feature_dim_no_absolute_x_control": int(x_no_x.shape[1]),
        "side_weak_label_all_roi_token_evidence": strip_private_arrays(side_all),
        "side_weak_label_no_absolute_x_control": strip_private_arrays(side_no_x),
        "weak_connection_risk_all_roi_token_evidence": strip_private_arrays(conn_all),
        "weak_depth_risk_all_roi_token_evidence": strip_private_arrays(depth_all),
    }
    upstream = upstream_readout(token_smoke, connected_precheck)
    label_definitions = {
        "side_weak_label": {
            "source": "b_hand_evidence_cache hand_crop_metadata.side",
            "mapping": {"0": "left", "1": "right"},
            "warning": "The current cache side labels are weak diagnostic labels, often assigned from image x relative to body bbox.",
        },
        "weak_connection_risk": {
            "positive_if_any": [
                f"roi_pixels < {int(args.min_risk_roi_pixels)}",
                f"smplx_visible_ratio_in_roi < {float(args.min_risk_visible_ratio)}",
                "body_overlap_ratio < 0.001",
                "component_count > 2",
                "largest_component_ratio < 0.75",
            ],
            "warning": "This is a weak evidence-risk label, not an Open3D connected-arm pass/fail label.",
        },
        "weak_depth_risk": {
            "positive_if_any": [
                f"smplx_depth_range >= {float(args.max_depth_range)}",
                f"smplx_max_extent >= {float(args.max_depth_range)}",
                f"smplx_pca_thin_ratio >= {float(args.max_pca_thin_ratio)}",
                "smplx_spatial_points <= 0 and smplx_visible_ratio_in_roi < 0.1",
            ],
            "warning": "This is a weak depth/span risk label from cached SMPL-X prior overlap, not a teacher depth target.",
        },
    }

    arrays_path = output_dir / "b_hand_label_evidence_learnability_arrays.npz"
    np.savez_compressed(
        arrays_path,
        features_all=x_all.astype(np.float32),
        features_no_absolute_x=x_no_x.astype(np.float32),
        feature_names=np.asarray(dataset["feature_names"]),
        feature_names_no_absolute_x=np.asarray([dataset["feature_names"][idx] for idx in no_x_idx.tolist()]),
        side_labels=dataset["side_labels"].astype(np.int64),
        connection_risk_labels=dataset["connection_labels"].astype(np.int64),
        depth_risk_labels=dataset["depth_labels"].astype(np.int64),
        side_ridge_pred_all=side_all["_ridge_pred"].astype(np.int64),
        side_ridge_pred_no_absolute_x=side_no_x["_ridge_pred"].astype(np.int64),
        connection_ridge_pred=conn_all["_ridge_pred"].astype(np.int64),
        depth_ridge_pred=depth_all["_ridge_pred"].astype(np.int64),
    )
    summary_path = output_dir / "b_hand_label_evidence_learnability_summary.json"
    summary_md_path = output_dir / "b_hand_label_evidence_learnability_summary.md"
    status_report = args.status_report.expanduser().resolve()
    summary = {
        **STRICT_FACTS,
        "task": "b_hand5_label_evidence_learnability_probe",
        "truthful_status": "research_only_learnability_probe_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "hand_evidence_cache": str(args.hand_evidence_cache.expanduser().resolve()),
            "hand_token_smoke": str(args.hand_token_smoke.expanduser().resolve()),
            "connected_precheck": str(args.connected_precheck.expanduser().resolve()),
        },
        "configuration": {
            "ridge": float(args.ridge),
            "evaluation": "fixed leave-one-out over cached ROI rows; no hyperparameter sweep",
            "min_risk_roi_pixels": int(args.min_risk_roi_pixels),
            "min_risk_visible_ratio": float(args.min_risk_visible_ratio),
            "max_depth_range": float(args.max_depth_range),
            "max_pca_thin_ratio": float(args.max_pca_thin_ratio),
        },
        "source_cache": {
            "scene_dir": cache.get("scene_dir"),
            "target_size": cache.get("target_size"),
            "view_indices": cache.get("view_indices"),
            "visible_view_summary": cache.get("visible_view_summary"),
        },
        "label_definitions": label_definitions,
        "results": results,
        "roi_records": dataset["records"],
        "upstream_gate_readout": upstream,
        "decision": decision_text(results, upstream),
        "outputs": {
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_path),
            "summary_md": str(summary_md_path),
            "status_report": str(status_report),
        },
    }
    summary = json_ready(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary_md_path, summary)
    write_markdown(status_report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
