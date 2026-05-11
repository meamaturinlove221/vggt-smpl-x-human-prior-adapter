from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QUERY_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D7_query_sdf_smoke_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_query_sdf_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "query_level_latent_sdf_smoke_only": True,
    "closed_form_no_training": True,
    "not_decoder_success": True,
    "not_visual_pass": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_state_write": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}

FAMILY_ORDER = ("full_body", "face_core", "hairline", "left_hand", "right_hand")
FAMILY_IDS = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
FAMILY_PRIOR_RISK = {
    "full_body": 0.05,
    "face_core": 0.00,
    "hairline": 0.30,
    "left_hand": 0.10,
    "right_hand": 0.30,
}
FAMILY_BASE_COLORS = {
    "full_body": np.asarray([165, 165, 165], dtype=np.float32),
    "face_core": np.asarray([245, 185, 40], dtype=np.float32),
    "hairline": np.asarray([155, 80, 225], dtype=np.float32),
    "left_hand": np.asarray([35, 145, 255], dtype=np.float32),
    "right_hand": np.asarray([255, 95, 55], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D7 query-level latent/SDF/occupancy smoke. "
            "It consumes the B-Fus3D6 query evidence cache plus the connected "
            "template payload, emits closed-form placeholder query SDF/occupancy "
            "diagnostics and risk-colored PLYs, and never trains, exports a "
            "teacher/candidate, writes registry state, writes prediction outputs, "
            "or calls cloud."
        )
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--normal-shell-radius", type=float, default=0.012)
    parser.add_argument("--occupancy-temperature", type=float, default=0.006)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--feature-bins", type=int, default=16)
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


def safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.clip(norms, eps, None)).astype(np.float32)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ).astype(np.float32)
    for corner_idx in range(3):
        np.add.at(normals, faces[:, corner_idx], face_normals)
    return safe_normalize(normals)


def robust_01(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    out = np.full(values.shape, float(default), dtype=np.float32)
    if not finite.any():
        return out
    lo, hi = np.percentile(values[finite], [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = np.clip((values[finite] - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    return out.astype(np.float32)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.float32)
    if not finite.any():
        return out
    center = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - center))
    scale = float(1.4826 * mad)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values[finite]))
    if not np.isfinite(scale) or scale < 1e-6:
        return out
    out[finite] = np.clip((values[finite] - center) / scale, -4.0, 4.0)
    return out.astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))).astype(np.float32)


def load_query_cache(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        required = {
            "query_indices",
            "query_positions",
            "query_part_ids",
            "query_families",
            "support",
            "token_ids",
            "uv",
            "depth",
            "mean_features",
            "variance_features",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"Missing query-cache arrays: {missing}")
        out = {
            "query_indices": np.asarray(payload["query_indices"], dtype=np.int64),
            "query_positions": np.asarray(payload["query_positions"], dtype=np.float32),
            "query_part_ids": np.asarray(payload["query_part_ids"], dtype=np.int64),
            "query_families": np.asarray(payload["query_families"]).astype(str),
            "support": np.asarray(payload["support"], dtype=np.int32),
            "token_ids": np.asarray(payload["token_ids"], dtype=np.int32),
            "uv": np.asarray(payload["uv"], dtype=np.float32),
            "depth": np.asarray(payload["depth"], dtype=np.float32),
            "mean_features": np.nan_to_num(
                np.asarray(payload["mean_features"], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "variance_features": np.nan_to_num(
                np.asarray(payload["variance_features"], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        }
        if "selected_view_indices" in payload.files:
            out["selected_view_indices"] = np.asarray(payload["selected_view_indices"], dtype=np.int32)
        else:
            out["selected_view_indices"] = np.arange(out["token_ids"].shape[1], dtype=np.int32)
        if "patch_grid" in payload.files:
            out["patch_grid"] = np.asarray(payload["patch_grid"], dtype=np.int32)
        if "patch_start_idx" in payload.files:
            out["patch_start_idx"] = np.asarray(payload["patch_start_idx"], dtype=np.int32)
    return out


def load_template_payload(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        files = set(payload.files)
        vertex_key = "hybrid_vertices" if "hybrid_vertices" in files else "vertices"
        face_key = "hybrid_faces" if "hybrid_faces" in files else "faces"
        vertices = np.asarray(payload[vertex_key], dtype=np.float32)
        faces = np.asarray(payload[face_key], dtype=np.int64)
        if "part_ids" in files and payload["part_ids"].shape[0] == vertices.shape[0]:
            part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
        else:
            part_ids = np.zeros((vertices.shape[0],), dtype=np.int64)
        part_names = np.asarray(payload["part_names"]).astype(str) if "part_names" in files else np.asarray([])
        part_families = (
            np.asarray(payload["part_families"]).astype(str) if "part_families" in files else np.asarray([])
        )
    return {
        "path": str(resolved),
        "vertex_key": vertex_key,
        "face_key": face_key,
        "vertices": vertices,
        "faces": faces,
        "part_ids": part_ids,
        "part_names": part_names,
        "part_families": part_families,
        "normals": compute_vertex_normals(vertices, faces),
    }


def chunk_reduce(features: np.ndarray, bins: int, reducer: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    bins = max(1, int(bins))
    chunks = np.array_split(np.arange(features.shape[1]), bins)
    rows = []
    for chunk in chunks:
        if chunk.size == 0:
            rows.append(np.zeros((features.shape[0],), dtype=np.float32))
        elif reducer == "rms":
            rows.append(np.sqrt(np.mean(np.square(features[:, chunk]), axis=1)).astype(np.float32))
        else:
            rows.append(np.mean(features[:, chunk], axis=1).astype(np.float32))
    reduced = np.stack(rows, axis=1).astype(np.float32)
    if reducer == "mean_z":
        cols = [robust_z(reduced[:, idx]) for idx in range(reduced.shape[1])]
        reduced = np.stack(cols, axis=1).astype(np.float32)
    elif reducer == "rms_01":
        cols = [robust_01(reduced[:, idx]) for idx in range(reduced.shape[1])]
        reduced = np.stack(cols, axis=1).astype(np.float32)
    return np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def view_stats(
    token_ids: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    target_size: float = 518.0,
) -> dict[str, np.ndarray]:
    token_ids = np.asarray(token_ids, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    valid = (token_ids >= 0) & np.isfinite(depth) & np.isfinite(uv).all(axis=2)
    query_count = token_ids.shape[0]
    depth_mean = np.zeros((query_count,), dtype=np.float32)
    depth_std = np.zeros((query_count,), dtype=np.float32)
    uv_spread = np.zeros((query_count,), dtype=np.float32)
    for idx in range(query_count):
        mask = valid[idx]
        if not mask.any():
            continue
        local_depth = depth[idx, mask]
        local_uv = uv[idx, mask]
        depth_mean[idx] = float(np.mean(local_depth))
        depth_std[idx] = float(np.std(local_depth)) if local_depth.size > 1 else 0.0
        if local_uv.shape[0] > 1:
            centered_uv = local_uv - np.mean(local_uv, axis=0, keepdims=True)
            uv_spread[idx] = float(np.sqrt(np.mean(np.sum(centered_uv * centered_uv, axis=1))) / max(target_size, 1.0))
    return {
        "valid_view_mask": valid.astype(bool),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "depth_std_norm": robust_01(depth_std),
        "uv_spread": uv_spread,
        "uv_spread_norm": robust_01(uv_spread),
    }


def build_latent_and_sdf(
    cache: dict[str, np.ndarray],
    template: dict[str, Any],
    shell_radius: float,
    occupancy_temperature: float,
    min_support: int,
    feature_bins: int,
) -> dict[str, np.ndarray | list[str]]:
    query_indices = cache["query_indices"]
    query_positions = cache["query_positions"]
    template_vertices = template["vertices"]
    normals = template["normals"][query_indices]
    normals = safe_normalize(normals)

    if query_indices.max(initial=-1) >= template_vertices.shape[0]:
        raise ValueError("Query indices exceed template vertex count.")
    anchor_error = np.linalg.norm(template_vertices[query_indices] - query_positions, axis=1).astype(np.float32)

    support = cache["support"].astype(np.float32)
    view_count = max(1, int(cache["token_ids"].shape[1]))
    support_ratio = np.clip(support / float(view_count), 0.0, 1.0).astype(np.float32)
    support_gap = np.clip((float(min_support) - support) / max(float(min_support), 1.0), 0.0, 1.0).astype(np.float32)
    feature_mean_chunks = chunk_reduce(cache["mean_features"], feature_bins, "mean_z")
    feature_var_chunks = chunk_reduce(cache["variance_features"], feature_bins, "rms_01")
    mean_energy = np.sqrt(np.mean(np.square(cache["mean_features"]), axis=1)).astype(np.float32)
    var_energy = np.sqrt(np.mean(np.maximum(cache["variance_features"], 0.0), axis=1)).astype(np.float32)
    mean_energy_z = robust_z(mean_energy)
    var_energy_norm = robust_01(var_energy)
    views = view_stats(cache["token_ids"], cache["uv"], cache["depth"])

    center = np.mean(query_positions, axis=0, keepdims=True)
    scale = float(np.percentile(np.linalg.norm(query_positions - center, axis=1), 95.0))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    query_xyz_norm = ((query_positions - center) / scale).astype(np.float32)
    geom_signal = (
        0.42 * query_xyz_norm[:, 1]
        + 0.26 * query_xyz_norm[:, 0]
        - 0.18 * query_xyz_norm[:, 2]
        + 0.14 * normals[:, 1]
    ).astype(np.float32)
    feature_signal = (
        0.28 * mean_energy_z
        + 0.18 * feature_mean_chunks[:, 0]
        - 0.14 * feature_mean_chunks[:, min(3, feature_mean_chunks.shape[1] - 1)]
        - 0.26 * var_energy_norm
    ).astype(np.float32)
    evidence_signal = np.tanh(0.45 * geom_signal + 0.45 * feature_signal + 0.35 * (support_ratio - 0.5))
    evidence_shift = (float(shell_radius) * 0.25 * support_ratio * evidence_signal).astype(np.float32)
    evidence_shift = np.clip(evidence_shift, -0.25 * float(shell_radius), 0.25 * float(shell_radius)).astype(np.float32)

    offsets = np.asarray([-float(shell_radius), 0.0, float(shell_radius)], dtype=np.float32)
    sample_positions = query_positions[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    template_signed_distance = np.broadcast_to(offsets[None, :], (query_positions.shape[0], offsets.size)).astype(
        np.float32
    )
    conditioned_sdf = (template_signed_distance - evidence_shift[:, None]).astype(np.float32)
    occupancy = sigmoid(-conditioned_sdf / max(float(occupancy_temperature), 1e-6))
    query_sdf = conditioned_sdf[:, 1].astype(np.float32)
    query_occupancy = occupancy[:, 1].astype(np.float32)

    family_one_hot = np.zeros((query_positions.shape[0], len(FAMILY_ORDER)), dtype=np.float32)
    family_ids = np.zeros((query_positions.shape[0],), dtype=np.int32)
    for idx, family in enumerate(cache["query_families"].astype(str).tolist()):
        family_id = FAMILY_IDS.get(family, 0)
        family_ids[idx] = int(family_id)
        family_one_hot[idx, family_id] = 1.0

    family_prior = np.asarray(
        [FAMILY_PRIOR_RISK.get(str(family), 0.15) for family in cache["query_families"].astype(str).tolist()],
        dtype=np.float32,
    )
    visibility_risk = np.clip(0.65 * support_gap + 0.35 * (1.0 - support_ratio), 0.0, 1.0).astype(np.float32)
    depth_risk = views["depth_std_norm"].astype(np.float32)
    depth_risk[support < 2] = np.maximum(depth_risk[support < 2], 0.65)
    uv_risk = views["uv_spread_norm"].astype(np.float32)
    feature_uncertainty = var_energy_norm.astype(np.float32)
    risk = np.clip(
        0.52 * visibility_risk
        + 0.18 * feature_uncertainty
        + 0.12 * depth_risk
        + 0.08 * uv_risk
        + 0.10 * family_prior,
        0.0,
        1.0,
    ).astype(np.float32)

    latent_parts = [
        query_xyz_norm,
        normals,
        support_ratio[:, None],
        support_gap[:, None],
        views["depth_mean"][:, None],
        views["depth_std_norm"][:, None],
        views["uv_spread_norm"][:, None],
        feature_mean_chunks,
        feature_var_chunks,
        family_one_hot,
    ]
    latent = np.concatenate(latent_parts, axis=1).astype(np.float32)
    latent_channel_names = (
        ["query_x_norm", "query_y_norm", "query_z_norm", "normal_x", "normal_y", "normal_z"]
        + ["support_ratio", "support_gap", "depth_mean_raw", "depth_std_norm", "uv_spread_norm"]
        + [f"mean_feature_chunk_z_{idx:02d}" for idx in range(feature_mean_chunks.shape[1])]
        + [f"variance_feature_chunk_norm_{idx:02d}" for idx in range(feature_var_chunks.shape[1])]
        + [f"family_{name}" for name in FAMILY_ORDER]
    )

    return {
        "query_normals": normals.astype(np.float32),
        "anchor_template_distance": anchor_error.astype(np.float32),
        "latent": latent.astype(np.float32),
        "latent_channel_names": np.asarray(latent_channel_names, dtype="<U48"),
        "family_ids": family_ids.astype(np.int32),
        "support_ratio": support_ratio.astype(np.float32),
        "support_gap": support_gap.astype(np.float32),
        "feature_mean_chunks": feature_mean_chunks.astype(np.float32),
        "feature_var_chunks": feature_var_chunks.astype(np.float32),
        "mean_feature_energy": mean_energy.astype(np.float32),
        "variance_feature_energy": var_energy.astype(np.float32),
        "depth_mean": views["depth_mean"].astype(np.float32),
        "depth_std": views["depth_std"].astype(np.float32),
        "uv_spread": views["uv_spread"].astype(np.float32),
        "evidence_shift": evidence_shift.astype(np.float32),
        "sample_offsets": offsets.astype(np.float32),
        "sample_positions": sample_positions.reshape(-1, 3).astype(np.float32),
        "sample_query_indices": np.repeat(np.arange(query_positions.shape[0], dtype=np.int32), offsets.size),
        "template_signed_distance": template_signed_distance.reshape(-1).astype(np.float32),
        "conditioned_sdf": conditioned_sdf.reshape(-1).astype(np.float32),
        "occupancy": occupancy.reshape(-1).astype(np.float32),
        "query_sdf": query_sdf.astype(np.float32),
        "query_occupancy": query_occupancy.astype(np.float32),
        "visibility_risk": visibility_risk.astype(np.float32),
        "feature_uncertainty": feature_uncertainty.astype(np.float32),
        "depth_risk": depth_risk.astype(np.float32),
        "uv_risk": uv_risk.astype(np.float32),
        "family_prior_risk": family_prior.astype(np.float32),
        "support_risk": risk.astype(np.float32),
    }


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def family_stats(cache: dict[str, np.ndarray], derived: dict[str, Any], min_support: int) -> dict[str, Any]:
    families = cache["query_families"].astype(str)
    support = cache["support"]
    risk = np.asarray(derived["support_risk"], dtype=np.float32)
    sdf = np.asarray(derived["query_sdf"], dtype=np.float32)
    occupancy = np.asarray(derived["query_occupancy"], dtype=np.float32)
    out: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        mask = families == family
        if not np.any(mask):
            continue
        values = support[mask]
        local_risk = risk[mask]
        high_risk = local_risk >= 0.50
        out[family] = {
            "query_count": int(mask.sum()),
            "support_ge_1": int((values >= 1).sum()),
            "support_ge_2": int((values >= 2).sum()),
            "support_ge_min": int((values >= int(min_support)).sum()),
            "mean_support": float(values.mean()) if values.size else 0.0,
            "max_support": int(values.max()) if values.size else 0,
            "mean_query_sdf": float(np.mean(sdf[mask])),
            "mean_occupancy": float(np.mean(occupancy[mask])),
            "mean_risk": float(np.mean(local_risk)),
            "max_risk": float(np.max(local_risk)),
            "high_risk_queries": int(high_risk.sum()),
            "blocker": bool((values < int(min_support)).any() or np.mean(local_risk) >= 0.35),
        }
    return out


def top_risk_rows(cache: dict[str, np.ndarray], derived: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    risk = np.asarray(derived["support_risk"], dtype=np.float32)
    order = np.argsort(-risk)[: int(limit)]
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(order.tolist(), start=1):
        reasons: list[str] = []
        support = int(cache["support"][row])
        family = str(cache["query_families"][row])
        if support < 3:
            reasons.append("support_lt_3")
        if family in {"hairline", "right_hand"}:
            reasons.append("known_weak_family_from_B_Fus3D6")
        if float(derived["feature_uncertainty"][row]) > 0.75:
            reasons.append("high_token_variance")
        if float(derived["depth_risk"][row]) > 0.75:
            reasons.append("high_depth_spread")
        rows.append(
            {
                "rank": rank,
                "cache_row": int(row),
                "query_index": int(cache["query_indices"][row]),
                "family": family,
                "part_id": int(cache["query_part_ids"][row]),
                "support": support,
                "query_sdf": float(derived["query_sdf"][row]),
                "occupancy": float(derived["query_occupancy"][row]),
                "risk": float(risk[row]),
                "reasons": reasons,
            }
        )
    return rows


def interpolate_colors(t: np.ndarray, lo: np.ndarray, mid: np.ndarray, hi: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32).reshape(-1), 0.0, 1.0)
    colors = np.zeros((t.shape[0], 3), dtype=np.float32)
    low_mask = t <= 0.5
    local = np.zeros_like(t)
    local[low_mask] = t[low_mask] / 0.5
    local[~low_mask] = (t[~low_mask] - 0.5) / 0.5
    colors[low_mask] = lo[None, :] * (1.0 - local[low_mask, None]) + mid[None, :] * local[low_mask, None]
    colors[~low_mask] = mid[None, :] * (1.0 - local[~low_mask, None]) + hi[None, :] * local[~low_mask, None]
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def colors_by_sdf(sdf: np.ndarray, shell_radius: float) -> np.ndarray:
    t = np.clip((np.asarray(sdf, dtype=np.float32) / max(float(shell_radius), 1e-6) + 1.0) * 0.5, 0.0, 1.0)
    return interpolate_colors(
        t,
        np.asarray([55, 110, 235], dtype=np.float32),
        np.asarray([245, 245, 245], dtype=np.float32),
        np.asarray([230, 70, 55], dtype=np.float32),
    )


def colors_by_risk(risk: np.ndarray) -> np.ndarray:
    return interpolate_colors(
        risk,
        np.asarray([40, 165, 95], dtype=np.float32),
        np.asarray([245, 205, 55], dtype=np.float32),
        np.asarray([220, 55, 55], dtype=np.float32),
    )


def colors_by_support(support: np.ndarray, view_count: int) -> np.ndarray:
    ratio = np.clip(np.asarray(support, dtype=np.float32) / max(float(view_count), 1.0), 0.0, 1.0)
    return interpolate_colors(
        ratio,
        np.asarray([160, 30, 30], dtype=np.float32),
        np.asarray([245, 200, 60], dtype=np.float32),
        np.asarray([55, 180, 100], dtype=np.float32),
    )


def colors_by_family(families: np.ndarray, support_ratio: np.ndarray) -> np.ndarray:
    families = np.asarray(families).astype(str)
    support_ratio = np.asarray(support_ratio, dtype=np.float32)
    colors = np.zeros((families.shape[0], 3), dtype=np.float32)
    for idx, family in enumerate(families.tolist()):
        base = FAMILY_BASE_COLORS.get(family, np.asarray([190, 190, 190], dtype=np.float32))
        intensity = 0.30 + 0.70 * float(np.clip(support_ratio[idx], 0.0, 1.0))
        colors[idx] = base * intensity
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def write_query_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    cache: dict[str, np.ndarray],
    derived: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    sdf = np.asarray(derived["query_sdf"], dtype=np.float32)
    occupancy = np.asarray(derived["query_occupancy"], dtype=np.float32)
    risk = np.asarray(derived["support_risk"], dtype=np.float32)
    support = np.asarray(cache["support"], dtype=np.int32)
    family_ids = np.asarray(derived["family_ids"], dtype=np.int32)
    part_ids = np.asarray(cache["query_part_ids"], dtype=np.int32)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment B-Fus3D7 research-only query-level SDF placeholder; not teacher/candidate.\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float sdf\n")
        handle.write("property float occupancy\n")
        handle.write("property float risk\n")
        handle.write("property int support\n")
        handle.write("property int family_id\n")
        handle.write("property int part_id\n")
        handle.write("end_header\n")
        for idx, (point, color) in enumerate(zip(points, colors, strict=False)):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(sdf[idx]):.8f} {float(occupancy[idx]):.8f} {float(risk[idx]):.8f} "
                f"{int(support[idx])} {int(family_ids[idx])} {int(part_ids[idx])}\n"
            )


def write_sample_ply(
    path: Path,
    sample_positions: np.ndarray,
    colors: np.ndarray,
    derived: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_positions = np.asarray(sample_positions, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    sdf = np.asarray(derived["conditioned_sdf"], dtype=np.float32)
    occupancy = np.asarray(derived["occupancy"], dtype=np.float32)
    query_rows = np.asarray(derived["sample_query_indices"], dtype=np.int32)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment B-Fus3D7 normal-shell SDF samples; placeholder only.\n")
        handle.write(f"element vertex {sample_positions.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float sdf\n")
        handle.write("property float occupancy\n")
        handle.write("property int query_row\n")
        handle.write("end_header\n")
        for idx, (point, color) in enumerate(zip(sample_positions, colors, strict=False)):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(sdf[idx]):.8f} {float(occupancy[idx]):.8f} {int(query_rows[idx])}\n"
            )


def write_markdown(path: Path, summary: dict[str, Any], report_title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {report_title}",
        "",
        "Status: `blocked_research_only_query_sdf_smoke`",
        "",
        "This is a closed-form query-level latent/SDF/occupancy wiring smoke over",
        "B-Fus3D6 query evidence. It is not decoder success, not a visual pass,",
        "not a teacher, not a candidate, and not a strict gate result.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Inputs",
        "",
        f"- query_cache: `{summary['inputs']['query_cache']}`",
        f"- template_payload: `{summary['inputs']['template_payload']}`",
        f"- template_vertices: `{summary['template']['vertices']}` via `{summary['template']['vertex_key']}`",
        f"- template_faces: `{summary['template']['faces']}` via `{summary['template']['face_key']}`",
        "",
        "## Query SDF / Occupancy Stats",
        "",
        "```json",
        json.dumps(summary["query_sdf_occupancy_stats"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Support-Aware Risk Map",
        "",
        "```json",
        json.dumps(summary["family_risk_map"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Blocker Conclusion",
        "",
        summary["blocker_conclusion"],
        "",
        "## Outputs",
        "",
    ]
    for value in summary["outputs"].values():
        if isinstance(value, list):
            for item in value:
                lines.append(f"- `{item}`")
        else:
            lines.append(f"- `{value}`")
    lines.extend(
        [
            "",
            "## Non-Repetition Guard",
            "",
            summary["non_repetition_guard"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_query_cache(args.query_cache)
    template = load_template_payload(args.template_payload)
    derived = build_latent_and_sdf(
        cache,
        template,
        float(args.normal_shell_radius),
        float(args.occupancy_temperature),
        int(args.min_support),
        int(args.feature_bins),
    )

    query_count = int(cache["query_positions"].shape[0])
    view_count = int(cache["token_ids"].shape[1])
    sample_count = int(derived["sample_positions"].shape[0])
    family_map = family_stats(cache, derived, int(args.min_support))
    top_rows = top_risk_rows(cache, derived)
    blocker_families = [family for family, stats in family_map.items() if bool(stats["blocker"])]
    insufficient_supervision = True

    arrays_path = output_dir / "b_fus3d_query_sdf_smoke_arrays.npz"
    np.savez_compressed(
        arrays_path,
        query_indices=cache["query_indices"].astype(np.int64),
        query_positions=cache["query_positions"].astype(np.float32),
        query_normals=np.asarray(derived["query_normals"], dtype=np.float32),
        query_part_ids=cache["query_part_ids"].astype(np.int64),
        query_families=cache["query_families"],
        family_ids=np.asarray(derived["family_ids"], dtype=np.int32),
        support=cache["support"].astype(np.int32),
        support_ratio=np.asarray(derived["support_ratio"], dtype=np.float32),
        support_gap=np.asarray(derived["support_gap"], dtype=np.float32),
        latent=np.asarray(derived["latent"], dtype=np.float32),
        latent_channel_names=np.asarray(derived["latent_channel_names"]),
        evidence_shift=np.asarray(derived["evidence_shift"], dtype=np.float32),
        query_sdf=np.asarray(derived["query_sdf"], dtype=np.float32),
        query_occupancy=np.asarray(derived["query_occupancy"], dtype=np.float32),
        support_risk=np.asarray(derived["support_risk"], dtype=np.float32),
        feature_uncertainty=np.asarray(derived["feature_uncertainty"], dtype=np.float32),
        depth_risk=np.asarray(derived["depth_risk"], dtype=np.float32),
        uv_risk=np.asarray(derived["uv_risk"], dtype=np.float32),
        family_prior_risk=np.asarray(derived["family_prior_risk"], dtype=np.float32),
        sample_positions=np.asarray(derived["sample_positions"], dtype=np.float32),
        sample_query_indices=np.asarray(derived["sample_query_indices"], dtype=np.int32),
        template_signed_distance=np.asarray(derived["template_signed_distance"], dtype=np.float32),
        conditioned_sdf=np.asarray(derived["conditioned_sdf"], dtype=np.float32),
        occupancy=np.asarray(derived["occupancy"], dtype=np.float32),
        selected_view_indices=np.asarray(cache["selected_view_indices"], dtype=np.int32),
    )

    query_points = cache["query_positions"]
    sdf_ply = output_dir / "query_points_colored_by_sdf.ply"
    risk_ply = output_dir / "query_points_colored_by_risk.ply"
    support_ply = output_dir / "query_points_colored_by_support.ply"
    family_ply = output_dir / "query_points_colored_by_family.ply"
    shell_ply = output_dir / "normal_shell_samples_colored_by_sdf.ply"
    write_query_ply(sdf_ply, query_points, colors_by_sdf(derived["query_sdf"], args.normal_shell_radius), cache, derived)
    write_query_ply(risk_ply, query_points, colors_by_risk(derived["support_risk"]), cache, derived)
    write_query_ply(support_ply, query_points, colors_by_support(cache["support"], view_count), cache, derived)
    write_query_ply(
        family_ply,
        query_points,
        colors_by_family(cache["query_families"], derived["support_ratio"]),
        cache,
        derived,
    )
    write_sample_ply(
        shell_ply,
        derived["sample_positions"],
        colors_by_sdf(derived["conditioned_sdf"], args.normal_shell_radius),
        derived,
    )

    risk_map_path = output_dir / "b_fus3d_query_support_aware_risk_map.json"
    summary_json_path = output_dir / "b_fus3d_query_sdf_smoke_summary.json"
    summary_md_path = output_dir / "b_fus3d_query_sdf_smoke_summary.md"
    report_path = args.status_report.expanduser().resolve()
    pyc_path = output_dir / "b_fus3d_query_sdf_smoke.pyc"

    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d7_query_level_latent_sdf_smoke",
        "truthful_status": "blocked_research_only_query_sdf_placeholder_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "query_cache": str(args.query_cache.expanduser().resolve()),
            "template_payload": str(args.template_payload.expanduser().resolve()),
            "selected_view_indices": [int(v) for v in np.asarray(cache["selected_view_indices"]).reshape(-1)],
        },
        "template": {
            "vertex_key": template["vertex_key"],
            "face_key": template["face_key"],
            "vertices": int(template["vertices"].shape[0]),
            "faces": int(template["faces"].shape[0]),
            "query_anchor_template_distance": scalar_stats(derived["anchor_template_distance"]),
        },
        "configuration": {
            "query_count": query_count,
            "view_count": view_count,
            "sample_count": sample_count,
            "normal_shell_radius": float(args.normal_shell_radius),
            "occupancy_temperature": float(args.occupancy_temperature),
            "min_support": int(args.min_support),
            "feature_bins": int(args.feature_bins),
            "latent_shape": [int(v) for v in np.asarray(derived["latent"]).shape],
            "latent_inputs": [
                "query_xyz",
                "template_vertex_normals",
                "support_ratio",
                "support_gap",
                "view_depth_stats",
                "uv_spread",
                "compressed_mean_features",
                "compressed_variance_features",
                "family_one_hot",
            ],
        },
        "query_sdf_occupancy_stats": {
            "query_sdf": scalar_stats(derived["query_sdf"]),
            "query_occupancy": scalar_stats(derived["query_occupancy"]),
            "evidence_shift": scalar_stats(derived["evidence_shift"]),
            "conditioned_sample_sdf": scalar_stats(derived["conditioned_sdf"]),
            "sample_occupancy": scalar_stats(derived["occupancy"]),
        },
        "support_stats": {
            "support": scalar_stats(cache["support"]),
            "support_ge_1": int((cache["support"] >= 1).sum()),
            "support_ge_2": int((cache["support"] >= 2).sum()),
            "support_ge_min": int((cache["support"] >= int(args.min_support)).sum()),
            "support_lt_min": int((cache["support"] < int(args.min_support)).sum()),
            "mean_support": float(np.mean(cache["support"])) if cache["support"].size else 0.0,
            "max_support": int(np.max(cache["support"])) if cache["support"].size else 0,
        },
        "risk_stats": {
            "support_aware_risk": scalar_stats(derived["support_risk"]),
            "visibility_risk": scalar_stats(derived["visibility_risk"]),
            "feature_uncertainty": scalar_stats(derived["feature_uncertainty"]),
            "depth_risk": scalar_stats(derived["depth_risk"]),
            "uv_risk": scalar_stats(derived["uv_risk"]),
            "high_risk_queries_ge_0_50": int((derived["support_risk"] >= 0.50).sum()),
            "medium_or_high_risk_queries_ge_0_33": int((derived["support_risk"] >= 0.33).sum()),
        },
        "family_risk_map": family_map,
        "top_risk_queries": top_rows,
        "blocker_families": blocker_families,
        "supervision_readout": {
            "has_inside_outside_sdf_supervision": False,
            "has_mesh_surface_target_for_new_sdf": False,
            "has_teacher_or_candidate_permission": False,
            "insufficient_supervision_for_real_surface_sdf": insufficient_supervision,
        },
        "outputs": {
            "arrays_npz": str(arrays_path),
            "risk_map_json": str(risk_map_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
            "status_report": str(report_path),
            "py_compile_cfile": str(pyc_path),
            "ply_files": [
                str(sdf_ply),
                str(risk_ply),
                str(support_ply),
                str(family_ply),
                str(shell_ply),
            ],
        },
        "blocker_conclusion": (
            "BLOCKER: the wiring now produces query-level latent, placeholder SDF, "
            "occupancy, and support-aware risk outputs from query points plus "
            "view-aware token evidence/support, but there is no inside/outside or "
            "true surface supervision in this smoke. Therefore this is not decoder "
            "success, not a visual pass, not a teacher/candidate, and cannot be "
            "promoted into formal cloud train/infer/export."
        ),
        "non_repetition_guard": (
            "This does not repeat B-Fus3D5 per-vertex offset MLP work: no residual "
            "vertex MLP, no step/hidden/weight sweep, no mask/photometric gamble. "
            "The smoke is query-level and closed-form, anchored on B-Fus3D6 query "
            "points, view-aware token features, support, and connected-template normals."
        ),
        "decision": "blocked_insufficient_sdf_supervision_research_only",
    }
    summary = json_ready(summary)
    risk_map = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "family_risk_map": summary["family_risk_map"],
        "top_risk_queries": summary["top_risk_queries"],
        "blocker_families": summary["blocker_families"],
        "decision": summary["decision"],
    }
    risk_map_path.write_text(json.dumps(risk_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary_md_path, summary, "B-Fus3D7 Query-Level Latent/SDF Smoke")
    write_markdown(report_path, summary, "B-Fus3D7 Query SDF Status")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
