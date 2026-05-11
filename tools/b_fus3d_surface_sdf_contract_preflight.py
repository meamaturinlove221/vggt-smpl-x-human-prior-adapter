from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TOKEN_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_QUERY_EVIDENCE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_LATENT_REAL = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_SHUFFLE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_ZERO = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D17_surface_sdf_contract_preflight_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_surface_sdf_contract_status.md")


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}

CONTRACT_FLAGS = {
    "research_only": True,
    "local_only": True,
    "contract_preflight_only": True,
    "no_train": True,
    "no_optimization": True,
    "no_mesh_extraction": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "no_cloud": True,
    "not_teacher": True,
    "not_candidate": True,
}

FORBIDDEN_PATH_TOKENS = ("strict_pass", "teacher_export", "candidate_export")
EXPECTED_QUERY_FAMILIES = ("full_body", "face_core", "hairline", "left_hand", "right_hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-Fus3D17 local contract-only preflight. It validates the input and "
            "output schema for a future surface-token/SDF backend with "
            "differentiable-render supervision and real/shuffle/zero controls. "
            "It never trains, optimizes, extracts a mesh, writes predictions, "
            "exports a teacher/candidate, writes a strict pass, updates a "
            "registry, or calls cloud."
        )
    )
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--latent-grid-real", type=Path, default=DEFAULT_LATENT_REAL)
    parser.add_argument("--latent-grid-shuffle", type=Path, default=DEFAULT_LATENT_SHUFFLE)
    parser.add_argument("--latent-grid-zero", type=Path, default=DEFAULT_LATENT_ZERO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
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
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def ensure_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in text:
            raise ValueError(f"Refusing output path containing forbidden token {token!r}: {path}")


def stat_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.ones(arr.shape, dtype=bool)
    if arr.size == 0:
        return {"count": 0, "finite": 0}
    if not finite.any() or not np.issubdtype(arr.dtype, np.number):
        return {"count": int(arr.size), "finite": int(finite.sum())}
    data = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(data.size),
        "min": float(data.min()),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.quantile(data, 0.50)),
        "mean": float(data.mean()),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(data.max()),
    }


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing required arrays: {missing}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def summarize_token_cache(path: Path) -> dict[str, Any]:
    payload = load_npz(path, ("tokens", "patch_start_idx", "selected_view_indices"))
    tokens = np.asarray(payload["tokens"])
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        raise ValueError(f"token cache expected [1,V,T,C], got {tokens.shape}")
    patch_start = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
    patch_count = int(tokens.shape[2] - patch_start)
    patch_grid = int(round(math.sqrt(max(patch_count, 1))))
    if patch_grid * patch_grid != patch_count:
        raise ValueError(f"patch_count={patch_count} is not a square grid")
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)
    return {
        "path": str(path.resolve()),
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "view_count": int(tokens.shape[1]),
        "token_count": int(tokens.shape[2]),
        "feature_dim": int(tokens.shape[3]),
        "patch_start_idx": patch_start,
        "patch_count": patch_count,
        "patch_grid": patch_grid,
        "selected_view_indices": selected.astype(int).tolist(),
        "token_value_stats": stat_array(tokens.astype(np.float32)),
    }


def summarize_query_evidence(path: Path, token_summary: dict[str, Any]) -> dict[str, Any]:
    required = (
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
        "selected_view_indices",
    )
    payload = load_npz(path, required)
    positions = np.asarray(payload["query_positions"])
    families = np.asarray(payload["query_families"]).astype(str).reshape(-1)
    support = np.asarray(payload["support"], dtype=np.int64).reshape(-1)
    mean_features = np.asarray(payload["mean_features"])
    variance_features = np.asarray(payload["variance_features"])
    token_ids = np.asarray(payload["token_ids"])
    uv = np.asarray(payload["uv"])
    depth = np.asarray(payload["depth"])
    selected = np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1)

    n = int(positions.shape[0])
    view_count = int(token_summary["view_count"])
    feature_dim = int(token_summary["feature_dim"])
    errors: list[str] = []
    if positions.shape != (n, 3):
        errors.append(f"query_positions must be [N,3], got {positions.shape}")
    if support.shape[0] != n:
        errors.append(f"support length {support.shape[0]} != N {n}")
    if token_ids.shape != (n, view_count):
        errors.append(f"token_ids expected {(n, view_count)}, got {token_ids.shape}")
    if uv.shape != (n, view_count, 2):
        errors.append(f"uv expected {(n, view_count, 2)}, got {uv.shape}")
    if depth.shape != (n, view_count):
        errors.append(f"depth expected {(n, view_count)}, got {depth.shape}")
    if mean_features.shape != (n, feature_dim):
        errors.append(f"mean_features expected {(n, feature_dim)}, got {mean_features.shape}")
    if variance_features.shape != (n, feature_dim):
        errors.append(f"variance_features expected {(n, feature_dim)}, got {variance_features.shape}")
    if selected.tolist() != list(token_summary["selected_view_indices"]):
        errors.append("query selected_view_indices do not match token cache")

    family_rows: dict[str, dict[str, Any]] = {}
    for family in sorted(set(families.tolist()) | set(EXPECTED_QUERY_FAMILIES)):
        mask = families == family
        count = int(mask.sum())
        family_rows[family] = {
            "count": count,
            "present": count > 0,
            "support_stats": stat_array(support[mask]) if count else {"count": 0, "finite": 0},
            "zero_support_count": int((support[mask] <= 0).sum()) if count else 0,
            "mean_support": float(support[mask].mean()) if count else 0.0,
            "two_view_or_more": int((support[mask] >= 2).sum()) if count else 0,
            "two_view_or_more_ratio": float((support[mask] >= 2).mean()) if count else 0.0,
        }
    missing_families = [family for family in EXPECTED_QUERY_FAMILIES if family_rows.get(family, {}).get("count", 0) <= 0]
    if missing_families:
        errors.append(f"missing required query families: {missing_families}")

    return {
        "path": str(path.resolve()),
        "query_count": n,
        "position_shape": list(positions.shape),
        "token_ids_shape": list(token_ids.shape),
        "uv_shape": list(uv.shape),
        "depth_shape": list(depth.shape),
        "mean_features_shape": list(mean_features.shape),
        "variance_features_shape": list(variance_features.shape),
        "feature_dim_matches_token_cache": mean_features.shape[-1] == feature_dim,
        "selected_view_indices": selected.astype(int).tolist(),
        "families": family_rows,
        "errors": errors,
        "compatible": not errors,
    }


def summarize_latent_control(path: Path, name: str) -> dict[str, Any]:
    required = (
        "points",
        "visible_count",
        "mask_count",
        "token_count",
        "occupancy_ratio",
        "rgb_variance",
        "rgb_range",
        "token_cosine",
        "evidence_score",
        "boundary_like",
        "selected_view_indices",
    )
    payload = load_npz(path, required)
    points = np.asarray(payload["points"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} points expected [M,3], got {points.shape}")
    m = int(points.shape[0])
    errors: list[str] = []
    for key in required:
        arr = np.asarray(payload[key])
        if key == "points" or key == "selected_view_indices":
            continue
        if arr.shape[0] != m:
            errors.append(f"{key} length {arr.shape[0]} != M {m}")
    return {
        "name": name,
        "path": str(path.resolve()),
        "point_count": m,
        "points_shape": list(points.shape),
        "bbox_min": points.min(axis=0).astype(float).tolist(),
        "bbox_max": points.max(axis=0).astype(float).tolist(),
        "selected_view_indices": np.asarray(payload["selected_view_indices"], dtype=np.int64).reshape(-1).astype(int).tolist(),
        "visible_count_stats": stat_array(payload["visible_count"]),
        "mask_count_stats": stat_array(payload["mask_count"]),
        "token_count_stats": stat_array(payload["token_count"]),
        "occupancy_ratio_stats": stat_array(payload["occupancy_ratio"]),
        "rgb_variance_stats": stat_array(payload["rgb_variance"]),
        "rgb_range_stats": stat_array(payload["rgb_range"]),
        "token_cosine_stats": stat_array(payload["token_cosine"]),
        "evidence_score_stats": stat_array(payload["evidence_score"]),
        "boundary_like_ratio": float(np.asarray(payload["boundary_like"], dtype=bool).mean()),
        "errors": errors,
        "compatible": not errors,
    }


def compare_controls(real: dict[str, Any], shuffle: dict[str, Any], zero: dict[str, Any]) -> dict[str, Any]:
    controls = {"real": real, "shuffle": shuffle, "zero": zero}
    errors: list[str] = []
    real_points = real["point_count"]
    real_views = real["selected_view_indices"]
    for name, summary in controls.items():
        if summary["point_count"] != real_points:
            errors.append(f"{name} point_count {summary['point_count']} != real {real_points}")
        if summary["selected_view_indices"] != real_views:
            errors.append(f"{name} selected_view_indices mismatch")
    real_token = real["token_cosine_stats"].get("mean")
    shuffle_token = shuffle["token_cosine_stats"].get("mean")
    zero_token = zero["token_cosine_stats"].get("mean")
    real_evidence = real["evidence_score_stats"].get("mean")
    shuffle_evidence = shuffle["evidence_score_stats"].get("mean")
    zero_evidence = zero["evidence_score_stats"].get("mean")
    return {
        "compatible": not errors,
        "errors": errors,
        "point_count": int(real_points),
        "selected_view_indices": real_views,
        "token_cosine_mean": {
            "real": real_token,
            "shuffle": shuffle_token,
            "zero": zero_token,
            "real_minus_shuffle": None if real_token is None or shuffle_token is None else float(real_token - shuffle_token),
        },
        "evidence_score_mean": {
            "real": real_evidence,
            "shuffle": shuffle_evidence,
            "zero": zero_evidence,
            "real_minus_shuffle": None if real_evidence is None or shuffle_evidence is None else float(real_evidence - shuffle_evidence),
            "real_minus_zero": None if real_evidence is None or zero_evidence is None else float(real_evidence - zero_evidence),
        },
        "interpretation": (
            "Contract check only. Real-vs-control separation can justify a later "
            "bounded learned/rendered smoke, but it is not visual or geometric pass evidence."
        ),
    }


def make_query_contract_schema(query_summary: dict[str, Any], token_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D17_surface_sdf_contract_v1",
        "purpose": "Future learned surface-token/SDF backend input schema; contract only.",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "inputs": {
            "token_cache": {
                "tokens": "[1, view_count, token_count, feature_dim]",
                "patch_start_idx": "[1]",
                "selected_view_indices": "[view_count]",
                "observed_shape": token_summary["shape"],
                "patch_grid": token_summary["patch_grid"],
            },
            "query_evidence": {
                "query_positions": "[query_count, 3]",
                "query_families": "[query_count]",
                "support": "[query_count]",
                "token_ids": "[query_count, view_count]",
                "uv": "[query_count, view_count, 2]",
                "depth": "[query_count, view_count]",
                "mean_features": "[query_count, feature_dim]",
                "variance_features": "[query_count, feature_dim]",
                "observed_query_count": query_summary["query_count"],
                "observed_families": query_summary["families"],
            },
        },
        "future_backend_api": {
            "query_inputs": {
                "xyz": "[N, 3]",
                "family_id": "[N]",
                "support": "[N]",
                "mean_features": "[N, C]",
                "variance_features": "[N, C]",
                "view_tokens": "[N, V, C] optional for non-pooled backend",
                "visibility": "[N, V]",
            },
            "named_outputs_required": {
                "sdf": "[N, 1]",
                "occupancy": "[N, 1]",
                "normal": "[N, 3]",
                "rgb": "[N, 3]",
                "confidence": "[N, 1]",
                "support": "[N, 1]",
                "family_logits": "[N, family_count]",
            },
            "forbidden_shortcuts": [
                "vertex-offset-only output",
                "confidence-only filtering to hide low-support regions",
                "VGGT depth/point/normal as hard teacher",
                "SMPL-X face/hair/clothing as hard teacher",
                "strict pass or teacher/candidate export from contract preflight",
            ],
        },
    }


def make_render_supervision_contract() -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D17_render_supervision_contract_v1",
        "purpose": "Future differentiable-render closure for a learned SDF/surface backend.",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "required_renderer_inputs": {
            "mesh_or_surface": "surface extracted from SDF or differentiable surface samples",
            "cameras": "same calibrated 4K4D/VGGT protocol cameras",
            "raw_rgb": "[view_count, H, W, 3]",
            "raw_mask_or_softmatte": "[view_count, H, W]",
            "part_or_family_ids": "body / face_core / hairline / left_hand / right_hand",
        },
        "named_diagnostics_required": {
            "mask": "[V, H, W]",
            "depth": "[V, H, W]",
            "normal": "[V, H, W, 3]",
            "rgb": "[V, H, W, 3]",
            "residuals": {
                "mask_bce": "scalar and per-view",
                "target_recall": "scalar and per-view",
                "overfill": "scalar and per-view",
                "rgb_residual": "scalar and per-view",
                "normal_depth_consistency": "scalar and per-view",
                "part_local_support": "per family and per view",
            },
        },
        "visual_review_required_before_any_export": [
            "full body front/side/back/iso",
            "head close",
            "face close",
            "hairline close",
            "left hand close",
            "right hand close",
        ],
        "fail_closed_conditions": [
            "real token-control does not beat shuffle/zero on rendered diagnostics",
            "Open3D remains shell/slab/template-like",
            "hands are detached or fragmentary",
            "head/face/hairline lack continuous modeled surface",
            "full body has large holes or incoherent slabs",
        ],
    }


def make_control_contract(control_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "B_Fus3D17_control_comparison_contract_v1",
        "purpose": "Controls required for any later learned/rendered B-Fus3D smoke.",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "controls": {
            "real": "unaltered VGGT token evidence plus same masks/RGB/cameras",
            "shuffle": "patch-token evidence shuffled; masks/RGB/cameras unchanged",
            "zero": "token evidence zeroed; masks/RGB/cameras unchanged",
        },
        "current_b15_readout": control_summary,
        "future_required_decisions": [
            "real must beat shuffle and zero on rendered diagnostics, not just scalar field fit loss",
            "shuffle/zero outputs must be rendered and reviewed, not ignored",
            "low-support face/hairline/hands must be reported explicitly",
            "any mesh extracted from controls must be labeled research-only and blocked from gate",
        ],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D17 Surface/SDF Contract Preflight",
        "",
        "Status: `research_only_contract_preflight_complete_no_train_no_export`",
        "",
        "This is a schema and contract preflight for the next B-Fus3D backend. It",
        "does not train, optimize, extract a mesh, write predictions, export a",
        "teacher/candidate, write a strict pass, update the registry, or call cloud.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_facts']['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_facts']['strict_teacher_passes']}",
        "formal_cloud_train_infer_export = blocked",
        "```",
        "",
        "## Compatibility",
        "",
        "```json",
        json.dumps(json_ready(summary["compatibility"]), indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Query Families",
        "",
    ]
    for family, row in summary["query_contract"]["families"].items():
        lines.append(
            f"- `{family}`: count={row['count']}, mean_support={row['mean_support']:.3f}, "
            f"two_view_or_more_ratio={row['two_view_or_more_ratio']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Control Readout",
            "",
            "```json",
            json.dumps(json_ready(summary["control_comparison"]), indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Written Contract Files",
            "",
        ]
    )
    for name, output_path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output_path}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_safe_path(args.output_dir)
    ensure_safe_path(args.status_report)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to refresh")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_summary = summarize_token_cache(args.token_cache)
    query_summary = summarize_query_evidence(args.query_evidence, token_summary)
    latent_real = summarize_latent_control(args.latent_grid_real, "real")
    latent_shuffle = summarize_latent_control(args.latent_grid_shuffle, "shuffle")
    latent_zero = summarize_latent_control(args.latent_grid_zero, "zero")
    control_summary = compare_controls(latent_real, latent_shuffle, latent_zero)

    compatibility_errors: list[str] = []
    for section in (query_summary, latent_real, latent_shuffle, latent_zero, control_summary):
        compatibility_errors.extend(section.get("errors", []))
    compatible = not compatibility_errors

    query_contract_schema = make_query_contract_schema(query_summary, token_summary)
    render_supervision_contract = make_render_supervision_contract()
    control_contract = make_control_contract(control_summary)

    outputs = {
        "summary": str((args.output_dir / "b_fus3d_surface_sdf_contract_summary.json").resolve()),
        "report": str((args.output_dir / "b_fus3d_surface_sdf_contract_report.md").resolve()),
        "query_contract_schema": str((args.output_dir / "query_contract_schema.json").resolve()),
        "render_supervision_contract": str((args.output_dir / "render_supervision_contract.json").resolve()),
        "control_comparison_contract": str((args.output_dir / "control_comparison_contract.json").resolve()),
        "status_report": str(args.status_report.resolve()),
    }

    summary = {
        "status": "research_only_contract_preflight_complete_no_train_no_export",
        "strict_facts": STRICT_FACTS,
        "contract_flags": CONTRACT_FLAGS,
        "contract_preflight_complete": bool(compatible),
        "trained_weights_written": False,
        "mesh_written": False,
        "predictions_written": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "token_contract": token_summary,
        "query_contract": query_summary,
        "latent_controls": {
            "real": latent_real,
            "shuffle": latent_shuffle,
            "zero": latent_zero,
        },
        "control_comparison": control_summary,
        "compatibility": {
            "compatible": bool(compatible),
            "errors": compatibility_errors,
            "expected_query_families": list(EXPECTED_QUERY_FAMILIES),
            "feature_dim_matches": bool(query_summary.get("feature_dim_matches_token_cache")),
            "view_indices_match": not any("selected_view_indices" in err for err in compatibility_errors),
        },
        "outputs": outputs,
        "decision": (
            "B17 contract is ready for a later bounded learned/rendered smoke; this "
            "artifact itself is not geometry evidence and cannot unblock cloud."
            if compatible
            else "B17 contract is not ready; fix schema/input compatibility before any learned smoke."
        ),
        "blocked_actions": [
            "train_or_optimize",
            "mesh_extraction",
            "prediction_npz_write",
            "teacher_export",
            "candidate_export",
            "strict_registry_write",
            "formal_cloud_train_infer_export",
        ],
    }

    write_json(args.output_dir / "query_contract_schema.json", query_contract_schema)
    write_json(args.output_dir / "render_supervision_contract.json", render_supervision_contract)
    write_json(args.output_dir / "control_comparison_contract.json", control_contract)
    write_json(args.output_dir / "b_fus3d_surface_sdf_contract_summary.json", summary)
    write_report(args.output_dir / "b_fus3d_surface_sdf_contract_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "compatible": compatible, "outputs": outputs}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
