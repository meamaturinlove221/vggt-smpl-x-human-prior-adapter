from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SURFACE_TOKEN_FEATURES = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D1_surface_token_smoke_hybrid6_layer23/surface_token_features.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D2_decoder_skeleton_smoke_hybrid6_layer23"
)

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
    "numpy_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
    "tiny_random_forward_is_diagnostic_only": True,
}

PART_HEADS = {
    "coarse_surface": {
        "kind": "metadata_only_part_head",
        "future_role": "local part surface query field",
        "blocked_outputs": ["mesh", "pointcloud", "depth", "normal", "mask", "candidate_npz"],
    },
    "visibility": {
        "kind": "metadata_only_part_head",
        "future_role": "view-conditioned part visibility",
        "blocked_outputs": ["rendered_mask", "strict_visibility_pass"],
    },
    "confidence": {
        "kind": "metadata_only_part_head",
        "future_role": "research confidence readout from token evidence",
        "blocked_outputs": ["gate_score", "teacher_score", "candidate_score"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only local B-Fus3D decoder skeleton smoke. It consumes pooled "
            "surface token features and writes diagnostic JSON/MD summaries for latent "
            "query and part-head wiring. It never trains, writes predictions, exports "
            "teacher/candidate artifacts, writes strict pass state, or calls cloud."
        )
    )
    parser.add_argument("--surface-token-features", type=Path, default=DEFAULT_SURFACE_TOKEN_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parts",
        default="",
        help="Comma-separated part names. Empty means use family_names from the NPZ.",
    )
    parser.add_argument("--query-dim", type=int, default=32)
    parser.add_argument("--queries-per-part", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument(
        "--status-report",
        type=Path,
        default=None,
        help="Optional extra markdown status report path, e.g. reports/20260507_b_fus3d_decoder_skeleton_status.md.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_parts(text: str) -> list[str]:
    parts: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in parts:
            parts.append(item)
    return parts


def feature_stats(feature: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not bool(finite.all()):
        arr = arr[finite]
    if arr.size == 0:
        return {"status": "no_finite_feature"}
    abs_arr = np.abs(arr)
    return {
        "status": "ok",
        "dim": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "l2": float(np.linalg.norm(arr)),
        "abs_p50": float(np.quantile(abs_arr, 0.50)),
        "abs_p95": float(np.quantile(abs_arr, 0.95)),
    }


def load_surface_features(path: Path, requested_parts: list[str]) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with np.load(resolved, allow_pickle=False) as payload:
        keys = list(payload.files)
        family_names = [str(v) for v in np.asarray(payload["family_names"]).reshape(-1)] if "family_names" in keys else []
        parts = requested_parts or family_names
        if not parts:
            parts = [
                key
                for key in keys
                if key not in {"family_names", "selected_view_indices", "patch_start_idx"}
            ]
        features: dict[str, np.ndarray] = {}
        missing_parts: list[str] = []
        for part in parts:
            if part not in keys:
                missing_parts.append(part)
                continue
            arr = np.asarray(payload[part], dtype=np.float32).reshape(-1)
            if arr.size == 0:
                missing_parts.append(part)
                continue
            features[part] = arr
        selected_view_indices = (
            [int(v) for v in np.asarray(payload["selected_view_indices"]).reshape(-1)]
            if "selected_view_indices" in keys
            else []
        )
        patch_start_idx = (
            int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
            if "patch_start_idx" in keys and np.asarray(payload["patch_start_idx"]).size
            else None
        )

    dims = sorted({int(arr.size) for arr in features.values()})
    if len(dims) > 1:
        raise ValueError(f"Surface features must share one dim, got {dims}")
    return {
        "resolved_path": str(resolved),
        "npz_keys": keys,
        "family_names": family_names,
        "requested_parts": parts,
        "missing_parts": missing_parts,
        "features": features,
        "feature_dim": int(dims[0]) if dims else 0,
        "selected_view_indices": selected_view_indices,
        "patch_start_idx": patch_start_idx,
    }


def stable_part_code(name: str) -> float:
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    return float((total % 997) / 997.0)


def tiny_random_forward(
    features: dict[str, np.ndarray],
    *,
    query_dim: int,
    queries_per_part: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    feature_dim = int(next(iter(features.values())).size) if features else 0
    if feature_dim <= 0:
        return {}, {"status": "no_features"}

    input_projection = rng.normal(0.0, 1.0 / math.sqrt(feature_dim), size=(feature_dim, query_dim)).astype(np.float32)
    head_weights = {
        name: rng.normal(0.0, 1.0 / math.sqrt(query_dim), size=(query_dim, 1)).astype(np.float32)
        for name in PART_HEADS
    }

    part_rows: dict[str, Any] = {}
    all_scores: list[float] = []
    for part_name, feature in features.items():
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        finite_feature = np.where(np.isfinite(feature), feature, 0.0)
        denom = float(np.linalg.norm(finite_feature))
        normalized = finite_feature / denom if denom > 1e-8 else finite_feature
        projected = normalized @ input_projection

        latent_queries = rng.normal(0.0, 0.02, size=(queries_per_part, query_dim)).astype(np.float32)
        latent_queries[:, 0] += stable_part_code(part_name)
        latent_queries += projected.reshape(1, query_dim)
        latent_queries = np.tanh(latent_queries).astype(np.float32)

        head_rows: dict[str, Any] = {}
        for head_name, head_meta in PART_HEADS.items():
            scores = (latent_queries @ head_weights[head_name]).reshape(-1)
            all_scores.extend(float(v) for v in scores)
            head_rows[head_name] = {
                **head_meta,
                "query_score_shape": [int(queries_per_part)],
                "score_stats": {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                },
                "emitted_prediction": False,
                "hard_gate_allowed": False,
            }

        part_rows[part_name] = {
            "status": "diagnostic_forward_only",
            "feature_stats": feature_stats(feature),
            "latent_query_metadata": {
                "query_count": int(queries_per_part),
                "query_dim": int(query_dim),
                "init": "deterministic_random_normal_plus_surface_token_projection",
                "stored_as_prediction": False,
                "mean": float(latent_queries.mean()),
                "std": float(latent_queries.std()),
                "abs_p95": float(np.quantile(np.abs(latent_queries), 0.95)),
            },
            "part_heads": head_rows,
            "blocked_from_gate": True,
        }

    scores_arr = np.asarray(all_scores, dtype=np.float32)
    forward_summary = {
        "status": "ran_numpy_tiny_random_forward",
        "seed": int(seed),
        "input_feature_dim": int(feature_dim),
        "query_dim": int(query_dim),
        "queries_per_part": int(queries_per_part),
        "part_count": int(len(part_rows)),
        "score_count": int(scores_arr.size),
        "score_mean": float(scores_arr.mean()) if scores_arr.size else None,
        "score_std": float(scores_arr.std()) if scores_arr.size else None,
        "emitted_prediction": False,
        "wrote_prediction_file": False,
    }
    return part_rows, forward_summary


def build_blockers(source: dict[str, Any], part_rows: dict[str, Any]) -> list[str]:
    blockers = [
        "strict_candidate_passes remains 0",
        "strict_teacher_passes remains 0",
        "formal cloud train/infer/export remains blocked",
        "no trained decoder weights or checkpoint exist",
        "tiny random forward is diagnostic only and cannot be scored as a candidate",
        "no mesh/depth/normal/mask/rendered prediction was written",
        "no teacher or candidate export was written",
        "no strict pass registry was written",
    ]
    for part in source.get("missing_parts", []):
        blockers.append(f"{part}: missing from surface_token_features.npz")
    for part, row in part_rows.items():
        if row.get("blocked_from_gate"):
            blockers.append(f"{part}: part head metadata is blocked from gate use")
    return blockers


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Decoder Skeleton Smoke",
        "",
        "Status: `research_only_decoder_skeleton`",
        "",
        "This is a local NumPy wiring smoke over pooled B-Fus3D surface tokens. It is not training, not inference, not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_FACTS, indent=2),
        "```",
        "",
        "## Inputs",
        "",
        f"- surface_token_features: `{summary['inputs']['surface_token_features']}`",
        f"- selected_view_indices: `{summary['inputs']['selected_view_indices']}`",
        f"- patch_start_idx: `{summary['inputs']['patch_start_idx']}`",
        "",
        "## Tiny Forward",
        "",
        f"- status: `{summary['tiny_forward']['status']}`",
        f"- seed: `{summary['tiny_forward']['seed']}`",
        f"- part_count: `{summary['tiny_forward']['part_count']}`",
        f"- query_dim: `{summary['tiny_forward']['query_dim']}`",
        f"- queries_per_part: `{summary['tiny_forward']['queries_per_part']}`",
        f"- emitted_prediction: `{summary['tiny_forward']['emitted_prediction']}`",
        "",
        "## Part Metadata",
        "",
    ]
    for part_name, row in summary["parts"].items():
        lines.extend(
            [
                f"### {part_name}",
                "",
                f"- status: `{row['status']}`",
                f"- feature_dim: `{row['feature_stats'].get('dim')}`",
                f"- latent_query_shape: `[{row['latent_query_metadata']['query_count']}, {row['latent_query_metadata']['query_dim']}]`",
                f"- part_heads: `{', '.join(row['part_heads'].keys())}`",
                f"- blocked_from_gate: `{row['blocked_from_gate']}`",
                "",
            ]
        )
    lines.extend(["## Blockers", ""])
    for blocker in summary["blockers"]:
        lines.append(f"- {blocker}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_safe_outputs(output_dir: Path, status_report: Path | None, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    if status_report is not None and status_report.exists() and not overwrite:
        raise FileExistsError(f"{status_report} exists; pass --overwrite")


def main() -> int:
    args = parse_args()
    if int(args.query_dim) <= 0:
        raise ValueError("--query-dim must be positive")
    if int(args.queries_per_part) <= 0:
        raise ValueError("--queries-per-part must be positive")

    output_dir = args.output_dir.resolve()
    status_report = args.status_report.resolve() if args.status_report is not None else None
    assert_safe_outputs(output_dir, status_report, bool(args.overwrite))
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_surface_features(args.surface_token_features, parse_parts(args.parts))
    parts, forward_summary = tiny_random_forward(
        source["features"],
        query_dim=int(args.query_dim),
        queries_per_part=int(args.queries_per_part),
        seed=int(args.seed),
    )
    summary = {
        "task": "b_fus3d_decoder_skeleton_smoke",
        "schema_version": 1,
        "status": "research_only_decoder_skeleton",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "surface_token_features": source["resolved_path"],
            "npz_keys": source["npz_keys"],
            "family_names": source["family_names"],
            "requested_parts": source["requested_parts"],
            "missing_parts": source["missing_parts"],
            "selected_view_indices": source["selected_view_indices"],
            "patch_start_idx": source["patch_start_idx"],
            "query_dim": int(args.query_dim),
            "queries_per_part": int(args.queries_per_part),
            "seed": int(args.seed),
        },
        "tiny_forward": forward_summary,
        "parts": parts,
        "outputs": {
            "summary_json": str(output_dir / "b_fus3d_decoder_skeleton_smoke_summary.json"),
            "report_md": str(output_dir / "b_fus3d_decoder_skeleton_smoke_report.md"),
            "status_report_md": str(status_report) if status_report is not None else None,
            "prediction_files_written": [],
            "teacher_exports_written": [],
            "candidate_exports_written": [],
            "strict_pass_files_written": [],
        },
        "blockers": build_blockers(source, parts),
        "next_allowed_action": (
            "Use this only as local decoder wiring metadata. A real local decoder would still need "
            "trained weights, rendered diagnostics, and strict gate review before any candidate/teacher/export claim."
        ),
    }

    write_json(output_dir / "b_fus3d_decoder_skeleton_smoke_summary.json", summary)
    write_markdown(output_dir / "b_fus3d_decoder_skeleton_smoke_report.md", summary)
    if status_report is not None:
        write_markdown(status_report, summary)

    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "summary": summary["outputs"]["summary_json"],
                    "report": summary["outputs"]["report_md"],
                    "status_report": summary["outputs"]["status_report_md"],
                    "prediction_files_written": [],
                    "strict_candidate_passes": 0,
                    "strict_teacher_passes": 0,
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
