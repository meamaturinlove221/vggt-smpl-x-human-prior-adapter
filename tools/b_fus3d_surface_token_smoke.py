from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "uses_vggt_depth_point_normal_as_hard_teacher": False,
    "writes_predictions_npz": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D surface-token smoke. Consumes saved VGGT aggregator tokens "
            "and ROI-to-token coverage, then aggregates diagnostic part/surface-token features. "
            "This never trains, exports a teacher/candidate, writes predictions, writes strict pass "
            "state, or calls cloud."
        )
    )
    parser.add_argument("--token-cache", type=Path, required=True, help="NPZ saved by b_fus3d_token_cache.py.")
    parser.add_argument("--roi-token-coverage", type=Path, required=True, help="roi_token_coverage.json from b_fus3d_token_cache.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--families",
        default="full_body,head,face,face_core,hairline,left_hand,right_hand",
        help="Comma-separated ROI families to aggregate.",
    )
    parser.add_argument("--min-views-soft", type=int, default=2, help="Soft diagnostic support threshold.")
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


def parse_families(text: str) -> list[str]:
    families: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if item and item not in families:
            families.append(item)
    if not families:
        raise ValueError("No ROI families requested")
    return families


def load_token_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"])
        patch_start_idx = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
        selected_view_indices = [int(v) for v in np.asarray(payload["selected_view_indices"]).reshape(-1)]
    if tokens.ndim != 4:
        raise ValueError(f"Expected token array [B,S,T,C], got {tokens.shape}")
    if int(tokens.shape[0]) != 1:
        raise ValueError(f"This smoke expects batch size 1, got {tokens.shape[0]}")
    return {
        "tokens": tokens,
        "patch_start_idx": patch_start_idx,
        "selected_view_indices": selected_view_indices,
        "shape": [int(v) for v in tokens.shape],
        "dtype": str(tokens.dtype),
    }


def feature_stats(feature: np.ndarray) -> dict[str, Any]:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    finite = np.isfinite(feature)
    if not bool(finite.all()):
        feature = feature[finite]
    if feature.size == 0:
        return {"status": "no_finite_feature"}
    abs_feature = np.abs(feature)
    return {
        "status": "ok",
        "dim": int(feature.size),
        "mean": float(feature.mean()),
        "std": float(feature.std()),
        "l2": float(np.linalg.norm(feature)),
        "abs_p50": float(np.quantile(abs_feature, 0.50)),
        "abs_p95": float(np.quantile(abs_feature, 0.95)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return None
    return float(np.dot(a, b) / denom)


def cosine_summary(features: list[np.ndarray]) -> dict[str, Any]:
    if len(features) < 2:
        return {"status": "not_enough_views", "pair_count": 0}
    values: list[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            value = cosine(features[i], features[j])
            if value is not None and math.isfinite(value):
                values.append(float(value))
    if not values:
        return {"status": "no_valid_pairs", "pair_count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "status": "ok",
        "pair_count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def aggregate_family(
    family: str,
    coverage_family: dict[str, Any],
    token_payload: dict[str, Any],
    *,
    min_views_soft: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    tokens = np.asarray(token_payload["tokens"])
    selected = list(token_payload["selected_view_indices"])
    view_to_pos = {int(view_idx): pos for pos, view_idx in enumerate(selected)}
    per_view: dict[str, Any] = {}
    view_features: list[np.ndarray] = []
    view_weights: list[float] = []
    views = coverage_family.get("views") if isinstance(coverage_family.get("views"), dict) else {}
    for view_text, row in views.items():
        view_idx = int(view_text)
        if view_idx not in view_to_pos:
            continue
        token_ids = row.get("aggregator_token_ids")
        if not isinstance(token_ids, list):
            token_ids = row.get("aggregator_token_ids_preview", [])
        token_ids = [int(v) for v in token_ids if 0 <= int(v) < int(tokens.shape[2])]
        if not token_ids:
            per_view[str(view_idx)] = {
                "status": row.get("status", "empty_roi"),
                "roi_pixels": int(row.get("roi_pixels", 0) or 0),
                "token_count": 0,
            }
            continue
        view_tokens = tokens[0, view_to_pos[view_idx], token_ids, :].astype(np.float32)
        pooled = view_tokens.mean(axis=0)
        token_count = int(len(token_ids))
        roi_pixels = int(row.get("roi_pixels", 0) or 0)
        weight = float(max(1, token_count))
        view_features.append(pooled)
        view_weights.append(weight)
        per_view[str(view_idx)] = {
            "status": "pooled",
            "roi_status": row.get("status"),
            "roi_pixels": roi_pixels,
            "token_count": token_count,
            "feature_stats": feature_stats(pooled),
            "token_ids_preview": token_ids[:48],
        }
    support_views = int(len(view_features))
    pooled_feature: np.ndarray | None = None
    if view_features:
        weights = np.asarray(view_weights, dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        stacked = np.stack(view_features, axis=0).astype(np.float32)
        pooled_feature = (stacked * weights[:, None]).sum(axis=0)
    status = "ok" if support_views >= int(min_views_soft) else "weak_view_support"
    return (
        {
            "family": family,
            "status": status,
            "support_views": support_views,
            "selected_view_count": int(len(selected)),
            "roi_mask_source": coverage_family.get("roi_mask_source"),
            "per_view": per_view,
            "pooled_feature_stats": feature_stats(pooled_feature) if pooled_feature is not None else {"status": "missing"},
            "cross_view_cosine": cosine_summary(view_features),
            "hard_gate_allowed": False,
        },
        pooled_feature,
    )


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Surface-Token Smoke",
        "",
        "Status: `research_only_surface_token_smoke`",
        "",
        "This is not a teacher, not a candidate, not training, and not a strict pass.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_FACTS, indent=2),
        "```",
        "",
        "## Family Readout",
        "",
    ]
    for family, row in summary["families"].items():
        lines.extend(
            [
                f"### {family}",
                "",
                f"- status: `{row['status']}`",
                f"- support_views: `{row['support_views']}/{row['selected_view_count']}`",
                f"- roi_source: `{row.get('roi_mask_source')}`",
                f"- cross_view_cosine: `{row['cross_view_cosine'].get('status')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Blockers",
            "",
        ]
    )
    for blocker in summary["blockers"]:
        lines.append(f"- {blocker}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    token_payload = load_token_cache(args.token_cache.resolve())
    coverage = json.loads(args.roi_token_coverage.resolve().read_text(encoding="utf-8"))
    families_requested = parse_families(args.families)
    features: dict[str, np.ndarray] = {}
    family_rows: dict[str, Any] = {}
    blockers: list[str] = []
    coverage_families = coverage.get("families") if isinstance(coverage.get("families"), dict) else {}
    for family in families_requested:
        if family not in coverage_families:
            family_rows[family] = {"family": family, "status": "missing_coverage_family", "hard_gate_allowed": False}
            blockers.append(f"{family}: missing ROI coverage family")
            continue
        row, pooled = aggregate_family(
            family,
            coverage_families[family],
            token_payload,
            min_views_soft=int(args.min_views_soft),
        )
        family_rows[family] = row
        if pooled is not None:
            features[family] = pooled.astype(np.float16)
        if row["status"] != "ok":
            blockers.append(f"{family}: {row['status']} ({row['support_views']} support views)")

    feature_path = output_dir / "surface_token_features.npz"
    if features:
        np.savez_compressed(
            feature_path,
            **features,
            family_names=np.asarray(list(features.keys())),
            selected_view_indices=np.asarray(token_payload["selected_view_indices"], dtype=np.int32),
            patch_start_idx=np.asarray([int(token_payload["patch_start_idx"])], dtype=np.int32),
        )
    else:
        feature_path = None
        blockers.append("No pooled family features were produced")

    summary = {
        "task": "b_fus3d_surface_token_smoke",
        "schema_version": 1,
        "status": "research_only_surface_token_smoke",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "token_cache": str(args.token_cache.resolve()),
            "roi_token_coverage": str(args.roi_token_coverage.resolve()),
            "families": families_requested,
        },
        "token_cache": {
            "shape": token_payload["shape"],
            "dtype": token_payload["dtype"],
            "selected_view_indices": token_payload["selected_view_indices"],
            "patch_start_idx": token_payload["patch_start_idx"],
        },
        "families": family_rows,
        "outputs": {
            "summary_json": str(output_dir / "b_fus3d_surface_token_smoke_summary.json"),
            "report_md": str(output_dir / "b_fus3d_surface_token_smoke_report.md"),
            "surface_token_features_npz": str(feature_path) if feature_path is not None else None,
        },
        "blockers": blockers,
        "next_allowed_action": (
            "If face/head/hairline have support and hands are diagnosed separately, build a tiny local decoder skeleton. "
            "Do not train formal candidate or export teacher until strict gates pass."
        ),
    }
    write_json(output_dir / "b_fus3d_surface_token_smoke_summary.json", summary)
    write_markdown_report(output_dir / "b_fus3d_surface_token_smoke_report.md", summary)
    print(json.dumps(json_ready({"summary": summary["outputs"]["summary_json"], "status": summary["status"], "blockers": blockers}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
