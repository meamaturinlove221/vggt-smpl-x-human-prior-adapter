from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
CLOUD_ROOT = REPO_ROOT / "output" / "surface_research_cloud_preflight"
DEFAULT_V24 = LOCAL_ROOT / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
DEFAULT_V25 = CLOUD_ROOT / "V25_research_vggt_predictions"
DEFAULT_OUT = LOCAL_ROOT / "V26_temporal_canonical_teacher"
DEFAULT_JSON = REPORTS / "20260508_v26_temporal_canonical_teacher.json"
DEFAULT_MD = REPORTS / "20260508_v26_temporal_canonical_teacher.md"


REGION_NAMES = ("body", "head", "face", "left_hand", "right_hand")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V26 Temporal Canonical SMPL-X Residual Teacher",
        "",
        f"Status: `{summary['status']}`",
        "",
        summary["decision"],
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.get("metrics", {}).items():
        lines.append(f"- {key}: `{jr(value)}`")
    lines.extend(["", "## Regions", ""])
    for name, row in summary.get("region_support", {}).items():
        lines.append(
            f"- {name}: canonical_support=`{row['canonical_support']}`, frames=`{row['frames_with_support']}`, "
            f"temporal_variance_mean=`{row.get('temporal_variance_mean')}`"
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{jr(value)}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v26_temporal_canonical_teacher" not in lower:
        raise ValueError(f"Refusing unsafe V26 output path: {resolved}")
    for token in ("predictions.npz", "candidate_package", "teacher_package", "strict_gate_registry", "strict_pass"):
        if token in lower:
            raise ValueError(f"Forbidden output token in path: {token}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def finite_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr.reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V26 temporal canonical residual teacher evidence.")
    parser.add_argument("--v24-targets", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v25-root", type=Path, default=DEFAULT_V25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    out = safe_output_dir(args.output_dir)
    v24 = load_npz(args.v24_targets)
    points_npz = load_npz(args.v25_root / "research_points_world.npz")
    depths_npz = load_npz(args.v25_root / "research_depths.npz")
    conf_npz = load_npz(args.v25_root / "research_confidence.npz")
    normals_npz = load_npz(args.v25_root / "research_normals.npz")

    frame_keys = [str(x) for x in points_npz["frame_keys"].tolist()]
    target_points_6v = np.asarray(v24["teacher_points_world"], dtype=np.float32)
    target_mask_6v = np.asarray(v24["teacher_mask"], dtype=bool)
    region_map_6v = np.asarray(v24["teacher_region_id_map"], dtype=np.uint8)
    # V25 has 12 views; V24 has 6 views. Use the first 6 views as the shared deterministic support contract.
    shared_views = min(6, int(points_npz[frame_keys[0]].shape[0]), int(target_points_6v.shape[0]))
    target_points = target_points_6v[:shared_views]
    target_mask = target_mask_6v[:shared_views]
    region_map = region_map_6v[:shared_views]

    frame_residuals = []
    frame_weights = []
    per_frame_stats: dict[str, Any] = {}
    for frame in frame_keys:
        frame_points = np.asarray(points_npz[frame], dtype=np.float32)[:shared_views]
        frame_conf = np.asarray(conf_npz[f"{frame}_world_points_conf"], dtype=np.float32)[:shared_views]
        finite = np.isfinite(frame_points).all(axis=-1) & target_mask
        # Canonical proxy: same image/SMPL-X region support coordinates. This is a research residual accumulator,
        # not a promoted strict canonical teacher.
        residual = np.zeros_like(target_points, dtype=np.float32)
        residual[finite] = frame_points[finite] - target_points[finite]
        weight = np.zeros(target_mask.shape, dtype=np.float32)
        if finite.any():
            conf = np.asarray(frame_conf, dtype=np.float32)
            weight[finite] = 1.0 / np.maximum(conf[finite], 1.0)
            weight[finite] = np.clip(weight[finite], 0.02, 1.0)
        frame_residuals.append(residual)
        frame_weights.append(weight)
        per_frame_stats[frame] = {
            "finite_support": int(finite.sum()),
            "residual_norm": finite_stats(np.linalg.norm(residual, axis=-1), finite),
            "depth": finite_stats(np.asarray(depths_npz[frame], dtype=np.float32)[:shared_views, ..., 0], finite),
        }

    residual_stack = np.stack(frame_residuals, axis=0)
    weight_stack = np.stack(frame_weights, axis=0)
    weight_sum = np.maximum(weight_stack.sum(axis=0), 1.0e-6)
    canonical_residual = (residual_stack * weight_stack[..., None]).sum(axis=0) / weight_sum[..., None]
    canonical_support = weight_stack.sum(axis=0) > 0
    temporal_var = np.var(residual_stack, axis=0).mean(axis=-1)
    target_frame_points = target_points + canonical_residual
    confidence = np.clip(weight_sum / max(1, len(frame_keys)), 0.0, 1.0).astype(np.float32)
    uncertainty = np.ones_like(confidence, dtype=np.float32)
    uncertainty[canonical_support] = np.clip(np.sqrt(temporal_var[canonical_support]) / 0.05, 0.03, 1.0)

    outputs_npz = out / "v26_temporal_canonical_teacher_targets.npz"
    np.savez_compressed(
        outputs_npz,
        canonical_residual=canonical_residual.astype(np.float32),
        canonical_support=canonical_support.astype(np.uint8),
        target_frame_points=target_frame_points.astype(np.float32),
        target_frame_region_id_map=region_map.astype(np.uint8),
        temporal_variance=temporal_var.astype(np.float32),
        temporal_confidence=confidence.astype(np.float32),
        temporal_uncertainty=uncertainty.astype(np.float32),
        frame_keys=np.asarray(frame_keys),
        shared_view_count=np.asarray(shared_views, dtype=np.int32),
        normal_available=np.asarray(bool(normals_npz.get("normal_available", np.asarray(False)).item())),
    )

    region_support: dict[str, Any] = {}
    blockers: list[str] = []
    for idx, name in enumerate(REGION_NAMES, start=1):
        mask = (region_map == idx) & canonical_support
        frame_counts = []
        for fi in range(len(frame_keys)):
            frame_counts.append(int(((region_map == idx) & (weight_stack[fi] > 0)).sum()))
        region_support[name] = {
            "canonical_support": int(mask.sum()),
            "per_frame_support": frame_counts,
            "frames_with_support": int(np.sum(np.asarray(frame_counts) > 0)),
            "temporal_variance": finite_stats(temporal_var, mask),
            "temporal_variance_mean": float(np.mean(temporal_var[mask])) if mask.any() else None,
        }
        if int(mask.sum()) <= 0:
            blockers.append(f"{name} has zero temporal canonical support")
        if int(np.sum(np.asarray(frame_counts) > 0)) < 3:
            blockers.append(f"{name} does not have support from all 3 frames")

    normal_available = bool(normals_npz.get("normal_available", np.asarray(False)).item())
    if not normal_available:
        blockers.append("V25 loaded model has no normal_head; temporal normals unavailable and not fabricated")

    # Normals unavailable is a strict-teacher blocker, but not a temporal evidence construction failure.
    non_normal_blockers = [b for b in blockers if "normal_head" not in b]
    status = "DONE_PASS" if not non_normal_blockers else "DONE_FAIL_ROUTED"
    decision = (
        "V26 constructed a three-frame research-only temporal residual accumulator. "
        "It is usable as research evidence but remains blocked for strict teacher promotion until normal evidence is available."
        if not normal_available
        else "V26 constructed a three-frame research-only temporal residual accumulator."
    )
    summary = {
        "task": "v26_temporal_canonical_smplx_residual_teacher",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "inputs": {
            "v24_targets": args.v24_targets,
            "v25_root": args.v25_root,
        },
        "outputs": {
            "temporal_targets_npz": outputs_npz,
            "summary_json": out / "summary.json",
            "report_json": args.output_json,
            "report_md": args.output_md,
        },
        "metrics": {
            "frame_keys": frame_keys,
            "shared_view_count": shared_views,
            "canonical_support_pixels": int(canonical_support.sum()),
            "canonical_residual_norm": finite_stats(np.linalg.norm(canonical_residual, axis=-1), canonical_support),
            "temporal_variance": finite_stats(temporal_var, canonical_support),
            "normal_available": normal_available,
        },
        "per_frame": per_frame_stats,
        "region_support": region_support,
        "blockers": blockers,
        "decision": decision,
    }
    write_json(out / "summary.json", summary)
    write_json(args.output_json, summary)
    write_md(args.output_md, summary)
    print(json.dumps(jr({"status": status, "json": args.output_json, "targets": outputs_npz}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
