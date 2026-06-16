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
DEFAULT_CASE = REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_V23 = LOCAL_ROOT / "V23_residual_surface_v2"
DEFAULT_OUT = LOCAL_ROOT / "V24_residual_teacher_v2"
DEFAULT_JSON = REPORTS / "20260508_v24_residual_teacher_v2.json"
DEFAULT_MD = REPORTS / "20260508_v24_residual_teacher_v2.md"


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
        "# V24 Residual Teacher V2",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only residual teacher targets regenerated from V23 complete region evidence. "
        "This script does not write formal predictions, strict registry entries, teacher packages, or candidate packages.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{jr(value)}`")
    lines.extend(["", "## Region Coverage", ""])
    for name, row in summary.get("region_coverage", {}).items():
        lines.append(
            f"- {name}: pixels=`{row['pixels']}`, views=`{row['views_with_pixels']}`, "
            f"sample_points=`{row['sample_points']}`"
        )
    lines.extend(["", "## Audit Readiness", ""])
    for key, value in summary.get("audit_readiness", {}).items():
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
    if "surface_research_preflight_local" not in lower or "v24_residual_teacher_v2" not in lower:
        raise ValueError(f"Refusing unsafe V24 output path: {resolved}")
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
    parser = argparse.ArgumentParser(description="Regenerate research-only residual teacher targets from V23.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--v23-dir", type=Path, default=DEFAULT_V23)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    out = safe_output_dir(args.output_dir)
    inputs = load_npz(args.case_root / "inputs.npz")
    targets = load_npz(args.case_root / "targets.npz")
    v23 = load_npz(args.v23_dir / "v23_residual_surface_v2_points.npz")
    masks = load_npz(args.v23_dir / "v23_repaired_region_evidence_masks.npz")

    prior_points = np.asarray(targets["prior_points"], dtype=np.float32)
    prior_depths = np.asarray(targets["prior_depths"], dtype=np.float32)
    prior_normals = np.asarray(targets["prior_normals"], dtype=np.float32)
    raw_mask = np.asarray(inputs["point_masks"], dtype=bool)
    h_shape = prior_points.shape[:-1]
    flat_count = int(np.prod(h_shape))

    sample_indices = np.asarray(v23["sample_indices"], dtype=np.int64).reshape(-1)
    valid_idx = sample_indices[(sample_indices >= 0) & (sample_indices < flat_count)]
    sample_points = np.asarray(v23["residual_surface_points"], dtype=np.float32).reshape(-1, 3)
    sample_normals = np.asarray(v23["normals"], dtype=np.float32).reshape(-1, 3)
    sample_region_ids = np.asarray(v23["sample_region_ids"], dtype=np.uint8).reshape(-1)
    sample_region_names = np.asarray(v23["sample_region_names"]).astype(str).reshape(-1)
    if valid_idx.size != sample_indices.size:
        keep = (sample_indices >= 0) & (sample_indices < flat_count)
        sample_points = sample_points[keep]
        sample_normals = sample_normals[keep]
        sample_region_ids = sample_region_ids[keep]
        sample_region_names = sample_region_names[keep]
        sample_indices = valid_idx

    teacher_points_flat = prior_points.reshape(-1, 3).copy()
    teacher_normals_flat = prior_normals.reshape(-1, 3).copy()
    teacher_region_flat = np.zeros((flat_count,), dtype=np.uint8)
    teacher_mask_flat = np.zeros((flat_count,), dtype=bool)
    teacher_points_flat[sample_indices] = sample_points
    n_len = np.linalg.norm(sample_normals, axis=-1, keepdims=True)
    teacher_normals_flat[sample_indices] = sample_normals / np.maximum(n_len, 1.0e-6)
    teacher_region_flat[sample_indices] = sample_region_ids
    teacher_mask_flat[sample_indices] = True

    teacher_points = teacher_points_flat.reshape(prior_points.shape).astype(np.float32)
    teacher_normals = teacher_normals_flat.reshape(prior_normals.shape).astype(np.float32)
    teacher_region_masks = teacher_region_flat.reshape(h_shape)
    teacher_mask = teacher_mask_flat.reshape(h_shape)
    teacher_visibility = teacher_mask.astype(np.float32)
    teacher_depths = prior_depths.copy()
    depth_plane = teacher_depths[..., 0] if teacher_depths.ndim == 4 else teacher_depths
    depth_plane[teacher_mask] = teacher_points[..., 2][teacher_mask]
    if teacher_depths.ndim == 4:
        teacher_depths[..., 0] = depth_plane
    else:
        teacher_depths = depth_plane

    residual_norm = np.linalg.norm(teacher_points - prior_points, axis=-1)
    uncertainty = np.ones(h_shape, dtype=np.float32)
    uncertainty[teacher_mask] = np.clip(residual_norm[teacher_mask] / 0.025, 0.03, 1.0)

    # Dense boolean region masks in fixed order for downstream audits.
    region_mask_stack = np.stack([(teacher_region_masks == (i + 1)) for i in range(len(REGION_NAMES))], axis=0)
    teacher_roi_source = np.zeros(h_shape, dtype=np.uint8)
    teacher_roi_source[teacher_mask] = teacher_region_masks[teacher_mask]

    targets_path = out / "v24_residual_teacher_targets_v2.npz"
    np.savez_compressed(
        targets_path,
        teacher_depths=teacher_depths.astype(np.float32),
        teacher_points_world=teacher_points.astype(np.float32),
        teacher_normals_world=teacher_normals.astype(np.float32),
        teacher_visibility=teacher_visibility.astype(np.float32),
        teacher_uncertainty=uncertainty.astype(np.float32),
        teacher_mask=teacher_mask,
        teacher_region_id_map=teacher_region_masks.astype(np.uint8),
        teacher_region_masks=region_mask_stack.astype(np.uint8),
        teacher_region_names=np.asarray(REGION_NAMES),
        teacher_roi_source=teacher_roi_source.astype(np.uint8),
        raw_mask=raw_mask,
    )

    region_coverage: dict[str, Any] = {}
    blockers: list[str] = []
    for idx, name in enumerate(REGION_NAMES, start=1):
        mask = teacher_region_masks == idx
        per_view = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(int).tolist()
        pixels = int(mask.sum())
        samples = int(np.sum(sample_region_names == name))
        region_coverage[name] = {
            "pixels": pixels,
            "sample_points": samples,
            "per_view_pixels": per_view,
            "views_with_pixels": int(np.sum(np.asarray(per_view) > 0)),
            "residual_norm": finite_stats(residual_norm, mask),
        }
        if pixels <= 0:
            blockers.append(f"{name} teacher region is empty")

    # Required support audit handles are generated by V24 audit script; this case builder verifies inputs exist.
    audit_readiness = {
        "has_6v_dense_teacher_arrays": True,
        "has_12v_tmf_scene": (REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf").is_dir(),
        "has_60v_support_case": any((REPO_ROOT / "output").glob("*60*")),
        "v23_surface_complete_regions": all(region_coverage[name]["pixels"] > 0 for name in REGION_NAMES),
        "forbidden_formal_writes": False,
    }

    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v24_residual_teacher_distillation_case_v2",
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
            "case_root": args.case_root,
            "v23_dir": args.v23_dir,
            "v23_surface_npz": args.v23_dir / "v23_residual_surface_v2_points.npz",
            "v23_masks_npz": args.v23_dir / "v23_repaired_region_evidence_masks.npz",
        },
        "outputs": {
            "targets_npz": targets_path,
            "summary_json": out / "summary.json",
            "report_json": args.output_json,
            "report_md": args.output_md,
        },
        "metrics": {
            "teacher_pixels": int(teacher_mask.sum()),
            "view_count": int(teacher_mask.shape[0]),
            "height": int(teacher_mask.shape[1]),
            "width": int(teacher_mask.shape[2]),
            "residual_norm": finite_stats(residual_norm, teacher_mask),
            "uncertainty": finite_stats(uncertainty, teacher_mask),
        },
        "region_coverage": region_coverage,
        "audit_readiness": audit_readiness,
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(args.output_json, summary)
    write_md(args.output_md, summary)
    print(json.dumps(jr({"status": status, "targets": targets_path, "json": args.output_json}), ensure_ascii=False))
    return 0 if status in {"DONE_PASS", "DONE_FAIL_ROUTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
