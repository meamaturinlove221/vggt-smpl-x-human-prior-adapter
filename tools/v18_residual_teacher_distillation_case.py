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
DEFAULT_V17 = LOCAL_ROOT / "V17_smplx_residual_surface_optimizer"
DEFAULT_OUT = LOCAL_ROOT / "V18_residual_teacher_distillation"
DEFAULT_JSON = REPORTS / "20260508_v18_residual_teacher_distillation_case.json"
DEFAULT_MD = REPORTS / "20260508_v18_residual_teacher_distillation_case.md"


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def safe_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v18_residual_teacher_distillation" not in lower:
        raise ValueError(f"Refusing unsafe V18 output path: {resolved}")
    for token in ("strict_pass", "strict_gate_registry", "candidate_export", "teacher_export", "formal_candidate"):
        if token in lower:
            raise ValueError(f"Forbidden output token: {token}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def stats(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask).astype(bool)]
    values = values.reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {"count": int(values.size), "finite": int(finite.size), "min": float(finite.min()), "mean": float(finite.mean()), "median": float(np.median(finite)), "max": float(finite.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V18 research-only residual-teacher distillation case.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--v17-dir", type=Path, default=DEFAULT_V17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    out = safe_out(args.output_dir)
    inputs = load_npz(args.case_root / "inputs.npz")
    targets = load_npz(args.case_root / "targets.npz")

    prior_points = targets["prior_points"].astype(np.float32)
    prior_depths = targets["prior_depths"].astype(np.float32)
    prior_normals = targets["prior_normals"].astype(np.float32)
    raw_points = targets["world_points"].astype(np.float32)
    visible = targets["smplx_native_visible_mask"].astype(bool)
    raw_mask = inputs["point_masks"].astype(bool)
    teacher_mask = visible & raw_mask & np.isfinite(prior_points).all(axis=-1) & np.isfinite(raw_points).all(axis=-1)

    residual = np.zeros_like(prior_points, dtype=np.float32)
    residual[teacher_mask] = raw_points[teacher_mask] - prior_points[teacher_mask]
    residual_norm = np.linalg.norm(residual, axis=-1)
    # Conservative residual teacher: use only a bounded fraction as a distillation target.
    bounded = np.clip(residual, -0.025, 0.025)
    teacher_points = prior_points + 0.20 * bounded
    teacher_depths = prior_depths.copy()
    finite_z = np.isfinite(teacher_points[..., 2])
    teacher_depths[finite_z] = teacher_points[..., 2][finite_z][..., None]
    teacher_normals = prior_normals.copy()
    uncertainty = np.ones_like(prior_depths[..., 0], dtype=np.float32)
    uncertainty[teacher_mask] = np.clip(residual_norm[teacher_mask] / 0.05, 0.05, 1.0)
    visibility = teacher_mask.astype(np.float32)

    target_path = out / "v18_residual_teacher_targets.npz"
    np.savez_compressed(
        target_path,
        teacher_points=teacher_points.astype(np.float32),
        teacher_depths=teacher_depths.astype(np.float32),
        teacher_normals=teacher_normals.astype(np.float32),
        teacher_visibility=visibility.astype(np.float32),
        teacher_uncertainty=uncertainty.astype(np.float32),
        teacher_mask=teacher_mask,
        residual_vectors=residual.astype(np.float32),
    )

    status = "v18_residual_teacher_case_ready_research_only" if int(teacher_mask.sum()) > 1000 else "v18_residual_teacher_case_sparse_research_only"
    summary = {
        "task": "v18_residual_teacher_distillation_case",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "case_root": args.case_root,
        "v17_dir": args.v17_dir,
        "outputs": {"targets_npz": target_path, "summary_json": out / "summary.json"},
        "metrics": {
            "teacher_pixels": int(teacher_mask.sum()),
            "view_count": int(teacher_mask.shape[0]),
            "residual_norm": stats(residual_norm, teacher_mask),
            "uncertainty": stats(uncertainty, teacher_mask),
            "v17_residual_ply_exists": (args.v17_dir / "v17_smplx_residual_surface_points.ply").is_file(),
        },
        "decision": "V18 built a bounded residual-teacher distillation target from V15/V17 research artifacts. This is not a strict teacher package.",
        "blockers": ["Requires full VGGT research training and visual/strict audits before any promotion."],
    }
    write_json(args.output_json, summary)
    write_json(out / "summary.json", summary)
    lines = [
        "# V18 Residual Teacher Distillation Case",
        "",
        f"Status: `{status}`",
        "",
        summary["decision"],
        "",
        "## Metrics",
        "",
    ]
    for k, v in summary["metrics"].items():
        lines.append(f"- {k}: `{jr(v)}`")
    lines.extend(["", "## Outputs", "", f"- targets_npz: `{target_path}`"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json, "targets": target_path}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
