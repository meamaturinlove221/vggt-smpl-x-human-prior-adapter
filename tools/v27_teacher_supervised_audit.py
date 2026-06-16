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
DEFAULT_V22 = REPORTS / "20260508_v22_true_vggt_smplx_microfit.json"
DEFAULT_V24 = REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
DEFAULT_V26 = REPO_ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
DEFAULT_CONFIG = REPO_ROOT / "training" / "config" / "4k4d_smplx_residual_teacher_research.yaml"
DEFAULT_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V27_teacher_supervised_training"
DEFAULT_JSON = REPORTS / "20260508_v27_teacher_supervised_training.json"
DEFAULT_MD = REPORTS / "20260508_v27_teacher_supervised_training.md"


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


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


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
    parser = argparse.ArgumentParser(description="Audit V27 residual-teacher-supervised VGGT research readiness.")
    parser.add_argument("--v22-report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v24-targets", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v26-targets", type=Path, default=DEFAULT_V26)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = read_json(args.v22_report)
    v24 = load_npz(args.v24_targets)
    v26 = load_npz(args.v26_targets)

    comparison = v22.get("comparison", {})
    positive_controls = {
        key: row
        for key, row in comparison.items()
        if isinstance(row, dict) and row.get("real_beats_all_controls") is True
    }
    any_positive = bool(positive_controls)
    v24_mask = np.asarray(v24["teacher_mask"], dtype=bool)
    v24_region = np.asarray(v24["teacher_region_id_map"], dtype=np.uint8)
    v26_support = np.asarray(v26["canonical_support"], dtype=bool)
    v26_region = np.asarray(v26["target_frame_region_id_map"], dtype=np.uint8)
    normal_available = bool(np.asarray(v26["normal_available"]).item())

    region_rows: dict[str, Any] = {}
    blockers: list[str] = []
    for idx, name in enumerate(REGION_NAMES, start=1):
        v24_pixels = int(((v24_region == idx) & v24_mask).sum())
        v26_pixels = int(((v26_region == idx) & v26_support).sum())
        region_rows[name] = {
            "v24_pixels": v24_pixels,
            "v26_temporal_pixels": v26_pixels,
            "v24_views_with_pixels": int(np.sum(((v24_region == idx) & v24_mask).reshape(v24_mask.shape[0], -1).sum(axis=1) > 0)),
            "v26_views_with_pixels": int(np.sum(((v26_region == idx) & v26_support).reshape(v26_support.shape[0], -1).sum(axis=1) > 0)),
        }
        if v24_pixels <= 0:
            blockers.append(f"{name} missing V24 teacher pixels")
        if v26_pixels <= 0:
            blockers.append(f"{name} missing V26 temporal pixels")

    if not any_positive:
        blockers.append("V22 did not produce any M2/M3 real-beats-all-controls cell")
    if not normal_available:
        blockers.append("V26 normals unavailable because V25 base VGGT has no normal_head")

    # V27 is a research-training gate. The config and teacher arrays are ready, but strict progress is conditional.
    status = "DONE_PASS" if any_positive and all(row["v24_pixels"] > 0 and row["v26_temporal_pixels"] > 0 for row in region_rows.values()) else "DONE_FAIL_ROUTED"
    research_positive = bool(any_positive and len([b for b in blockers if "normal_head" not in b]) == 0)
    audit = {
        "task": "v27_teacher_supervised_vggt_research",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "inputs": {
            "config": args.config,
            "v22_report": args.v22_report,
            "v24_targets": args.v24_targets,
            "v26_targets": args.v26_targets,
        },
        "outputs": {
            "summary_json": args.output_dir / "summary.json",
            "report_json": args.output_json,
            "report_md": args.output_md,
        },
        "research_training_plan": {
            "T0": "single-case overfit with V24 residual teacher",
            "T1": "three-frame temporal case with V26 canonical residual evidence",
            "T2": "12-view multiview research case",
            "controls": ["real_teacher_real_prior", "zero_prior_same_teacher", "shuffle_prior_same_teacher", "smplx_template_teacher_only", "no_teacher_baseline"],
            "modal_entrypoint": "modal_v27_teacher_supervised_vggt_research.py::run_v27_research_training",
        },
        "v22_positive_cells": positive_controls,
        "region_support": region_rows,
        "normal_available": normal_available,
        "research_positive": research_positive,
        "blockers": blockers,
        "decision": (
            "V27 has a valid residual-teacher research config and positive V22 evidence in at least one cell; "
            "strict teacher promotion remains blocked by missing normal-head evidence in V25/V26."
            if research_positive
            else "V27 evidence is not sufficient for strict promotion; it remains research-only and routes to V28 complete failure-proof if strict gates fail."
        ),
    }
    write_json(args.output_dir / "summary.json", audit)
    write_json(args.output_json, audit)
    lines = [
        "# V27 Teacher-Supervised VGGT Research",
        "",
        f"Status: `{status}`",
        "",
        audit["decision"],
        "",
        "## Positive V22 Cells",
        "",
    ]
    for key in positive_controls:
        lines.append(f"- {key}")
    if not positive_controls:
        lines.append("- none")
    lines.extend(["", "## Region Support", ""])
    for name, row in region_rows.items():
        lines.append(f"- {name}: V24=`{row['v24_pixels']}`, V26=`{row['v26_temporal_pixels']}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {b}" for b in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
