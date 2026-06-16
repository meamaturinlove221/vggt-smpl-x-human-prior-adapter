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
DEFAULT_TARGETS = REPO_ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
DEFAULT_JSON = REPORTS / "20260508_v26_temporal_consistency_audit.json"
DEFAULT_MD = REPORTS / "20260508_v26_temporal_consistency_audit.md"
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
    parser = argparse.ArgumentParser(description="Audit V26 temporal canonical teacher consistency.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    with np.load(args.targets, allow_pickle=False) as z:
        canonical_support = np.asarray(z["canonical_support"], dtype=bool)
        region_map = np.asarray(z["target_frame_region_id_map"], dtype=np.uint8)
        temporal_variance = np.asarray(z["temporal_variance"], dtype=np.float32)
        confidence = np.asarray(z["temporal_confidence"], dtype=np.float32)
        uncertainty = np.asarray(z["temporal_uncertainty"], dtype=np.float32)
        normal_available = bool(np.asarray(z["normal_available"]).item())
        frame_keys = [str(x) for x in np.asarray(z["frame_keys"]).tolist()]

    blockers: list[str] = []
    region_rows: dict[str, Any] = {}
    for idx, name in enumerate(REGION_NAMES, start=1):
        mask = (region_map == idx) & canonical_support
        pixels = int(mask.sum())
        per_view = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(int).tolist()
        region_rows[name] = {
            "pixels": pixels,
            "views_with_support": int(np.sum(np.asarray(per_view) > 0)),
            "per_view_pixels": per_view,
            "temporal_variance": finite_stats(temporal_variance, mask),
            "confidence": finite_stats(confidence, mask),
            "uncertainty": finite_stats(uncertainty, mask),
        }
        if pixels <= 0:
            blockers.append(f"{name} missing temporal support")

    strict_blockers = list(blockers)
    if not normal_available:
        strict_blockers.append("normal evidence unavailable from V25; strict temporal teacher cannot pass")
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    audit = {
        "task": "v26_temporal_consistency_audit",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "inputs": {"targets_npz": args.targets},
        "frame_keys": frame_keys,
        "canonical_support_pixels": int(canonical_support.sum()),
        "normal_available": normal_available,
        "research_blockers": blockers,
        "strict_blockers": strict_blockers,
        "region_support": region_rows,
    }
    write_json(args.output_json, audit)
    lines = [
        "# V26 Temporal Consistency Audit",
        "",
        f"Status: `{status}`",
        "",
        f"- normal_available: `{normal_available}`",
        f"- canonical_support_pixels: `{audit['canonical_support_pixels']}`",
        "",
        "## Regions",
        "",
    ]
    for name, row in region_rows.items():
        lines.append(f"- {name}: pixels=`{row['pixels']}`, views=`{row['views_with_support']}`")
    lines.extend(["", "## Research Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    lines.extend(["", "## Strict Blockers", ""])
    lines.extend([f"- {item}" for item in strict_blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
