from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize A5 teacher-gate failures across existing audit outputs. This is a "
            "research-only failure audit; it does not run reconstruction, export teacher/candidate, "
            "write predictions, write strict registry, or call cloud."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("output/surface_research_preflight_local"))
    parser.add_argument("--pattern", default="A5_*teacher_gate*/teacher_gate_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarize_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    by_roi: dict[str, dict[str, list[float]]] = {}
    gates_failed: dict[str, int] = {}
    total_raw_hits = 0
    total_compat_hits = 0
    total_roi_pixels = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        roi = str(entry.get("roi_kind"))
        bucket = by_roi.setdefault(
            roi,
            {
                "raw_hit_pixels": [],
                "compat_hit_pixels": [],
                "roi_pixels": [],
                "raw_p50": [],
                "raw_p90": [],
                "compat_p50": [],
                "compat_p90": [],
                "raw_largest_ratio": [],
                "compat_largest_ratio": [],
            },
        )
        raw = entry.get("raw_visible") if isinstance(entry.get("raw_visible"), dict) else {}
        compat = entry.get("depth_compatible") if isinstance(entry.get("depth_compatible"), dict) else {}
        roi_pixels = int(entry.get("roi_pixels", 0) or 0)
        raw_hits = int(raw.get("hit_pixels", 0) or 0)
        compat_hits = int(compat.get("hit_pixels", 0) or 0)
        total_raw_hits += raw_hits
        total_compat_hits += compat_hits
        total_roi_pixels += roi_pixels
        bucket["roi_pixels"].append(float(roi_pixels))
        bucket["raw_hit_pixels"].append(float(raw_hits))
        bucket["compat_hit_pixels"].append(float(compat_hits))
        for source, prefix in ((raw, "raw"), (compat, "compat")):
            residual = source.get("depth_residual") if isinstance(source.get("depth_residual"), dict) else {}
            comps = source.get("components") if isinstance(source.get("components"), dict) else {}
            p50 = number(residual.get("p50"))
            p90 = number(residual.get("p90"))
            largest = number(comps.get("largest_component_ratio"))
            if p50 is not None:
                bucket[f"{prefix}_p50"].append(p50)
            if p90 is not None:
                bucket[f"{prefix}_p90"].append(p90)
            if largest is not None:
                bucket[f"{prefix}_largest_ratio"].append(largest)
        gate = entry.get("gate") if isinstance(entry.get("gate"), dict) else {}
        for key, value in gate.items():
            if key == "pass":
                continue
            if value is False:
                gates_failed[key] = gates_failed.get(key, 0) + 1
    roi_rows: dict[str, Any] = {}
    for roi, bucket in by_roi.items():
        roi_pixels_sum = float(sum(bucket["roi_pixels"]))
        raw_sum = float(sum(bucket["raw_hit_pixels"]))
        compat_sum = float(sum(bucket["compat_hit_pixels"]))
        roi_rows[roi] = {
            "roi_pixels_sum": int(roi_pixels_sum),
            "raw_hit_sum": int(raw_sum),
            "compat_hit_sum": int(compat_sum),
            "raw_coverage_sum": float(raw_sum / max(roi_pixels_sum, 1.0)),
            "compat_coverage_sum": float(compat_sum / max(roi_pixels_sum, 1.0)),
            "compat_over_raw": float(compat_sum / max(raw_sum, 1.0)),
            "raw_depth_p50": summarize_values(bucket["raw_p50"]),
            "raw_depth_p90": summarize_values(bucket["raw_p90"]),
            "compat_depth_p50": summarize_values(bucket["compat_p50"]),
            "compat_depth_p90": summarize_values(bucket["compat_p90"]),
            "raw_largest_component_ratio": summarize_values(bucket["raw_largest_ratio"]),
            "compat_largest_component_ratio": summarize_values(bucket["compat_largest_ratio"]),
        }
    return {
        "name": path.parent.name,
        "path": str(path),
        "source_path": payload.get("source_path"),
        "gate": payload.get("gate"),
        "entry_count": int(len(entries)),
        "target_views": payload.get("target_views"),
        "roi_kinds": payload.get("roi_kinds"),
        "total_roi_pixels": int(total_roi_pixels),
        "total_raw_hits": int(total_raw_hits),
        "total_compat_hits": int(total_compat_hits),
        "total_raw_coverage": float(total_raw_hits / max(total_roi_pixels, 1)),
        "total_compat_coverage": float(total_compat_hits / max(total_roi_pixels, 1)),
        "compat_over_raw": float(total_compat_hits / max(total_raw_hits, 1)),
        "failed_gate_counts": gates_failed,
        "roi_summary": roi_rows,
    }


def classify(row: dict[str, Any]) -> str:
    raw = float(row.get("total_raw_coverage", 0.0) or 0.0)
    compat = float(row.get("total_compat_coverage", 0.0) or 0.0)
    compat_over_raw = float(row.get("compat_over_raw", 0.0) or 0.0)
    if raw < 0.02:
        return "too_sparse_before_depth_compatibility"
    if compat < 0.02 and compat_over_raw < 0.10:
        return "raw_hits_exist_but_depth_compatibility_collapses"
    if compat < 0.20:
        return "depth_compatible_coverage_too_low_or_fragmented"
    return "needs_visual_component_audit"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A5 Teacher-Gate Failure Summary",
        "",
        "This is a research-only summary. It does not run reconstruction, export a teacher/candidate, write predictions, write strict pass state, or call cloud.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_FACTS, indent=2),
        "```",
        "",
        "## Runs",
        "",
    ]
    for row in summary["runs"]:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- classification: `{row['classification']}`",
                f"- total raw coverage: `{row['total_raw_coverage']:.4f}`",
                f"- total depth-compatible coverage: `{row['total_compat_coverage']:.4f}`",
                f"- compatible/raw: `{row['compat_over_raw']:.4f}`",
                f"- failed gates: `{row['failed_gate_counts']}`",
                "",
            ]
        )
    lines.extend(["## Decision", "", summary["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.root.glob(args.pattern))
    runs = []
    for path in paths:
        row = summarize_one(path)
        row["classification"] = classify(row)
        runs.append(row)
    decision = (
        "Existing A5 COLMAP/CUDA teacher-gate outputs should remain frozen as teacher-negative. "
        "The dominant pattern is raw hits followed by depth-compatible collapse or severe sparsity, "
        "so the next non-redundant A-line action is a coordinate/depth-range bridge audit or a different dense backend class, "
        "not adjacent-view/threshold/fusion parameter loops."
    )
    summary = {
        "task": "summarize_a5_teacher_gate_failures",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "root": str(args.root.resolve()),
        "pattern": args.pattern,
        "run_count": int(len(runs)),
        "runs": runs,
        "decision": decision,
        "outputs": {
            "summary_json": str(output_dir / "a5_teacher_gate_failure_summary.json"),
            "report_md": str(output_dir / "a5_teacher_gate_failure_summary.md"),
        },
    }
    write_json(output_dir / "a5_teacher_gate_failure_summary.json", summary)
    write_markdown(output_dir / "a5_teacher_gate_failure_summary.md", summary)
    print(json.dumps({"summary": summary["outputs"]["summary_json"], "run_count": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
