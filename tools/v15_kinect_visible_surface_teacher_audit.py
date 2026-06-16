from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
OUTPUT = REPO_ROOT / "output"

DEFAULT_V14_ROLLUP = REPORTS / "20260508_v14_execution_rollup.json"
DEFAULT_K14_REPORT = REPORTS / "V14_K14" / "20260508_k14_kinect_teacher_gate_autopsy.json"
DEFAULT_REGISTRY = REPORTS / "20260508_v14_strict_gate_registry_refresh.json"
DEFAULT_TEACHER_GATE = (
    OUTPUT
    / "normal_line_multiview_20260504"
    / "teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews"
    / "teacher_gate_summary.json"
)
DEFAULT_RAW_ASSET = (
    OUTPUT
    / "local_teacher_probes"
    / "0012_11_kinect_tsdf_asset_frame0_v01"
    / "kinect_smc_tsdf_summary.json"
)
DEFAULT_REPORT_JSON = REPORTS / "20260508_v15_kline_kinect_visible_surface_teacher_audit.json"
DEFAULT_REPORT_MD = REPORTS / "20260508_v15_kline_kinect_visible_surface_teacher_audit.md"

READ_ONLY_CONTRACT = {
    "research_only": True,
    "no_predictions_write": True,
    "no_teacher_targets_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_package_write": True,
    "no_strict_pass_write": True,
    "formal_cloud_unblocked": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_json_lenient(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return {"_parse_error": True, "_path": str(path)}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in {float("inf"), float("-inf")} else None


def strict_counts(registry: dict[str, Any], rollup: dict[str, Any]) -> dict[str, int]:
    counts = registry.get("counts") if isinstance(registry.get("counts"), dict) else {}
    return {
        "strict_candidate_passes": int(
            counts.get("strict_candidate_passes", rollup.get("strict_candidate_passes", 0)) or 0
        ),
        "strict_teacher_passes": int(
            counts.get("strict_teacher_passes", rollup.get("strict_teacher_passes", 0)) or 0
        ),
        "visible_surface_teacher_audits": int(counts.get("visible_surface_teacher_audits", 0) or 0),
        "visible_surface_teacher_passes": int(counts.get("visible_surface_teacher_passes", 0) or 0),
        "kinect_coord_audits": int(counts.get("kinect_coord_audits", 0) or 0),
        "kinect_coord_passes": int(counts.get("kinect_coord_passes", 0) or 0),
    }


def summarize_entries(entries: list[Any]) -> dict[str, Any]:
    by_roi: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        roi = str(item.get("roi_kind") or "unknown")
        row = by_roi.setdefault(
            roi,
            {
                "views": 0,
                "entry_passes": 0,
                "max_raw_coverage": None,
                "max_depth_compatible_coverage": None,
                "max_depth_compatible_hit_pixels": 0,
                "min_hole_ratio": None,
                "max_largest_component_ratio": None,
                "median_depth_residual_values": [],
                "failed_checks": {},
                "gate_thresholds": item.get("gate_thresholds", {}),
            },
        )
        row["views"] += 1
        gate = item.get("gate") if isinstance(item.get("gate"), dict) else {}
        if bool(gate.get("pass")):
            row["entry_passes"] += 1
        for key, ok in gate.items():
            if key != "pass" and not bool(ok):
                row["failed_checks"][key] = int(row["failed_checks"].get(key, 0)) + 1

        raw = item.get("raw_visible") if isinstance(item.get("raw_visible"), dict) else {}
        depth = item.get("depth_compatible") if isinstance(item.get("depth_compatible"), dict) else {}
        raw_cov = as_float(raw.get("coverage"))
        depth_cov = as_float(depth.get("coverage"))
        hole = as_float(depth.get("hole_ratio"))
        hit_pixels = int(depth.get("hit_pixels", 0) or 0)
        components = depth.get("components") if isinstance(depth.get("components"), dict) else {}
        lcc = as_float(components.get("largest_component_ratio"))
        residual = depth.get("depth_residual") if isinstance(depth.get("depth_residual"), dict) else {}
        p50 = as_float(residual.get("p50"))

        row["max_raw_coverage"] = max_value(row["max_raw_coverage"], raw_cov)
        row["max_depth_compatible_coverage"] = max_value(row["max_depth_compatible_coverage"], depth_cov)
        row["min_hole_ratio"] = min_value(row["min_hole_ratio"], hole)
        row["max_largest_component_ratio"] = max_value(row["max_largest_component_ratio"], lcc)
        row["max_depth_compatible_hit_pixels"] = max(int(row["max_depth_compatible_hit_pixels"]), hit_pixels)
        if p50 is not None:
            row["median_depth_residual_values"].append(p50)

    for row in by_roi.values():
        vals = row.pop("median_depth_residual_values")
        row["best_median_depth_residual"] = min(vals) if vals else None
        row["failed_checks"] = dict(sorted(row["failed_checks"].items()))
    return dict(sorted(by_roi.items()))


def max_value(left: float | None, right: float | None) -> float | None:
    if right is None:
        return left
    if left is None:
        return right
    return max(left, right)


def min_value(left: float | None, right: float | None) -> float | None:
    if right is None:
        return left
    if left is None:
        return right
    return min(left, right)


def existing_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path.resolve()) if path.exists() else str(path),
            "exists": bool(path.exists()),
        }
        for name, path in paths.items()
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["strict_pass_counts"]
    gate = summary["existing_strict_teacher_gate"]
    lines = [
        "# V15 K-line Kinect Visible Surface Teacher Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only. This report reads existing Kinect/V14 artifacts and writes no teacher targets, candidate package, registry entry, predictions, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Strict Counts",
        "",
        f"- Strict candidate passes: `{counts['strict_candidate_passes']}`",
        f"- Strict teacher passes: `{counts['strict_teacher_passes']}`",
        f"- Orphan visible-surface teacher audits: `{counts['visible_surface_teacher_audits']}`",
        f"- Orphan visible-surface teacher passes: `{counts['visible_surface_teacher_passes']}`",
        f"- Kinect coordinate audit passes: `{counts['kinect_coord_passes']}`",
        "",
        "## Existing Teacher Gate",
        "",
        f"- Numeric pass: `{gate.get('numeric_pass')}`",
        f"- Explicit visual pass: `{gate.get('visual_pass')}`",
        f"- Overall pass: `{gate.get('pass')}`",
        f"- Source: `{gate.get('source_path')}`",
        "",
        "| ROI | Views | Entry passes | Max depth-compatible coverage | Max hit pixels | Best median residual | Failed checks |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for roi, row in summary.get("roi_visible_surface_metrics", {}).items():
        failed = ", ".join(f"{key}:{value}" for key, value in row.get("failed_checks", {}).items())
        lines.append(
            f"| {roi} | {row.get('views')} | {row.get('entry_passes')} | "
            f"{fmt(row.get('max_depth_compatible_coverage'))} | "
            f"{row.get('max_depth_compatible_hit_pixels')} | "
            f"{fmt(row.get('best_median_depth_residual'))} | {failed or 'none'} |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    lines.extend(["", "## No-write Policy", ""])
    lines.append(f"- Teacher package written: `{summary['writes']['teacher_package_written']}`")
    lines.append(f"- Candidate package written: `{summary['writes']['candidate_package_written']}`")
    lines.append(f"- Registry written: `{summary['writes']['registry_written']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    number = as_float(value)
    return "NA" if number is None else f"{number:.6f}"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rollup = load_json_lenient(args.v14_rollup)
    k14 = load_json_lenient(args.k14_report)
    registry = load_json_lenient(args.registry)
    teacher_gate = load_json_lenient(args.teacher_gate_summary)
    raw_asset = load_json_lenient(args.raw_asset_summary)
    counts = strict_counts(registry, rollup)

    gate = teacher_gate.get("gate") if isinstance(teacher_gate.get("gate"), dict) else {}
    entries = teacher_gate.get("entries") if isinstance(teacher_gate.get("entries"), list) else []
    roi_metrics = summarize_entries(entries)
    visual_review = teacher_gate.get("visual_review") if isinstance(teacher_gate.get("visual_review"), dict) else {}
    official_k14 = k14.get("official_teacher_gate") if isinstance(k14.get("official_teacher_gate"), dict) else {}
    k14_failed = list((official_k14.get("failed_checks") or {}).keys())

    blockers: list[str] = []
    if not teacher_gate:
        blockers.append("Existing Kinect TSDF strict teacher-gate summary was not found or could not be parsed.")
    if not bool(gate.get("numeric_pass")):
        blockers.append("Existing Kinect visible-surface teacher numeric gate is false.")
    if not bool(gate.get("visual_pass")):
        blockers.append("Explicit Open3D visual teacher review is missing or failed.")
    if not bool(gate.get("pass")):
        blockers.append("Existing same-protocol Kinect teacher gate overall pass is false.")
    if counts["strict_teacher_passes"] <= 0:
        blockers.append("V14 strict registry reports zero strict teacher passes.")
    if counts["strict_candidate_passes"] <= 0:
        blockers.append("V14 strict registry reports zero strict candidate passes.")
    if k14_failed:
        blockers.append(f"K14 official Kinect camera/depth gate failed checks: {', '.join(k14_failed)}.")
    face = roi_metrics.get("face_core", {})
    hair = roi_metrics.get("hairline", {})
    head_face = roi_metrics.get("head_face", {})
    if as_float(face.get("max_depth_compatible_coverage")) in (None, 0.0) or (
        as_float(face.get("max_depth_compatible_coverage")) or 0.0
    ) < 0.58:
        blockers.append(
            f"Face-core depth-compatible coverage remains below threshold: max={fmt(face.get('max_depth_compatible_coverage'))}."
        )
    if (as_float(hair.get("max_depth_compatible_coverage")) or 0.0) <= 0.0:
        blockers.append("Hairline depth-compatible coverage is zero in the existing strict teacher gate.")
    if (as_float(head_face.get("max_depth_compatible_coverage")) or 0.0) < 0.58:
        blockers.append(
            f"Head/face depth-compatible coverage remains below threshold: max={fmt(head_face.get('max_depth_compatible_coverage'))}."
        )

    status = "v15_kline_kinect_visible_surface_teacher_blocked"
    if bool(gate.get("pass")) and counts["strict_teacher_passes"] > 0:
        status = "v15_kline_kinect_visible_surface_teacher_existing_strict_pass_detected_no_package_written"

    return {
        "task": "v15_kline_kinect_visible_surface_teacher_audit",
        "created_utc": utc_now(),
        "status": status,
        **READ_ONLY_CONTRACT,
        "inputs": {
            "v14_rollup": str(args.v14_rollup.resolve()) if args.v14_rollup.exists() else str(args.v14_rollup),
            "k14_report": str(args.k14_report.resolve()) if args.k14_report.exists() else str(args.k14_report),
            "registry": str(args.registry.resolve()) if args.registry.exists() else str(args.registry),
            "teacher_gate_summary": str(args.teacher_gate_summary.resolve())
            if args.teacher_gate_summary.exists()
            else str(args.teacher_gate_summary),
            "raw_asset_summary": str(args.raw_asset_summary.resolve())
            if args.raw_asset_summary.exists()
            else str(args.raw_asset_summary),
        },
        "input_artifacts": existing_artifacts(
            {
                "v14_rollup": args.v14_rollup,
                "k14_report": args.k14_report,
                "registry": args.registry,
                "teacher_gate_summary": args.teacher_gate_summary,
                "raw_asset_summary": args.raw_asset_summary,
            }
        ),
        "strict_pass_counts": counts,
        "strict_candidate_passes": counts["strict_candidate_passes"],
        "strict_teacher_passes": counts["strict_teacher_passes"],
        "existing_strict_teacher_gate": {
            "source_kind": teacher_gate.get("source_kind"),
            "source_path": teacher_gate.get("source_path"),
            "predictions_npz": teacher_gate.get("predictions_npz"),
            "scene_dir": teacher_gate.get("scene_dir"),
            "numeric_pass": bool(gate.get("numeric_pass")),
            "visual_pass": bool(gate.get("visual_pass")),
            "pass": bool(gate.get("pass")),
            "visual_review_path": visual_review.get("path"),
            "visual_review_pass": bool(visual_review.get("pass")),
            "visual_review_reason": visual_review.get("reason"),
        },
        "roi_visible_surface_metrics": roi_metrics,
        "k14_official_teacher_gate": {
            "pass": bool(official_k14.get("pass")),
            "teacher_targets_written": bool(official_k14.get("teacher_targets_written")),
            "failed_checks": k14_failed,
            "alignment_residual_p50": official_k14.get("alignment_residual_p50"),
            "alignment_residual_p50_max": official_k14.get("alignment_residual_p50_max"),
            "distance_to_base_p50": official_k14.get("distance_to_base_p50"),
            "distance_to_base_p50_max": official_k14.get("distance_to_base_p50_max"),
        },
        "raw_kinect_asset_context": {
            "truthful_status": raw_asset.get("truthful_status"),
            "mesh": raw_asset.get("mesh"),
            "selected_kinect_cameras": raw_asset.get("selected_kinect_cameras"),
            "notes": raw_asset.get("notes"),
        },
        "writes": {
            "teacher_package_written": False,
            "candidate_package_written": False,
            "teacher_targets_written": False,
            "predictions_written": False,
            "registry_written": False,
        },
        "decision": (
            "Existing Kinect TSDF and K14 artifacts are real bounded evidence, but they do not form a strict "
            "visible-surface teacher. V15 K-line remains blocked and no teacher/candidate package is written."
        ),
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V15 K-line read-only Kinect visible-surface teacher audit.")
    parser.add_argument("--v14-rollup", type=Path, default=DEFAULT_V14_ROLLUP)
    parser.add_argument("--k14-report", type=Path, default=DEFAULT_K14_REPORT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--teacher-gate-summary", type=Path, default=DEFAULT_TEACHER_GATE)
    parser.add_argument("--raw-asset-summary", type=Path, default=DEFAULT_RAW_ASSET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "strict_candidate_passes": summary["strict_candidate_passes"],
                "strict_teacher_passes": summary["strict_teacher_passes"],
                "output_json": str(args.output_json.resolve()),
                "output_md": str(args.output_md.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
