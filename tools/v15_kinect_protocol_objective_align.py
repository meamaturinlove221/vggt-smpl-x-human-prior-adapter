from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

DEFAULT_V14_ROLLUP = REPORTS / "20260508_v14_execution_rollup.json"
DEFAULT_K14_REPORT = REPORTS / "V14_K14" / "20260508_k14_kinect_teacher_gate_autopsy.json"
DEFAULT_REGISTRY = REPORTS / "20260508_v14_strict_gate_registry_refresh.json"
DEFAULT_REPORT_JSON = REPORTS / "20260508_v15_kline_kinect_protocol_objective_align.json"
DEFAULT_REPORT_MD = REPORTS / "20260508_v15_kline_kinect_protocol_objective_align.md"

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
        "kinect_coord_audits": int(counts.get("kinect_coord_audits", 0) or 0),
        "kinect_coord_passes": int(counts.get("kinect_coord_passes", 0) or 0),
        "raw_sensor_fullbody_hand_audits": int(counts.get("raw_sensor_fullbody_hand_audits", 0) or 0),
        "raw_sensor_fullbody_hand_passes": int(counts.get("raw_sensor_fullbody_hand_passes", 0) or 0),
    }


def objective_row(name: str, value: Any, required: Any, passed: bool, authority: str, note: str) -> dict[str, Any]:
    return {
        "objective": name,
        "value": value,
        "required": required,
        "pass": bool(passed),
        "authority": authority,
        "note": note,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["strict_pass_counts"]
    official = summary["official_kinect_teacher_objective"]
    lines = [
        "# V15 K-line Kinect Protocol Objective Alignment",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only. This report aligns the K-line decision to the existing strict Kinect teacher objective and writes no package, registry, teacher targets, or predictions.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Strict Counts",
        "",
        f"- Strict candidate passes: `{counts['strict_candidate_passes']}`",
        f"- Strict teacher passes: `{counts['strict_teacher_passes']}`",
        f"- Kinect coordinate audit passes: `{counts['kinect_coord_passes']}`",
        f"- Raw sensor full-body/hand passes: `{counts['raw_sensor_fullbody_hand_passes']}`",
        "",
        "## Objective Table",
        "",
        "| Objective | Value | Required | Pass | Authority |",
        "|---|---:|---:|---|---|",
    ]
    for row in summary.get("objective_rows", []):
        lines.append(
            f"| {row['objective']} | {fmt(row.get('value'))} | {fmt(row.get('required'))} | "
            f"`{row.get('pass')}` | {row.get('authority')} |"
        )
    lines.extend(
        [
            "",
            "## Official Gate",
            "",
            f"- Official teacher pass: `{official.get('pass')}`",
            f"- Teacher targets written: `{official.get('teacher_targets_written')}`",
            f"- Failed checks: `{', '.join(official.get('failed_checks') or [])}`",
            f"- Visibility pass: `{summary.get('visibility_context', {}).get('per_view_visibility_pass')}`",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    lines.extend(["", "## No-write Policy", ""])
    lines.append(f"- Teacher package written: `{summary['writes']['teacher_package_written']}`")
    lines.append(f"- Candidate package written: `{summary['writes']['candidate_package_written']}`")
    lines.append(f"- Registry written: `{summary['writes']['registry_written']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "NA" if value is None else str(value)
    return f"{number:.6f}"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rollup = load_json_lenient(args.v14_rollup)
    k14 = load_json_lenient(args.k14_report)
    registry = load_json_lenient(args.registry)
    counts = strict_counts(registry, rollup)

    official = k14.get("official_teacher_gate") if isinstance(k14.get("official_teacher_gate"), dict) else {}
    protocol = k14.get("k2_protocol_mismatch") if isinstance(k14.get("k2_protocol_mismatch"), dict) else {}
    crop = k14.get("crop_roi_autopsy") if isinstance(k14.get("crop_roi_autopsy"), dict) else {}
    camera = k14.get("camera_autopsy") if isinstance(k14.get("camera_autopsy"), dict) else {}
    depth = k14.get("depth_autopsy") if isinstance(k14.get("depth_autopsy"), dict) else {}
    registry_context = k14.get("registry_context") if isinstance(k14.get("registry_context"), dict) else {}

    failed_checks = list((official.get("failed_checks") or {}).keys())
    objective_rows = [
        objective_row(
            "K2 one-way NN median to G3",
            protocol.get("k2_best_score_to_g3_median"),
            "diagnostic only",
            False,
            "not_authoritative",
            "Nearest-neighbor proximity does not test calibrated camera/depth teacher eligibility.",
        ),
        objective_row(
            "K2 bbox volume ratio vs G3",
            protocol.get("k2_volume_ratio_vs_g3"),
            "not tiny",
            False,
            "diagnostic_only",
            "Small cloud inside a large anchor can look close one-way.",
        ),
        objective_row(
            "official alignment residual p50",
            official.get("alignment_residual_p50"),
            official.get("alignment_residual_p50_max"),
            bool(((official.get("failed_checks") or {}).get("alignment_residual_p50") or {}).get("ok")),
            "strict_teacher_objective",
            "Camera-axis alignment must land in the VGGT/base prediction world.",
        ),
        objective_row(
            "official alignment residual p95",
            official.get("alignment_residual_p95"),
            official.get("alignment_residual_p95_max"),
            bool(((official.get("failed_checks") or {}).get("alignment_residual_p95") or {}).get("ok")),
            "strict_teacher_objective",
            "Tail residual must also stay bounded.",
        ),
        objective_row(
            "official distance-to-base p50",
            official.get("distance_to_base_p50"),
            official.get("distance_to_base_p50_max"),
            bool(((official.get("failed_checks") or {}).get("distance_to_base_p50") or {}).get("ok")),
            "strict_teacher_objective",
            "Projected depth must be compatible with the reference predictions.",
        ),
        objective_row(
            "official distance-to-base p95",
            official.get("distance_to_base_p95"),
            official.get("distance_to_base_p95_max"),
            bool(((official.get("failed_checks") or {}).get("distance_to_base_p95") or {}).get("ok")),
            "strict_teacher_objective",
            "Tail depth mismatch must stay bounded.",
        ),
    ]

    visibility_pass = bool((crop.get("view_pass_ratio") or 0.0) >= 0.8 and (crop.get("min_hit_ratio") or 0.0) >= 0.7)
    official_pass = bool(official.get("pass"))
    blockers: list[str] = []
    if not official_pass:
        blockers.append("The official Kinect teacher objective did not pass.")
    if failed_checks:
        blockers.append(f"Official Kinect checks failed: {', '.join(failed_checks)}.")
    if counts["strict_teacher_passes"] <= 0:
        blockers.append("V14 strict registry has zero strict teacher passes.")
    if counts["strict_candidate_passes"] <= 0:
        blockers.append("V14 strict registry has zero strict candidate passes.")
    if protocol.get("k2_best_score_to_g3_median") is not None:
        blockers.append("K2 low one-way G3 proximity is diagnostic and cannot override the official camera/depth gate.")
    if visibility_pass:
        blockers.append("K3 visibility/crop coverage is adequate, so the remaining blocker is protocol alignment/depth, not ROI hits.")
    if counts["kinect_coord_passes"] > 0:
        blockers.append("Kinect coordinate audit passes are ROI diagnostics and are not strict teacher passes.")

    return {
        "task": "v15_kline_kinect_protocol_objective_align",
        "created_utc": utc_now(),
        "status": "v15_kline_protocol_objective_alignment_blocked",
        **READ_ONLY_CONTRACT,
        "inputs": {
            "v14_rollup": str(args.v14_rollup.resolve()) if args.v14_rollup.exists() else str(args.v14_rollup),
            "k14_report": str(args.k14_report.resolve()) if args.k14_report.exists() else str(args.k14_report),
            "registry": str(args.registry.resolve()) if args.registry.exists() else str(args.registry),
        },
        "strict_pass_counts": counts,
        "strict_candidate_passes": counts["strict_candidate_passes"],
        "strict_teacher_passes": counts["strict_teacher_passes"],
        "selected_v15_objective": "existing official Kinect camera/depth strict teacher objective",
        "diagnostic_objectives_not_authoritative": [
            "K2 one-way nearest-neighbor score to G3",
            "bbox-normalized region point counts",
            "Kinect coordinate ROI audit passes",
            "visibility-only hit coverage",
        ],
        "objective_rows": objective_rows,
        "official_kinect_teacher_objective": {
            "pass": official_pass,
            "teacher_targets_written": bool(official.get("teacher_targets_written")),
            "failed_checks": failed_checks,
            "threshold_ratios": official.get("threshold_ratios"),
            "alignment_source": official.get("alignment_source"),
            "transform_mode": official.get("transform_mode"),
        },
        "k2_diagnostic_context": {
            "best_candidate": protocol.get("k2_best_candidate"),
            "best_score_to_g3_median": protocol.get("k2_best_score_to_g3_median"),
            "metric_is_one_way_nn_to_g3": protocol.get("k2_metric_is_one_way_nn_to_g3"),
            "k2_volume_ratio_vs_g3": protocol.get("k2_volume_ratio_vs_g3"),
            "k2_extent_ratio_vs_g3": protocol.get("k2_extent_ratio_vs_g3"),
            "interpretation": protocol.get("interpretation"),
        },
        "visibility_context": {
            "per_view_visibility_pass": visibility_pass,
            "view_pass_ratio": crop.get("view_pass_ratio"),
            "view_passes": crop.get("view_passes"),
            "view_total": crop.get("view_total"),
            "min_hit_ratio": crop.get("min_hit_ratio"),
            "mean_hit_ratio": crop.get("mean_hit_ratio"),
            "interpretation": crop.get("interpretation"),
        },
        "camera_context": {
            "alignment_source": camera.get("alignment_source"),
            "similarity_scale": camera.get("similarity_scale"),
            "camera_correspondences": camera.get("camera_correspondences"),
            "interpretation": camera.get("interpretation"),
        },
        "depth_context": {
            "temporal_sweep_best_frame": depth.get("temporal_sweep_best_frame"),
            "temporal_sweep_best_metrics": depth.get("temporal_sweep_best_metrics"),
            "interpretation": depth.get("interpretation"),
        },
        "registry_context": registry_context,
        "writes": {
            "teacher_package_written": False,
            "candidate_package_written": False,
            "teacher_targets_written": False,
            "predictions_written": False,
            "registry_written": False,
        },
        "decision": (
            "V15 K-line uses the official Kinect camera/depth teacher objective, not the K2 one-way G3 proximity "
            "diagnostic. Under that objective the route remains blocked; no strict pass or package is written."
        ),
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V15 K-line read-only Kinect protocol objective audit.")
    parser.add_argument("--v14-rollup", type=Path, default=DEFAULT_V14_ROLLUP)
    parser.add_argument("--k14-report", type=Path, default=DEFAULT_K14_REPORT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
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
