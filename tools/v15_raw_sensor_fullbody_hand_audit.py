from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
OUTPUT = REPO_ROOT / "output"

DEFAULT_V14_ROLLUP = REPORTS / "20260508_v14_execution_rollup.json"
DEFAULT_K14_REPORT = REPORTS / "V14_K14" / "20260508_k14_kinect_teacher_gate_autopsy.json"
DEFAULT_REGISTRY = REPORTS / "20260508_v14_strict_gate_registry_refresh.json"
DEFAULT_RAW_ASSET = (
    OUTPUT
    / "local_teacher_probes"
    / "0012_11_kinect_tsdf_asset_frame0_v01"
    / "kinect_smc_tsdf_summary.json"
)
DEFAULT_TEACHER_MESH = (
    OUTPUT
    / "local_teacher_probes"
    / "kinect_tsdf_v21_original6v_camaxes"
    / "kinect_tsdf_teacher_mesh_summary.json"
)
DEFAULT_REPORT_JSON = REPORTS / "20260508_v15_kline_raw_sensor_fullbody_hand_audit.json"
DEFAULT_REPORT_MD = REPORTS / "20260508_v15_kline_raw_sensor_fullbody_hand_audit.md"

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
        "raw_sensor_fullbody_hand_audits": int(counts.get("raw_sensor_fullbody_hand_audits", 0) or 0),
        "raw_sensor_fullbody_hand_passes": int(counts.get("raw_sensor_fullbody_hand_passes", 0) or 0),
        "smplx_weak_anchor_audits": int(counts.get("smplx_weak_anchor_audits", 0) or 0),
        "smplx_weak_anchor_passes": int(counts.get("smplx_weak_anchor_passes", 0) or 0),
        "kinect_coord_audits": int(counts.get("kinect_coord_audits", 0) or 0),
        "kinect_coord_passes": int(counts.get("kinect_coord_passes", 0) or 0),
    }


def numeric_stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v == v]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": min(clean),
        "median": median(clean),
        "mean": mean(clean),
        "max": max(clean),
    }


def raw_asset_metrics(raw_asset: dict[str, Any]) -> dict[str, Any]:
    per_camera = raw_asset.get("per_camera") if isinstance(raw_asset.get("per_camera"), list) else []
    valid_pixels: list[float] = []
    valid_ratios: list[float] = []
    med_depths: list[float] = []
    for row in per_camera:
        if not isinstance(row, dict):
            continue
        valid = as_float(row.get("valid_pixels"))
        ratio = as_float(row.get("valid_ratio"))
        depths = row.get("depth_percentiles_m") if isinstance(row.get("depth_percentiles_m"), list) else []
        if valid is not None:
            valid_pixels.append(valid)
        if ratio is not None:
            valid_ratios.append(ratio)
        if len(depths) >= 2 and as_float(depths[1]) is not None:
            med_depths.append(float(depths[1]))
    mesh = raw_asset.get("mesh") if isinstance(raw_asset.get("mesh"), dict) else {}
    screenshots = raw_asset.get("screenshots") if isinstance(raw_asset.get("screenshots"), list) else []
    return {
        "truthful_status": raw_asset.get("truthful_status"),
        "selected_kinect_cameras": raw_asset.get("selected_kinect_cameras"),
        "camera_count": len(per_camera),
        "total_valid_depth_pixels": int(sum(valid_pixels)),
        "valid_pixels_stats": numeric_stats(valid_pixels),
        "valid_ratio_stats": numeric_stats(valid_ratios),
        "median_depth_m_stats": numeric_stats(med_depths),
        "mesh": mesh,
        "screenshot_count": len(screenshots),
        "screenshots_exist": [
            {"path": str(path), "exists": Path(str(path)).is_file()} for path in screenshots
        ],
        "notes": raw_asset.get("notes"),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["strict_pass_counts"]
    metrics = summary["raw_kinect_asset_metrics"]
    gates = summary["gates"]
    lines = [
        "# V15 K-line Raw Sensor Full-body/Hand Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only. This report reads existing raw Kinect sensor artifacts and V14 registry state. It writes no teacher targets, candidate package, registry entry, predictions, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Strict Counts",
        "",
        f"- Strict candidate passes: `{counts['strict_candidate_passes']}`",
        f"- Strict teacher passes: `{counts['strict_teacher_passes']}`",
        f"- Raw sensor full-body/hand audits: `{counts['raw_sensor_fullbody_hand_audits']}`",
        f"- Raw sensor full-body/hand passes: `{counts['raw_sensor_fullbody_hand_passes']}`",
        f"- SMPL-X weak-anchor passes: `{counts['smplx_weak_anchor_passes']}`",
        "",
        "## Raw Kinect Asset",
        "",
        f"- Truthful status: `{metrics.get('truthful_status')}`",
        f"- Kinect cameras: `{metrics.get('camera_count')}`",
        f"- Total valid depth pixels: `{metrics.get('total_valid_depth_pixels')}`",
        f"- Mesh vertices: `{(metrics.get('mesh') or {}).get('vertices')}`",
        f"- Mesh triangles: `{(metrics.get('mesh') or {}).get('triangles')}`",
        f"- Screenshot count: `{metrics.get('screenshot_count')}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in gates.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    lines.extend(["", "## No-write Policy", ""])
    lines.append(f"- Teacher package written: `{summary['writes']['teacher_package_written']}`")
    lines.append(f"- Candidate package written: `{summary['writes']['candidate_package_written']}`")
    lines.append(f"- Registry written: `{summary['writes']['registry_written']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rollup = load_json_lenient(args.v14_rollup)
    k14 = load_json_lenient(args.k14_report)
    registry = load_json_lenient(args.registry)
    raw_asset = load_json_lenient(args.raw_asset_summary)
    teacher_mesh = load_json_lenient(args.teacher_mesh_summary)
    counts = strict_counts(registry, rollup)
    key_outcomes = rollup.get("key_outcomes") if isinstance(rollup.get("key_outcomes"), dict) else {}
    official = k14.get("official_teacher_gate") if isinstance(k14.get("official_teacher_gate"), dict) else {}
    failed_checks = list((official.get("failed_checks") or {}).keys())

    metrics = raw_asset_metrics(raw_asset)
    mesh = metrics.get("mesh") if isinstance(metrics.get("mesh"), dict) else {}
    asset_available = bool(raw_asset and int(mesh.get("vertices", 0) or 0) > 0 and int(mesh.get("triangles", 0) or 0) > 0)
    multiview_depth_available = bool(metrics.get("camera_count", 0) >= 6 and metrics.get("total_valid_depth_pixels", 0) > 0)
    raw_audit_present = counts["raw_sensor_fullbody_hand_audits"] > 0
    raw_audit_pass = counts["raw_sensor_fullbody_hand_passes"] > 0
    strict_teacher_available = counts["strict_teacher_passes"] > 0
    strict_candidate_available = counts["strict_candidate_passes"] > 0
    hand_ownership_ready = bool(key_outcomes.get("hand_ownership_ready"))
    hair_ownership_ready = bool(key_outcomes.get("hair_ownership_ready"))

    gates = {
        "raw_kinect_asset_available": asset_available,
        "raw_kinect_multiview_depth_available": multiview_depth_available,
        "existing_raw_sensor_fullbody_hand_audit_present": raw_audit_present,
        "existing_raw_sensor_fullbody_hand_pass": raw_audit_pass,
        "existing_strict_teacher_pass_available": strict_teacher_available,
        "existing_strict_candidate_pass_available": strict_candidate_available,
        "hand_ownership_ready": hand_ownership_ready,
        "hair_ownership_ready": hair_ownership_ready,
        "overall_package_eligible": False,
    }

    blockers: list[str] = []
    if not asset_available:
        blockers.append("Raw Kinect TSDF asset summary or mesh evidence is missing.")
    if not raw_audit_present:
        blockers.append("V14 strict registry reports zero raw Kinect sensor full-body/hand audits.")
    if not raw_audit_pass:
        blockers.append("V14 strict registry reports zero raw Kinect sensor full-body/hand passes.")
    if not strict_teacher_available:
        blockers.append("No existing strict teacher pass is available for raw Kinect training.")
    if not strict_candidate_available:
        blockers.append("No existing strict candidate pass is available for package/cloud promotion.")
    if not hand_ownership_ready:
        blockers.append("V14 rollup reports no strict hand ownership asset.")
    if not hair_ownership_ready:
        blockers.append("V14 rollup reports no strict hair topology ownership asset.")
    if failed_checks:
        blockers.append(f"K14 official Kinect teacher gate failed checks: {', '.join(failed_checks)}.")
    if counts["smplx_weak_anchor_passes"] > 0:
        blockers.append("SMPL-X weak-anchor pass exists only as a weak topology diagnostic, not as raw Kinect sensor or strict teacher pass.")

    return {
        "task": "v15_kline_raw_sensor_fullbody_hand_audit",
        "created_utc": utc_now(),
        "status": "v15_kline_raw_sensor_fullbody_hand_blocked_no_strict_audit",
        **READ_ONLY_CONTRACT,
        "inputs": {
            "v14_rollup": str(args.v14_rollup.resolve()) if args.v14_rollup.exists() else str(args.v14_rollup),
            "k14_report": str(args.k14_report.resolve()) if args.k14_report.exists() else str(args.k14_report),
            "registry": str(args.registry.resolve()) if args.registry.exists() else str(args.registry),
            "raw_asset_summary": str(args.raw_asset_summary.resolve())
            if args.raw_asset_summary.exists()
            else str(args.raw_asset_summary),
            "teacher_mesh_summary": str(args.teacher_mesh_summary.resolve())
            if args.teacher_mesh_summary.exists()
            else str(args.teacher_mesh_summary),
        },
        "strict_pass_counts": counts,
        "strict_candidate_passes": counts["strict_candidate_passes"],
        "strict_teacher_passes": counts["strict_teacher_passes"],
        "raw_kinect_asset_metrics": metrics,
        "teacher_mesh_context": {
            "truthful_status": teacher_mesh.get("truthful_status"),
            "real_mesh": teacher_mesh.get("real_mesh"),
            "vggt_mesh": teacher_mesh.get("vggt_mesh"),
            "alignment": teacher_mesh.get("alignment"),
            "target_alignment": teacher_mesh.get("target_alignment"),
            "notes": teacher_mesh.get("notes"),
        },
        "k14_official_teacher_gate": {
            "pass": bool(official.get("pass")),
            "teacher_targets_written": bool(official.get("teacher_targets_written")),
            "failed_checks": failed_checks,
            "alignment_residual_p50": official.get("alignment_residual_p50"),
            "distance_to_base_p50": official.get("distance_to_base_p50"),
        },
        "gates": gates,
        "writes": {
            "teacher_package_written": False,
            "candidate_package_written": False,
            "teacher_targets_written": False,
            "predictions_written": False,
            "registry_written": False,
        },
        "decision": (
            "The raw Kinect asset exists as bounded sensor evidence, but V14 has no raw Kinect full-body/hand "
            "audit pass and no strict teacher or candidate pass. V15 K-line remains blocked and writes no package."
        ),
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V15 K-line read-only raw Kinect full-body/hand audit.")
    parser.add_argument("--v14-rollup", type=Path, default=DEFAULT_V14_ROLLUP)
    parser.add_argument("--k14-report", type=Path, default=DEFAULT_K14_REPORT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--raw-asset-summary", type=Path, default=DEFAULT_RAW_ASSET)
    parser.add_argument("--teacher-mesh-summary", type=Path, default=DEFAULT_TEACHER_MESH)
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
