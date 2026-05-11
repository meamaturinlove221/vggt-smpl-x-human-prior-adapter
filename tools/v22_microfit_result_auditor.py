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
DEFAULT_REPORT_JSON = REPORTS / "20260508_v22_true_vggt_smplx_microfit.json"
DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v22_true_vggt_smplx_microfit.audit.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v22_true_vggt_smplx_microfit.audit.md"

REQUIRED_METHODS = ("M2", "M3")
REQUIRED_CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")
FORBIDDEN_OUTPUT_TOKENS = (
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
    "strict_pass",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": repr(exc)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def safe_report_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent != REPORTS.resolve():
        raise ValueError(f"Refusing V22 audit report outside reports/: {resolved}")
    if not resolved.name.startswith("20260508_v22_true_vggt_smplx_microfit."):
        raise ValueError(f"Unexpected V22 audit report name: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def metric_file_ok(path: Path, expected: dict[str, str]) -> dict[str, Any]:
    data = read_json(path)
    problems: list[str] = []
    if not path.is_file():
        problems.append("metric file missing")
    for key, value in expected.items():
        if data.get(key) != value:
            problems.append(f"{key} mismatch: expected {value!r}, got {data.get(key)!r}")
    if data.get("task") != "v22_true_vggt_smplx_microfit_metric":
        problems.append("metric task marker missing")
    if not data.get("research_only"):
        problems.append("metric does not declare research_only")
    if int(data.get("executed_steps", 0) or 0) <= 0:
        problems.append("metric executed_steps <= 0")
    for field in ("initial", "final"):
        value = (data.get(field) or {}).get("loss")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            problems.append(f"{field}.loss is not finite")
    trainable = data.get("trainable") or {}
    if int(trainable.get("trainable_params", 0) or 0) <= 0:
        problems.append("trainable_params <= 0")
    batch = data.get("batch") or {}
    if int(batch.get("valid_pixels", 0) or 0) <= 0:
        problems.append("valid_pixels <= 0")
    output_meta = data.get("output_meta") or {}
    if int(output_meta.get("aggregated_token_layers", 0) or 0) <= 0:
        problems.append("VGGT Aggregator token layers missing")
    control = data.get("control")
    control_meta = data.get("control_meta") or {}
    if control == "random-region" and not control_meta.get("region_boxes"):
        problems.append("random-region control missing region_boxes")
    if control == "prior-dropout" and "map_keep_fraction" not in control_meta:
        problems.append("prior-dropout control missing map_keep_fraction")
    return {
        "metric_path": str(path),
        "exists": path.is_file(),
        "ok": not problems,
        "problems": problems,
        "final_loss": (data.get("final") or {}).get("loss"),
        "initial_loss": (data.get("initial") or {}).get("loss"),
        "executed_steps": data.get("executed_steps"),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# V22 Microfit Result Audit",
        "",
        f"Status: `{audit['final_status']}`",
        "",
        audit.get("decision", ""),
        "",
        "## Coverage",
        "",
        f"- report_exists: `{audit['report'].get('exists')}`",
        f"- metric_file_count: `{audit['metrics'].get('metric_file_count')}`",
        f"- required_metric_count: `{audit['metrics'].get('required_metric_count')}`",
        f"- required_methods: `{audit['required_methods']}`",
        f"- required_controls: `{audit['required_controls']}`",
        f"- executed_viewsets: `{audit['metrics'].get('executed_viewsets')}`",
        "",
        "## Metric Problems",
        "",
    ]
    failed = [row for row in audit.get("metric_checks", []) if not row.get("ok")]
    if failed:
        for row in failed:
            lines.append(f"- `{row.get('metric_path')}`: {row.get('problems')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    blockers = audit.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    lines.extend(["", "## Outputs", ""])
    for key, value in audit.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def audit_report(report_json: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    report_json = report_json.expanduser().resolve()
    output_json = safe_report_path(output_json)
    output_md = safe_report_path(output_md)
    summary = read_json(report_json)
    blockers: list[str] = []
    report_row = file_row(report_json)
    if not report_row["is_file"]:
        blockers.append(f"V22 report JSON missing: {report_json}")
    if summary.get("_read_error"):
        blockers.append(f"Could not parse V22 report JSON: {summary['_read_error']}")

    local_dir = Path((summary.get("outputs") or {}).get("local_output_dir", REPO_ROOT / "missing")).expanduser()
    cloud_dir = Path((summary.get("outputs") or {}).get("cloud_output_dir", REPO_ROOT / "missing")).expanduser()
    path_rows = {
        "local_output_dir": file_row(local_dir),
        "cloud_output_dir": file_row(cloud_dir),
        "cloud_guard": file_row(cloud_dir / "cloud_guard.json"),
        "summary_json": file_row(local_dir / "summary.json"),
        "summary_md": file_row(local_dir / "summary.md"),
    }

    for label, row in path_rows.items():
        lower = str(row["path"]).replace("\\", "/").lower()
        if any(token in lower for token in FORBIDDEN_OUTPUT_TOKENS):
            blockers.append(f"Forbidden formal-output token found in {label}: {row['path']}")
    if "surface_research_preflight_local" not in str(path_rows["local_output_dir"]["path"]).replace("\\", "/").lower():
        blockers.append("Local output is not under surface_research_preflight_local.")
    if "v22_true_vggt_smplx_microfit" not in str(path_rows["local_output_dir"]["path"]).replace("\\", "/").lower():
        blockers.append("Local output is not under the V22-owned directory.")

    if not summary.get("research_only"):
        blockers.append("Top-level summary does not declare research_only.")
    for flag in (
        "no_teacher_export",
        "no_candidate_export",
        "no_registry_write",
        "no_package_write",
        "no_strict_pass_write",
    ):
        if not summary.get(flag):
            blockers.append(f"Research guard flag missing/false: {flag}")
    if not summary.get("execution", {}).get("executed"):
        blockers.append("Execution marker is false; this is only a route file.")

    viewset_runs = summary.get("viewset_runs") or []
    executed_viewsets = [row.get("viewset_id") for row in viewset_runs if row.get("executed")]
    missing_required_viewsets = [
        row.get("viewset_id")
        for row in viewset_runs
        if row.get("viewset_id") in {"existing6", "hand_head6"} and not row.get("executed")
    ]
    if missing_required_viewsets:
        blockers.append(f"Required view sets were not executed: {missing_required_viewsets}")
    if not executed_viewsets:
        blockers.append("No view set was executed.")

    metric_checks: list[dict[str, Any]] = []
    expected_cells: list[tuple[str, str, str]] = []
    for viewset_id in executed_viewsets:
        for method in REQUIRED_METHODS:
            for control in REQUIRED_CONTROLS:
                expected_cells.append((method, viewset_id, control))

    for method, viewset_id, control in expected_cells:
        path = local_dir / f"{method}_{viewset_id}_{control}.metrics.json"
        metric_checks.append(metric_file_ok(path, {"method_id": method, "viewset_id": viewset_id, "control": control}))

    failed_metrics = [row for row in metric_checks if not row.get("ok")]
    if failed_metrics:
        blockers.append(f"{len(failed_metrics)} expected metric cells are missing or invalid.")

    required_metric_count = len(expected_cells)
    metric_file_count = len([row for row in metric_checks if row.get("exists")])
    if required_metric_count == 0 or metric_file_count < required_metric_count:
        blockers.append(f"Metric coverage incomplete: {metric_file_count}/{required_metric_count}.")

    if summary.get("final_status") == "DONE_PASS" and blockers:
        final_status = "DONE_FAIL_ROUTED"
        decision = "Auditor downgraded V22 because the bounded run evidence is incomplete or violates the research guard."
    elif blockers and metric_file_count == 0:
        final_status = "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
        decision = "Auditor found no valid bounded metric evidence."
    elif blockers:
        final_status = "DONE_FAIL_ROUTED"
        decision = "Auditor found partial bounded evidence, but not enough for a V22 research pass."
    else:
        final_status = "DONE_PASS"
        decision = "Auditor verified concrete bounded M2/M3 metric files, all required controls, executed view sets, and research-only guards."

    audit = {
        "task": "v22_microfit_result_auditor",
        "created_utc": utc_now(),
        "final_status": final_status,
        "decision": decision,
        "report": report_row,
        "required_methods": list(REQUIRED_METHODS),
        "required_controls": list(REQUIRED_CONTROLS),
        "metrics": {
            "executed_viewsets": executed_viewsets,
            "missing_required_viewsets": missing_required_viewsets,
            "required_metric_count": required_metric_count,
            "metric_file_count": metric_file_count,
            "valid_metric_count": len([row for row in metric_checks if row.get("ok")]),
        },
        "metric_checks": metric_checks,
        "path_checks": path_rows,
        "source_summary_status": summary.get("final_status"),
        "source_decision": summary.get("decision"),
        "blockers": blockers,
        "outputs": {
            "audit_json": str(output_json),
            "audit_md": str(output_md),
            "source_report_json": str(report_json),
        },
    }
    write_json(output_json, audit)
    write_markdown(output_md, audit)
    print(json.dumps({"final_status": final_status, "valid_metric_count": audit["metrics"]["valid_metric_count"], "audit_json": str(output_json)}, ensure_ascii=False))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V22 true VGGT SMPL-X microfit research outputs.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_report(args.report_json, args.output_json, args.output_md)
    return 0 if audit.get("final_status") == "DONE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
