from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("output/surface_research_preflight_local")
DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_JSON_REPORT = Path("reports/20260507_v7_region_referee_agent.json")
DEFAULT_MD_REPORT = Path("reports/20260507_v7_region_referee_agent.md")

STATUS = "research_only_v7_region_referee_strict_gate_red"
REQUIRED_CONTROLS = ("real", "shuffle", "zero", "mask_only")
REQUIRED_REGIONS = ("full", "head", "face", "hairline", "hands")
THREED_SUFFIXES = {".ply", ".obj", ".stl", ".glb", ".gltf", ".npz"}
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}

DLINE_NAME_PATTERNS = (
    "b_fus3d",
    "fus3d",
    "b_gs",
    "b_hair",
    "b_hand",
    "dline",
    "v6",
    "v7",
    "region",
)
REPORT_NAME_PATTERNS = (
    "20260507_b_fus3d",
    "20260507_b_gs",
    "20260507_b_hair",
    "20260507_b_hand",
    "20260507_dline",
    "20260507_v6",
    "20260507_v7",
)
CONTROL_PATTERNS = {
    "real": re.compile(r"(^|[^a-z0-9])real([^a-z0-9]|$)"),
    "shuffle": re.compile(r"(^|[^a-z0-9])shuffle([^a-z0-9]|$)"),
    "zero": re.compile(r"(^|[^a-z0-9])zero([^a-z0-9]|$)"),
    "mask_only": re.compile(r"mask[-_\s]?only"),
    "random": re.compile(r"(^|[^a-z0-9])random([^a-z0-9]|$)"),
}
REGION_PATTERNS = {
    "full": re.compile(r"(^|[^a-z0-9])full(body)?([^a-z0-9]|$)"),
    "head": re.compile(r"(^|[^a-z0-9])head([^a-z0-9]|$)|head_hair|headtop"),
    "face": re.compile(r"(^|[^a-z0-9])face([^a-z0-9]|$)|head_face"),
    "hairline": re.compile(r"hairline|headtop|head_hair|hair"),
    "hands": re.compile(r"(^|[^a-z0-9])hands?([^a-z0-9]|$)|left_hand|right_hand"),
}
SCAFFOLD_WORDS = (
    "template",
    "smplx",
    "scaffold",
    "connected_payload",
    "weak",
    "landmark",
    "prior",
)
GENEALOGY_WORDS = (
    "input",
    "upstream",
    "source",
    "template",
    "smplx",
    "scaffold",
    "payload",
    "latent",
    "evidence",
    "summary",
    "depth",
    "colmap",
    "kinect",
    "hair0",
    "hand0",
    "hand1",
    "hand2",
    "hand3",
    "hand4",
    "hand5",
    "hand6",
    "fus3d",
)
METRIC_WORDS = (
    "iou",
    "overfill",
    "recall",
    "rgb_residual",
    "normal_grad",
    "depth_valid",
    "vertices",
    "faces",
    "count",
    "extent",
    "support",
    "score",
    "selected",
    "component",
    "pass",
    "success",
    "gate_color",
    "real_minus",
    "better_than",
)
FORBIDDEN_WORDS = (
    "teacher",
    "candidate",
    "predictions",
    "prediction",
    "strict_pass",
    "pass",
    "export",
)
DISALLOWED_PATH_WORDS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions",
    "prediction_export",
)
PROTECTIVE_WORDS = (
    "research_only",
    "no_teacher_export",
    "no_candidate_export",
    "no_predictions_write",
    "no_prediction",
    "no_strict_pass_write",
    "not_teacher",
    "not_candidate",
    "blocked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only v7 D-line region referee for local research artifacts. "
            "It inventories v6/v7 outputs and always fails closed on strict gate state."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--include-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-artifacts", type=int, default=96)
    parser.add_argument("--max-text-chars", type=int, default=220_000)
    parser.add_argument("--max-json-bytes", type=int, default=5_000_000)
    parser.add_argument("--skip-guard", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def read_text_sample(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception as exc:  # pragma: no cover - defensive inventory path
        return f"<read_error:{type(exc).__name__}:{exc}>"


def load_json(path: Path, max_bytes: int) -> Any | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def should_scan_dir(path: Path) -> bool:
    name = path.name.lower()
    return any(pattern in name for pattern in DLINE_NAME_PATTERNS)


def should_scan_report(path: Path) -> bool:
    name = path.name.lower()
    return any(pattern in name for pattern in REPORT_NAME_PATTERNS)


def sorted_files(root: Path) -> list[Path]:
    try:
        return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda item: str(item).lower())
    except FileNotFoundError:
        return []


def token_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    lower = text.lower()
    return [name for name, pattern in patterns.items() if pattern.search(lower)]


def suffix_counts(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def sample_paths(files: list[Path], limit: int = 16) -> list[str]:
    return [rel(path) for path in files[:limit]]


def compact_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def walk_json(
    value: Any,
    prefix: str = "",
    *,
    metric_limit: int = 120,
    genealogy_limit: int = 80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    genealogy: list[dict[str, Any]] = []
    risk_fields: list[dict[str, Any]] = []

    def visit(node: Any, path: str) -> None:
        lower_path = path.lower()
        if len(metrics) >= metric_limit and len(genealogy) >= genealogy_limit and len(risk_fields) >= 120:
            return
        if isinstance(node, dict):
            for key, item in node.items():
                next_path = f"{path}.{key}" if path else str(key)
                key_lower = str(key).lower()
                if key_lower in {"views", "renders"} and isinstance(item, list):
                    continue
                visit(item, next_path)
            return
        if isinstance(node, list):
            if len(node) > 40 and all(isinstance(item, dict) for item in node[:10]):
                return
            for index, item in enumerate(node[:80]):
                visit(item, f"{path}[{index}]")
            return

        value_text = str(node)
        lower_value = value_text.lower()
        if (
            len(metrics) < metric_limit
            and any(word in lower_path for word in METRIC_WORDS)
            and isinstance(node, (int, float, bool, str))
        ):
            if isinstance(node, str) and len(node) > 180:
                metric_value: Any = node[:177] + "..."
            else:
                metric_value = compact_scalar(node)
            metrics.append({"path": path, "value": metric_value})
        if (
            len(genealogy) < genealogy_limit
            and isinstance(node, str)
            and (any(word in lower_path for word in GENEALOGY_WORDS) or any(word in lower_value for word in GENEALOGY_WORDS))
        ):
            genealogy.append({"path": path, "value": node[:240]})
        if len(risk_fields) < 120 and any(word in lower_path or word in lower_value for word in FORBIDDEN_WORDS):
            risk_fields.append({"path": path, "value": compact_scalar(node) if not isinstance(node, str) else node[:180]})

    visit(value, prefix)
    return metrics, genealogy, risk_fields


def extract_control_metrics(payloads: list[tuple[Path, Any]]) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    for path, payload in payloads:
        if not isinstance(payload, dict):
            continue
        control_block = payload.get("controls")
        if not isinstance(control_block, dict):
            continue
        for control_name, value in control_block.items():
            canonical = str(control_name).lower().replace("-", "_")
            if canonical not in CONTROL_PATTERNS or not isinstance(value, dict):
                continue
            metrics: dict[str, Any] = {}
            for key, item in value.items():
                key_lower = str(key).lower()
                if key_lower == "views":
                    continue
                if any(word in key_lower for word in METRIC_WORDS):
                    if isinstance(item, (dict, list)):
                        metrics[key] = summarize_nested_metric(item)
                    else:
                        metrics[key] = compact_scalar(item)
            controls.setdefault(canonical, {"source": rel(path), "metrics": {}})
            controls[canonical]["metrics"].update(metrics)
    return controls


def summarize_nested_metric(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, (int, float, bool, str)):
                out[str(key)] = compact_scalar(item)
        return out or "<nested>"
    if isinstance(value, list):
        return {"count": len(value)}
    return compact_scalar(value)


def scan_forbidden(paths_text: str, payloads: list[tuple[Path, Any]], text_sample: str) -> dict[str, Any]:
    lower_paths = paths_text.lower()
    lower_text = text_sample.lower()
    path_hits = sorted({word for word in DISALLOWED_PATH_WORDS if word in lower_paths})
    broad_path_hits = sorted({word for word in FORBIDDEN_WORDS if word in lower_paths})
    content_hits = sorted({word for word in FORBIDDEN_WORDS if word in lower_text})
    protective_hits = sorted({word for word in PROTECTIVE_WORDS if word in lower_text or word in lower_paths})

    disallowed_files: list[str] = []
    diagnostic_candidate_files: list[str] = []
    for raw in paths_text.splitlines():
        lower = raw.lower()
        if any(word in lower for word in DISALLOWED_PATH_WORDS):
            disallowed_files.append(raw)
        elif "candidate" in lower:
            if "raw_free_candidates" in lower:
                diagnostic_candidate_files.append(raw)
            else:
                disallowed_files.append(raw)

    suspicious_fields: list[dict[str, Any]] = []
    strict_count_fields: list[dict[str, Any]] = []
    for source, payload in payloads:
        if not isinstance(payload, (dict, list)):
            continue
        _, _, risk_fields = walk_json(payload, prefix=source.name)
        for item in risk_fields:
            path = str(item["path"]).lower()
            value = item.get("value")
            if path.endswith("strict_candidate_passes") or path.endswith("strict_teacher_passes"):
                strict_count_fields.append({"source": rel(source), **item})
                if isinstance(value, (int, float)) and value > 0:
                    suspicious_fields.append({"source": rel(source), "reason": "strict_pass_count_positive", **item})
            if isinstance(value, bool) and value is True:
                if any(word in path for word in ("writes_teacher", "writes_candidate", "writes_predictions", "writes_strict_registry", "cloud_allowed")):
                    suspicious_fields.append({"source": rel(source), "reason": "forbidden_boolean_true", **item})
            if isinstance(value, str):
                value_lower = value.lower()
                if ("teacher_export" in path or "candidate_export" in path or "formal_cloud" in path) and "blocked" not in value_lower:
                    suspicious_fields.append({"source": rel(source), "reason": "export_or_cloud_not_blocked", **item})
                if "strict_pass" in path and "no_strict_pass_write" not in path and "blocked" not in value_lower:
                    suspicious_fields.append({"source": rel(source), "reason": "strict_pass_field_not_protective", **item})

    return {
        "path_hits": path_hits,
        "broad_path_hits": broad_path_hits,
        "content_hits": content_hits,
        "protective_hits": protective_hits,
        "disallowed_files": disallowed_files[:40],
        "diagnostic_candidate_files": diagnostic_candidate_files[:20],
        "strict_count_fields": strict_count_fields[:20],
        "suspicious_fields": suspicious_fields[:40],
        "unprotected_risk": bool(path_hits or disallowed_files or suspicious_fields),
    }


def classify_artifact_dir(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    files = sorted_files(path)
    all_path_text = "\n".join(rel(item) for item in files)
    text_chunks: list[str] = []
    json_payloads: list[tuple[Path, Any]] = []
    for file_path in files:
        if file_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if sum(len(chunk) for chunk in text_chunks) < args.max_text_chars:
            text_chunks.append(read_text_sample(file_path, max(1_000, args.max_text_chars // 8)))
        if file_path.suffix.lower() == ".json":
            payload = load_json(file_path, args.max_json_bytes)
            if payload is not None:
                json_payloads.append((file_path, payload))
    text_sample = "\n".join(text_chunks)[: args.max_text_chars]
    search_text = f"{path.name}\n{all_path_text}\n{text_sample}".lower()

    three_d_files = [item for item in files if item.suffix.lower() in THREED_SUFFIXES]
    contact_files = [
        item
        for item in files
        if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".json"}
        and ("contact" in item.name.lower() or "contact_sheet" in str(item).lower())
    ]
    visual_review_files = [
        item
        for item in files
        if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and any(word in str(item).lower() for word in ("open3d_review", "review", "render", "camera_view"))
    ]
    controls_found = token_hits(search_text, CONTROL_PATTERNS)
    regions_found = token_hits(search_text, REGION_PATTERNS)

    metrics: list[dict[str, Any]] = []
    genealogy: list[dict[str, Any]] = []
    for source, payload in json_payloads:
        metric_rows, genealogy_rows, _ = walk_json(payload, prefix=source.name)
        for row in metric_rows:
            if len(metrics) >= 140:
                break
            metrics.append({"source": rel(source), **row})
        for row in genealogy_rows:
            if len(genealogy) >= 80:
                break
            genealogy.append({"source": rel(source), **row})

    control_metrics = extract_control_metrics(json_payloads)
    forbidden = scan_forbidden(all_path_text, json_payloads, text_sample)

    scaffold_hits = sorted({word for word in SCAFFOLD_WORDS if word in search_text})
    scaffold_dependency = bool(scaffold_hits)
    dominant_scaffold_risk = any(
        phrase in search_text
        for phrase in (
            "scaffold_only",
            "template_or_smplx_scaffold",
            "dominant support",
            "template scaffold",
            "weak_depth_or_landmark_evidence_is_not_success",
        )
    )

    missing_controls = [name for name in REQUIRED_CONTROLS if name not in controls_found]
    missing_regions = [name for name in REQUIRED_REGIONS if name not in regions_found]
    blocked_reasons = ["strict_gate_red_global"]
    if not three_d_files:
        blocked_reasons.append("missing_3d_output")
    if not contact_files:
        blocked_reasons.append("missing_contact_sheet")
    if missing_controls:
        blocked_reasons.append("missing_required_controls:" + ",".join(missing_controls))
    if missing_regions:
        blocked_reasons.append("missing_required_regions:" + ",".join(missing_regions))
    if dominant_scaffold_risk:
        blocked_reasons.append("scaffold_or_template_dependency_not_success")
    if forbidden["unprotected_risk"]:
        blocked_reasons.append("forbidden_token_or_field_risk_present")

    return {
        "path": rel(path),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files if item.exists()),
        "suffix_counts": suffix_counts(files),
        "has_3d_output": bool(three_d_files),
        "three_d_outputs": sample_paths(three_d_files, limit=18),
        "has_contact_sheet": bool(contact_files),
        "contact_sheets": sample_paths(contact_files, limit=18),
        "has_visual_review_pngs": bool(visual_review_files),
        "visual_review_png_count": len(visual_review_files),
        "visual_review_samples": sample_paths(visual_review_files, limit=12),
        "controls_found": controls_found,
        "missing_required_controls": missing_controls,
        "control_metrics": control_metrics,
        "regions_found": regions_found,
        "missing_required_regions": missing_regions,
        "scaffold_dependency": scaffold_dependency,
        "scaffold_hits": scaffold_hits,
        "dominant_scaffold_risk": dominant_scaffold_risk,
        "genealogy_refs": genealogy[:80],
        "metric_refs": metrics[:140],
        "forbidden_risk_scan": forbidden,
        "can_enter_strict_gate": False,
        "blocked_reasons": blocked_reasons,
    }


def scan_report_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    text = read_text_sample(path, args.max_text_chars)
    lower = text.lower()
    controls_found = token_hits(text, CONTROL_PATTERNS)
    regions_found = token_hits(text, REGION_PATTERNS)
    forbidden = scan_forbidden(rel(path), [], text)
    return {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "controls_found": controls_found,
        "regions_found": regions_found,
        "mentions_3d_output": any(word in lower for word in (".ply", ".obj", ".npz", "pointcloud", "mesh", "surface")),
        "mentions_contact_sheet": "contact_sheet" in lower or "contact sheet" in lower,
        "mentions_scaffold_or_genealogy": any(word in lower for word in SCAFFOLD_WORDS + GENEALOGY_WORDS),
        "forbidden_risk_scan": forbidden,
    }


def run_guard(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True)
    try:
        payload: dict[str, Any] = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["exit_code"] = proc.returncode
    payload["command"] = " ".join(command)
    return payload


def aggregate(artifacts: list[dict[str, Any]], reports: list[dict[str, Any]], guards: dict[str, Any]) -> dict[str, Any]:
    all_controls = sorted({item for artifact in artifacts for item in artifact["controls_found"]})
    all_regions = sorted({item for artifact in artifacts for item in artifact["regions_found"]})
    dirs_with_3d = sum(1 for artifact in artifacts if artifact["has_3d_output"])
    dirs_with_contact = sum(1 for artifact in artifacts if artifact["has_contact_sheet"])
    dirs_with_scaffold = sum(1 for artifact in artifacts if artifact["scaffold_dependency"])
    dirs_with_dominant_scaffold = sum(1 for artifact in artifacts if artifact["dominant_scaffold_risk"])
    dirs_with_unprotected_risk = sum(
        1 for artifact in artifacts if artifact["forbidden_risk_scan"]["unprotected_risk"]
    )
    candidate_guard = guards.get("candidate", {})
    teacher_guard = guards.get("teacher_supervised", {})
    strict_gate_red = (
        not bool(candidate_guard.get("cloud_allowed", False))
        and not bool(teacher_guard.get("cloud_allowed", False))
        and int(candidate_guard.get("strict_candidate_passes", 0) or 0) == 0
        and int(candidate_guard.get("strict_teacher_passes", 0) or 0) == 0
    )
    return {
        "artifact_count": len(artifacts),
        "report_count": len(reports),
        "dirs_with_3d_output": dirs_with_3d,
        "dirs_with_contact_sheet": dirs_with_contact,
        "dirs_with_scaffold_dependency": dirs_with_scaffold,
        "dirs_with_dominant_scaffold_risk": dirs_with_dominant_scaffold,
        "dirs_with_unprotected_forbidden_risk": dirs_with_unprotected_risk,
        "controls_found_anywhere": all_controls,
        "missing_controls_anywhere": [name for name in REQUIRED_CONTROLS if name not in all_controls],
        "regions_found_anywhere": all_regions,
        "missing_regions_anywhere": [name for name in REQUIRED_REGIONS if name not in all_regions],
        "strict_gate_red": strict_gate_red,
        "strict_candidate_passes": int(candidate_guard.get("strict_candidate_passes", 0) or 0),
        "strict_teacher_passes": int(candidate_guard.get("strict_teacher_passes", 0) or 0),
        "formal_cloud_train_infer_export": "blocked",
        "teacher_export": "blocked",
        "candidate_export": "blocked",
        "predictions_export": "blocked",
        "strict_pass_write": "blocked",
        "can_enter_strict_gate": False,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    guard_candidate = summary["guards"].get("candidate", {})
    guard_teacher = summary["guards"].get("teacher_supervised", {})
    artifact_rows = []
    for artifact in summary["artifacts"]:
        artifact_rows.append(
            "| {path} | {files} | {three_d} | {contact} | {controls} | {regions} | {scaffold} | {risk} |".format(
                path=artifact["path"],
                files=artifact["file_count"],
                three_d="yes" if artifact["has_3d_output"] else "no",
                contact="yes" if artifact["has_contact_sheet"] else "no",
                controls=", ".join(artifact["controls_found"]) or "-",
                regions=", ".join(artifact["regions_found"]) or "-",
                scaffold="yes" if artifact["scaffold_dependency"] else "no",
                risk="yes" if artifact["forbidden_risk_scan"]["unprotected_risk"] else "no",
            )
        )

    lines = [
        "# v7 D-Line Region Referee",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Scope: read-only local referee over v6/v7 D-line research artifacts. This run did not train,",
        "launch cloud, export teacher/candidate/prediction packages, mutate the strict registry, or write pass state.",
        "",
        "## Strict Gate",
        "",
        "```text",
        f"candidate command = {guard_candidate.get('command', 'skipped')}",
        f"candidate exit_code = {guard_candidate.get('exit_code', 'skipped')}",
        f"candidate cloud_allowed = {guard_candidate.get('cloud_allowed', False)}",
        f"strict_candidate_passes = {guard_candidate.get('strict_candidate_passes', 0)}",
        f"strict_teacher_passes = {guard_candidate.get('strict_teacher_passes', 0)}",
        f"teacher-supervised command = {guard_teacher.get('command', 'skipped')}",
        f"teacher-supervised exit_code = {guard_teacher.get('exit_code', 'skipped')}",
        f"teacher-supervised cloud_allowed = {guard_teacher.get('cloud_allowed', False)}",
        "formal_cloud_train_infer_export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "predictions_export = blocked",
        "strict_pass_write = blocked",
        "```",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(agg, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Artifact Matrix",
        "",
        "| path | files | 3D | contact sheet | controls | regions | scaffold refs | unprotected risk |",
        "|---|---:|---|---|---|---|---|---|",
        *artifact_rows,
        "",
        "## Commands",
        "",
        "```text",
        *summary["commands_run"],
        "```",
        "",
        "## Outputs",
        "",
        "```text",
        summary["json_report"],
        summary["md_report"],
        "```",
        "",
        "Full per-artifact metrics, genealogy references, control evidence, and forbidden-token",
        "details are in the JSON report.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cwd = Path.cwd().resolve()
    root = args.root
    reports_root = args.reports_root

    artifact_dirs = []
    if root.exists():
        artifact_dirs = [path for path in root.iterdir() if path.is_dir() and should_scan_dir(path)]
    artifact_dirs = sorted(artifact_dirs, key=lambda item: item.name.lower())[: args.max_artifacts]
    artifacts = [classify_artifact_dir(path, args) for path in artifact_dirs]

    report_files: list[Path] = []
    if args.include_reports and reports_root.exists():
        report_files = [path for path in reports_root.iterdir() if path.is_file() and should_scan_report(path)]
    report_files = sorted(report_files, key=lambda item: item.name.lower())[: args.max_artifacts]
    report_rows = [scan_report_file(path, args) for path in report_files]

    commands_run = [
        "python tools\\research_artifact_referee_v7_region.py",
    ]
    if args.skip_guard:
        guards = {
            "candidate": {"command": "skipped", "exit_code": None, "cloud_allowed": False, "strict_candidate_passes": 0, "strict_teacher_passes": 0},
            "teacher_supervised": {"command": "skipped", "exit_code": None, "cloud_allowed": False, "strict_candidate_passes": 0, "strict_teacher_passes": 0},
        }
    else:
        candidate_command = [sys.executable, "tools/check_cloud_gate_status.py", "--json"]
        teacher_command = [sys.executable, "tools/check_cloud_gate_status.py", "--teacher-supervised", "--json"]
        guards = {
            "candidate": run_guard(candidate_command),
            "teacher_supervised": run_guard(teacher_command),
        }
        commands_run.extend(
            [
                "python tools\\check_cloud_gate_status.py --json",
                "python tools\\check_cloud_gate_status.py --teacher-supervised --json",
            ]
        )

    summary: dict[str, Any] = {
        "status": STATUS,
        "agent": "Agent A / v7 D-line region referee",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(cwd),
        "scope": (
            "Read-only scan for 3D outputs, contact sheets, real/shuffle/zero/mask-only controls, "
            "full/head/face/hairline/hands regions, scaffold genealogy, metrics, and forbidden "
            "teacher/candidate/predictions/pass/export risks."
        ),
        "roots": {
            "artifact_root": rel(root),
            "reports_root": rel(reports_root),
        },
        "commands_run": commands_run,
        "guards": guards,
        "artifacts": artifacts,
        "reports_scanned": report_rows,
        "strict_facts": {
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud_train_infer_export": "blocked",
            "teacher_export": "blocked",
            "candidate_export": "blocked",
            "predictions_export": "blocked",
            "strict_pass_write": "blocked",
        },
        "forbidden_actions_taken": {
            "cloud": False,
            "train": False,
            "infer": False,
            "teacher_export": False,
            "candidate_export": False,
            "predictions_export": False,
            "strict_pass_write": False,
            "strict_registry_write": False,
        },
        "json_report": rel(args.json_report),
        "md_report": rel(args.md_report),
    }
    summary["aggregate"] = aggregate(artifacts, report_rows, guards)
    summary["verdict"] = (
        "STRICT_GATE_RED: v6/v7 D-line artifacts remain research-only. "
        "No artifact is promoted to strict pass, teacher, candidate, prediction, cloud, or export state."
    )

    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.md_report, json_ready(summary))

    print(
        json.dumps(
            {
                "status": STATUS,
                "artifact_count": len(artifacts),
                "report_count": len(report_rows),
                "strict_gate_red": summary["aggregate"]["strict_gate_red"],
                "strict_candidate_passes": summary["aggregate"]["strict_candidate_passes"],
                "strict_teacher_passes": summary["aggregate"]["strict_teacher_passes"],
                "json_report": rel(args.json_report),
                "md_report": rel(args.md_report),
                "verdict": summary["verdict"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
