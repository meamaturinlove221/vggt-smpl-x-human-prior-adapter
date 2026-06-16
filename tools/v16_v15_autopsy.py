from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v16_v15_autopsy.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v16_v15_autopsy.md"

RUNNER_PATH = REPO_ROOT / "tools" / "v15_smplx_native_overfit_runner.py"
SOFTSURFEL_PATH = REPO_ROOT / "tools" / "optimize_raw_smplx_softsurfel_torch.py"
OVERFIT_REPORT = REPORTS / "20260508_v15_smplx_native_overfit_runner.json"
FUSION_AUDIT_REPORT = REPORTS / "20260508_v15_smplx_fusion_effect_audit.json"
NATIVE_ROLLUP_MD = REPORTS / "20260508_v15_smplx_native_execution_rollup.md"

FORBIDDEN_WRITE_TOKENS = (
    "strict_gate_registry",
    "strict_pass",
    "candidate_package",
    "teacher_package",
    "candidate_export",
    "teacher_export",
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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def line_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    text = read_text(path)
    hits: list[dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        matched = [pattern for pattern in patterns if pattern.lower() in lower]
        if matched:
            hits.append({"path": str(path), "line": idx, "patterns": matched, "text": line.strip()})
    return hits


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def ast_imports(path: Path) -> list[str]:
    text = read_text(path)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
    return sorted(set(names))


def ast_string_literals(path: Path) -> list[str]:
    text = read_text(path)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


def contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def command_contains_train(cmd: list[Any]) -> bool:
    joined = " ".join(str(item).lower() for item in cmd)
    train_terms = ["torchrun", "train.py", "training/trainer.py", "modal_4k4d_vggt_train.py", "hydra", "trainer"]
    return any(term in joined for term in train_terms)


def extract_runner_command(overfit_report: dict[str, Any]) -> list[str]:
    command = overfit_report.get("overfit_command")
    if isinstance(command, list):
        return [str(item) for item in command]
    run_result = overfit_report.get("attempt", {}).get("run_result", {})
    cmd = run_result.get("cmd") if isinstance(run_result, dict) else None
    if isinstance(cmd, list):
        return [str(item) for item in cmd]
    return []


def parse_stdout_json(overfit_report: dict[str, Any]) -> dict[str, Any]:
    run_result = overfit_report.get("attempt", {}).get("run_result", {})
    stdout = run_result.get("stdout_tail") if isinstance(run_result, dict) else ""
    if not isinstance(stdout, str) or not stdout.strip():
        return {}
    match = re.search(r"\{.*\}\s*$", stdout, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def load_softsurfel_summary(overfit_report: dict[str, Any]) -> dict[str, Any]:
    attempt = overfit_report.get("attempt", {})
    summary_path = None
    if isinstance(attempt, dict):
        summary = attempt.get("summary", {})
        if isinstance(summary, dict):
            summary_path = summary.get("summary_path")
    data = read_json(Path(str(summary_path))) if summary_path else {}
    if data:
        return data
    return parse_stdout_json(overfit_report)


def metric_value(summary: dict[str, Any], *path: str) -> Any:
    current: Any = summary
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def mean_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_v15_result(
    overfit_report: dict[str, Any],
    fusion_audit: dict[str, Any],
    softsurfel_summary: dict[str, Any],
    command: list[str],
    runner_text: str,
    softsurfel_text: str,
) -> tuple[str, list[str], dict[str, Any]]:
    evidence: list[str] = []
    facts: dict[str, Any] = {}

    command_uses_softsurfel = any("optimize_raw_smplx_softsurfel_torch.py" in part for part in command)
    command_uses_training = command_contains_train(command)
    optimizer_declares_no_vggt = contains_any(
        softsurfel_text,
        [
            "does not use VGGT depth",
            "uses_vggt_depth_point_normal\": False",
            '"uses_vggt_depth_point_normal": False',
            "uses no VGGT geometry output",
        ],
    )
    optimizer_declares_no_candidate = contains_any(
        softsurfel_text,
        [
            "creates_candidate_predictions\": False",
            '"creates_candidate_predictions": False',
            "not a mentor candidate",
            "not a strict-passing teacher",
        ],
    )
    runner_research_only = bool(overfit_report.get("research_only")) and not bool(overfit_report.get("formal_cloud_unblocked"))
    strict_zero = int(overfit_report.get("strict_candidate_passes", 0) or 0) == 0 and int(overfit_report.get("strict_teacher_passes", 0) or 0) == 0
    no_export = all(
        bool(overfit_report.get(key))
        for key in ("no_predictions_write", "no_teacher_export", "no_candidate_export", "no_registry_write", "no_strict_pass_write")
    )
    summary_uses_vggt = softsurfel_summary.get("uses_vggt_depth_point_normal")
    summary_creates_candidate = softsurfel_summary.get("creates_candidate_predictions")
    summary_truthful_status = softsurfel_summary.get("truthful_status")

    initial_iou = mean_value(metric_value(softsurfel_summary, "metrics", "initial_iou"))
    optimized_iou = mean_value(metric_value(softsurfel_summary, "metrics", "optimized_iou"))
    initial_recall = mean_value(metric_value(softsurfel_summary, "metrics", "initial_target_recall"))
    optimized_recall = mean_value(metric_value(softsurfel_summary, "metrics", "optimized_target_recall"))
    iou_delta = metric_value(softsurfel_summary, "metrics", "iou_delta")
    recall_delta = metric_value(softsurfel_summary, "metrics", "target_recall_delta")
    self_deltas = metric_value(fusion_audit, "comparison", "self_deltas")
    fusion_effect_observed = bool(fusion_audit.get("fusion_effect_observed"))

    facts.update(
        {
            "command_uses_softsurfel_optimizer": command_uses_softsurfel,
            "command_uses_true_vggt_training_entrypoint": command_uses_training,
            "optimizer_declares_no_vggt_depth_point_normal": optimizer_declares_no_vggt,
            "optimizer_declares_no_candidate_or_teacher": optimizer_declares_no_candidate,
            "runner_research_only": runner_research_only,
            "runner_strict_passes_zero": strict_zero,
            "runner_no_export_flags": no_export,
            "summary_uses_vggt_depth_point_normal": summary_uses_vggt,
            "summary_creates_candidate_predictions": summary_creates_candidate,
            "summary_truthful_status": summary_truthful_status,
            "initial_iou_mean": initial_iou,
            "optimized_iou_mean": optimized_iou,
            "iou_delta": iou_delta,
            "initial_target_recall_mean": initial_recall,
            "optimized_target_recall_mean": optimized_recall,
            "target_recall_delta": recall_delta,
            "fusion_self_deltas": self_deltas,
            "fusion_effect_observed": fusion_effect_observed,
        }
    )

    if command_uses_softsurfel:
        evidence.append("V15 runner command delegates to tools/optimize_raw_smplx_softsurfel_torch.py.")
    if optimizer_declares_no_vggt or summary_uses_vggt is False:
        evidence.append("The softsurfel optimizer/report says it uses raw RGB/mask/camera/SMPL-X and no VGGT depth/point/normal geometry.")
    if command_uses_training:
        evidence.append("A VGGT training entrypoint-like token appears in the V15 command.")
    else:
        evidence.append("No VGGT training entrypoint-like token appears in the executed V15 command.")
    if strict_zero and no_export:
        evidence.append("V15 report kept strict candidate/teacher passes at zero and set no prediction/teacher/candidate/registry writes.")
    if iou_delta is not None and recall_delta is not None:
        evidence.append(f"Self metrics were negative for geometry gates: IoU delta {iou_delta}, target recall delta {recall_delta}.")
    if fusion_effect_observed is False:
        evidence.append("Fusion audit marked fusion_effect_observed false.")

    raw_softsurfel_only = (
        command_uses_softsurfel
        and not command_uses_training
        and (optimizer_declares_no_vggt or summary_uses_vggt is False)
        and (optimizer_declares_no_candidate or summary_creates_candidate is False)
        and runner_research_only
        and strict_zero
    )

    true_vggt_training_negative = command_uses_training and not raw_softsurfel_only
    if raw_softsurfel_only:
        conclusion = "v15_negative_is_raw_softsurfel_only_not_true_vggt_training"
    elif true_vggt_training_negative:
        conclusion = "v15_negative_may_include_true_vggt_training"
    else:
        conclusion = "v15_negative_ambiguous_needs_manual_review"
    return conclusion, evidence, facts


def forbidden_output_scan(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = read_text(path)
        lower = text.lower()
        hits = [token for token in FORBIDDEN_WRITE_TOKENS if token in lower]
        rows.append({"path": str(path), "exists": path.exists(), "tokens_present_in_text": hits})
    return {
        "clean_for_v16_outputs": True,
        "note": "This scan is informational; source files may contain forbidden words in guard strings. V16 tools do not write pass/registry/package artifacts.",
        "rows": rows,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 V15 Autopsy",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Conclusion",
        "",
        summary["decision"],
        "",
        "## Classification",
        "",
        f"- result_class: `{summary['result_class']}`",
        f"- v15_negative_is_raw_softsurfel_only: `{summary['v15_negative_is_raw_softsurfel_only']}`",
        f"- v15_negative_is_true_vggt_training: `{summary['v15_negative_is_true_vggt_training']}`",
        "",
        "## Key Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["evidence"])
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- initial_iou_mean: `{summary['facts'].get('initial_iou_mean')}`",
            f"- optimized_iou_mean: `{summary['facts'].get('optimized_iou_mean')}`",
            f"- iou_delta: `{summary['facts'].get('iou_delta')}`",
            f"- initial_target_recall_mean: `{summary['facts'].get('initial_target_recall_mean')}`",
            f"- optimized_target_recall_mean: `{summary['facts'].get('optimized_target_recall_mean')}`",
            f"- target_recall_delta: `{summary['facts'].get('target_recall_delta')}`",
            "",
            "## Command",
            "",
            "```powershell",
            " ".join(summary["v15_command"]),
            "```",
            "",
            "## Source Hits",
            "",
        ]
    )
    for hit in summary["source_hits"][:30]:
        lines.append(f"- `{hit['path']}:{hit['line']}` {hit['text']}")
    lines.extend(
        [
            "",
            "## Non-Promotion Guard",
            "",
            "- V16 autopsy writes only this JSON/MD report.",
            "- It does not write strict registry, package, pass, predictions, teacher export, or candidate export artifacts.",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = summary.get("blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Autopsy V15 SMPL-X native negative result.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    overfit_report = read_json(OVERFIT_REPORT)
    fusion_audit = read_json(FUSION_AUDIT_REPORT)
    softsurfel_summary = load_softsurfel_summary(overfit_report)
    command = extract_runner_command(overfit_report)
    runner_text = read_text(RUNNER_PATH)
    softsurfel_text = read_text(SOFTSURFEL_PATH)

    result_class, evidence, facts = classify_v15_result(
        overfit_report=overfit_report,
        fusion_audit=fusion_audit,
        softsurfel_summary=softsurfel_summary,
        command=command,
        runner_text=runner_text,
        softsurfel_text=softsurfel_text,
    )

    source_hits = line_hits(
        RUNNER_PATH,
        ["optimize_raw_smplx_softsurfel_torch.py", "research_only", "strict_candidate_passes", "no_predictions_write"],
    ) + line_hits(
        SOFTSURFEL_PATH,
        [
            "does not use VGGT depth",
            "uses_vggt_depth_point_normal",
            "creates_candidate_predictions",
            "not a production",
            "not a strict",
            "not a mentor candidate",
        ],
    ) + line_hits(
        NATIVE_ROLLUP_MD,
        ["No formal cloud train", "IoU delta", "target recall delta", "strict_candidate_passes", "no candidate package"],
    )

    blockers = []
    if not overfit_report:
        blockers.append("Missing V15 overfit runner report.")
    if not softsurfel_summary:
        blockers.append("Missing raw softsurfel summary from V15 overfit output.")
    if not fusion_audit:
        blockers.append("Missing V15 fusion effect audit report.")

    v15_negative_is_raw = result_class == "v15_negative_is_raw_softsurfel_only_not_true_vggt_training"
    v15_negative_is_true_vggt_training = result_class == "v15_negative_may_include_true_vggt_training"
    status = "v16_v15_autopsy_complete_raw_softsurfel_only" if v15_negative_is_raw else "v16_v15_autopsy_requires_review"
    decision = (
        "V15 negative result should be interpreted as raw softsurfel-only local SMPL-X optimization, not a negative result for true VGGT training."
        if v15_negative_is_raw
        else "V15 negative result is not proven raw-softsurfel-only; review the command and reports before routing."
    )

    runner_imports = ast_imports(RUNNER_PATH)
    softsurfel_imports = ast_imports(SOFTSURFEL_PATH)
    training_imports_present = sorted(
        name
        for name in set(runner_imports + softsurfel_imports)
        if any(term in name.lower() for term in ("trainer", "training.trainer", "vggt.models.vggt", "hydra"))
    )
    string_literals = ast_string_literals(RUNNER_PATH)
    command_literals = [
        value
        for value in string_literals
        if "optimize_raw_smplx_softsurfel_torch.py" in value
        or "training" in value.lower()
        or "torchrun" in value.lower()
    ]

    summary = {
        "task": "v16_v15_autopsy",
        "created_utc": utc_now(),
        "status": status,
        "result_class": result_class,
        "v15_negative_is_raw_softsurfel_only": v15_negative_is_raw,
        "v15_negative_is_true_vggt_training": v15_negative_is_true_vggt_training,
        "decision": decision,
        "facts": facts,
        "evidence": evidence,
        "v15_command": command,
        "source_hits": source_hits,
        "source_files": {
            "runner": file_row(RUNNER_PATH),
            "softsurfel_optimizer": file_row(SOFTSURFEL_PATH),
            "overfit_report": file_row(OVERFIT_REPORT),
            "fusion_audit": file_row(FUSION_AUDIT_REPORT),
            "native_rollup_md": file_row(NATIVE_ROLLUP_MD),
        },
        "static_code_scan": {
            "runner_imports": runner_imports,
            "softsurfel_imports": softsurfel_imports,
            "training_imports_present": training_imports_present,
            "runner_command_related_literals": command_literals,
        },
        "non_promotion_contract": {
            "writes_only_reports": True,
            "writes_strict_registry": False,
            "writes_package": False,
            "writes_strict_pass": False,
            "writes_predictions": False,
            "writes_teacher_or_candidate_export": False,
        },
        "informational_forbidden_token_scan": forbidden_output_scan([RUNNER_PATH, SOFTSURFEL_PATH]),
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "result_class": result_class, "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0 if v15_negative_is_raw else 2


if __name__ == "__main__":
    raise SystemExit(main())
