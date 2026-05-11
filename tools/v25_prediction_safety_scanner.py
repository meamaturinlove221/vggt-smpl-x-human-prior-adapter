from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OWNED_FILES = (
    REPO_ROOT / "modal_v25_research_vggt_predictions_3frames.py",
    REPO_ROOT / "tools/v25_research_prediction_intake.py",
    REPO_ROOT / "tools/v25_prediction_safety_scanner.py",
    REPO_ROOT / "reports/20260508_v25_research_predictions_3frames.json",
    REPO_ROOT / "reports/20260508_v25_research_predictions_3frames.md",
)
OUTPUT_ROOTS = (
    REPO_ROOT / "output/surface_research_cloud_preflight/V25_research_vggt_predictions",
    REPO_ROOT / "output/surface_research_preflight_local/V25_research_vggt_predictions",
)

FORBIDDEN_FILE_NAMES = {"predictions.npz"}
FORBIDDEN_OUTPUT_PATH_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
    "package_normal_candidate_gate",
    "registry_refresh",
)
REQUIRED_SUMMARY_FLAGS = {
    "research_only": True,
    "no_predictions_write": True,
    "no_registry_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_strict_pass_write": True,
}
ALLOWED_CODE_PREDICTIONS_NPZ_REFERENCES = {
    "modal_v25_research_vggt_predictions_3frames.py": [
        "predictions.npz is written",
        "Refusing to upload formal prediction file",
        "Safety stop: remote V25 output contains predictions.npz",
        "V25 must not write or address predictions.npz.",
    ],
    "v25_research_prediction_intake.py": [
        "predictions.npz is forbidden",
        "or `predictions.npz` is accepted",
    ],
    "v25_prediction_safety_scanner.py": [
        "predictions.npz",
    ],
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def scan_output_root(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not root.exists():
        findings.append({"path": str(root.resolve()), "severity": "warning", "reason": "V25 output root does not exist yet"})
        return findings
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        lower = rel.lower()
        if path.is_file() and path.name.lower() in FORBIDDEN_FILE_NAMES:
            findings.append({"path": str(path.resolve()), "severity": "error", "reason": "forbidden predictions.npz found"})
        tokens = [token for token in FORBIDDEN_OUTPUT_PATH_TOKENS if token in lower]
        if tokens:
            findings.append(
                {
                    "path": str(path.resolve()),
                    "severity": "error",
                    "reason": f"forbidden formal token in V25 output path: {', '.join(tokens)}",
                }
            )
    return findings


def scan_code_references(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json"}:
        return findings
    text = path.read_text(encoding="utf-8", errors="replace")
    if "np.savez" in text and "predictions.npz" in text:
        findings.append(
            {
                "path": str(path.resolve()),
                "severity": "error",
                "reason": "code appears capable of writing predictions.npz",
            }
        )
    if path.suffix.lower() == ".py" and "predictions.npz" in text:
        allowed = ALLOWED_CODE_PREDICTIONS_NPZ_REFERENCES.get(path.name, [])
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "predictions.npz" not in line:
                continue
            if not any(fragment in line for fragment in allowed):
                findings.append(
                    {
                        "path": str(path.resolve()),
                        "line": line_no,
                        "severity": "error",
                        "reason": "unexpected predictions.npz reference in V25 code",
                        "line_text": line.strip(),
                    }
                )
    return findings


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_json_error": repr(exc)}


def validate_summary(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    payload = load_json(path)
    if not payload:
        findings.append({"path": str(path.resolve()), "severity": "warning", "reason": "summary json missing or empty"})
        return findings
    if "_json_error" in payload:
        findings.append({"path": str(path.resolve()), "severity": "error", "reason": payload["_json_error"]})
        return findings
    for key, expected in REQUIRED_SUMMARY_FLAGS.items():
        if payload.get(key) is not expected:
            findings.append(
                {
                    "path": str(path.resolve()),
                    "severity": "error",
                    "reason": f"summary flag {key} is not {expected}",
                }
            )
    if payload.get("formal_cloud_unblocked") is not False:
        findings.append(
            {
                "path": str(path.resolve()),
                "severity": "error",
                "reason": "formal_cloud_unblocked must be false",
            }
        )
    return findings


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V25 Prediction Safety Scanner",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Scope: V25 research-only runner, intake, reports, and V25 output roots.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Findings",
        "",
    ]
    findings = summary.get("findings") or []
    if findings:
        for item in findings:
            location = item.get("path")
            if item.get("line"):
                location = f"{location}:{item['line']}"
            lines.append(f"- `{item.get('severity')}` {location}: {item.get('reason')}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan V25 outputs for research-only safety contract violations.")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "output/surface_research_preflight_local/V25_research_vggt_predictions/safety_scan_summary.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "output/surface_research_preflight_local/V25_research_vggt_predictions/safety_scan_summary.md")
    args = parser.parse_args()

    findings: list[dict[str, Any]] = []
    files = [file_info(path) for path in OWNED_FILES]
    for path in OWNED_FILES:
        findings.extend(scan_code_references(path))
    for root in OUTPUT_ROOTS:
        findings.extend(scan_output_root(root))
    findings.extend(validate_summary(REPO_ROOT / "reports/20260508_v25_research_predictions_3frames.json"))
    findings.extend(validate_summary(CLOUD_ROOT_SUMMARY := OUTPUT_ROOTS[0] / "research_summary.json"))
    if not CLOUD_ROOT_SUMMARY.is_file():
        findings.append({"path": str(CLOUD_ROOT_SUMMARY.resolve()), "severity": "warning", "reason": "remote/downloaded research_summary.json not present"})

    errors = [item for item in findings if item.get("severity") == "error"]
    status = "safety_scan_pass_research_only" if not errors else "safety_scan_fail"
    summary = {
        "task": "v25_prediction_safety_scanner",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "owned_files": files,
        "output_roots": [file_info(root) for root in OUTPUT_ROOTS],
        "findings": findings,
        "decision": (
            "V25 scan passed: no predictions.npz or formal promotion output was found in the V25 scope."
            if status == "safety_scan_pass_research_only"
            else "V25 scan failed: remove or explain the listed formal-output violation before using the research artifacts."
        ),
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "errors": errors}), ensure_ascii=False))
    return 0 if status == "safety_scan_pass_research_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
