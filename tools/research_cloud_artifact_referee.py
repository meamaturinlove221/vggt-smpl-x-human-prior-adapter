from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.research_cloud_common import FORBIDDEN_PATH_WORDS, REQUIRED_RESEARCH_FLAGS, json_ready, repo_root, write_json


REPO_ROOT = repo_root()
DEFAULT_ROOT = REPO_ROOT / "output" / "surface_research_cloud_preflight"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "research_cloud_artifact_referee"
DEFAULT_STATUS_REPORT = REPO_ROOT / "reports" / "20260507_v8_research_cloud_artifact_referee.md"
DEFAULT_STATUS_JSON = REPO_ROOT / "reports" / "20260507_v8_research_cloud_artifact_referee.json"
STATUS = "research_cloud_artifact_referee_research_only"

THREED_SUFFIXES = {".ply", ".obj", ".npz", ".glb", ".gltf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
REQUIRED_CONTROLS = ("real", "shuffle", "zero")
OPTIONAL_CONTROLS = ("random", "mask_only")
REGION_WORDS = ("full", "head", "face", "hairline", "hand", "hands", "clothing")
PROTECTIVE_WORDS = (
    "research_only",
    "no_export",
    "no_predictions_write",
    "no_registry_write",
    "no_teacher_export",
    "no_candidate_export",
    "blocked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Referee V8 research-cloud artifacts. Fails closed and never promotes strict pass/export state."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_gate() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "tools/check_research_cloud_gate_status.py", "--write-default-manifest", "--json"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = int(proc.returncode)
    return payload


def read_text(path: Path, limit: int = 240_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def load_json(path: Path) -> Any | None:
    try:
        if path.stat().st_size > 6_000_000:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def walk_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(walk_scalars(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value[:120]):
            rows.extend(walk_scalars(item, f"{prefix}[{index}]"))
    else:
        rows.append((prefix, value))
    return rows


def classify_artifact_dir(path: Path) -> dict[str, Any]:
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item).lower())
    path_text = "\n".join(str(item.relative_to(path)).replace("\\", "/").lower() for item in files)
    text = path.name.lower() + "\n" + path_text
    json_payloads: list[tuple[Path, Any]] = []
    for file in files:
        if file.suffix.lower() in TEXT_SUFFIXES:
            text += "\n" + read_text(file, 16_000).lower()
        if file.suffix.lower() == ".json":
            payload = load_json(file)
            if payload is not None:
                json_payloads.append((file, payload))

    controls = sorted({name for name in (*REQUIRED_CONTROLS, *OPTIONAL_CONTROLS) if name in text})
    regions = sorted({word for word in REGION_WORDS if word in text})
    three_d = [file for file in files if file.suffix.lower() in THREED_SUFFIXES]
    contact = [file for file in files if file.suffix.lower() in IMAGE_SUFFIXES and ("contact" in file.name.lower() or "sheet" in file.name.lower())]
    reports = [file for file in files if file.suffix.lower() in {".md", ".json"} and ("summary" in file.name.lower() or "report" in file.name.lower())]

    flags: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    genealogy_hits: list[dict[str, Any]] = []
    suspicious_fields: list[dict[str, Any]] = []
    for source, payload in json_payloads:
        if isinstance(payload, dict):
            for key in REQUIRED_RESEARCH_FLAGS:
                if key in payload:
                    flags[key] = payload[key]
        for scalar_path, value in walk_scalars(payload):
            lower_path = scalar_path.lower()
            if any(word in lower_path for word in ("iou", "overfill", "recall", "score", "margin", "visible", "connected", "component")):
                if len(metrics) < 100:
                    metrics.append({"source": str(source.relative_to(path)), "path": scalar_path, "value": value})
            if any(word in lower_path for word in ("source", "input", "genealogy", "template", "smplx", "vggt", "dataset", "camera", "teacher")):
                if len(genealogy_hits) < 80:
                    genealogy_hits.append({"source": str(source.relative_to(path)), "path": scalar_path, "value": value})
            if any(word in lower_path for word in ("writes_predictions", "writes_teacher", "writes_candidate", "writes_registry", "strict_pass")) and value is True:
                suspicious_fields.append({"source": str(source.relative_to(path)), "path": scalar_path, "value": value})

    forbidden_path_hits = sorted({word for word in FORBIDDEN_PATH_WORDS if word in path_text})
    protective = any(word in text for word in PROTECTIVE_WORDS)
    missing_flags = [key for key, expected in REQUIRED_RESEARCH_FLAGS.items() if flags.get(key) is not expected]
    missing = []
    if not three_d:
        missing.append("missing_3d_output")
    if not contact:
        missing.append("missing_contact_sheet")
    if not reports:
        missing.append("missing_summary_or_report")
    if not set(REQUIRED_CONTROLS).issubset(controls) and any(key in path.name.lower() for key in ("fus3d", "hand", "hair")):
        missing.append("missing_real_shuffle_zero_controls")
    if any(key in path.name.lower() for key in ("fus3d", "hand", "hair", "dense")) and not regions:
        missing.append("missing_region_terms")
    if missing_flags:
        missing.append("missing_or_false_research_flags:" + ",".join(missing_flags))
    if forbidden_path_hits and not protective:
        missing.append("forbidden_path_words_without_protective_context:" + ",".join(forbidden_path_hits))
    if suspicious_fields:
        missing.append("suspicious_true_write_fields")

    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "total_bytes": sum(file.stat().st_size for file in files),
        "has_3d_output": bool(three_d),
        "three_d_samples": [str(item.relative_to(path)) for item in three_d[:12]],
        "has_contact_sheet": bool(contact),
        "contact_sheet_samples": [str(item.relative_to(path)) for item in contact[:12]],
        "summary_report_samples": [str(item.relative_to(path)) for item in reports[:12]],
        "controls_found": controls,
        "regions_found": regions,
        "research_flags_found": flags,
        "metric_refs": metrics,
        "genealogy_refs": genealogy_hits,
        "forbidden_path_hits": forbidden_path_hits,
        "protective_context": protective,
        "suspicious_fields": suspicious_fields,
        "can_promote_to_formal": False,
        "blocked_reasons": ["strict_formal_guard_not_checked_by_research_referee", *missing],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# V8 Research Cloud Artifact Referee",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This referee checks only research-cloud preflight artifacts. It cannot promote teacher, candidate,",
        "prediction, registry, or strict-pass state.",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(agg, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(summary["research_gate"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Artifact Summary",
        "",
        "```json",
        json.dumps(
            [
                {
                    "path": item["path"],
                    "has_3d_output": item["has_3d_output"],
                    "has_contact_sheet": item["has_contact_sheet"],
                    "controls_found": item["controls_found"],
                    "regions_found": item["regions_found"],
                    "blocked_reasons": item["blocked_reasons"],
                }
                for item in summary["artifacts"]
            ],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.expanduser().resolve()
    artifact_dirs = [
        item
        for item in sorted(root.iterdir(), key=lambda path: path.name.lower())
        if item.is_dir() and item.resolve() != args.output_dir.resolve()
    ] if root.is_dir() else []
    artifacts = [classify_artifact_dir(path) for path in artifact_dirs]
    research_gate = run_gate()
    aggregate = {
        "artifact_count": len(artifacts),
        "dirs_with_3d_output": sum(1 for item in artifacts if item["has_3d_output"]),
        "dirs_with_contact_sheet": sum(1 for item in artifacts if item["has_contact_sheet"]),
        "dirs_with_controls": sum(1 for item in artifacts if {"real", "shuffle", "zero"}.issubset(set(item["controls_found"]))),
        "dirs_with_forbidden_path_hits": sum(1 for item in artifacts if item["forbidden_path_hits"]),
        "dirs_with_suspicious_fields": sum(1 for item in artifacts if item["suspicious_fields"]),
        "research_cloud_allowed": bool(research_gate.get("research_gate", {}).get("allowed", False)),
        "formal_cloud_train_infer_export": "blocked",
        "teacher_export": "blocked",
        "candidate_export": "blocked",
        "predictions_export": "blocked",
        "strict_pass_write": "blocked",
    }
    summary = {
        "status": STATUS,
        "root": str(root),
        "aggregate": aggregate,
        "research_gate": research_gate,
        "artifacts": artifacts,
        "verdict": "research cloud artifacts remain research-only; formal promotion is blocked",
    }
    write_json(args.output_dir / "research_cloud_artifact_referee_summary.json", summary)
    write_report(args.output_dir / "research_cloud_artifact_referee_report.md", summary)
    write_json(args.status_json, summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": STATUS, **aggregate}), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
