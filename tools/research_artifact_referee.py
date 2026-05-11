from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("output/surface_research_preflight_local")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "research_artifact_referee_20260507"
DEFAULT_STATUS_REPORT = Path("reports/20260507_research_artifact_referee_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}
RISK_WORDS = ("strict_pass", "teacher_export", "candidate_export", "cloud_allowed", "CLOUD ALLOWED")
CONTROL_WORDS = ("real", "shuffle", "zero", "random")
REGION_WORDS = ("full", "head", "face", "hairline", "hand", "hands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed referee for local research artifacts. It never writes pass/export state."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_guard(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, *args], cwd=Path.cwd(), text=True, capture_output=True)
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = int(proc.returncode)
    return payload


def classify_dir(path: Path) -> dict[str, Any]:
    files = [p for p in path.rglob("*") if p.is_file()]
    suffix_counts: dict[str, int] = {}
    for file in files:
        suffix_counts[file.suffix.lower() or "<none>"] = suffix_counts.get(file.suffix.lower() or "<none>", 0) + 1
    names = " ".join(p.name.lower() for p in files[:2000])
    text_sample = ""
    for file in files:
        if file.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        try:
            text_sample += "\n" + file.read_text(encoding="utf-8", errors="ignore")[:4000]
        except Exception:
            pass
        if len(text_sample) > 30000:
            break
    lower = (names + "\n" + text_sample).lower()
    risk_hits = [word for word in RISK_WORDS if word.lower() in lower]
    protective = all(token in lower for token in ("research_only", "no_teacher_export", "no_candidate_export")) or "blocked" in lower
    controls = [word for word in CONTROL_WORDS if word in lower]
    regions = [word for word in REGION_WORDS if word in lower]
    has_3d = any(p.suffix.lower() in {".ply", ".obj", ".npz"} for p in files)
    has_contact = any("contact" in p.name.lower() and p.suffix.lower() in {".png", ".jpg", ".json"} for p in files)
    can_enter_strict = False
    reasons = []
    if not has_3d:
        reasons.append("missing_3d_artifact")
    if not has_contact:
        reasons.append("missing_contact_sheet")
    if len(set(controls) & {"real", "shuffle", "zero"}) < 3 and any(key in path.name.lower() for key in ("fus3d", "hair", "gs")):
        reasons.append("missing_real_shuffle_zero_controls")
    if risk_hits and not protective:
        reasons.append("risk_words_without_protective_context")
    if STRICT_FACTS["strict_candidate_passes"] == 0:
        reasons.append("strict_candidate_passes_zero")
    return {
        "path": str(path.resolve()),
        "file_count": int(len(files)),
        "suffix_counts": suffix_counts,
        "has_3d_artifact": bool(has_3d),
        "has_contact_sheet": bool(has_contact),
        "controls_found": controls,
        "regions_found": regions,
        "risk_hits": risk_hits,
        "protective_context": bool(protective),
        "can_enter_strict_gate": bool(can_enter_strict),
        "blocked_reasons": reasons,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Research Artifact Referee",
        "",
        "Status: `research_only_referee_strict_gate_red`",
        "",
        "This referee scans local research artifacts for 3D outputs, contact sheets,",
        "controls, pass/export/cloud risk words, and strict-gate eligibility. It does",
        "not write pass state, export candidates/teachers, or launch cloud jobs.",
        "",
        "## Guard",
        "",
        "```json",
        json.dumps(summary["guards"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Verdict",
        "",
        "```text",
        summary["verdict"],
        "```",
        "",
        "## Artifacts",
        "",
        "```json",
        json.dumps(summary["artifacts"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    interesting = [
        p for p in args.root.iterdir()
        if p.is_dir()
        and any(key in p.name.lower() for key in ("b_gs", "fus3d", "hair", "hand", "a5"))
    ]
    rows = [classify_dir(path) for path in sorted(interesting, key=lambda p: p.name.lower())]
    guards = {
        "candidate": run_guard(["tools/check_cloud_gate_status.py", "--json"]),
        "teacher_supervised": run_guard(["tools/check_cloud_gate_status.py", "--teacher-supervised", "--json"]),
    }
    strict_red = not bool(guards["candidate"].get("cloud_allowed")) and int(guards["candidate"].get("strict_candidate_passes", 0) or 0) == 0
    verdict = (
        "strict gate red; research artifacts remain blocked from formal cloud/export"
        if strict_red
        else "guard is not red; inspect manually before proceeding"
    )
    summary = {
        "status": "research_only_referee_strict_gate_red",
        "strict_facts": STRICT_FACTS,
        "root": str(args.root.resolve()),
        "guards": guards,
        "artifacts": rows,
        "verdict": verdict,
    }
    (args.output_dir / "research_artifact_referee_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "research_artifact_referee_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "artifact_count": len(rows), "verdict": verdict}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
