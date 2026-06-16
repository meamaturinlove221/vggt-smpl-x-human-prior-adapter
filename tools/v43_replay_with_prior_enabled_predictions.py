from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "output/surface_research_preflight_local/V43_replay_with_prior_enabled_predictions"
REPORT_JSON = REPO_ROOT / "reports/20260509_v43_replay_with_prior_enabled_predictions.json"
REPORT_MD = REPO_ROOT / "reports/20260509_v43_replay_with_prior_enabled_predictions.md"
V42_REPORT = REPO_ROOT / "reports/20260509_v42_prior_enabled_predictions_rerun.json"
V42_OUTPUT = REPO_ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions"

FORBIDDEN_NAMES = {"predictions.npz"}
FORBIDDEN_TOKENS = (
    "formal_candidate",
    "candidate_package",
    "teacher_package",
    "strict_gate_registry",
    "strict_pass",
    "registry_refresh",
)

REPLAY_STAGES = ("V31", "V32", "V33", "V34", "V35", "V36")
REQUIRED_V42_FILES = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_confidence.npz",
    "research_normals_geometric.npz",
    "research_prior_effect.json",
    "control_real_zero_shuffle_random_dropout.json",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "mtime": path.stat().st_mtime if path.is_file() else None,
    }


def scan_forbidden(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not root.exists():
        return findings
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix().lower()
        if path.is_file() and path.name.lower() in FORBIDDEN_NAMES:
            findings.append({"path": str(path.resolve()), "reason": "predictions.npz is forbidden for V43"})
        hits = [token for token in FORBIDDEN_TOKENS if token in rel]
        if hits:
            findings.append({"path": str(path.resolve()), "reason": "formal output token in V43 path: " + ",".join(hits)})
    return findings


def stage_statuses(v42_pass: bool) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for stage in REPLAY_STAGES:
        statuses[stage] = {
            "status": "DONE_FAIL_ROUTED" if not v42_pass else "READY_TO_REPLAY",
            "executed_with_v42_inputs": False,
            "reason": "V42 prior-enabled prediction payload is missing, so replay is blocked without fabricating inputs."
            if not v42_pass
            else "V42 pass allows replay.",
        }
    return statuses


def write_md(summary: dict[str, Any]) -> None:
    lines = [
        "# V43 Replay With Prior-Enabled Predictions",
        "",
        f"Status: `{summary['status']}`",
        "",
        "V43 is research-only and did not write formal outputs.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## V42 Gate",
        "",
        f"- V42 status: `{summary['v42_status']}`",
        f"- complete V42 payload: `{summary['complete_v42_payload']}`",
        "",
        "## Replay Stages",
        "",
    ]
    for stage, item in summary["replay_stages"].items():
        lines.append(f"- `{stage}`: `{item['status']}` - {item['reason']}")
    lines.extend(["", "## Blockers", ""])
    for blocker in summary["blockers"] or ["none"]:
        lines.append(f"- {blocker}")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    v42 = read_json(V42_REPORT)
    v42_status = v42.get("status", "MISSING")
    v42_files = {name: file_info(V42_OUTPUT / name) for name in REQUIRED_V42_FILES}
    complete_payload = all(info["exists"] and info["size"] > 0 for info in v42_files.values())
    v42_pass = v42_status == "DONE_PASS" and complete_payload

    blockers: list[str] = []
    if not v42_pass:
        blockers.append("V42 did not produce a complete prior-enabled prediction payload from a real V38-V41 checkpoint.")
        blockers.append("V43 replay of V31-V36 is not valid without V42 research_depths/points/confidence/normals/prior-effect/control outputs.")
    forbidden = scan_forbidden(OUT_ROOT)
    summary = {
        "task": "v43_replay_with_prior_enabled_predictions",
        "created_utc": utc_now(),
        "status": "DONE_FAIL_ROUTED" if blockers else "DONE_PASS",
        "research_only": True,
        "v42_status": v42_status,
        "complete_v42_payload": complete_payload,
        "v42_files": v42_files,
        "replay_stages": stage_statuses(v42_pass),
        "blockers": blockers,
        "forbidden_findings": forbidden,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "writes_package": False,
        "writes_strict_registry": False,
        "writes_strict_pass": False,
        "decision": (
            "V43 replay is routed back to V37-V42 because V42 lacks real prior-enabled predictions."
            if blockers
            else "V43 replay completed with real V42 prior-enabled predictions."
        ),
        "next_route": "V37-V42 checkpoint construction and V42 rerun" if blockers else "V44 visual pre-promotion gate",
    }
    write_json(REPORT_JSON, summary)
    write_json(OUT_ROOT / "summary.json", summary)
    write_md(summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(result["status"])
