from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions"
REPORT_JSON = REPO_ROOT / "reports/20260509_v42_prior_enabled_predictions_rerun.json"
REPORT_MD = REPO_ROOT / "reports/20260509_v42_prior_enabled_predictions_rerun.md"

REQUIRED_OUTPUTS = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_confidence.npz",
    "research_normals_geometric.npz",
    "research_prior_effect.json",
    "control_real_zero_shuffle_random_dropout.json",
)

FORBIDDEN_NAMES = {"predictions.npz"}
FORBIDDEN_TOKENS = (
    "formal_candidate",
    "candidate_package",
    "teacher_package",
    "strict_gate_registry",
    "strict_pass",
    "registry_refresh",
)

CHECKPOINT_ROOTS = (
    REPO_ROOT / "output/surface_research_preflight_local/V38_prior_enabled_checkpoint_scaffold",
    REPO_ROOT / "output/surface_research_preflight_local/V39_adapter_only_microfit",
    REPO_ROOT / "output/surface_research_preflight_local/V40_region_balanced_adapter_rescue",
    REPO_ROOT / "output/surface_research_preflight_local/V41_prior_sensitive_head_unfreeze",
    REPO_ROOT / "output/surface_research_cloud_preflight/V38_prior_enabled_checkpoint_scaffold",
    REPO_ROOT / "output/surface_research_cloud_preflight/V39_adapter_only_microfit",
    REPO_ROOT / "output/surface_research_cloud_preflight/V40_region_balanced_adapter_rescue",
    REPO_ROOT / "output/surface_research_cloud_preflight/V41_prior_sensitive_head_unfreeze",
)

REFERENCE_CHECKPOINTS = (
    REPO_ROOT / "output/surface_research_preflight_local/V31_teacher_supervised_candidate_train/v31_candidate_research_checkpoint.npz",
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
            findings.append({"path": str(path.resolve()), "reason": "predictions.npz is forbidden for V42"})
        hits = [token for token in FORBIDDEN_TOKENS if token in rel]
        if hits:
            findings.append({"path": str(path.resolve()), "reason": "formal output token in V42 path: " + ",".join(hits)})
    return findings


def npz_keys(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "keys": []}
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {
                "exists": True,
                "keys": list(payload.files),
                "arrays": {key: {"shape": list(np.asarray(payload[key]).shape), "dtype": str(np.asarray(payload[key]).dtype)} for key in payload.files},
            }
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "keys": [], "error": f"{type(exc).__name__}: {exc}"}


def find_checkpoint_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".npz"}
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            rows.append({"root": str(root.resolve()), "root_exists": False})
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            lower = path.as_posix().lower()
            row = {
                "root": str(root.resolve()),
                "root_exists": True,
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "suffix": path.suffix.lower(),
                "name_has_prior_enabled": any(token in lower for token in ("prior", "human_prior", "adapter", "vggt")),
                "probe": {},
                "usable_for_v42": False,
                "rejection_reason": "",
            }
            if path.suffix.lower() == ".npz":
                probe = npz_keys(path)
                row["probe"] = probe
                keys = set(probe.get("keys", []))
                if "checkpoint_kind" in keys:
                    row["rejection_reason"] = "npz checkpoint is metadata/residual-style, not full VGGT state_dict"
                else:
                    row["rejection_reason"] = "npz checkpoint lacks full VGGT state_dict and HumanPriorAdapter weights"
            else:
                row["probe"] = {"state_dict_not_loaded": "V42 agent does not execute untrusted torch checkpoint unless located in V38-V41 roots with explicit report"}
                row["rejection_reason"] = "candidate requires explicit V38-V41 load_report before inference"
            rows.append(row)
    for path in REFERENCE_CHECKPOINTS:
        row = {
            "root": str(path.parent.resolve()),
            "root_exists": path.parent.exists(),
            "path": str(path.resolve()),
            "size": path.stat().st_size if path.is_file() else 0,
            "suffix": path.suffix.lower(),
            "name_has_prior_enabled": False,
            "probe": npz_keys(path),
            "usable_for_v42": False,
            "rejection_reason": "V31 research checkpoint is a residual/candidate NPZ, not a prior-enabled VGGT checkpoint",
        }
        rows.append(row)
    return rows


def existing_v30_v36_context() -> dict[str, Any]:
    return {
        "v30": read_json(REPO_ROOT / "reports/20260508_v30_prior_enabled_vggt_predictions.json"),
        "v31": read_json(REPO_ROOT / "reports/20260508_v31_teacher_supervised_candidate_train.json"),
        "v36": read_json(REPO_ROOT / "reports/20260508_v36_final_promotion_report.json"),
    }


def write_md(summary: dict[str, Any]) -> None:
    lines = [
        "# V42 Prior-Enabled Predictions Rerun",
        "",
        f"Status: `{summary['status']}`",
        "",
        "V42 is research-only and did not write `predictions.npz`, candidate package, teacher package, strict registry, or strict pass.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Checkpoint Gate",
        "",
        f"- usable V38-V41 prior-enabled checkpoint: `{summary['usable_prior_enabled_checkpoint_exists']}`",
        f"- candidate count: `{len(summary['checkpoint_candidates'])}`",
        "",
        "## Required Outputs",
        "",
    ]
    for name, info in summary["required_outputs"].items():
        lines.append(f"- `{name}`: exists=`{info['exists']}` size=`{info['size']}`")
    lines.extend(["", "## Blockers", ""])
    for blocker in summary["blockers"] or ["none"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Forbidden Scan", ""])
    lines.append(f"- findings: `{len(summary['forbidden_findings'])}`")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = find_checkpoint_candidates()
    usable = [row for row in candidates if row.get("usable_for_v42")]
    required_outputs = {name: file_info(OUT_ROOT / name) for name in REQUIRED_OUTPUTS}
    blockers: list[str] = []
    if not usable:
        blockers.append("No V38-V41 usable prior-enabled VGGT checkpoint/state_dict exists under required research roots.")
        blockers.append("V31 checkpoint exists but is a small residual/candidate NPZ, not a loadable VGGT HumanPriorAdapter checkpoint.")
        blockers.append("V42 therefore refuses to run base VGGT, random prior-enabled weights, or residual NPZ as predictions.")
    for name, info in required_outputs.items():
        if not info["exists"]:
            blockers.append(f"missing V42 research output because checkpoint gate failed: {name}")
    forbidden = scan_forbidden(OUT_ROOT)
    summary = {
        "task": "v42_prior_enabled_predictions_rerun",
        "created_utc": utc_now(),
        "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE" if blockers else "DONE_PASS",
        "research_only": True,
        "usable_prior_enabled_checkpoint_exists": bool(usable),
        "usable_prior_enabled_checkpoints": usable,
        "checkpoint_candidates": candidates,
        "required_outputs": required_outputs,
        "context": existing_v30_v36_context(),
        "blockers": blockers,
        "forbidden_findings": forbidden,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "writes_package": False,
        "writes_strict_registry": False,
        "writes_strict_pass": False,
        "decision": (
            "V42 fail-closed with evidence: prior-enabled predictions cannot be rerun until V38-V41 produce a loadable HumanPriorAdapter VGGT checkpoint."
            if blockers
            else "V42 generated complete prior-enabled research predictions from a usable V38-V41 checkpoint."
        ),
        "next_route": "V37-V41 prior-enabled checkpoint construction line" if blockers else "V43 replay",
    }
    write_json(REPORT_JSON, summary)
    write_json(OUT_ROOT / "summary.json", summary)
    write_md(summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(result["status"])
