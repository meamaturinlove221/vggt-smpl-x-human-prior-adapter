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
DEFAULT_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V28_final_gate"
DEFAULT_JSON = REPORTS / "20260508_v28_strict_teacher_candidate_gate.json"
DEFAULT_MD = REPORTS / "20260508_v28_strict_teacher_candidate_gate.md"


REQUIRED_REPORTS = {
    "v21_contract": REPORTS / "20260508_v21_completion_contract.json",
    "v22_microfit": REPORTS / "20260508_v22_true_vggt_smplx_microfit.json",
    "v23_residual": REPORTS / "20260508_v23_residual_surface_v2.json",
    "v24_teacher": REPORTS / "20260508_v24_residual_teacher_v2.json",
    "v25_predictions": REPORTS / "20260508_v25_research_predictions_3frames.json",
    "v26_temporal": REPORTS / "20260508_v26_temporal_canonical_teacher.json",
    "v27_training": REPORTS / "20260508_v27_teacher_supervised_training.json",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def path_forbidden_hits(root: Path) -> list[str]:
    hits: list[str] = []
    if not root.exists():
        return hits
    forbidden_names = {"predictions.npz", "candidate_package", "teacher_package", "strict_gate_registry"}
    for path in root.rglob("*"):
        lower = path.name.lower()
        if lower in forbidden_names or any(token in path.as_posix().lower() for token in ("strict_pass", "formal_candidate")):
            hits.append(str(path))
    return hits[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="V28 strict teacher/candidate gate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {name: read_json(path) for name, path in REQUIRED_REPORTS.items()}
    statuses = {name: data.get("status") for name, data in reports.items()}
    missing = [name for name, path in REQUIRED_REPORTS.items() if not path.is_file()]
    blockers: list[str] = []
    if missing:
        blockers.append(f"missing prerequisite reports: {missing}")
    for name, status in statuses.items():
        if status is None:
            blockers.append(f"{name} has no status")
        elif str(status) not in {"DONE_PASS", "DONE_FAIL_ROUTED"} and "ready" not in str(status) and "pass" not in str(status).lower():
            blockers.append(f"{name} status is not a completed V21-V27 state: {status}")

    v27 = reports.get("v27_training", {})
    v26 = reports.get("v26_temporal", {})
    v24 = reports.get("v24_teacher", {})
    v22 = reports.get("v22_microfit", {})
    v25 = reports.get("v25_predictions", {})
    if not bool(v27.get("research_positive")):
        blockers.append("V27 research_positive is false")
    if not bool(v26.get("metrics", {}).get("normal_available")):
        blockers.append("V26/V25 normal evidence unavailable; strict teacher normal gate cannot pass")
    if not bool(v24.get("audit_readiness", {}).get("v23_surface_complete_regions")):
        blockers.append("V24 teacher does not have complete region surfaces")
    if not bool(v22.get("comparison", {}).get("any_m2_m3_viewset_control_positive")) and not bool(
        v22.get("comparison", {}).get("any_trainable_method_control_positive")
    ):
        blockers.append("V22 has no positive trainable microfit cell")
    if "human_prior_channels=0" in json.dumps(v25, ensure_ascii=False):
        blockers.append("V25 Modal model used base VGGT with human_prior_channels=0")

    scan_roots = [
        REPO_ROOT / "output" / "surface_research_preflight_local",
        REPO_ROOT / "output" / "surface_research_cloud_preflight" / "V25_research_vggt_predictions",
    ]
    forbidden_hits: list[str] = []
    for root in scan_roots:
        forbidden_hits.extend(path_forbidden_hits(root))

    strict_teacher_pass = False
    strict_candidate_pass = False
    if forbidden_hits:
        blockers.append(f"forbidden output hits found under research roots: {len(forbidden_hits)}")
    if blockers:
        status = "DONE_FAIL_ROUTED"
    else:
        status = "DONE_PASS"
        strict_teacher_pass = True

    summary = {
        "task": "v28_strict_teacher_candidate_gate",
        "created_utc": utc_now(),
        "status": status,
        "research_only": not (strict_teacher_pass or strict_candidate_pass),
        "strict_candidate_passes": 1 if strict_candidate_pass else 0,
        "strict_teacher_passes": 1 if strict_teacher_pass else 0,
        "formal_cloud_unblocked": bool(strict_teacher_pass or strict_candidate_pass),
        "candidate_package_path": None,
        "teacher_package_path": None,
        "registry_entry_path": None,
        "writes_strict_registry": False,
        "writes_package": False,
        "writes_strict_pass": False,
        "prerequisite_statuses": statuses,
        "strict_teacher_precheck_pass": strict_teacher_pass,
        "strict_candidate_precheck_pass": strict_candidate_pass,
        "forbidden_scan": {
            "roots": scan_roots,
            "hit_count": len(forbidden_hits),
            "hits": forbidden_hits,
        },
        "blockers": blockers,
        "all_required_branches_executed": not missing,
        "failure_proof_complete": bool(blockers) and not missing,
        "failed_region": "strict_normal_or_model_prior_route" if blockers else None,
        "next_manual_mentor_decision_needed": bool(blockers) and not missing,
        "decision": (
            "V28 failed closed after all V21-V27 branches completed. No strict registry/package/pass was written."
            if blockers
            else "V28 strict gate passed and may be handed to the package builder."
        ),
    }
    write_json(args.output_json, summary)
    write_json(args.output_dir / "strict_gate_summary.json", summary)
    lines = [
        "# V28 Strict Teacher/Candidate Gate",
        "",
        f"Status: `{status}`",
        "",
        summary["decision"],
        "",
        "## Prerequisites",
        "",
    ]
    for name, status_value in statuses.items():
        lines.append(f"- {name}: `{status_value}`")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
            f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
            f"- formal_cloud_unblocked: `{summary['formal_cloud_unblocked']}`",
            f"- forbidden_scan_hits: `{len(forbidden_hits)}`",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend([f"- {b}" for b in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json, "strict_teacher_passes": summary["strict_teacher_passes"], "strict_candidate_passes": summary["strict_candidate_passes"]}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
