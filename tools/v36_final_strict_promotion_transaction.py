from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V36_final_promotion"
DEFAULT_JSON = REPORTS / "20260508_v36_final_promotion_report.json"
DEFAULT_MD = REPORTS / "20260508_v36_final_promotion_report.md"


REQUIRED = {
    "v29_normal": REPORTS / "20260508_v29_normal_route_rescue.json",
    "v30_prior_predictions": REPORTS / "20260508_v30_prior_enabled_vggt_predictions.json",
    "v31_training": REPORTS / "20260508_v31_teacher_supervised_candidate_train.json",
    "v32_inference": REPORTS / "20260508_v32_candidate_inference_region_audit.json",
    "v33_head_face": REPORTS / "20260508_v33_head_face_detail_route.json",
    "v34_hand": REPORTS / "20260508_v34_smplx_native_hand_route.json",
    "v35_60view": REPORTS / "20260508_v35_60view_support_expansion.json",
    "v36_forbidden_scan": REPORTS / "20260508_v36_forbidden_output_scan.json",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def stage_pass(data: dict[str, Any]) -> bool:
    status = str(data.get("status") or data.get("final_status") or "")
    if status == "DONE_PASS":
        return True
    return bool(data.get("strict_teacher_precheck_pass") or data.get("strict_candidate_precheck_pass"))


def main() -> int:
    parser = argparse.ArgumentParser(description="V36 final strict promotion transaction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {name: read_json(path) for name, path in REQUIRED.items()}
    missing = [name for name, path in REQUIRED.items() if not path.is_file()]
    statuses = {name: data.get("status") or data.get("final_status") for name, data in loaded.items()}
    blockers: list[str] = []
    if missing:
        blockers.append(f"missing V29-V36 prerequisite reports: {missing}")
    for name, data in loaded.items():
        if name == "v36_forbidden_scan":
            if data.get("hit_count", 1) != 0:
                blockers.append("forbidden output scan has hits")
            continue
        if not stage_pass(data):
            blockers.append(f"{name} is not pass-ready: {statuses.get(name)}")
    # Hard strict conditions: V30/V31/V32 must be genuine, not only routed evidence.
    if not bool(loaded.get("v30_prior_predictions", {}).get("human_prior_channels_gt_zero")):
        blockers.append("V30 did not verify human_prior_channels > 0")
    if not bool(loaded.get("v31_training", {}).get("checkpoint_exists")):
        blockers.append("V31 checkpoint missing")
    if not bool(loaded.get("v32_inference", {}).get("candidate_artifacts_exist")):
        blockers.append("V32 candidate inference artifacts missing")

    strict_pass = not blockers
    status = "DONE_PASS" if strict_pass else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
    candidate_package_path = args.output_dir / "candidate_package_v36" if strict_pass else None
    registry_entry_path = args.output_dir / "strict_registry_entry_v36.json" if strict_pass else None
    if strict_pass:
        candidate_package_path.mkdir(parents=True, exist_ok=True)
        write_json(registry_entry_path, {"strict_candidate_pass": True, "created_utc": utc_now()})

    summary = {
        "task": "v36_final_strict_promotion_transaction",
        "created_utc": utc_now(),
        "status": status,
        "all_routes_executed": not missing,
        "strict_candidate_passes": 1 if strict_pass else 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": strict_pass,
        "candidate_package_path": candidate_package_path,
        "teacher_package_path": None,
        "registry_entry_path": registry_entry_path,
        "writes_strict_registry": strict_pass,
        "writes_package": strict_pass,
        "writes_strict_pass": strict_pass,
        "mentor_decision_needed": not strict_pass,
        "remaining_blockers": blockers,
        "stage_statuses": statuses,
        "decision": (
            "V36 strict promotion passed and wrote candidate package/registry."
            if strict_pass
            else "V36 executed all available routes and failed with hard evidence. No strict package/registry/pass was written."
        ),
    }
    write_json(args.output_json, summary)
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "# V36 Final Promotion Report",
        "",
        f"Status: `{status}`",
        "",
        summary["decision"],
        "",
        f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        f"- formal_cloud_unblocked: `{summary['formal_cloud_unblocked']}`",
        f"- candidate_package_path: `{candidate_package_path}`",
        f"- registry_entry_path: `{registry_entry_path}`",
        "",
        "## Stage Statuses",
        "",
    ]
    for name, st in statuses.items():
        lines.append(f"- {name}: `{st}`")
    lines.extend(["", "## Remaining Blockers", ""])
    lines.extend([f"- {b}" for b in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "strict_candidate_passes": summary["strict_candidate_passes"], "strict_teacher_passes": summary["strict_teacher_passes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
