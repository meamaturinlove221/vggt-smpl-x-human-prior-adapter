from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_GATE = REPORTS / "20260508_v28_strict_teacher_candidate_gate.json"
DEFAULT_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V28_final_gate"
DEFAULT_JSON = REPORTS / "20260508_v28_final_package_builder.json"
DEFAULT_MD = REPORTS / "20260508_v28_final_package_builder.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="V28 final package builder or fail-closed proof pack.")
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = read_json(args.gate_json)
    pass_count = int(gate.get("strict_teacher_passes", 0)) + int(gate.get("strict_candidate_passes", 0))
    safe_pack = args.output_dir / "failure_proof_pack"
    copied: dict[str, Any] = {}
    for src in [
        REPORTS / "20260508_v21_completion_contract.md",
        REPORTS / "20260508_v22_true_vggt_smplx_microfit.md",
        REPORTS / "20260508_v23_residual_surface_v2.md",
        REPORTS / "20260508_v24_residual_teacher_v2.md",
        REPORTS / "20260508_v24_teacher_visual_review_pack.md",
        REPORTS / "20260508_v25_research_predictions_3frames.md",
        REPORTS / "20260508_v26_temporal_canonical_teacher.md",
        REPORTS / "20260508_v27_teacher_supervised_training.md",
        REPORTS / "20260508_v28_strict_teacher_candidate_gate.md",
    ]:
        dst = copy_if_exists(src, safe_pack / "reports")
        copied[src.name] = dst

    visual_sources = [
        REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_teacher_depth_contact_sheet.png",
        REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_teacher_region_contact_sheet.png",
        REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_teacher_uncertainty_contact_sheet.png",
    ]
    for src in visual_sources:
        dst = copy_if_exists(src, safe_pack / "visual_review")
        copied[src.name] = dst

    if pass_count > 0:
        status = "DONE_PASS"
        package_path = args.output_dir / "strict_package_placeholder"
        package_path.mkdir(exist_ok=True)
        decision = "Strict pass is available; a formal package directory was created."
    else:
        status = "DONE_FAIL_ROUTED"
        package_path = None
        decision = "No strict pass is available; only a failure-proof research pack was created. No package/registry/pass write occurred."

    summary = {
        "task": "v28_final_package_builder",
        "created_utc": utc_now(),
        "status": status,
        "research_only": pass_count == 0,
        "strict_candidate_passes": int(gate.get("strict_candidate_passes", 0)),
        "strict_teacher_passes": int(gate.get("strict_teacher_passes", 0)),
        "formal_cloud_unblocked": bool(gate.get("formal_cloud_unblocked", False)),
        "candidate_package_path": None,
        "teacher_package_path": None,
        "registry_entry_path": None,
        "writes_strict_registry": False,
        "writes_package": pass_count > 0,
        "writes_strict_pass": False,
        "failure_proof_pack": safe_pack,
        "formal_package_path": package_path,
        "copied_artifacts": copied,
        "decision": decision,
    }
    write_json(args.output_json, summary)
    write_json(args.output_dir / "package_builder_summary.json", summary)
    lines = [
        "# V28 Final Package Builder",
        "",
        f"Status: `{status}`",
        "",
        decision,
        "",
        f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"- formal_cloud_unblocked: `{summary['formal_cloud_unblocked']}`",
        f"- failure_proof_pack: `{safe_pack}`",
        f"- formal_package_path: `{package_path}`",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json, "failure_proof_pack": safe_pack}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
