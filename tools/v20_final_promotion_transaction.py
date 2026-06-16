from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_OUT = LOCAL_ROOT / "V20_final_promotion_transaction"
DEFAULT_JSON = REPORTS / "20260508_v20_final_promotion_transaction.json"
DEFAULT_MD = REPORTS / "20260508_v20_final_promotion_transaction.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def main() -> int:
    parser = argparse.ArgumentParser(description="V20 final strict promotion transaction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    sources = {
        "v16_router": REPORTS / "20260508_v16_dline_failure_router.json",
        "v16_microfit": REPORTS / "20260508_v16_vggt_smplx_microfit_runner.json",
        "v17_residual": REPORTS / "20260508_v17_smplx_residual_surface_optimizer.json",
        "v18_distill": REPORTS / "20260508_v18_residual_teacher_distillation_case.json",
        "v19_temporal": REPORTS / "20260508_v19_temporal_canonical_residual_teacher.json",
    }
    loaded = {name: read_json(path) for name, path in sources.items()}
    statuses = {name: data.get("status") for name, data in loaded.items()}

    strict_ready = False
    blockers = []
    if not str(statuses.get("v17_residual", "")).endswith("ready"):
        blockers.append("V17 residual route is not research-ready.")
    if "ready" not in str(statuses.get("v18_distill", "")):
        blockers.append("V18 residual distillation case is not ready.")
    if "predictions_missing" in str(statuses.get("v19_temporal", "")):
        blockers.append("V19 adjacent-frame VGGT predictions are missing.")
    if not bool(loaded.get("v16_microfit", {}).get("comparison", {}).get("any_trainable_method_control_positive")):
        blockers.append("V16 trainable microfit did not beat zero/shuffle controls.")

    summary = {
        "task": "v20_final_promotion_transaction",
        "created_utc": utc_now(),
        "status": "v20_promotion_fail_closed_no_strict_write",
        "research_only": True,
        "formal_cloud_unblocked": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "candidate_package_path": None,
        "teacher_package_path": None,
        "registry_entry_path": None,
        "writes_strict_registry": False,
        "writes_package": False,
        "writes_strict_pass": False,
        "sources": {name: str(path.resolve()) for name, path in sources.items()},
        "source_statuses": statuses,
        "strict_ready": strict_ready,
        "blockers": blockers,
        "decision": "V20 ran the promotion transaction and failed closed. Upstream artifacts remain research-only and do not satisfy strict teacher/candidate gates.",
    }
    write_json(args.output_json, summary)
    write_json(out / "summary.json", summary)
    lines = [
        "# V20 Final Promotion Transaction",
        "",
        f"Status: `{summary['status']}`",
        "",
        summary["decision"],
        "",
        "## Source Statuses",
        "",
    ]
    for name, status in statuses.items():
        lines.append(f"- {name}: `{status}`")
    lines.extend(["", "## Non-Promotion Guard", "", "- strict_candidate_passes: `0`", "- strict_teacher_passes: `0`", "- writes_package: `False`", "- writes_strict_registry: `False`", "", "## Blockers", ""])
    lines.extend([f"- {b}" for b in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": summary["status"], "json": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
