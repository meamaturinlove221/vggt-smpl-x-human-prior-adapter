from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "20260509_v42_cloud_attempt_audit.json"
REPORT_MD = ROOT / "reports" / "20260509_v42_cloud_attempt_audit.md"
OUT = ROOT / "output" / "surface_research_preflight_local" / "V42_cloud_attempt_audit"
V42_OUT = ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions"
V41B = ROOT / "output" / "surface_research_preflight_local" / "V41b_real_prior_enabled_checkpoint"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "mtime": path.stat().st_mtime if path.is_file() else None,
    }


def dir_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rows.append(file_row(item))
    return rows


def main() -> None:
    required = {
        name: file_row(V42_OUT / name)
        for name in (
            "research_depths.npz",
            "research_points_world.npz",
            "research_confidence.npz",
            "research_normals_geometric.npz",
            "research_prior_effect.json",
            "control_real_zero_shuffle_random_dropout.json",
        )
    }
    missing = [name for name, row in required.items() if not row["exists"] or row["size"] <= 0]
    summary = {
        "task": "v42_cloud_attempt_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE" if missing else "DONE_PASS",
        "research_only": True,
        "commands_attempted": [
            "modal run modal_v42_prior_enabled_predictions.py",
            "modal run modal_v42_prior_enabled_predictions.py with PYTHONUTF8/PYTHONIOENCODING",
        ],
        "attempt_result": "local command timed out while uploading/running the 5GB V41b checkpoint; Modal app showed 0 active tasks and no V42 payload files were downloaded",
        "v41b_checkpoint": file_row(V41B / "checkpoint.pt"),
        "v41b_load_report": file_row(V41B / "load_report.json"),
        "v42_output_files": required,
        "v42_output_listing": dir_rows(V42_OUT),
        "missing_outputs": missing,
        "forbidden_outputs_written": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "writes_package": False,
        "writes_strict_registry": False,
        "writes_strict_pass": False,
        "decision": (
            "V42 has a prior-enabled VGGT checkpoint dependency, but the cloud inference payload was not produced in this run. "
            "The missing arrays remain a hard blocker for V43-V50 promotion."
        ),
    }
    write_json(REPORT_JSON, summary)
    write_json(OUT / "summary.json", summary)
    lines = [
        "# V42 Cloud Attempt Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        summary["decision"],
        "",
        "## Missing Outputs",
        "",
    ]
    lines.extend([f"- `{name}`" for name in missing] if missing else ["- none"])
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(summary["status"])


if __name__ == "__main__":
    main()
