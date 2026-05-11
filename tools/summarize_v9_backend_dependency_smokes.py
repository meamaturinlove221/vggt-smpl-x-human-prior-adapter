from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "output" / "surface_research_cloud_preflight" / "V9_backend_dependency_smokes"
REPORT_JSON = ROOT / "reports" / "20260507_v9_backend_dependency_smoke_status.json"
REPORT_MD = ROOT / "reports" / "20260507_v9_backend_dependency_smoke_status.md"

SELECTED = {
    "2dgs_cuda": "2dgs_summary.json",
    "mast3r_slam_cuda": "mast3r_slam_summary.json",
    "hair_gs_cuda": "hair_gs_summary.json",
}


def _load_summary(subdir: str, filename: str) -> dict:
    path = SMOKE_ROOT / subdir / filename
    if not path.is_file():
        return {"path": str(path), "status": "missing_summary", "decision": "No summary file was produced."}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["path"] = str(path)
    return data


def _failed_steps(summary: dict) -> list[dict]:
    rows = []
    for name, step in summary.get("steps", {}).items():
        rc = step.get("returncode")
        if rc not in (0, None):
            tail = (step.get("stderr_tail") or step.get("stdout_tail") or "").replace("\r", "")
            rows.append({"step": name, "returncode": rc, "tail": tail[-1600:]})
    return rows


def main() -> int:
    entries = []
    for subdir, filename in SELECTED.items():
        summary = _load_summary(subdir, filename)
        entries.append(
            {
                "backend": summary.get("backend", subdir),
                "path": summary.get("path"),
                "status": summary.get("status"),
                "decision": summary.get("decision"),
                "research_only": summary.get("research_only"),
                "no_export": summary.get("no_export"),
                "no_predictions_write": summary.get("no_predictions_write"),
                "no_teacher_export": summary.get("no_teacher_export"),
                "no_candidate_export": summary.get("no_candidate_export"),
                "no_registry_write": summary.get("no_registry_write"),
                "no_strict_pass_write": summary.get("no_strict_pass_write"),
                "failed_steps": _failed_steps(summary),
            }
        )

    report = {
        "task": "v9_backend_dependency_smoke_status",
        "research_only": True,
        "formal_cloud_unblocked": False,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "d_root_repos_confirmed": [
            "D:\\2d-gaussian-splatting-main",
            "D:\\MASt3R-SLAM-main",
            "D:\\must3r-main",
        ],
        "dependency_smokes": entries,
        "live_modal_tasks_after_cleanup": 0,
        "monitoring_note": "Use tools/monitor_v9_backend_smokes.py with background modal logs instead of waiting on a blocking modal run.",
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# V9 Backend Dependency Smoke Status",
        "",
        "Research-only dependency smoke report. No teacher, candidate, predictions, registry, or strict pass artifact was written.",
        "",
        "## D-root Repos",
    ]
    for repo in report["d_root_repos_confirmed"]:
        lines.append(f"- `{repo}`")
    lines.extend(["", "## Cloud Smoke Results"])
    for entry in entries:
        lines.extend(
            [
                f"- `{entry['backend']}`: `{entry['status']}`",
                f"  - summary: `{entry['path']}`",
                f"  - decision: {entry['decision']}",
            ]
        )
        for failed in entry["failed_steps"][:4]:
            tail = " ".join(failed["tail"].split())[-420:]
            lines.append(f"  - failed step `{failed['step']}` rc={failed['returncode']}: {tail}")
    lines.extend(
        [
            "",
            "## Monitoring Fix",
            "",
            "`tools/monitor_v9_backend_smokes.py` now reads background Modal logs and summary files so dependency smokes do not need to be observed as blocking shell calls.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(REPORT_JSON), "md": str(REPORT_MD), "entries": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
