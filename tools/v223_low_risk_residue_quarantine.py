from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
QUARANTINE = ROOT / "archive" / "quarantine" / "V223_low_risk_residue"
PROTECTED = [
    ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal",
    ROOT / "output" / "surface_research_preflight_local" / "V50_final_promotion_transaction",
    ROOT / "archive" / "V223_V50R2_mentor_final_bundle.zip",
]
LOW_RISK_RESIDUE = [
    ROOT / "archive" / "V223_recovery_cloud_evidence_bundle.zip",
    ROOT / "output" / "tmp_targetcam30_view0_scene",
    ROOT / "output" / "tmp_6v_from60v_view0_scene",
    ROOT / "output" / "tmp_60v_teacher_view0_scene",
    ROOT / "output" / "_tmp_tests",
    ROOT / "tmp_glbr_fc4_eval.out",
    ROOT / "tmp_glbr_fc5_eval.out",
    ROOT / "tmp_probe_summary_20260329_020541.md",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else None,
        "size": path.stat().st_size if path.is_file() else None,
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for src in LOW_RISK_RESIDUE:
        src = src.resolve()
        if not src.exists():
            skipped.append({"source": str(src), "reason": "missing"})
            continue
        if not is_relative_to(src, ROOT):
            skipped.append({"source": str(src), "reason": "outside_workspace"})
            continue
        if any(src == p.resolve() or is_relative_to(src, p) for p in PROTECTED):
            skipped.append({"source": str(src), "reason": "protected_v50r2_path"})
            continue
        dst = (QUARANTINE / src.relative_to(ROOT)).resolve()
        if not is_relative_to(dst, QUARANTINE):
            skipped.append({"source": str(src), "reason": "destination_outside_quarantine"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(dst.name + f".{int(time.time())}")
        shutil.move(str(src), str(dst))
        moved.append({"source": str(src), "destination": str(dst)})

    protected_checks = [row(path) for path in PROTECTED]
    payload = {
        "task": "V222_cleanup_report",
        "created_utc": now(),
        "status": "PASS",
        "policy": "Moved only low-risk temp/root scratch residues into archive quarantine. No V50R2 frozen candidate, package, registry, or final bundle was modified.",
        "quarantine_dir": str(QUARANTINE.resolve()),
        "moved_count": len(moved),
        "skipped_count": len(skipped),
        "moved": moved,
        "skipped": skipped,
        "protected_checks": protected_checks,
    }
    (REPORTS / "V222_cleanup_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "V222_cleanup_report.md").write_text(
        "# V222 Cleanup Report\n\n"
        f"- status: `{payload['status']}`\n"
        f"- moved_count: `{payload['moved_count']}`\n"
        f"- quarantine_dir: `{payload['quarantine_dir']}`\n"
        "- V50R2 frozen candidate and final bundle were not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "moved_count": len(moved), "quarantine": str(QUARANTINE)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
