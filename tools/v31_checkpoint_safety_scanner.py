#!/usr/bin/env python
"""Safety scanner for V31 research checkpoint outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train"
REPORT_JSON = ROOT / "reports" / "20260508_v31_checkpoint_safety_scanner.json"
REPORT_MD = ROOT / "reports" / "20260508_v31_checkpoint_safety_scanner.md"
FORBIDDEN_NAMES = {
    "predictions.npz",
    "strict_gate_registry_entry.json",
    "candidate_package.json",
    "teacher_package.json",
}


def main() -> None:
    hits = []
    if OUT.exists():
        for path in OUT.rglob("*"):
            if path.name in FORBIDDEN_NAMES or "formal_candidate" in str(path).lower() or "strict_registry" in str(path).lower():
                hits.append(str(path))
    summary = {
        "status": "DONE_PASS" if not hits else "DONE_FAIL_ROUTED",
        "output_root": str(OUT),
        "forbidden_hit_count": len(hits),
        "forbidden_hits": hits,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text("# V31 Checkpoint Safety Scanner\n\nstatus: `{}`\n\nforbidden_hit_count: `{}`\n".format(summary["status"], len(hits)), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "forbidden_hit_count": len(hits)}, indent=2))


if __name__ == "__main__":
    main()
