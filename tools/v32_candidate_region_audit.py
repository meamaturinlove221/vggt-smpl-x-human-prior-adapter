#!/usr/bin/env python
"""Audit V32 candidate region coverage."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
REPORT_JSON = ROOT / "reports" / "20260508_v32_candidate_region_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v32_candidate_region_audit.md"


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    blockers = []
    for name, metric in summary["region_metrics"].items():
        if metric["pixel_count"] <= 0:
            blockers.append(f"{name}_empty")
        if metric["normal_nonzero_ratio"] < 0.8:
            blockers.append(f"{name}_normal_sparse")
    audit = {"status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED", "blockers": blockers, "region_metrics": summary["region_metrics"]}
    REPORT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REPORT_MD.write_text("# V32 Candidate Region Audit\n\nstatus: `{}`\n\nblockers: `{}`\n".format(audit["status"], blockers), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
