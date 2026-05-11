#!/usr/bin/env python
"""Record V40/V41 route decisions after V39."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT / "reports" / "20260509_v39_adapter_microfit.json"
REPORTS = ROOT / "reports"


def write_pair(name: str, payload: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"20260509_{name}.json"
    md_path = REPORTS / f"20260509_{name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"status: `{payload['status']}`",
        f"decision: `{payload['decision']}`",
        f"reason: {payload['reason']}",
        "",
        "## Upstream Evidence",
        "",
        f"- v39_status: `{payload['v39_status']}`",
        f"- v39_next_route: `{payload['v39_next_route']}`",
        f"- v39_region_win_count: `{payload['v39_region_win_count']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    v39 = json.loads(V39.read_text(encoding="utf-8")) if V39.is_file() else {}
    v39_pass = v39.get("status") == "DONE_PASS" and v39.get("next_route") == "V42_READY_FROM_V39"
    base = {
        "research_only": True,
        "v39_report": str(V39),
        "v39_status": v39.get("status"),
        "v39_next_route": v39.get("next_route"),
        "v39_region_win_count": v39.get("region_win_count"),
        "v39_real_wins_controls": v39.get("real_wins_controls"),
        "formal_outputs_written": False,
    }
    if v39_pass:
        v40 = {
            **base,
            "status": "DONE_PASS",
            "decision": "NOT_REQUIRED",
            "reason": "V39 adapter-only microfit already beat zero/shuffle/random-region/prior-dropout controls in all required regions.",
            "would_route_if_failed": "region-balanced adapter rescue",
        }
        v41 = {
            **base,
            "status": "DONE_PASS",
            "decision": "NOT_REQUIRED",
            "reason": "V40 escalation is not required because V39 met the control and region criteria; therefore prior-sensitive head unfreeze is unnecessary.",
            "would_route_if_failed": "prior-sensitive depth/point head unfreeze",
            "checkpoint_full_prior_enabled": None,
        }
    else:
        v40 = {
            **base,
            "status": "DONE_FAIL_ROUTED",
            "decision": "REQUIRED",
            "reason": "V39 did not meet the region/control criteria.",
            "would_route_if_failed": "region-balanced adapter rescue",
        }
        v41 = {
            **base,
            "status": "DONE_FAIL_ROUTED",
            "decision": "HELD_FOR_V40",
            "reason": "V41 depends on V40 failing or being insufficient.",
            "would_route_if_failed": "prior-sensitive depth/point head unfreeze",
        }
    write_pair("v40_region_balanced_adapter_rescue", v40)
    write_pair("v41_prior_sensitive_head_unfreeze", v41)
    print(json.dumps({"v40": v40["status"], "v41": v41["status"], "decision": v40["decision"]}, indent=2))


if __name__ == "__main__":
    main()
