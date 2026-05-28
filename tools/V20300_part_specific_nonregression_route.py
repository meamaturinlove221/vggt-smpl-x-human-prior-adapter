from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created_at = now()
    route = """# V20300000000000000000 Auto-Evolved Part-Specific Non-Regression Route

Status: fail-closed continuation.

V202 ran on Modal A10 with visible-surface non-regression and connected weak-region infill. It improved over V200 in score, but the mentor board still shows baseline as the cleanest visible human surface. V202 adds connected points but still contaminates clothing, leg, and foot boundaries and remains below V194/pose-frame/topology controls.

Do not continue global connected infill or selection-only postprocessing.

Next required route:

```text
baseline visible surface non-regression
    + part-specific weak-region heads
    + clothing boundary lock
    + leg/foot endpoint lock
    + small per-part infill quota
    + source-upright full-scene board
```

Hard gates:

- torso/clothing and leg/foot cannot regress against VGGT baseline;
- infill quota must be part-specific, not global;
- each local crop must show baseline / true / best control;
- true must beat baseline and hard controls in both source-upright visual board and topology metrics;
- face detail remains not applicable.

This is not an external hard block. It is a model objective/part-local decoder failure.
"""
    decision = {
        "created_at": created_at,
        "status": "V20300_PART_SPECIFIC_NONREGRESSION_ROUTE_READY_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "v202_result": "Modal A10 run improved over V200 but still failed full-scene mentor visual and hard controls.",
        "root_cause": "Global connected infill still contaminates clothing and leg/foot visible boundaries.",
        "next_required_route": "part-specific visible non-regression and local weak-region infill with clothing/leg/foot hard locks",
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
    }
    write(GOALS / "V20300000000000000000_auto_evolved_part_specific_nonregression_route.md", route)
    write_json(REPORTS / "V20300000000000000000_part_specific_nonregression_route_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
