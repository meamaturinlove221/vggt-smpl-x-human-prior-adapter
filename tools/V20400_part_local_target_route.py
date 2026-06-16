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
    route = """# V20400000000000000000 Auto-Evolved Part-Local Target Route

Status: fail-closed continuation.

V203 Modal A10 part-specific non-regression produced cleaner visible surfaces, but it nearly collapsed back to the VGGT baseline and still failed hard controls. V202 added more connected infill but contaminated clothing and leg/foot boundaries. The route is now trapped between:

- clean visible baseline with no clear improvement;
- noisy connected infill with boundary regression.

Next route must add explicit part-local targets instead of global infill or quota-only selection.

Required next architecture:

```text
VGGT visible baseline surface
    + strict no-regression mask
    + per-part weak target regions
    + SMPL-X adjacency-local target bands
    + clothing/leg/foot boundary target masks
    + part-local infill decoder
    + source-upright full-scene mentor board
```

Hard gates:

- true must not regress visible baseline in clothing/leg/foot/head-hair regions;
- true must add visible part-local improvement in at least two regions;
- same-topology/shuffled/thickness-only cannot be close or better;
- full-scene source-upright board remains primary visual evidence;
- face detail remains not applicable.

This is a model/data target definition problem, not an external hard block.
"""
    decision = {
        "created_at": created_at,
        "status": "V20400_PART_LOCAL_TARGET_ROUTE_READY_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "v202_result": "connected infill improved score but polluted visible clothing/leg/foot boundaries",
        "v203_result": "part-specific quotas protected surface but collapsed near baseline and failed controls",
        "root_cause": "no explicit part-local target tells the student where improvement is allowed and required",
        "next_required_route": "construct part-local weak target masks and train local infill heads with no-regression locks",
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
    }
    write(GOALS / "V20400000000000000000_auto_evolved_part_local_target_route.md", route)
    write_json(REPORTS / "V20400000000000000000_part_local_target_route_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
