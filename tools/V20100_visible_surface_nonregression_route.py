from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created_at = now()
    route = """# V20100000000000000000 Auto-Evolved Visible-Surface Non-Regression Route

Status: fail-closed continuation.

V198 source-upright render audit showed the real VGGT baseline is already the most mentor-readable full-scene human surface for the current back/side-back cases. V196/V197/V200 filtering routes preserved more baseline surface but did not outperform the baseline or hard controls, and in several views damaged leg/foot/clothing continuity.

Therefore the next route must not be another nearest/moderate-offset/fixed-ratio point selection pass.

Required next architecture:

```text
VGGT baseline visible RGB surface
        -> hard no-regression mask for high-confidence visible points
        -> SMPL-X posed adjacency and part graph
        -> connected weak-region infill decoder
        -> leg/foot/clothing continuity losses
        -> full-scene source-upright mentor board
```

Hard gates:

- visible baseline high-confidence points must stay unchanged;
- leg/foot and clothing continuity cannot regress relative to baseline;
- infill must be connected to weak regions, not a detached cloud;
- source-upright full-scene board is required, but render pass is auxiliary;
- true must beat baseline, same-topology, shuffled, and previous V194/V200 routes;
- face detail remains not applicable.

No external hard block: the current blocker is model objective/representation, not missing assets.
"""
    failures = []
    for path in [
        REPORTS / "V19600000000000000000_surface_locked_sparse_decision.json",
        REPORTS / "V19700000000000000000_moderate_offset_surface_decision.json",
        REPORTS / "V19800000000000000000_source_upright_visual_decision.json",
        REPORTS / "V20000000000000000000_visible_baseline_locked_decision.json",
    ]:
        if not path.exists():
            failures.append({"missing": str(path)})
    decision = {
        "created_at": created_at,
        "status": "V20100_VISIBLE_SURFACE_NONREGRESSION_ROUTE_READY_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "input_failures": failures,
        "v196_result": "sparse completion became cleaner but weaker than baseline/controls",
        "v197_result": "moderate offset selection was nearly visually identical to V196 and still failed controls",
        "v198_result": "source-upright render made baseline mentor-readable but did not change geometry gate",
        "v200_result": "visible baseline lock reduced noise but damaged visible leg/foot continuity and still lost to baseline/controls",
        "next_required_route": "train visible-surface non-regression plus connected weak-region infill; stop selection-only postprocessing",
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
    }
    write_text(GOALS / "V20100000000000000000_auto_evolved_visible_surface_nonregression_route.md", route)
    write_json(REPORTS / "V20100000000000000000_visible_surface_nonregression_route_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
