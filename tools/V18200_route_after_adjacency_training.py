from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
REPORTS = REPO / "reports"
DOCS = REPO / "docs" / "goals"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    created_at = now()
    v180 = read_json(REPORTS / "V18000000000000000000_adjacency_collision_decision.json")
    v181 = read_json(REPORTS / "V18100000000000000000_training_decision.json")
    failures = [
        "V181 remains billboard_fail_v2=true / combined_fail_v4=true.",
        "V181 does not consistently beat V173; 0012 and current_v895 regress versus V173.",
        "Invalid distant overlap remains high across all four cases.",
        "The V181 board shows torn multi-layer sheets rather than a stable human-main 3D point cloud.",
        "Direct pair loss on final shell points is insufficient; the representation itself must decode part-pair exclusion and semantic graph occupancy.",
    ]
    decision = {
        "created_at": created_at,
        "status": "V18200_ADJACENCY_TRAINING_FAIL_CLOSED_ROUTE_TO_PART_PAIR_EXCLUSION_DECODER",
        "mentor_ready": False,
        "external_hard_block": False,
        "v180_status": v180.get("status"),
        "v181_status": v181.get("status"),
        "failure_reasons": failures,
        "allowed_face_claim": "head/face contour and hair region only",
        "forbidden_claims": [
            "facial detail improved",
            "mentor-ready",
            "external hard block",
            "route exhausted",
            "metric pass",
            "viewer/render fix success",
        ],
        "next_route": {
            "name": "V18250 part-pair exclusion decoder / semantic graph occupancy",
            "must_not_do": [
                "increase thickness-only loss",
                "global part shift",
                "procedural occupancy repair",
                "viewer-only repair",
                "projection-only pass",
            ],
            "must_do": [
                "keep V173 as current best source candidate",
                "add per-part occupancy fields before point decoding",
                "add invalid-pair exclusion logits for head-foot, torso-foot, arm-leg, and left-right endpoints",
                "retain valid adjacent contact for head-torso, torso-arm, torso-leg, and leg-foot",
                "decode each part in a semantic graph frame instead of copying one global shell",
                "evaluate with full-scene mentor boards and V180 metric v4",
            ],
        },
    }
    write_json(REPORTS / "V18200000000000000000_route_decision.json", decision)
    route = f"""# V182 Route After Adjacency Training

Created: {created_at}

## Status

`V18200_ADJACENCY_TRAINING_FAIL_CLOSED_ROUTE_TO_PART_PAIR_EXCLUSION_DECODER`

V181 is not mentor-ready and is not an external hard block.

## Why V181 Fails

- V181 was trained on Modal A10 for 300 steps / 8192 points, so this is a real training attempt.
- The result still has `billboard_fail_v2=true` and `combined_fail_v4=true`.
- It improves some cases but regresses 0012/current against V173.
- Invalid distant part overlap remains high.
- The board still looks like torn multi-layer sheets, not a stable 3D human point cloud.

## Correct Next Step

Do not keep increasing pair-loss weight or thickness. The next route must move the topology signal into the representation:

1. Per-part occupancy fields before point decoding.
2. Part-pair exclusion logits for semantically distant overlaps.
3. Valid-adjacency contact logits for real body edges.
4. Semantic graph frame decoding instead of a single global shell copy.
5. Full-scene mentor board and same-scene hard controls as the main gate.

## Final Policy

No mentor-ready claim is allowed from V181. No external hard block is allowed because the files, Modal A10, and repo writes are available. Continue via V18250.
"""
    REPORTS.joinpath("V18200000000000000000_route_state.md").write_text(route, encoding="utf-8")
    goal = f"""# V18250 Auto-Evolved Part-Pair Exclusion Decoder Route

Created: {created_at}

This route continues the anti-billboard topology-volume goal after V181 failed closed.

## Hard Constraint

Main mentor evidence remains a human-main full-scene RGB point cloud with partial real environment. Metrics, projection, render, thickness, adjacency scores, and local crops are auxiliary only.

## Repair Target

The model must stop decoding one global multi-shell that lets semantically distant parts overlap. It must decode body parts in a semantic graph frame with explicit part-pair occupancy/exclusion.

## Required Work

1. Build a part-pair exclusion decoder contract.
2. Add invalid-pair exclusion heads for head-foot, torso-foot, arm-leg, and left-right endpoints.
3. Add valid-contact heads for head-torso, torso-arm, torso-leg, and leg-foot.
4. Train or smoke a model-owned student without teacher/raw Kinect inference.
5. Compare against V173, V181, VGGT baseline, same-topology, shuffled, thickness-only, and posthoc.
6. Generate full-scene mentor board, same-scene controls, turntable/cross-section, local 3D morphology closeups, and environment gate.
7. Fail closed unless the visual gate and V180/V4-style causality gates both pass.

## No Agent Rule

Do not launch agents/subagents unless the user explicitly changes this run's permission.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    DOCS.joinpath("V18250000000000000000_auto_evolved_part_pair_exclusion_decoder_route.md").write_text(goal, encoding="utf-8")
    print(json.dumps({"status": decision["status"], "mentor_ready": False, "external_hard_block": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
