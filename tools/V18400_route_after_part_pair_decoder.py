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
    v183 = read_json(REPORTS / "V18300000000000000000_training_decision.json")
    decision = {
        "created_at": created_at,
        "status": "V18400_PART_PAIR_DECODER_FAIL_CLOSED_ROUTE_TO_CANONICAL_SURFEL_GRAPH_OCCUPANCY",
        "mentor_ready": False,
        "external_hard_block": False,
        "v183_status": v183.get("status"),
        "root_cause": [
            "Point-anchor shell decoding keeps inheriting billboard geometry from the visible RGB/VGGT sheet.",
            "Part-pair exclusion heads improve selected scores but do not create stable whole-body topology.",
            "The full-scene board still shows torn multi-layer sheets instead of a coherent human volume.",
            "The route needs a canonical SMPL-X surfel/graph occupancy support, not another point-anchor loss term.",
        ],
        "must_preserve": [
            "real VGGT full-forward / world points / RGB / confidence",
            "SMPL-X feature binding",
            "no raw Kinect or teacher points at inference",
            "full-scene RGB point cloud with visible environment as mentor main evidence",
            "face-invisible claim guard",
        ],
        "next_route": "V18450 canonical SMPL-X surfel graph occupancy decoder",
        "summary": "V183 was a real Modal A10 training attempt but still failed the anti-billboard mentor gate. Continue with canonical surfel/graph occupancy rather than point-anchor shell decoding.",
    }
    write_json(REPORTS / "V18400000000000000000_route_decision.json", decision)
    route = f"""# V184 Route After Part-Pair Exclusion Decoder

Created: {created_at}

## Decision

`V18400_PART_PAIR_DECODER_FAIL_CLOSED_ROUTE_TO_CANONICAL_SURFEL_GRAPH_OCCUPANCY`

V183 is not mentor-ready and is not an external hard block.

## Why V183 Still Fails

- It is a real Modal A10 training run, not a smoke-only result.
- It adds part-pair exclusion and semantic graph heads, but the output remains a torn multi-layer sheet.
- All four cases still have billboard/combined gate failures.
- Distant-part overlap remains too high.
- The visual board does not show a stable human-main 3D point cloud.

## Required Next Architecture

The next route must switch from point-anchor shell decoding to canonical SMPL-X surfel/graph occupancy:

1. Canonical SMPL-X surfel bank with body-part graph edges.
2. VGGT feature/RGB/confidence sampling onto surfels.
3. Surfel occupancy, visibility, residual, normal, and RGB heads.
4. Body-part graph continuity and exclusion losses.
5. Full-scene insertion into real VGGT environment points.
6. Same-scene baseline/true/controls visual boards.

## Forbidden Next Moves

- No viewer-only repair.
- No thickness-only repair.
- No global point shift.
- No procedural occupancy final.
- No projection-only pass.
- No facial detail claim for the current back/side-back views.
"""
    REPORTS.joinpath("V18400000000000000000_route_state.md").write_text(route, encoding="utf-8")
    goal = f"""# V18450 Auto-Evolved Canonical SMPL-X Surfel Graph Occupancy Route

Created: {created_at}

This route continues the V13050-V600 anti-billboard topology-volume goal after V183 failed closed.

## Objective

Build a canonical SMPL-X surfel/graph occupancy student that produces a model-owned human-main full-scene RGB point cloud with partial real environment and same-scene controls.

## Required Model Route

1. Build or reuse a canonical SMPL-X surfel bank with body part, graph edges, normal, tangent, binormal, and local frame.
2. Sample real VGGT features/RGB/confidence/world-point support onto surfels.
3. Predict surfel occupancy, visibility, residual, local thickness, and RGB correction.
4. Use body-part graph continuity and distant-part exclusion losses.
5. Insert the occupied surfel result into real VGGT scene/environment points.
6. Compare against VGGT baseline, same-topology, shuffled, thickness-only, posthoc, and tiny controls.
7. Generate full-scene mentor board, controls board, turntable/cross-section board, local 3D morphology closeups, environment gate, viewer, report, bundles, and cleanup.

## Hard Gates

- No raw Kinect/teacher points at inference.
- Face detail is not applicable; claim only head/face contour and hair region.
- Metrics are auxiliary.
- If the main board is still billboard/sheet/torn, fail closed and continue.
- Do not return external hard block for visual failure.

## No Agent Rule

Do not launch agents/subagents unless the user explicitly re-authorizes them in the current turn.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    DOCS.joinpath("V18450000000000000000_auto_evolved_canonical_surfel_graph_occupancy_route.md").write_text(goal, encoding="utf-8")
    print(json.dumps({"status": decision["status"], "mentor_ready": False, "external_hard_block": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
