# V187 Auto-Evolved Spatial Anchor Route

Current status:

`V18600_PART_COVERAGE_CANONICAL_SURFEL_FAIL_CLOSED_CONTINUE`

V186 fixed the V185 part coverage failure but exposed a new failure:

```text
part coverage pass
spatial anchoring fail
visible human body not preserved
candidate becomes a floating topology-volume cloud
```

This is not mentor-ready and not an external hard block.

## Root Cause

The canonical surfel/graph decoder can fill all semantic body parts, but the occupied surfels are not sufficiently constrained by the real VGGT visible body anchors and the posed SMPL-X local frame. The model is learning coverage before it has a strong spatial attachment policy.

## Repair

Build a visible-anchor constrained canonical surfel occupancy route:

1. Keep VGGT high-confidence visible human points as no-drift anchors.
2. Use SMPL-X posed local frame as a hard spatial coordinate system.
3. Add off-body / floating-surface penalty.
4. Add visible-anchor distance loss for occupied surfels.
5. Add per-part coverage only after anchoring constraints are satisfied.
6. Keep environment from real VGGT scene points.
7. Compare against V185, V186, V173, V183, baseline, same-topology, shuffled, and thickness-only.

## Required Outputs

- `tools/V18700_visible_anchor_canonical_surfel_training.py`
- `modal_v18700_visible_anchor_canonical_surfel_training.py`
- `reports/V18700000000000000000_training_manifest.csv`
- `reports/V18700000000000000000_visible_anchor_scores.csv`
- `reports/V18700000000000000000_training_decision.json`
- `boards/V18700000000000000000_visible_anchor_board.png`
- `boards/V18700000000000000000_visible_anchor_turntable_cross_section.png`

## Gates

Fail closed if:

- true is still billboard-like;
- true is a floating volume cloud;
- true does not visually beat the VGGT baseline;
- same-topology / shuffled / thickness-only / V186 are close or stronger;
- local evidence is contour-only but written as fine detail;
- face detail is claimed despite invisible source face.

## Final Policy

Continue toward:

`V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED`

Only return:

`V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION`

for real missing assets, Modal/GPU outage, file permission failure, corrupt unrecoverable inputs, disk failure, or repo write failure.
