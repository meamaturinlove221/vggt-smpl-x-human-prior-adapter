# V180 Adjacency-Aware Collision Route

Created: 2026-05-28T20:42:35+00:00

## Decision

Status: `V18000_ADJACENCY_AWARE_COLLISION_METRIC_FAIL_CLOSED_CONTINUE`

V180 is diagnostic only. It does not make the route mentor-ready because the hard mentor visual gate still requires a human-main full-scene RGB point cloud that visibly beats VGGT baseline and hard controls.

## What Changed

- V178 treated part bbox overlap too coarsely.
- V179 tried direct part separation and harmed at least one case.
- V180 separates valid adjacent contact from invalid distant part overlap.
- Adjacent edges, such as head-torso, torso-arm, torso-leg, and leg-foot, are allowed to touch.
- Semantically distant pairs, such as head-foot, arm-leg, left-right endpoints, and torso-foot, receive the main collision penalty.

## Current Result

- Mentor ready: `false`
- External hard block: `false`
- Failure count: `12`

## Next Route

If V180 still fails, the next route is not a viewer/thickness repair. It should train an adjacency-aware topology loss:

1. Preserve V173 multi-shell output as the current best source candidate.
2. Add valid-edge contact loss for body topology adjacency.
3. Add invalid-pair separation loss for semantically distant part overlaps.
4. Add cross-part occupancy exclusion for head/foot, arm/leg, left/right endpoints, and torso/foot.
5. Keep projection as auxiliary only.
6. Fail closed unless V138-style full-scene and same-scene controls visibly pass.
