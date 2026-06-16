# V187 Spatial Anchor Route Decision

## Conclusion

V186 is a real Modal A10 training run, but it remains fail-closed.

It fixed the immediate V185 failure mode:

- V185 decoded only partial semantic support, with `part_presence` around 0.5.
- V186 uses part-balanced surfel selection, coverage loss, and per-part decode quotas.
- V186 reaches `part_presence_score=1.0` for the four cases.

But the mentor visual gate still fails. The V186 board shows a floating part-balanced volume cloud, not a stable human-main full-scene RGB point cloud. The candidate has more coverage, but it is not sufficiently anchored to the visible VGGT body, posed SMPL-X frame, and same-scene environment.

## Evidence

- `reports/V18600000000000000000_runtime_environment.json`
- `reports/V18600000000000000000_training_decision.json`
- `reports/V18600000000000000000_part_coverage_canonical_surfel_scores.csv`
- `boards/V18600000000000000000_part_coverage_canonical_surfel_board.png`
- `boards/V18600000000000000000_part_coverage_canonical_surfel_turntable_cross_section.png`

## Failure Register

- V186 is not mentor-ready.
- V186 is not an external hard block.
- Part coverage is fixed, but spatial anchoring fails.
- The result is closer to a floating topology-volume cloud than a readable human body.
- The full-scene mentor visual still does not clearly beat the VGGT baseline.
- The route must not return as `review-ready`, `limitation disclosed`, or `route exhausted`.

## Next Route

V187 must shift from `part coverage` to `visible-anchor constrained canonical surfel occupancy`.

Required repairs:

- preserve high-confidence VGGT visible body anchors;
- constrain canonical surfel occupancy to posed SMPL-X local frames;
- attach part-balanced shells to visible RGB/body evidence instead of filling surfel coverage globally;
- penalize floating/off-body occupied surfels;
- keep the full-scene RGB point cloud as the mentor main evidence;
- keep projection, metrics, coverage, and thickness auxiliary only.

Allowed claims remain limited to head/face contour and hair region because the source face is not visible.

## Hard Fail

If V187 produces a covered but floating body cloud, it must fail closed and route to representation repair. It must not be called mentor-ready.
