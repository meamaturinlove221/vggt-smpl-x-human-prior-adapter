# V18900000000000000000 Non-Fallback Visible Anchor Failure Route

Created: 2026-05-28T22:22:06+00:00

## Conclusion

V188 restored the missing V950/V536/V161/V140 assets and V187 was rerun on Modal A10 without feature-bank fallback.

This removes the previous asset-restoration uncertainty, but it does not create mentor-ready evidence.

## Current Evidence

- Modal A10 run confirmed: `True`
- feature_bank_fallback_used: `False`
- restored four-case V950/V536 assets: `True`
- V187 true combined_fail_v4 cases: `3/4`
- close/better controls: `0012_11_frame001::part_coverage_canonical_surfel_true, 0013_01_frame001::part_coverage_canonical_surfel_true, 0021_03_frame001::part_coverage_canonical_surfel_true, current_v895_0021_03::part_coverage_canonical_surfel_true`
- advisor board: `D:\vggt\vggt-canonical-surfel-adapter\boards\V18700000000000000000_visible_anchor_board.png`
- turntable/cross-section board: `D:\vggt\vggt-canonical-surfel-adapter\boards\V18700000000000000000_visible_anchor_turntable_cross_section.png`

## Failure Interpretation

The current failure is no longer a missing-asset hard block. The non-fallback visible-anchor student still produces a torn / tilted volume cloud rather than a natural human-main full-scene RGB point cloud.

The training history also shows shell / coverage / occupancy-completeness terms are effectively inactive in the final checkpoints for these cases. This means the model is optimizing anchor proximity without learning a posed, body-frame topology-volume occupancy structure.

## Next Route

Continue with anchored pose-frame occupancy repair:

1. Decode occupancy in posed SMPL local frames rather than free visible anchors.
2. Make front/back/side shell, part continuity, and cross-section occupancy active training objectives.
3. Add a full-scene upright pose-frame alignment gate before mentor rendering.
4. Keep real VGGT environment insertion, same-scene controls, and face-detail overclaim guards.
5. Fail closed if the main board is still billboard / torn cloud / tilted shell.

## Forbidden Returns

- mentor-ready from V187;
- fallback/surrogate evidence;
- render-only pass;
- thickness-only pass;
- projection-only pass;
- route exhausted;
- visual failure as external hard block.
