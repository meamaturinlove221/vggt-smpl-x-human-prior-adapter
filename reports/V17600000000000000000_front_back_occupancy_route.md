# V176 Front/Back Occupancy Route

## Conclusion

V173 Modal A10 multi-shell decoding is the best current candidate, but it still fails closed. V175 layer-balanced resampling did not solve the route; it reduced the accidental shell-drop issue but also weakened the strongest V173 scores.

Current route state:

- V173 true beats VGGT baseline on all four cases.
- V173 true beats same-topology / shuffled / thickness-only on 3/4 cases.
- 0013_01_frame001 still loses to same-topology and shuffled.
- `billboard_fail_v2` remains true for all cases.
- Face detail remains not applicable; allowed claim is only head/face contour and hair region.

## Failure Diagnosis

The remaining metric blocker is no longer raw thickness. V173 already improves `pca_thickness_ratio`, `multi_layer_section_ratio`, and `dense_section_ratio`.

The weak part is `front_back_separation_ratio`:

- 0012_11 V173 true: low front/back separation despite high overall score.
- 0013_01 V173 true: same-topology and shuffled have stronger front/back or dense section structure.
- V175 layer balancing did not consistently improve this, so the problem is not just decoded point budget trimming.

## Required Next Repair

Do not continue thickness-only or shell-count-only repair.

The next model/decoder must make front/back occupancy explicit:

1. Predict front/back layer assignment per anchor, not only shell coordinates.
2. Add a differentiable thin-axis extreme-bin occupancy loss.
3. Preserve part continuity while forcing both front and back layers to survive final point-budget sampling.
4. Compare against same-topology and shuffled in-batch, especially on 0013_01.
5. Select checkpoints only if visual boards and V140/V174-style gates agree.

## Forbidden Return

Do not return mentor-ready from V173/V175. Both remain checkpoints.

Do not return external hard block. Modal is available and V173 A10 ran successfully.

Continue with V176/V177 front-back occupancy training or an equivalent topology-volume representation repair.
