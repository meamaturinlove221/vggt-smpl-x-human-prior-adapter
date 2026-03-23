# ZJU Geometry View Sweep Round 2

This note records the second sparse-view sweep after round 1 had shown an important limitation:

- the mentor's `depth + camera -> unproject -> render` direction was clearly real
- but the sparse `6src_hist` and `12src_nested` profiles were still too dependent on the old fixed source subset from the original `Camera_B5` case

Round 2 keeps the model and render pipeline unchanged and changes only the sparse source-selection policy.

## Setup

Sequence:

- `CoreView_390`

Frames:

- `0`
- `150`
- `300`
- `450`
- `600`
- `750`
- `900`
- `1080`
- `1170`

Profiles:

- `6src_hist`
- `12src_nested`

What changed versus round 1:

- round 1 reused the old fixed sparse source subset from the historical `Camera_B5` report
- round 2 uses `source_policy=rotate_template_offsets`
- that policy first maps the template sparse subset onto the real ZJU camera ring order
- then rotates the same relative offset pattern to the current target camera

What did **not** change:

- original VGGT checkpoint
- same `point_map` vs `depth_unproject` render comparison
- same target image and mask
- same MAE / coverage based decision rule

Artifacts:

- [round2 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.md)
- [round2 json](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.json)
- [round2 csv](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.csv)
- [round1 vs round2 comparison](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_vs_round2_targetaware_v1/comparison.md)

## Round-2 Overall Readout

Across all `414` sparse target-aware cases:

- `depth_unproject` wins: `76`
- `point_map` wins: `226`
- ties: `112`

By profile:

- `6src_hist`: `62` depth wins / `83` point wins / `62` ties
- `12src_nested`: `14` depth wins / `143` point wins / `50` ties

Average gain summary:

- `6src_hist`: average geometry gain `-0.0008`, average coverage gain `+0.0218`
- `12src_nested`: average geometry gain `-0.0009`, average coverage gain `-0.0297`

So round 2 is not a universal sparse-view win, but it is also not a null result.

## What Improved Relative To Round 1

On the common sparse cases that exist in both rounds:

- common cases: `252`
- round 1 depth wins: `32`
- round 2 depth wins: `55`
- improved-to-depth transitions: `49`
- regressed-from-depth transitions: `26`

Most important split:

- `6src_hist`: depth wins improved from `28 -> 46` on the same `153` common cases
- `6src_hist`: average geometry gain delta `+0.000896`
- `6src_hist`: average coverage gain delta `+0.024030`

This is the clearest sign that the fixed sparse subset itself was part of the problem.

For `12src_nested`:

- depth wins improved from `4 -> 9` on the same `99` common cases
- but average geometry gain delta is still `-0.000598`
- and average coverage gain delta is still `-0.002107`

That means a target-aware rotation of the old 12-view template is **not** enough to rescue that profile.

## Frame-Level Pattern

For `6src_hist`, the target-aware policy improved the geometry-chain support on almost every frame of the common-case comparison:

- `0`: depth wins `3 -> 4`
- `450`: depth wins `3 -> 5`
- `600`: depth wins `4 -> 6`
- `750`: depth wins `3 -> 6`
- `900`: depth wins `2 -> 5`
- `1080`: depth wins `5 -> 7`
- `1170`: depth wins `1 -> 5`

This is especially useful because `1170` was one of the hardest sparse regions in round 1.

For `12src_nested`, the changes are much weaker and less consistent:

- only small depth-win increases on some frames
- geometry gain often remains negative
- the tail frame `1170` is still poor

## Interpretation

Round 2 gives a more precise answer to the mentor's question.

The answer is now:

- yes, the render-point-source question was worth testing
- yes, sparse failures were partly caused by the old fixed source subset
- but that explanation is strong mainly for `6src_hist`, not for `12src_nested`

So the geometry-chain status after round 2 is:

- `23cam_fullset` from round 1 is still the strongest evidence for `depth + camera`
- `6src_hist` becomes noticeably more supportive once the source subset follows the target camera
- `12src_nested` still does not behave like a robust geometry-friendly sparse profile

## Current Recommendation

1. Keep the project on original VGGT and keep the geometry-chain direction.
2. Do not restore the old ghost stack.
3. For sparse inference-side checks, prefer `6src_hist` with target-aware source selection over the old fixed subset.
4. Do not treat `12src_nested` as a reliable sparse baseline yet.
5. The next sparse-view experiment should focus on a better 12-view source policy or on sparse-view geometry diagnostics, not on image-side ghost losses.
