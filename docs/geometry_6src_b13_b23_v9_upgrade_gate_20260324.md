# Geometry 6src B13 B23 V9 Upgrade Gate 2026-03-24

## Goal

- Freeze `v8` as the current local frame-aware baseline.
- Check whether the strongest post-`v8` `frame 1170` residuals,
  `Camera_B13` and `Camera_B23`, still have a local
  inference/source-policy repair.
- Only promote a new override if it improves the target cases locally and
  survives an all-target rollout-safety sweep.

## Search Readout

- `B13 @ frame 1170` one-swap `uniform + nearest` search:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B13_frame1170_uniform_nearest_family_v1/summary.md)
  - best candidate for the target frame:
    - `s1_021 = swap Camera_B1 -> Camera_B15`
  - target-frame delta vs `v8` baseline:
    - full depth-point MAE delta: `-0.007289`
    - `fg_human` delta: `-0.009684`
    - `bg_far` delta: `-0.007401`
    - `bg_bottom_band` delta: `-0.000208`
- `B23 @ frame 1170` one-swap `uniform + nearest` search:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B23_frame1170_uniform_nearest_family_v1/summary.md)
  - best promoted candidate for the target frame:
    - `s1_016 = swap Camera_B9 -> Camera_B21`
  - target-frame delta vs `v8` baseline:
    - full depth-point MAE delta: `-0.003310`
    - `fg_human` delta: `-0.003699`
    - `bg_far` delta: `-0.003408`
    - `bg_bottom_band` delta: `-0.001020`

## Controlled Local Contrast

- `B13` four-frame probe:
  - [s1_005 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B13_s1_005_frames0_600_1080_1170_v1/summary.md)
  - [s1_021 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B13_s1_021_frames0_600_1080_1170_v1/summary.md)
  - comparison against `v8`:
    - [v8 vs s1_005](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v8_vs_b13_s1_005_frames0_600_1080_1170/comparison.md)
    - [v8 vs s1_021](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v8_vs_b13_s1_021_frames0_600_1080_1170/comparison.md)
  - readout:
    - `s1_021` is better than `s1_005`
    - it keeps `frame 1080` as `depth_unproject`
    - it flips `frame 1170` from `point_map -> depth_unproject`
    - it softens `frame 0` to `tie`
    - but it does not cleanly transfer to all `B13` frames
- `B23` four-frame probe:
  - [s1_016 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B23_s1_016_frames0_600_1080_1170_v1/summary.md)
  - [s1_008 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B23_s1_008_frames0_600_1080_1170_v1/summary.md)
  - comparison against `v8`:
    - [v8 vs s1_016](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v8_vs_b23_s1_016_frames0_600_1080_1170/comparison.md)
    - [v8 vs s1_008](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v8_vs_b23_s1_008_frames0_600_1080_1170/comparison.md)
  - readout:
    - `s1_016` is the better local `B23` family
    - it flips `frame 1170` from `point_map -> depth_unproject`
    - it softens `frame 0` to `tie`
    - but it does not cleanly transfer to every `B23` frame either

## Promotion Decision

- I did not promote a broad all-frame `B13` or `B23` family.
- I did promote two narrow frame-aware repairs for `frame 1170` only:
  - `Camera_B13 @ frame 1170 -> swap Camera_B1 -> Camera_B15`
  - `Camera_B23 @ frame 1170 -> swap Camera_B9 -> Camera_B21`
- New manifest:
  - [zju_6src_hardcontrol_hybrid_v9_b1_b2_b4_b12_frameaware_b13_b15_b23_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v9_b1_b2_b4_b12_frameaware_b13_b15_b23_frames0_600_1080_1170.json)

## All-Target Safety

- Local all-target `v9` sweep:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round15_6src_uniform_override_b1_b2_b4_b12_frameaware_b13_b15_b23_alltargets_frames0_600_1080_1170_v9/summary.md)
- `v8 -> v9` comparison:
  - [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v8_vs_v9_alltargets_frames0_600_1080_1170/comparison.md)
- key readout:
  - all-target decisions move from `depth = 46, point = 10, tie = 36`
    to `depth = 48, point = 8, tie = 36`
  - average geometry gain moves from `0.000194 -> 0.000309`
  - average coverage gain moves from `0.074477 -> 0.077294`
  - `improved_to_depth = 2`
  - `regressed_from_depth = 0`
  - only two cases change:
    - `Camera_B13 @ frame 1170: point_map -> depth_unproject`
    - `Camera_B23 @ frame 1170: point_map -> depth_unproject`

## Current Residuals

- `v9` reduces the remaining `point_map` residuals from `10` to `8`.
- Current local residual list:
  - `frame 0`: `Camera_B13`, `Camera_B23`
  - `frame 600`: `Camera_B11`, `Camera_B13`
  - `frame 1080`: `Camera_B23`
  - `frame 1170`: `Camera_B11`, `Camera_B16`, `Camera_B7`

## Decision

- `v9` is now the current local frame-aware main manifest.
- Cloud stays closed.
- The next local residual axis is no longer `B13/B23 @ frame 1170`.
- The next best local target is the post-`v9` residual cluster, especially:
  - `frame 1170 / Camera_B11`
  - then the repeated `Camera_B13` and `Camera_B23` residuals on the other frames
