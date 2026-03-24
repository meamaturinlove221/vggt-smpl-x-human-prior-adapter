# Geometry 6src Frame-Aware v6 Multiframe Transfer Gate 2026-03-24

## Goal

- Close the broader local multi-frame transfer question that remained open after
  the earlier `v5` carry-over manifest.
- Keep the work fully local.
- This closes the old `frame 1080 / B12 @ frame 600` local blocker; any future
  cloud run should depend on a new training/ablation question, not on this
  inference-only source-policy gate.

## Follow-Up Inputs

- prior `v5` multiframe gate:
  [geometry_6src_hybrid_v5_multiframe_local_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_v5_multiframe_local_gate_20260324.md)
- `B12 @ frame 600` nearest-ring probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/policy_probe_b12_frame600_nearest_v1/summary.md)
- `B12 @ frame 600` broader nearest-family search:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B12_frame600_nearest_family_v1/summary.md)
- direct controlled probe for the winning frame-aware variant:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260324/B12_frame600_stage2_nearestfamily_std750k/summary.md)
- new manifest:
  [zju_6src_hardcontrol_hybrid_v6_b1_b4_b12_frameaware_b15_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v6_b1_b4_b12_frameaware_b15_frames0_600_1080_1170.json)

## Frame-Aware Fix Applied

- The only unresolved `v5` residual was `Camera_B12 @ frame 600`.
- The earlier `uniform/rotate/hybrid` family was too narrow for that case.
- A broader nearest-family search ran `262` variants and found `45`
  guard-pass `depth_unproject` candidates.
- The selected frame-aware replacement is `s2_231`:
  - target: `Camera_B12`
  - frame: `600`
  - sources:
    `Camera_B11, Camera_B15, Camera_B16, Camera_B14, Camera_B3, Camera_B5`
- Direct probe readout for that variant against the plain `uniform` reference:
  - decision: `depth_unproject`
  - full depth-point MAE delta: `-0.004803`
  - full depth-point coverage delta: `+0.080619`
  - `fg_human` MAE delta: `-0.001975`
  - `bg_far` MAE delta: `-0.004638`
  - `bg_bottom_band` MAE delta: `-0.014572`

## Targeted 16-Case Transfer Readout

- baseline summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round10_6src_uniform_b1_b4_b12_b15_frames0_600_1080_1170_v1/summary.md)
- `v5` summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round10_6src_uniform_override_b1_b4_b12_b15_frames0_600_1080_1170_v5/summary.md)
- `v6` summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round11_6src_uniform_override_b1_b4_b12_frameaware_b15_frames0_600_1080_1170_v6/summary.md)
- baseline vs `v6` comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_v6_b1_b4_b12_b15_frames0_600_1080_1170/comparison.md)
- `v5` vs `v6` comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v5_vs_v6_b1_b4_b12_b15_frames0_600_1080_1170/comparison.md)

## Targeted Result

- baseline:
  - `depth_unproject = 0 / 16`
  - `point_map = 14 / 16`
  - `tie = 2 / 16`
- `v5`:
  - `depth_unproject = 7 / 16`
  - `point_map = 1 / 16`
  - `tie = 8 / 16`
- `v6`:
  - `depth_unproject = 8 / 16`
  - `point_map = 0 / 16`
  - `tie = 8 / 16`

Readout:

- `v6` removes the last explicit `point_map` residual in the targeted
  multi-frame slice.
- Relative to `v5`, `v6` adds exactly one more `point -> depth` flip:
  - `Camera_B12 @ frame 600`
- `v6` minus `v5`:
  - average geometry gain delta: `+0.000173`
  - average coverage gain delta: `+0.002013`
  - `improved_to_depth = 1`
  - `regressed_from_depth = 0`

## All-Target 92-Case Rollout Safety

- baseline summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round12_6src_uniform_alltargets_frames0_600_1080_1170_v1/summary.md)
- `v6` summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round12_6src_uniform_override_b1_b4_b12_frameaware_b15_alltargets_frames0_600_1080_1170_v6/summary.md)
- comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_v6_alltargets_frames0_600_1080_1170/comparison.md)

## All-Target Result

- baseline:
  - `depth_unproject = 34 / 92`
  - `point_map = 28 / 92`
  - `tie = 30 / 92`
  - average depth-point geometry gain: `-0.000530`
  - average depth-point coverage gain: `+0.051624`
- `v6`:
  - `depth_unproject = 42 / 92`
  - `point_map = 14 / 92`
  - `tie = 36 / 92`
  - average depth-point geometry gain: `-0.000018`
  - average depth-point coverage gain: `+0.065943`
- `v6` minus baseline:
  - average geometry gain delta: `+0.000512`
  - average coverage gain delta: `+0.014319`
  - `point -> depth = 8`
  - `point -> tie = 6`
  - `regressed_from_depth = 0`
  - `tie -> point = 0`

## Scope Check

- The only changed cases are the intended `16` override cases:
  - `Camera_B1, Camera_B4, Camera_B12, Camera_B15`
  - across `frame 0 / 600 / 1080 / 1170`
- No non-target camera changes at all.
- The changed-case breakdown is:
  - `Camera_B12`: `4 point -> depth`
  - `Camera_B15`: `2 point -> depth`, `2 point -> tie`
  - `Camera_B1`: `1 point -> depth`, `3 point -> tie`
  - `Camera_B4`: `1 point -> depth`, `3 tie -> tie` with higher coverage and
    no `point_map` regression

## Residual After v6

- `v6` closes the repaired hard/control multi-frame slice, but it does not make
  the whole `6src` world uniformly depth-favored.
- Remaining all-target `point_map` cases after `v6`: `14`
- Frame distribution after `v6`:
  - `frame 0`: `8 depth / 3 point / 12 tie`
  - `frame 600`: `9 depth / 3 point / 11 tie`
  - `frame 1080`: `15 depth / 2 point / 6 tie`
  - `frame 1170`: `10 depth / 6 point / 7 tie`
- The strongest residual target cluster is now:
  - `Camera_B2`: `4 / 4 point`
  - `Camera_B13`: `3 / 4 point`
  - `Camera_B23`: `3 / 4 point`
  - `Camera_B11`: `2 / 4 point`
  - plus narrower `frame 1170` residuals such as `Camera_B7` and `Camera_B16`

## Decision

- The broader local multi-frame transfer question for the repaired
  `B1 / B4 / B12 / B15` slice is now closed.
- The old `B12 @ frame 600` blocker is no longer active.
- This is still a local-only result.
- This result does not itself call for any cloud launch.
- Whether to go cloud next now depends only on a new training/ablation
  hypothesis; as long as the follow-up remains inference-only source-policy
  cleanup, keep it local.
- The next local source-policy diagnosis should start from the post-`v6`
  residual cluster, especially the `frame 1170` and `B2 / B13 / B23 / B11`
  cases, not from the already-closed `B12 @ frame 600` path.
