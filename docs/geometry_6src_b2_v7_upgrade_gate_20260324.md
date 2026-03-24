# Geometry 6src B2 v7 Upgrade Gate 2026-03-24

> Historical stage. The later `frame 1080 / 1170` continuation and final
> `v8` promotion are recorded in
> [geometry_6src_b2_v8_completion_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_v8_completion_gate_20260324.md).

## Goal

- Continue the post-`v6` local residual cleanup from the strongest remaining
  target: `Camera_B2`.
- Determine whether `B2` has a guard-pass custom source-set repair that is
  strong enough to be promoted into the main local override manifest.
- Keep the work fully local; do not open cloud from this inference-only
  source-policy search.

## Inputs

- first-pass generic policy probe:
  [geometry_6src_b2_policy_probe_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_policy_probe_20260324.md)
- frame-600 custom search:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B2_frame600_nearest_family_v2/summary.md)
- frame-600 controlled probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260324/B2_frame600_stage2_nearestfamily_std750k_v1/summary.md)
- four-frame candidate probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/b2_s1_013_probe_frames0_600_1080_1170_v1/summary.md)
- baseline vs four-frame candidate:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_b2_s1_013_frames0_600_1080_1170/comparison.md)
- `v6` vs `v7` all-target comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v6_vs_v7_alltargets_frames0_600_1080_1170/comparison.md)

## Frame-600 Custom Search

- search target: `CoreView_390 / frame 600 / Camera_B2 / 6src_hist`
- reference family: `nearest_ring`
- reference sources:
  `Camera_B1, Camera_B3, Camera_B14, Camera_B4, Camera_B13, Camera_B23`
- candidate pool:
  `Camera_B1, Camera_B3, Camera_B14, Camera_B4, Camera_B13, Camera_B23, Camera_B11, Camera_B9, Camera_B19, Camera_B21, Camera_B20, Camera_B16, Camera_B5`
- search size: `358` variants
- guard-pass count: `60`

Best guard-pass candidate from the search:

- variant: `s1_013`
- source cameras:
  `Camera_B1, Camera_B14, Camera_B4, Camera_B13, Camera_B23, Camera_B16`
- relative to the `nearest` reference:
  - decision: `depth_unproject`
  - full depth-point MAE delta: `-0.002416`
  - `fg_human` MAE delta: `+0.000930`
  - `bg_far` MAE delta: `-0.002659`
  - `bg_bottom_band` MAE delta: `-0.007788`

## Controlled Frame-600 Readout

- controlled variants:
  - `uniform`
  - `nearest`
  - `s1_013`
  - `s2_227`
  - `s2_320`

Selection result:

- `s1_013` is the most balanced local candidate for promotion testing.
- Relative to `uniform` at `frame 600`:
  - decision: `depth_unproject`
  - full depth-point MAE delta: `-0.000867`
  - `fg_human` MAE delta: `-0.010065`
  - `bg_bottom_band` MAE delta: `-0.003582`
- `s2_320` is stronger on full-frame MAE, but it does not improve
  `bg_bottom_band`.
- `nearest` remains only `tie` and keeps the bottom-band regression.

## Four-Frame Transfer Check

- probe manifest:
  [zju_b2_s1_013_probe_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_b2_s1_013_probe_frames0_600_1080_1170.json)
- result on `frame 0 / 600 / 1080 / 1170`:
  - baseline: `depth = 0, point = 4, tie = 0`
  - candidate: `depth = 3, point = 0, tie = 1`
  - average geometry gain delta vs baseline: `+0.001643`
  - average coverage gain delta vs baseline: `+0.064800`

Per-frame readout:

- `frame 0`: `point -> depth`
- `frame 600`: `point -> depth`
- `frame 1080`: `point -> tie`
- `frame 1170`: `point -> depth`

## Guard Decision

- `s1_013` is good enough to promote partially, but not across all four frames.
- Safe promotion signals:
  - `frame 0`: full improves, `bg_bottom_band` improves, `fg_human` delta stays
    within the local `0.003` guard
  - `frame 600`: full improves, `bg_bottom_band` improves, `fg_human` improves
- Not safe enough for full four-frame rollout:
  - `frame 1080`: only `tie`, and `bg_bottom_band` slightly regresses
  - `frame 1170`: full and bottom improve, but `fg_human` MAE delta exceeds the
    local `0.003` guard

## v7 Partial Upgrade

- accepted manifest:
  [zju_6src_hardcontrol_hybrid_v7_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v7_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json)
- added overrides:
  - `Camera_B2 @ frame 0`
  - `Camera_B2 @ frame 600`
- kept on baseline for now:
  - `Camera_B2 @ frame 1080`
  - `Camera_B2 @ frame 1170`

## All-Target v6 -> v7 Safety

- `v6` all-target result:
  - `depth = 42 / 92`
  - `point = 14 / 92`
  - `tie = 36 / 92`
- `v7` all-target result:
  - `depth = 44 / 92`
  - `point = 12 / 92`
  - `tie = 36 / 92`
- `v7` minus `v6`:
  - average geometry gain delta: `+0.000036`
  - average coverage gain delta: `+0.002371`
  - `point -> depth = 2`
  - `regressed_from_depth = 0`
- changed cases:
  - `Camera_B2 @ frame 0`: `point -> depth`
  - `Camera_B2 @ frame 600`: `point -> depth`
- no non-target changes and no collateral regressions

## Decision

- `B2` is no longer a pure unresolved `4 / 4 point` residual.
- A safe partial `v7` upgrade is now accepted locally for `frame 0 / 600`.
- The remaining `B2` residual is narrowed to:
  - `frame 1080`
  - `frame 1170`
- The next local residual question should start from those two remaining `B2`
  frames, then widen to `B13 / B23 / B11` only if they become the stronger
  unresolved cluster.
- `frame 1080 / B12 @ frame 600` no longer block cloud on their own.
- Whether to start cloud again now depends only on a new training/ablation
  question; as long as the work remains inference-only source-policy cleanup,
  keep it local.
