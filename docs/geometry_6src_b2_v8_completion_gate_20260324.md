# Geometry 6src B2 v8 Completion Gate 2026-03-24

## Goal

- Continue the post-`v7` local residual cleanup on the last two unresolved
  `Camera_B2` frames:
  - `frame 1080`
  - `frame 1170`
- Decide whether either frame now has a guard-pass custom source-set repair that
  is strong enough to be promoted into the main local frame-aware override
  manifest.
- Keep the work fully local and keep cloud closed while this remains an
  inference-only source-policy cleanup loop.

## Inputs

- prior partial `v7` gate:
  [geometry_6src_b2_v7_upgrade_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_v7_upgrade_gate_20260324.md)
- `frame 1080` search:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B2_frame1080_s1013_family_v1/summary.md)
- `frame 1170` search:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B2_frame1170_s1013_family_v1/summary.md)
- `frame 1080` controlled probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260324/B2_frame1080_stage2_s1013family_std750k_v1/summary.md)
- `frame 1170` controlled probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260324/B2_frame1170_stage2_s1013family_std750k_v1/summary.md)
- `v7` all-target sweep:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round13_6src_uniform_override_b1_b2_b4_b12_frameaware_b15_alltargets_frames0_600_1080_1170_v7/summary.md)
- `v8` all-target sweep:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round14_6src_uniform_override_b1_b2_b4_b12_frameaware_b15_alltargets_frames0_600_1080_1170_v8/summary.md)
- `v7` vs `v8` comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v7_vs_v8_alltargets_frames0_600_1080_1170/comparison.md)

## Frame-1080 Search And Probe

- search family: narrow `s1_013`-centered custom source-set search
- search size: `358` variants
- top guard-pass candidates included:
  - `s1_029`
  - `s2_263`
  - `s2_253`

Controlled probe result against `uniform`:

- `s1_029`
  - decision: `depth_unproject`
  - full depth-point MAE delta vs `uniform`: `-0.010464`
  - `fg_human` delta vs `uniform`: `-0.043022`
  - `bg_far` delta vs `uniform`: `-0.008659`
  - `bg_bottom_band` delta vs `uniform`: `-0.014670`
- `s2_263`
  - also guard-pass and strong
  - but `s1_029` was chosen for promotion because it gives the larger
    full-frame improvement while still materially improving both
    `fg_human` and `bg_bottom_band`

Accepted `frame 1080` candidate:

- override name: `s1_029_s1013_family`
- source cameras:
  `Camera_B1, Camera_B14, Camera_B4, Camera_B13, Camera_B16, Camera_B3`

## Frame-1170 Search And Probe

- search family: narrow `s1_013`-centered custom source-set search
- search size: `358` variants
- top guard-pass candidates included:
  - `s2_130`
  - `s1_032`
  - `s1_017`

Controlled probe result against `uniform`:

- `s2_130`
  - decision: `depth_unproject`
  - full depth-point MAE delta vs `uniform`: `-0.005726`
  - `fg_human` delta vs `uniform`: `-0.014884`
  - `bg_far` delta vs `uniform`: `-0.004497`
  - `bg_bottom_band` delta vs `uniform`: `-0.028361`
- `s1_032`
  - full improves
  - but `fg_human` delta vs `uniform` is `+0.008058`, so it is not safe
    enough for promotion
- `s1_017`
  - full and bottom improve
  - but `fg_human` delta vs `uniform` is `+0.005737`, so it is also not safe
    enough for promotion

Accepted `frame 1170` candidate:

- override name: `s2_130_s1013_family`
- source cameras:
  `Camera_B14, Camera_B4, Camera_B13, Camera_B23, Camera_B3, Camera_B21`

## Local Runner Fix

- During the first `v8` all-target sweep attempt, the evaluation cases failed
  because `run_zju_geometry_view_sweep.py` inherited a stale checkpoint string
  from the shell path instead of resolving the real local checkpoint.
- The runner now resolves checkpoint paths locally before dispatching
  per-case compares, using the same local detector as the custom probe chain.
- This fixes a real local issue in the `view_sweep` runner, not just this one
  experiment directory.

## v8 Promotion

- accepted manifest:
  [zju_6src_hardcontrol_hybrid_v8_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v8_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json)
- new `B2` additions beyond `v7`:
  - `Camera_B2 @ frame 1080 -> s1_029_s1013_family`
  - `Camera_B2 @ frame 1170 -> s2_130_s1013_family`

## All-Target v7 -> v8 Safety

- `v8` sweep status:
  - rows: `92`
  - failures: `0`
- `v7` all-target result:
  - `depth = 44 / 92`
  - `point = 12 / 92`
  - `tie = 36 / 92`
- `v8` all-target result:
  - `depth = 46 / 92`
  - `point = 10 / 92`
  - `tie = 36 / 92`
- `v8` minus `v7`:
  - average geometry gain delta: `+0.000176`
  - average coverage gain delta: `+0.006163`
  - `point -> depth = 2`
  - `regressed_from_depth = 0`
- changed cases:
  - `Camera_B2 @ frame 1080`: `point_map -> depth_unproject`
  - `Camera_B2 @ frame 1170`: `point_map -> depth_unproject`
- there are no non-target changes and no collateral regressions

## Decision

- The old post-`v7` `B2` residual is now closed locally.
- `Camera_B2` is no longer part of the remaining `point_map` residual set on
  this `6src / frame 0,600,1080,1170` sweep.
- The next local residual cluster is now outside `B2`, with the strongest
  remaining `point_map` cases concentrated on:
  - `frame 1170`: `Camera_B13`, `Camera_B7`, `Camera_B11`, `Camera_B16`,
    `Camera_B23`
  - `frame 0`: `Camera_B13`, `Camera_B23`
  - `frame 600`: `Camera_B11`, `Camera_B13`
  - `frame 1080`: `Camera_B23`
- Cloud remains closed:
  - this step was pure local source-policy cleanup
  - there is still no new training/ablation question selected
  - and the remaining residual cleanup is still active locally
