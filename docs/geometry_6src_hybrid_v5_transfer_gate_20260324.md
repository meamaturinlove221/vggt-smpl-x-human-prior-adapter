# Geometry 6src Hybrid v5 Transfer Gate 2026-03-24

## Goal

- Validate the repaired local `6src hybrid v5` source-policy batch beyond the
  original hard/control set, without opening any cloud run.
- Use the same `CoreView_390 / frame 1080 / 6src_hist` sweep pipeline on all
  `23` target cameras.

## Inputs

- baseline all-target sweep:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round9_6src_uniform_alltargets_frame1080_v1/summary.md)
- candidate all-target sweep:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round9_6src_uniform_override_b1_b4_b12_b15_alltargets_frame1080_v5/summary.md)
- direct comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_v5_alltargets_frame1080/comparison.md)

## Result

- baseline `uniform_ring`:
  - `depth_unproject = 11 / 23`
  - `point_map = 6 / 23`
  - `tie = 6 / 23`
  - average depth-point geometry gain: `-0.000117`
  - average depth-point coverage gain: `+0.062199`
- candidate `uniform_ring + v5 overrides`:
  - `depth_unproject = 15 / 23`
  - `point_map = 2 / 23`
  - `tie = 6 / 23`
  - average depth-point geometry gain: `+0.000550`
  - average depth-point coverage gain: `+0.074123`
- candidate minus baseline:
  - average geometry gain delta: `+0.000667`
  - average coverage gain delta: `+0.011924`
  - improved to depth: `4`
  - regressed from depth: `0`

## Changed Cases

- Only `4` cases changed, and they are exactly the override targets:
  - `Camera_B1`: `point_map -> depth_unproject`
    - geometry gain delta: `+0.002195`
    - coverage gain delta: `+0.052970`
  - `Camera_B4`: `point_map -> depth_unproject`
    - geometry gain delta: `+0.001823`
    - coverage gain delta: `+0.076355`
  - `Camera_B12`: `point_map -> depth_unproject`
    - geometry gain delta: `+0.002008`
    - coverage gain delta: `+0.055664`
  - `Camera_B15`: `point_map -> depth_unproject`
    - geometry gain delta: `+0.009317`
    - coverage gain delta: `+0.089258`
- The remaining `19` cameras are numerically unchanged:
  - average geometry gain delta: `0.000000`
  - average coverage gain delta: `0.000000`

## Decision

- The local `6src hybrid v5` batch now passes the all-target rollout-safety
  gate for `frame 1080`.
- This does **not** mean the policy magically generalized to every camera by
  learning a new rule; it means the targeted override manifest repaired the
  four failing cameras and introduced no collateral regressions elsewhere.
- So the local blocker is now cleared at three levels:
  - hard/control current-current sweep
  - strict same-source legacy-native backfill
  - all-target rollout-safety sweep
- Cloud should still remain off until the next experiment axis is chosen
  explicitly.
