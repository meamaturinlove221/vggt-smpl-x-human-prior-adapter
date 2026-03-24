# Geometry 6src Hybrid v5 Multiframe Local Gate 2026-03-24

> Historical note: this `v5`-only multiframe gate was later completed by the
> frame-aware `v6` follow-up in
> [geometry_6src_frameaware_v6_multiframe_transfer_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_frameaware_v6_multiframe_transfer_gate_20260324.md).

## Goal

- Extend the repaired local `6src hybrid v5` source-policy readout beyond the
  earlier `frame 1080` rollout-safety slice.
- Keep the work fully local.
- Keep cloud blocked until the broader transfer question is either fixed or
  clearly bounded.

## Local Fix Applied

- [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py)
  now accepts override manifests with `frame_id = 0`.
- The previous loader treated `0` as falsy and rejected valid ZJU frame-zero
  override cases.
- This bug is now fixed and was verified by successfully running the new
  multi-frame override sweep on `frame 0, 600, 1080, 1170`.

## Multi-Frame Transfer Scope

- target cameras: `Camera_B1, Camera_B4, Camera_B12, Camera_B15`
- frames: `0, 600, 1080, 1170`
- profile: `6src_hist`
- baseline summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round10_6src_uniform_b1_b4_b12_b15_frames0_600_1080_1170_v1/summary.md)
- candidate summary:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round10_6src_uniform_override_b1_b4_b12_b15_frames0_600_1080_1170_v5/summary.md)
- comparison:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_v5_b1_b4_b12_b15_frames0_600_1080_1170/comparison.md)

## Overall Result

- baseline:
  - `depth_unproject = 0 / 16`
  - `point_map = 14 / 16`
  - `tie = 2 / 16`
  - average depth-point geometry gain: `-0.002419`
  - average depth-point coverage gain: `-0.024763`
- candidate:
  - `depth_unproject = 7 / 16`
  - `point_map = 1 / 16`
  - `tie = 8 / 16`
  - average depth-point geometry gain: `+0.000355`
  - average depth-point coverage gain: `+0.055556`
- candidate minus baseline:
  - average geometry gain delta: `+0.002774`
  - average coverage gain delta: `+0.080319`
  - `point -> depth = 7`
  - `point -> tie = 6`
  - `regressed_from_depth = 0`

## By Frame

- `frame 0`
  - baseline: `3 point + 1 tie`
  - candidate: `1 depth + 3 tie`
  - average geometry gain delta: `+0.002063`
  - average coverage gain delta: `+0.050370`
- `frame 600`
  - baseline: `3 point + 1 tie`
  - candidate: `1 point + 3 tie`
  - average geometry gain delta: `+0.001884`
  - average coverage gain delta: `+0.045038`
- `frame 1080`
  - baseline: `4 point`
  - candidate: `4 depth`
  - average geometry gain delta: `+0.003836`
  - average coverage gain delta: `+0.068562`
- `frame 1170`
  - baseline: `4 point`
  - candidate: `2 depth + 2 tie`
  - average geometry gain delta: `+0.003312`
  - average coverage gain delta: `+0.157306`

## By Target

- `Camera_B12`
  - candidate outcomes: `3 depth + 1 point`
  - strongest cross-frame transfer among the four repaired cameras
- `Camera_B15`
  - candidate outcomes: `2 depth + 2 tie`
- `Camera_B1`
  - candidate outcomes: `1 depth + 3 tie`
- `Camera_B4`
  - candidate outcomes: `1 depth + 3 tie`

## Residual Case: `B12 @ frame 600`

- focused narrow probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260324/B12_frame600_stage1_std750k/summary.md)
- bounded family search:
  [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B12_frame600_uniform_family_v1/summary.md)

Readout:

- The carried-over `hybrid_b23_b8` candidate is still the best result among the
  previously known `B12` local variants at `frame 600`, but it remains
  `point_map`.
- The bounded `1-2 swap` search over the controlled
  `uniform/rotate/hybrid` family ran `64` variants.
- Decision counts in that search:
  - `depth_unproject = 0`
  - `tie = 0`
  - `point_map = 64`
- So the `frame 600 / Camera_B12` residual is real: it is not just one missed
  narrow swap inside the current source-family pool.

## Decision

- The local `v5` source-policy repair is now clearly stronger than plain
  `uniform_ring` across the tested multi-frame slice.
- But it is **not** yet a full multi-frame closure:
  - `frame 1080` is fully repaired
  - `frame 1170` is mostly repaired
  - `frame 0` and especially `frame 600` still stop at `tie` or `point_map`
- Under the current rule, cloud remains blocked.
- The next local-only step should be one of:
  - broaden the `frame 600` candidate pool beyond the current
    `uniform/rotate/hybrid` family
  - or accept that the current `v5` manifest is a strong `frame 1080`
    source-policy repair, not yet a general multi-frame fix
