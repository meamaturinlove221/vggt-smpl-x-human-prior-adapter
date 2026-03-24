# Geometry 6src B2 Policy Probe 2026-03-24

> Historical note: this generic-policy probe was later followed by the custom
> search and partial `v7` upgrade in
> [geometry_6src_b2_v7_upgrade_gate_20260324.md](/f:/vggt/vggt-main/docs/geometry_6src_b2_v7_upgrade_gate_20260324.md).

## Goal

- Start the next local residual diagnosis from the strongest post-`v6`
  unresolved target: `Camera_B2`.
- Test whether this residual is still cheaply recoverable through an existing
  policy family before opening a more expensive custom source-set search.
- Keep the work fully local; this inference-only probe does not justify any
  cloud action by itself.

## Inputs

- baseline multiframe all-target sweep:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round12_6src_uniform_alltargets_frames0_600_1080_1170_v1/summary.md)
- `nearest_ring` probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/policy_probe_b2_frames0_600_1080_1170_nearest_v1/summary.md)
- `rotate_template_offsets` probe:
  [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/policy_probe_b2_frames0_600_1080_1170_rotate_v1/summary.md)
- baseline vs `nearest_ring`:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_b2_nearest_frames0_600_1080_1170/comparison.md)
- baseline vs `rotate_template_offsets`:
  [comparison.md](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/uniform_vs_b2_rotate_frames0_600_1080_1170/comparison.md)

## Baseline

- Under the current post-`v6` `uniform_ring` readout, `Camera_B2` remains
  `point_map = 4 / 4` on `frame 0 / 600 / 1080 / 1170`.
- Average depth-point geometry gain: `-0.000865`
- Average depth-point coverage gain: `-0.022921`

## Nearest-Ring Readout

- `nearest_ring` moves the four-case `B2` slice to:
  - `depth_unproject = 0 / 4`
  - `point_map = 2 / 4`
  - `tie = 2 / 4`
- Relative to baseline:
  - average geometry gain delta: `-0.003112`
  - average coverage gain delta: `+0.028769`
  - `point -> tie = 2`
  - `point -> depth = 0`
- Frame-level readout:
  - `frame 0`: `point -> tie`
  - `frame 600`: `point -> tie`
  - `frame 1080`: stays `point_map`, and geometry worsens
  - `frame 1170`: stays `point_map`

## Rotate-Template Readout

- `rotate_template_offsets` gives:
  - `depth_unproject = 0 / 4`
  - `point_map = 4 / 4`
  - `tie = 0 / 4`
- Relative to baseline:
  - average geometry gain delta: `-0.001062`
  - average coverage gain delta: `-0.001761`
  - no favorable decision flips at all

## Decision

- `Camera_B2` is still source-policy-sensitive, but not in the same easy way as
  the repaired `B12 @ frame 600` case.
- Off-the-shelf `nearest_ring` only buys coverage and only reaches `tie` on
  `frame 0 / 600`; it does not produce any `depth_unproject` win.
- `rotate_template_offsets` is not a useful next path for `B2`.
- So the next local step for `B2` should be a targeted custom source-set search,
  not another generic policy family sweep.
