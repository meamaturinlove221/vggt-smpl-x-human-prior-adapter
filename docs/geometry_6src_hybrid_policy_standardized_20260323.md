# 6src Hybrid Policy Standardized Status 2026-03-23

## Goal

This note replaces the first `6src` custom source-swap readout as the
decision-grade artifact for the local-only sparse-policy follow-up.

Its purpose is simple:

- standardize the earlier probe to the same render setting used by the normal
  compare/sweep path
- keep only the source-policy leads that still survive that standardization
- keep cloud off until the local result is actually stable

## Local Fix Applied

The original
[run_zju_custom_source_set_region_probe.py](/f:/vggt/vggt-main/scripts/run_zju_custom_source_set_region_probe.py)
used `render_max_points = 500000` by default.

The normal compare/sweep path used `750000`.

That mismatch was a local apples-to-oranges bug. It has now been fixed at the
probe entrypoint, and the critical `B4` / `B12` cases were re-run under the
standard `750000` setting.

To move beyond one-off probe runs, the standard sweep runner also now supports
per-target source overrides:

- [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py)
- [zju_6src_hardcontrol_hybrid_v2_b12_only.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v2_b12_only.json)

## Standardized Critical Cases

### `Camera_B4`

Standardized rerun:

- [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B4_v2_std750k/summary.md)

Best narrow variant:

- `swap_b2_to_b19`

Delta vs standardized `uniform`:

- full-frame depth-minus-point MAE: `+0.000465`
- `fg_human` depth-minus-point MAE: `-0.007705`
- `bg_far` depth-minus-point MAE: `+0.001078`
- `bg_bottom_band` depth-minus-point MAE: `-0.003045`
- decision: still `point_map`

Readout:

- `B4` does improve on `bg_bottom_band`
- but it gives back more than it saves on full-frame / `bg_far`
- so `B4` must **not** be kept as a current hybrid lead

### `Camera_B12`

Standardized rerun:

- [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B12_stage2_std750k/summary.md)

Surviving hybrid:

- `hybrid_b23_b8`

Delta vs standardized `uniform`:

- full-frame depth-minus-point MAE: `-0.002008`
- `fg_human` depth-minus-point MAE: `+0.000088`
- `bg_far` depth-minus-point MAE: `-0.002109`
- `bg_bottom_band` depth-minus-point MAE: `-0.003177`
- decision: `point_map -> depth_unproject`

Readout:

- `B12` remains a valid local source-policy rescue
- the gain is not coming from sacrificing the human region
- this is the only narrow override that still survives standardization

## Standardized Batch Follow-Up

The surviving override was then pushed back through the normal local sweep path
as a `B12`-only hybrid manifest.

Outputs:

- [sweep summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round5_6src_uniform_override_b12only_hardcontrol_v2/summary.md)
- [region summary.md](/f:/vggt/vggt-main/output/geometry_region_hybrid6src_b12only_via_sweep_20260323/summary.md)
- [contrast summary.md](/f:/vggt/vggt-main/output/geometry_region_policy_contrast_20260323/uniform_vs_hybrid6src_b12only_via_sweep_hardcontrol/summary.md)

Batch readout relative to `6src_uniform`:

- decision counts:
  - `uniform`: `point_map = 4`, `depth_unproject = 1`
  - `b12-only hybrid`: `point_map = 3`, `depth_unproject = 2`
- full-frame average depth-minus-point MAE:
  - `0.001521 -> 0.001119`
  - delta: `-0.000402`
- full-frame average depth-minus-point coverage:
  - `0.010582 -> 0.021715`
  - delta: `+0.011133`
- region aggregate delta:
  - `fg_human` MAE: `+0.000018`
  - `fg_edge` MAE: `-0.000363`
  - `bg_far` MAE: `-0.000422`
  - `bg_bottom_band` MAE: `-0.000635`

Case-level meaning:

- only `Camera_B12` flips
- `Camera_B5` remains a `depth_unproject` win
- `Camera_B1`, `Camera_B4`, and `Camera_B15` remain unresolved `point_map`
  cases

## Decision

The corrected local conclusion is now:

- keep `6src_uniform` as the base sparse policy
- keep only the `B12 -> hybrid_b23_b8` override as the current validated local
  refinement
- reject the earlier `B4` override as non-standardized / non-surviving
- do **not** launch Modal from this evidence yet

This is progress, but it is not a full local repair.

The hard/control subset is still only at:

- `depth_unproject = 2 / 5`
- `point_map = 3 / 5`

So cloud remains blocked by rule.

## Next Local Step

The next local-only work should stay narrow and standardized:

- continue probing the unresolved `Camera_B1`, `Camera_B4`, and `Camera_B15`
  cases under the same `750000` render setting
- keep using the standard sweep path for any candidate that looks promising
- do not reopen `ghost`
- do not reopen global depth-threshold tuning
- do not start a new cloud run until the remaining local failures are either
  fixed or clearly bounded
