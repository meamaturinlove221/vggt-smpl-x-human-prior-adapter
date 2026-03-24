# 6src Hard/Control Local Completion 2026-03-24

## Goal

This note closes the local-only `6src` hard/control source-policy repair loop.

The remaining blocker after the `B1+B4+B12` batch was `Camera_B15`.

The requirement stayed unchanged:

- keep all work local
- do not open any new cloud run before the local failure set is fully repaired

## Inputs

- previous `B1+B4+B12` batch:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round7_6src_uniform_override_b1_b4_b12_hardcontrol_v4/summary.md)
- previous local state note:
  - [geometry_next_step_after_20260323.md](/f:/vggt/vggt-main/docs/geometry_next_step_after_20260323.md)
- new expanded `B15` search:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260323/B15_uniform_near_hist_swaps_v2/summary.md)
- new finalized local manifest:
  - [zju_6src_hardcontrol_hybrid_v5_b1_b4_b12_b15.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v5_b1_b4_b12_b15.json)

## `B15` Expanded Search

Search setup:

- reference:
  - `Camera_B11,Camera_B9,Camera_B7,Camera_B21,Camera_B4,Camera_B14`
- candidate pool:
  - `Camera_B11,Camera_B9,Camera_B7,Camera_B21,Camera_B4,Camera_B14,Camera_B22,Camera_B3,Camera_B12,Camera_B13,Camera_B16,Camera_B10`
- search depth:
  - `max_swaps = 3`
- guard:
  - `fg_human_guard_max_delta = 0.001`

Result:

- searched variants:
  - `662`
- guard-pass variants:
  - `554`
- guard-pass `depth_unproject` variants:
  - `14`

Best standardized candidate:

- variant:
  - `s3_279`
- source set:
  - `Camera_B21,Camera_B4,Camera_B14,Camera_B12,Camera_B13,Camera_B10`
- delta vs `uniform`:
  - full depth-point MAE: `-0.009317`
  - `fg_human` depth-point MAE: `-0.026460`
  - `bg_far` depth-point MAE: `-0.008073`
  - `bg_bottom_band` depth-point MAE: `-0.007997`
- decision:
  - `depth_unproject`

This is the first local `B15` result that is both:

- guard-pass on `fg_human`
- and a real `depth_unproject` decision

## Standardized Sweep

The `B15` candidate was promoted into the same standardized override manifest
used for the rest of the hard/control set:

- [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round8_6src_uniform_override_b1_b4_b12_b15_hardcontrol_v5/summary.md)

Result:

- `depth_unproject = 5 / 5`
- `point_map = 0 / 5`
- `ties = 0 / 5`

Per-case decisions:

- `Camera_B1`: `depth_unproject`
- `Camera_B4`: `depth_unproject`
- `Camera_B5`: `depth_unproject`
- `Camera_B12`: `depth_unproject`
- `Camera_B15`: `depth_unproject`

`Camera_B15` standardized readout:

- [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round8_6src_uniform_override_b1_b4_b12_b15_hardcontrol_v5/6src_hist/frame_001080_Camera_B15/summary.md)
- point MAE: `0.0496`
- depth MAE: `0.0435`
- point coverage: `0.4180`
- depth coverage: `0.4748`

So the `B15` fix survives the normal sweep path and is not just a search-only
artifact.

## Region-Level Readout

Batch summary:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_hybrid6src_b1_b4_b12_b15_via_sweep_20260324/summary.md)

Uniform contrast:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_policy_contrast_20260324/uniform_vs_hybrid6src_b1_b4_b12_b15_via_sweep_hardcontrol/summary.md)

`B1+B4+B12 -> B1+B4+B12+B15` contrast:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_policy_contrast_20260324/hybrid6src_b1_b4_b12_vs_b1_b4_b12_b15_via_sweep_hardcontrol/summary.md)

Key numbers:

- full decision counts:
  - `uniform`: `point_map = 4`, `depth_unproject = 1`
  - `v5 hybrid`: `depth_unproject = 5`
- average depth-point MAE:
  - `uniform`: `+0.001521`
  - `v5 hybrid`: `-0.001548`
  - delta: `-0.003068`
- average depth-point coverage:
  - `uniform`: `+0.010582`
  - `v5 hybrid`: `+0.065431`
  - delta: `+0.054849`

Region aggregate under `v5`:

- `fg_human` avg depth-point MAE:
  - `-0.006458`
- `fg_edge` avg depth-point MAE:
  - `-0.000403`
- `bg_far` avg depth-point MAE:
  - `-0.001282`
- `bg_bottom_band` avg depth-point MAE:
  - `+0.006143`

Interpretation:

- the local source-policy repair is now enough to make the full-frame decision
  flip to `depth_unproject` on all `5 / 5` hard/control cases
- `bg_bottom_band` is still the weakest region
- but it no longer overturns the full-frame decision on the repaired set

## Decision

The local `6src` hard/control repair gate is now passed.

That means:

- the project is no longer blocked by this local source-policy failure set
- there is no reason to reopen the old `ghost` stack
- there is no reason to reopen the rejected global depth-threshold branch

But this does **not** mean a cloud run should start automatically.

The correct operational state after this local completion is:

- keep Modal clean until the next cloud-worthy question is chosen explicitly
- treat the local source-policy repair as complete evidence, not as a reason to
  launch an unrelated cloud batch by default
