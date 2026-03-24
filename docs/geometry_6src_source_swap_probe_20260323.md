# 6src Source-Swap Probe Status 2026-03-23

## Goal

This note records the next local-only step after the `6src` leave-one-out and
`6src_uniform` follow-ups.

## Correction

This note should no longer be used as the final decision artifact by itself.

The first version of this probe used the original
[run_zju_custom_source_set_region_probe.py](/f:/vggt/vggt-main/scripts/run_zju_custom_source_set_region_probe.py)
default `render_max_points = 500000`, while the main compare/sweep path used
`750000`.

After standardizing that local inconsistency to `750000` and re-running the
critical cases:

- `Camera_B4` no longer flips to `depth_unproject`
- `Camera_B12` still supports a narrow hybrid override
- the corrected batch-level readout is now documented in:
  - [geometry_6src_hybrid_policy_standardized_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_hybrid_policy_standardized_20260323.md)

So the sections below remain useful as historical probe context, but the
standardized follow-up above is the current source of truth.

At that point the project already knew:

- `6src` failures were still dominated by `bg_bottom_band`
- `uniform_ring` helped `bg_far`, but not enough on the bottom band
- for `Camera_B4` and `Camera_B12`, `uniform_ring` still kept the same source
  camera that the earlier leave-one-out probe most wanted to remove

So the next question became:

> can a very narrow source-set refinement rescue the remaining `6src`
> bottom-band failure without any new training and without cloud?

## New Tool

A reusable local-only probe runner was added:

- [run_zju_custom_source_set_region_probe.py](/f:/vggt/vggt-main/scripts/run_zju_custom_source_set_region_probe.py)

What it does:

- takes one ZJU case `report.json`
- runs any number of custom `src_cameras` variants on the same target view
- keeps the same `point_map` vs `depth_unproject` compare pipeline
- writes a per-case summary ranked by:
  - `bg_bottom_band` change vs reference
  - then full-frame MAE change vs reference

This stays fully local:

- no training
- no Modal
- no `ghost`

## Cases Probed

Two `6src_hist` targets were used because they were the sharpest unresolved
uniform-policy failures:

- `Camera_B4`
- `Camera_B12`

Outputs:

- [B4 summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B4/summary.md)
- [B12 single-swap summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B12/summary.md)
- [B12 hybrid summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B12_stage2/summary.md)

## Main Readout

### `Camera_B4`: one targeted source swap is enough

Reference `uniform` source set:

- `Camera_B2,Camera_B15,Camera_B10,Camera_B18,Camera_B20,Camera_B22`

Best narrow variant:

- `swap_b2_to_b19`
- source set:
  - `Camera_B19,Camera_B15,Camera_B10,Camera_B18,Camera_B20,Camera_B22`

Delta vs `uniform`:

- full-frame depth-minus-point MAE:
  - `-0.001597`
- `fg_human` depth-minus-point MAE:
  - `-0.000413`
- `bg_far` depth-minus-point MAE:
  - `-0.001807`
- `bg_bottom_band` depth-minus-point MAE:
  - `-0.006212`
- decision:
  - `point_map -> depth_unproject`

Interpretation:

- this is not just a tiny foreground trade
- the variant improves full frame, far background, and bottom band at the same
  time
- foreground is essentially preserved

`swap_b2_to_b12` also flips the decision, but `swap_b2_to_b19` is the cleaner
bottom-band rescue.

### `Camera_B12`: one swap is not enough

Reference `uniform` source set:

- `Camera_B16,Camera_B18,Camera_B20,Camera_B5,Camera_B3,Camera_B13`

Single-source replacements for the suspicious `Camera_B18` were tested:

- `swap_b18_to_b23`
- `swap_b18_to_b8`
- `swap_b18_to_b2`

Result:

- none beat `uniform`
- all remain `point_map`-favored
- the best single swap is effectively flat on bottom-band and worse on full
  frame / foreground

So for `B12`, the remaining failure is not a one-camera bug inside `uniform`.

### `Camera_B12`: a narrow two-source hybrid does work

The next local check was a tiny hybrid ladder between `uniform` and `rotate`.

Best hybrid:

- `hybrid_b23_b8`
- source set:
  - `Camera_B16,Camera_B18,Camera_B23,Camera_B5,Camera_B8,Camera_B13`

Relative to `uniform`:

- full-frame depth-minus-point MAE:
  - `-0.002192`
- `fg_human` depth-minus-point MAE:
  - `-0.000293`
- `bg_far` depth-minus-point MAE:
  - `-0.002319`
- `bg_bottom_band` depth-minus-point MAE:
  - `-0.005353`
- decision:
  - `point_map -> depth_unproject`

Interpretation:

- `B12` was not fundamentally resistant to source-policy rescue
- it just needed a multi-source refinement rather than a single-source swap
- the useful move is still narrow and inference-side

## What This Means

This probe sharpens the current diagnosis again.

We now know:

1. the unresolved `6src` bottom-band gap is still source-set-driven
2. the needed fix is target-specific
3. some cases are fixed by a single targeted source swap
4. harder cases may require a two-source hybrid, not a generic global rule
5. this still points to source policy, not back to training-side generic loss
   tuning

## Decision

After this probe:

- keep cloud off
- do not reopen `ghost`
- do not return to `zju_min_depth_conf`
- do not promote the failed auxiliary-region training branch

The next safe local step is now clearer:

> build and test a sharper `6src` hybrid source-policy rule on the hard/control
> subset before any new cloud run.

That next step should be framed as source-policy refinement, not as another
training ablation.
