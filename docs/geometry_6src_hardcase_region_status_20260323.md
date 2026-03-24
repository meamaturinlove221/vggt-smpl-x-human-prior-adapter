# 6src Hard-Case Region Status 2026-03-23

## Goal

This note records the next offline diagnosis step after the reliable-region
local gate.

The question here was not "does `depth + camera` work at all?"

That part was already supported.

The narrower question was:

> on the strict same-source `6src` hard cameras where current `point_map`
> still beats current `depth_unproject`, is the failure mainly a foreground-human
> issue, or is it being driven by background / bottom-band regions?

This directly follows the mentor's concern that the ZJU depth target and the
depth-derived point cloud may be less reliable on ground / background regions.

## Inputs

### Hard-case source

- strict same-source legacy backfill summary:
  - [summary.md](/f:/vggt/vggt-main/output/legacy_rotate6src_backfill_from_current/batch4_20260323/summary/summary.md)
- selected hard targets:
  - `Camera_B15`
  - `Camera_B12`
  - `Camera_B1`
  - `Camera_B4`

These are the four `6src_hist` cameras where current `point_map` still beats
current `depth_unproject` under the matched-source legacy backfill.

### Full compare regeneration

The original sweep directories for these cases did not include `predictions.npz`,
so region slicing could not be run directly from the sweep artifacts.

To keep the scope local and reproducible, each hard case was re-materialized
locally from its stored `synthetic_report.json` using the original VGGT compare
pipeline:

- [run_zju_geometry_baseline_from_report.ps1](/f:/vggt/vggt-main/scripts/run_zju_geometry_baseline_from_report.ps1)

Outputs:

- [B15 summary.json](/f:/vggt/vggt-main/output/geometry_hardcases_fullcompare_20260323/CoreView_390_frame_001080_Camera_B15_6src_hist/summary.json)
- [B12 summary.json](/f:/vggt/vggt-main/output/geometry_hardcases_fullcompare_20260323/CoreView_390_frame_001080_Camera_B12_6src_hist/summary.json)
- [B1 summary.json](/f:/vggt/vggt-main/output/geometry_hardcases_fullcompare_20260323/CoreView_390_frame_001080_Camera_B1_6src_hist/summary.json)
- [B4 summary.json](/f:/vggt/vggt-main/output/geometry_hardcases_fullcompare_20260323/CoreView_390_frame_001080_Camera_B4_6src_hist/summary.json)

All four were regenerated locally on the same original checkpoint hash.

### Batch aggregation

A reusable summarizer was added:

- [summarize_zju_region_case_summaries.py](/f:/vggt/vggt-main/scripts/summarize_zju_region_case_summaries.py)

Hard-case batch outputs:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_hardcases_6src_20260323/summary.md)
- [summary.json](/f:/vggt/vggt-main/output/geometry_region_hardcases_6src_20260323/summary.json)
- [summary.csv](/f:/vggt/vggt-main/output/geometry_region_hardcases_6src_20260323/summary.csv)

Control reference already existed:

- [B5 6src region_metrics.json](/f:/vggt/vggt-main/output/geometry_region_diagnostics_controlcases_20260323/CoreView_390_frame_001080_Camera_B5_6src_hist/region_metrics.json)

## Main Readout

### Full-frame result

Across the four hard cameras:

- full decision counts: `point_map = 4 / 4`
- average full-frame MAE:
  - point: `0.055550`
  - depth: `0.058713`
  - depth minus point: `+0.003163`
- average full-frame coverage:
  - point: `0.454571`
  - depth: `0.388241`
  - depth minus point: `-0.066329`

So the full-frame loss still clearly favors `point_map`.

### Region breakdown

But the region breakdown is not uniform.

#### `fg_human`

- MAE winner counts:
  - `depth_unproject = 3`
  - `point_map = 1`
- coverage winner counts:
  - `depth_unproject = 4`
- average depth minus point MAE: `-0.003462`
- average depth minus point coverage: `+0.035593`

Interpretation:

- in these hard `6src` cameras, `depth_unproject` is usually **better on the human body itself**
- it also has better foreground coverage on every hard case

#### `fg_edge`

- MAE winner counts:
  - `depth_unproject = 2`
  - `point_map = 2`
- coverage winner counts:
  - `depth_unproject = 3`
  - `point_map = 1`
- average depth minus point MAE: `+0.002256`

Interpretation:

- silhouette edge quality is mixed
- this is not the main explanation for the full-frame failure

#### `bg_far`

- MAE winner counts:
  - `point_map = 4`
- coverage winner counts:
  - `point_map = 4`
- average depth minus point MAE: `+0.003440`
- average depth minus point coverage: `-0.075836`

Interpretation:

- all four hard cases regress in the far background for `depth_unproject`

#### `bg_bottom_band`

- MAE winner counts:
  - `point_map = 4`
- coverage winner counts:
  - `point_map = 4`
- average depth minus point MAE: `+0.010246`
- average depth minus point coverage: `-0.237034`

Interpretation:

- this is the strongest and most consistent failure region
- the bottom-band degradation is much larger than the foreground-human effect

## Control vs Hard Split

The existing `6src_hist / Camera_B5` control case already showed a different
pattern:

- `fg_human`: `depth_unproject` wins MAE
- `bg_far`: `depth_unproject` wins MAE
- `bg_bottom_band`: `point_map` wins MAE

By contrast, the four hard cases show:

- `fg_human`: `depth_unproject` still usually wins
- `bg_far`: `point_map` wins on all `4 / 4`
- `bg_bottom_band`: `point_map` wins on all `4 / 4`

That means the hard-case regression is not just "depth is bad everywhere."

It is much more specific:

> the sparse hard-camera failures are dominated by background degradation, and
> especially by bottom-band / ground-adjacent degradation.

## Decision

This strengthens the earlier mentor-aligned interpretation:

- the project should **not** move back to the old `ghost` stack
- the project should **not** treat the current failure as mainly a foreground-human geometry problem
- the current `6src` hard-camera gap looks much more like:
  - sparse-view source-policy sensitivity
  - plus bottom/background weakness in the depth-derived branch

## Practical Implication

The new reliable-region code path remains valuable conceptually, but the first
short local training gate was not good enough to justify cloud time.

After this hard-case region diagnosis, the current priority remains:

1. keep the geometry-first direction
2. keep local/offline diagnosis first
3. do not launch cloud for this branch yet
4. use the hard-case split to guide the next minimal change

In plain terms:

- if the next experiment touches supervision, it should be justified by this
  bottom/background failure pattern
- if the next experiment stays inference-side, it should continue around
  `6src` sparse hard cameras and source policy rather than image-side ghost logic

