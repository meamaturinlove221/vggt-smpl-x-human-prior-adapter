# 6src Source Leave-One-Out Status 2026-03-23

## Goal

This note records the next local-only diagnosis step after the region-policy
contrast.

At that point the project already knew:

- `6src` sparse failures are dominated by `bg_bottom_band`
- source policy clearly affects `bg_far`
- `6src_uniform` helps, but does not solve the bottom-band problem

The next question was narrower:

> in the failing `6src` branch, is the remaining bottom-band error still
> sensitive to which individual source camera is present?

This is the smallest useful check before any new cloud action, because if the
answer is "yes", the next step should stay on source-policy / view-selection
 work rather than jumping to training.

## New Tool

A reusable local diagnostic runner was added:

- [run_zju_source_leave_one_out_region_diagnostics.py](/f:/vggt/vggt-main/scripts/run_zju_source_leave_one_out_region_diagnostics.py)

What it does:

- takes one or more ZJU case `report.json` files
- runs the normal geometry compare pipeline on:
  - the original baseline source set
  - each leave-one-source-out variant
- keeps the evaluation fixed on:
  - `point_map` vs `depth_unproject`
  - full-frame metrics
  - region metrics
- ranks dropped sources by whether they improve:
  - `bg_bottom_band`
  - full-frame geometry
  - and how much foreground tradeoff they introduce

This stays fully local:

- no training
- no Modal
- no old `ghost` stack

## Batch Run

The first full batch was run on the current `6src_hist` hard/control subset:

- `Camera_B1`
- `Camera_B4`
- `Camera_B5`
- `Camera_B12`
- `Camera_B15`

Outputs:

- [batch summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/summary.md)
- [batch summary.json](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/summary.json)

Per-case summaries:

- [B1 summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/CoreView_390_frame_001080_Camera_B1_6src_hist/summary.md)
- [B4 summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/CoreView_390_frame_001080_Camera_B4_6src_hist/summary.md)
- [B5 summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/CoreView_390_frame_001080_Camera_B5_6src_hist/summary.md)
- [B12 summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/CoreView_390_frame_001080_Camera_B12_6src_hist/summary.md)
- [B15 summary.md](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_20260323_6src_hardcontrol/CoreView_390_frame_001080_Camera_B15_6src_hist/summary.md)

## Main Readout

### Batch-level result

Across:

- `5` cases
- `30` leave-one-out variants

the results are:

- bottom-band-improving variants: `7 / 30`
- variants that improve both bottom-band MAE and full-frame MAE: `3 / 30`

So the bottom-band failure is **not rigid**.

It still moves under source selection.

That alone is important, because it means the next step does not need to jump to
cloud training just to keep making progress.

### Best drop per case

- `Camera_B1`
  - best drop: `Camera_B22`
  - bottom-band delta: `-0.000054`
  - full-frame delta: `-0.000267`
- `Camera_B4`
  - best drop: `Camera_B2`
  - bottom-band delta: `-0.001294`
  - full-frame delta: `+0.000960`
- `Camera_B5`
  - best drop: `Camera_B19`
  - bottom-band delta: `-0.001348`
  - full-frame delta: `-0.000605`
- `Camera_B12`
  - best drop: `Camera_B18`
  - bottom-band delta: `-0.005863`
  - full-frame delta: `+0.000178`
- `Camera_B15`
  - best drop: `Camera_B18`
  - bottom-band delta: `-0.004392`
  - full-frame delta: `-0.000114`

Interpretation:

- some cases do have a single-source bottleneck
- but the helpful drop is not the same for every target
- so there is no clean universal "always remove camera X" rule

### Foreground tradeoff

This is the most important caution.

For several of the strongest bottom-band improvements:

- `fg_human` gets worse
- sometimes by a clearly non-trivial amount

Examples:

- `Camera_B5`, drop `Camera_B19`
  - `bg_bottom_band` improves
  - full-frame also improves
  - but `fg_human` delta is `+0.010465`
- `Camera_B15`, drop `Camera_B18`
  - `bg_bottom_band` improves strongly
  - full-frame improves slightly
  - but `fg_human` delta is `+0.017081`
- `Camera_B12`, drop `Camera_B18`
  - `bg_bottom_band` improves a lot
  - full-frame gets slightly worse
  - `fg_human` delta is `+0.025255`

So the correct conclusion is **not**:

> just prune one camera and the problem is solved.

It is:

> `6src` bottom-band weakness is still source-sensitive, but source removal often
> trades bottom-band cleanup for foreground damage.

## What This Means

This result sharpens the current project diagnosis again.

We now know:

1. the remaining `6src` bottom-band problem is still partly source-selection-sensitive
2. but the effect is case-specific
3. and naive source pruning is not a free win because it often hurts the person region

### Relation to the existing `6src_uniform` policy

A quick local source-set cross-check was also done against the existing
`round4_6src_uniform_hardcontrol_v1` reports.

Result:

- for `Camera_B1`, the best leave-one-out source was `Camera_B22`
  - `6src_uniform` already excludes it
- for `Camera_B5`, the best leave-one-out source was `Camera_B19`
  - `6src_uniform` already excludes it
- for `Camera_B15`, the best leave-one-out source was `Camera_B18`
  - `6src_uniform` already excludes it
- for `Camera_B4`, the best leave-one-out source was `Camera_B2`
  - `6src_uniform` still keeps it
- for `Camera_B12`, the best leave-one-out source was `Camera_B18`
  - `6src_uniform` still keeps it

Interpretation:

- `6src_uniform` is not arbitrary
- in `3 / 5` cases it already removes the source camera that the local
  leave-one-out diagnosis would most like to drop
- the remaining `2 / 5` cases are exactly where there is still room for a
  sharper source-policy refinement beyond plain `uniform_ring`

So the next step should stay local and should be more targeted than:

- another generic loss sweep
- another cloud fine-tune
- a crude "drop one source everywhere" rule

## Recommended Next Step

The next safe local hypothesis is:

> refine `6src` source policy in a way that targets bottom-band cleanup while
> preserving foreground support, instead of raw single-source pruning.

That means one of:

- a source-policy rule that rebalances the ring without removing the strongest
  foreground-supporting views
- a new compare-only diagnostic that explains *why* the harmful source cameras
  differ by target
- or a very narrow render-side bottom-band treatment, but only after this
  source-selection evidence is fully used

## Operational Decision

After this leave-one-out batch:

- keep cloud off
- keep the work local
- do not reopen `ghost`
- do not relaunch the failed auxiliary-region training branch

The current best reason to stay local is now concrete:

> `6src` still contains actionable source-policy structure, and we have not yet
> exhausted that cheaper path.
