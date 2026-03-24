# Sparse Region Contrast 2026-03-23

## Goal

This note contrasts the two strongest sparse-view readouts now available:

- failing sparse branch:
  - `6src_hist` hard cameras
- stronger sparse branch:
  - matched-source `12src_uniform`

The purpose is to answer a narrower project question:

> what actually distinguishes the sparse branch that still fails from the sparse
> branch that is already good enough to keep as the current geometry-first lead?

## Inputs

### Failing branch

- [geometry_6src_hardcase_region_status_20260323.md](/f:/vggt/vggt-main/docs/geometry_6src_hardcase_region_status_20260323.md)
- [6src hard summary.md](/f:/vggt/vggt-main/output/geometry_region_hardcases_6src_20260323/summary.md)

### Stronger branch

- [12src uniform legacy backfill summary.md](/f:/vggt/vggt-main/output/legacy_uniform_backfill_from_current/batch4_20260323/summary/summary.md)
- [12src uniform region summary.md](/f:/vggt/vggt-main/output/geometry_region_uniform_12src_20260323/summary.md)

## Full-Frame Contrast

### `6src` hard cases

- full decision counts: `point_map = 4 / 4`
- average depth minus point MAE: `+0.003163`
- average depth minus point coverage: `-0.066329`

### `12src_uniform` matched-source cases

- full decision counts:
  - `depth_unproject = 1`
  - `tie = 3`
- average depth minus point MAE: `-0.001371`
- average depth minus point coverage: `-0.003008`

Interpretation:

- `12src_uniform` has already moved the full-frame result from "point wins clearly"
  toward "depth wins or ties"
- the remaining coverage penalty becomes very small compared with the `6src` hard set

## Region Contrast

### `fg_human`

`6src` hard:

- depth wins MAE on `3 / 4`
- average depth minus point MAE: `-0.003462`
- average depth minus point coverage: `+0.035593`

`12src_uniform`:

- depth wins MAE on `4 / 4`
- average depth minus point MAE: `-0.032657`
- average depth minus point coverage: `+0.217607`

Interpretation:

- both branches already show that `depth_unproject` is not fundamentally bad on the human region
- the stronger `12src_uniform` branch amplifies that advantage very clearly

### `fg_edge`

`6src` hard:

- average depth minus point MAE: `+0.002256`
- average depth minus point coverage: `+0.027131`

`12src_uniform`:

- average depth minus point MAE: `-0.003629`
- average depth minus point coverage: `+0.118914`

Interpretation:

- edge quality is one of the places where `12src_uniform` is also materially better
- this suggests the better sparse source policy helps both stability and usable coverage around the person

### `bg_far`

`6src` hard:

- point wins MAE on `4 / 4`
- average depth minus point MAE: `+0.003440`
- average depth minus point coverage: `-0.075836`

`12src_uniform`:

- depth wins MAE on `3 / 4`
- average depth minus point MAE: `-0.000069`
- average depth minus point coverage: `-0.017985`

Interpretation:

- the far-background collapse seen in `6src` hard cases is largely absent in `12src_uniform`
- this is one of the clearest discriminators between the bad sparse branch and the better sparse branch

### `bg_bottom_band`

`6src` hard:

- point wins MAE on `4 / 4`
- average depth minus point MAE: `+0.010246`
- average depth minus point coverage: `-0.237034`

`12src_uniform`:

- point wins MAE on `3 / 4`
- average depth minus point MAE: `+0.001036`
- average depth minus point coverage: `-0.025681`

Interpretation:

- bottom-band weakness still exists even in the stronger branch
- but its magnitude is dramatically smaller than in `6src` hard cameras

This is important because it means:

- the mentor's bottom/background concern is real
- but it is not acting alone
- source policy / sparse-view geometry quality strongly modulates how severe that weakness becomes

## Working Conclusion

The current evidence now supports a more precise statement than before:

> the main difference between the failing sparse branch and the stronger sparse
> branch is not the human region itself; it is whether the geometry branch can
> avoid large background and especially bottom-band collapse under sparse-view conditions.

That makes the current priority order clearer:

1. keep `depth + camera -> unproject` as the main geometry direction
2. keep `12src_uniform` as the best current sparse geometry-friendly branch
3. treat `6src` hard cameras as a dedicated failure set
4. avoid reopening `ghost`
5. avoid cloud until a new local hypothesis is sharp enough

## Practical Next Gate

If the next experiment is training-side, it should now satisfy both:

1. it is motivated by the bottom/background weakness visible in the hard set
2. it does not ignore the fact that source policy already changes the severity of that weakness

So the next small experiment should not be a generic loss sweep.

It should be one of:

- a bottom/background-specific geometry treatment justified by the hard-case split
- or another source-policy / sparse-hard-camera diagnostic that narrows why `6src`
  collapses much more strongly than `12src_uniform`

