# Geometry Region Policy Contrast 2026-03-23

## Goal

This note consolidates the new region-level source-policy evidence into one
place.

The project already knew two things:

- `depth_unproject` is often better on the person itself
- sparse failures are often driven by background, especially bottom-band regions

What was still missing was a direct, same-format region comparison for:

- `12src_rotate -> 12src_uniform`
- `6src_rotate -> 6src_uniform`

That gap is now filled locally, without training and without cloud.

## New Outputs

### Region batch regeneration

The missing matched `12src_rotate` region batch was re-materialized locally:

- [12src rotate region summary.md](/f:/vggt/vggt-main/output/geometry_region_rotate12src_20260323/summary.md)

This uses the same four matched targets already used in the sparse-policy line:

- `Camera_B3`
- `Camera_B5`
- `Camera_B8`
- `Camera_B19`

### Reusable region-batch comparison tool

A new comparer was added:

- [compare_zju_region_batch_summaries.py](/f:/vggt/vggt-main/scripts/compare_zju_region_batch_summaries.py)

This compares two aggregated region batch summaries and reports:

- full-frame depth-minus-point MAE / coverage deltas
- per-region aggregate deltas
- matched per-target deltas for:
  - full frame
  - `bg_far`
  - `bg_bottom_band`

### Comparison outputs

- [12src region policy contrast](/f:/vggt/vggt-main/output/geometry_region_policy_contrast_20260323/rotate_vs_uniform12src_matched/summary.md)
- [6src region policy contrast](/f:/vggt/vggt-main/output/geometry_region_policy_contrast_20260323/rotate_vs_uniform6src_hardcontrol/summary.md)

## Main Readout

### `12src`: source policy helps almost everywhere that matters

`12src_uniform` vs `12src_rotate`:

- full-frame average depth-minus-point MAE:
  - `-0.000369 -> -0.001371`
- full-frame average depth-minus-point coverage:
  - `-0.034007 -> -0.003008`

Region deltas:

- `fg_human` MAE improves:
  - `-0.029262 -> -0.032657`
- `fg_edge` flips from point-favored to depth-favored:
  - `+0.002820 -> -0.003629`
- `bg_far` improves:
  - `+0.000644 -> -0.000069`
- `bg_bottom_band` also improves materially:
  - `+0.004842 -> +0.001036`
  - coverage gap:
    - `-0.119343 -> -0.025681`

Interpretation:

- for `12src`, `uniform_ring` is not just a far-background fix
- it improves the person region, the edge region, the far background, and even
  the bottom band
- `bg_bottom_band` is still not fully solved, but it is much less damaging than
  under rotate

### `6src`: source policy mainly fixes `bg_far`, but bottom-band remains stubborn

`6src_uniform` vs `6src_rotate`:

- full-frame average depth-minus-point MAE:
  - `+0.002257 -> +0.001521`
- full-frame average depth-minus-point coverage:
  - `-0.042405 -> +0.010582`

Region deltas:

- `bg_far` improves clearly:
  - `+0.003440 -> +0.001500`
  - coverage gap:
    - `-0.075836 -> +0.011434`
- `bg_bottom_band` improves only a little:
  - `+0.010246 -> +0.008889`
  - coverage gap:
    - `-0.237034 -> -0.219947`
- `fg_human` regresses:
  - `-0.003462 -> +0.000250`
- `fg_edge` also regresses:
  - `+0.002256 -> +0.003589`

Interpretation:

- for `6src`, `uniform_ring` is a partial rescue, not a clean promotion
- it strongly helps `bg_far`
- it barely dents the bottom-band problem
- and it gives back some foreground / edge advantage

## Working Conclusion

The current evidence is now sharper than the earlier broad statement
"source policy matters."

What it now supports is:

1. `12src_uniform` is a real promotion over `12src_rotate`, including at the
   region level.
2. `6src_uniform` is not a real promotion; it is only a partial mitigation.
3. `bg_far` is strongly source-policy-sensitive.
4. `bg_bottom_band` is partly source-policy-sensitive, but much more stubborn,
   especially in the `6src` regime.

So the practical diagnosis becomes:

> source policy can greatly reduce the geometry collapse, but the unresolved
> problem is no longer "background in general." It is increasingly
> `6src`-conditioned bottom-band weakness.

## Next Safe Step

Given these results, the safest next local direction is:

- keep `12src_uniform` as the stronger sparse geometry reference
- keep `6src` as the failure-focused diagnosis branch
- do not reopen `ghost`
- do not move the failed auxiliary-region training branch to cloud
- focus the next hypothesis specifically on why `6src` bottom-band remains bad
  even after source-policy improvement

In practice, that means the next experiment should be justified as one of:

- a `6src` bottom-band-specific inference/render diagnosis
- a `6src` source-camera policy refinement aimed specifically at ground-adjacent
  failure
- a very narrow training-side change only if it is motivated by this remaining
  `6src` bottom-band gap rather than by generic full-frame loss movement
