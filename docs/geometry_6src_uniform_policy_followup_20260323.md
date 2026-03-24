# 6src Uniform-Policy Follow-Up 2026-03-23

## Goal

This note records the next local/offline follow-up after the `6src` hard-case
region diagnosis.

The question was:

> if `6src` hard-camera failure is strongly background / bottom-band driven,
> does a more even sparse source policy help reduce that collapse?

The tested change was deliberately narrow:

- keep `view_profile = 6src_hist`
- keep `frame_id = 1080`
- keep the same original VGGT checkpoint
- keep the same geometry-branch compare pipeline
- only change the sparse source policy:
  - from `rotate_template_offsets`
  - to `uniform_ring`

## Local Current-Current Follow-Up

Targets:

- control:
  - `Camera_B5`
- hard cases:
  - `Camera_B1`
  - `Camera_B4`
  - `Camera_B12`
  - `Camera_B15`

Current-current subset comparison:

- [summary.md](/f:/vggt/vggt-main/output/geometry_sparse_policy_followup_20260323/rotate_vs_uniform6src_hardcontrol_frame1080/summary.md)

Key full-frame readout:

- cases: `5`
- average geometry gain:
  - rotate: `-0.002257`
  - uniform: `-0.001521`
  - delta: `+0.000736`
- average coverage gain:
  - rotate: `-0.042405`
  - uniform: `+0.010582`
  - delta: `+0.052987`
- improved geometry cases: `4 / 5`
- worsened geometry cases: `1 / 5`

Per case:

- `Camera_B5`: depth still wins, and coverage improves strongly
- `Camera_B1`: still point-favored, but less negative than rotate
- `Camera_B4`: still point-favored, but the gap narrows materially
- `Camera_B12`: still point-favored, but the gap narrows
- `Camera_B15`: still point-favored, but the gap narrows

So `uniform_ring` is a real positive move for this subset, but it does not flip
the hard cases on full-frame MAE.

## Region-Level Follow-Up

To avoid relying on full-frame metrics alone, the same five cases were
re-materialized locally with full compare outputs and region diagnostics.

Uniform-region batch:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_uniform6src_20260323/summary.md)

Matched rotate-region batch:

- [summary.md](/f:/vggt/vggt-main/output/geometry_region_rotate6src_hardcontrol_20260323/summary.md)

## Region Contrast

### Full frame

Rotate:

- avg depth minus point MAE: `+0.002257`
- avg depth minus point coverage: `-0.042405`

Uniform:

- avg depth minus point MAE: `+0.001521`
- avg depth minus point coverage: `+0.010582`

Interpretation:

- `uniform_ring` improves both the full-frame MAE gap and the full-frame coverage gap

### `bg_far`

Rotate:

- avg depth minus point MAE: `+0.003440`
- avg depth minus point coverage: `-0.075836`

Uniform:

- avg depth minus point MAE: `+0.001500`
- avg depth minus point coverage: `+0.011434`

Interpretation:

- this is the clearest improvement
- the far-background collapse becomes much milder
- coverage even flips from negative to positive on average

### `bg_bottom_band`

Rotate:

- avg depth minus point MAE: `+0.010246`
- avg depth minus point coverage: `-0.237034`

Uniform:

- avg depth minus point MAE: `+0.008889`
- avg depth minus point coverage: `-0.219947`

Interpretation:

- the bottom-band failure improves slightly
- but it remains the strongest unresolved region
- every case still remains `point_map`-favored there

### `fg_human`

Rotate:

- avg depth minus point MAE: `-0.003462`
- avg depth minus point coverage: `+0.035593`

Uniform:

- avg depth minus point MAE: `+0.000250`
- avg depth minus point coverage: `+0.005291`

Interpretation:

- `uniform_ring` gives back some of the foreground-human advantage that rotate had
- this is a real tradeoff, not a free win

### `fg_edge`

Rotate:

- avg depth minus point MAE: `+0.002256`
- avg depth minus point coverage: `+0.027131`

Uniform:

- avg depth minus point MAE: `+0.003589`
- avg depth minus point coverage: `-0.027835`

Interpretation:

- edge quality also gets worse under `uniform_ring`

## Working Conclusion

`uniform_ring` for `6src` is useful, but only in a specific way:

- it helps the full-frame result
- it strongly helps `bg_far`
- it only slightly helps `bg_bottom_band`
- and it gives back some of the earlier foreground / edge advantage

So the correct conclusion is not:

> replace `6src_rotate` with `6src_uniform` everywhere

It is:

> the `6src` hard-camera problem is partly source-policy-sensitive, especially
> in the far background, but the bottom-band failure remains strong enough that
> source policy alone is not enough.

## Decision

After this follow-up:

- do **not** launch cloud
- do **not** reopen `ghost`
- do **not** start another generic loss sweep

The next useful hypothesis should be sharper:

1. treat `bg_far` as partly source-policy-sensitive
2. treat `bg_bottom_band` as the most stubborn unresolved failure region
3. if the next change is training-side, it should be justified mainly by the
   remaining bottom-band weakness, not by a vague full-frame objective change

