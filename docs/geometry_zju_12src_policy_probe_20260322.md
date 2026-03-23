# ZJU 12src Policy Probe

This note answers the next question that remained after round 2:

> if `12src_nested` is still weak after rotating the old sparse template, is it because the views should be more local, or because they should be more uniformly distributed over the ring?

I kept the same original VGGT checkpoint and the same branch-comparison pipeline, and tested two new `12src` source-selection policies:

- `nearest_ring`
- `uniform_ring`

Reference baseline:

- round-2 `rotate_template_offsets`

Artifacts:

- [round2 rotate summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.md)
- [round3 nearest summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round3_12src_nearest_v1/summary.md)
- [round3 uniform summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round3_12src_uniform_v1/summary.md)
- [rotate vs nearest](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_rotate_vs_round3_nearest_12src_v1/comparison.md)
- [rotate vs uniform](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_rotate_vs_round3_uniform_12src_v1/comparison.md)
- [nearest vs uniform](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round3_nearest_vs_uniform_12src_v1/comparison.md)

## Overall Result

`nearest_ring`:

- `23` depth wins
- `122` point wins
- `62` ties
- average geometry gain: `-0.000452`
- average coverage gain: `-0.021911`

`uniform_ring`:

- `48` depth wins
- `107` point wins
- `52` ties
- average geometry gain: `-0.000248`
- average coverage gain: `-0.021101`

Reference `rotate_template_offsets` from round 2:

- `14` depth wins
- `143` point wins
- `50` ties
- average geometry gain: `-0.000894`
- average coverage gain: `-0.029728`

## Direct Comparison

Relative to round-2 rotated template:

- `nearest_ring` improves depth wins from `14 -> 23`
- `nearest_ring` improves geometry gain by `+0.000442`
- `nearest_ring` improves coverage gain by `+0.007816`

- `uniform_ring` improves depth wins from `14 -> 48`
- `uniform_ring` improves geometry gain by `+0.000647`
- `uniform_ring` improves coverage gain by `+0.008627`

Relative to `nearest_ring`:

- `uniform_ring` improves depth wins from `23 -> 48`
- `uniform_ring` improves geometry gain by `+0.000205`
- `uniform_ring` improves coverage gain by `+0.000811`

So both new policies help, but `uniform_ring` helps much more.

## Frame-Level Pattern For `uniform_ring`

Per-frame `depth_unproject` wins:

- `0`: `1`
- `150`: `3`
- `300`: `7`
- `450`: `6`
- `600`: `5`
- `750`: `7`
- `900`: `5`
- `1080`: `8`
- `1170`: `6`

Compared to round 2 on the same `12src` cases, `uniform_ring` increases depth wins on every frame:

- `150`: `0 -> 3`
- `300`: `2 -> 7`
- `450`: `1 -> 6`
- `600`: `1 -> 5`
- `750`: `2 -> 7`
- `900`: `2 -> 5`
- `1080`: `4 -> 8`
- `1170`: `1 -> 6`

This means the improvement is broad, not just carried by one easy frame.

## Interpretation

The main answer from this probe is:

- `12src` was not primarily missing local target-neighbor overlap
- `12src` improves more when the sparse source set is distributed more uniformly over the rig

That does **not** make `12src` a solved geometry-friendly regime yet:

- average geometry gain is still slightly negative
- average coverage gain is still slightly negative
- `point_map` still wins more cases than `depth_unproject`

But the result is still useful because it removes one ambiguity:

- if `12src` is to keep improving, the next policy should be coverage-oriented, not local-neighbor-oriented

## Current Recommendation

1. Keep `uniform_ring` as the best current `12src` source policy.
2. Do not keep `nearest_ring` as the main `12src` baseline.
3. Keep `6src_hist + rotate_template_offsets` as the cleaner sparse geometry-friendly profile for now.
4. Treat `12src_uniform` as an improved but still incomplete sparse baseline.
5. If more offline work is done, focus on:
   - stronger uniform-coverage `12src` variants
   - confidence-threshold and sparse geometry diagnostics
   - not ghost-loss restoration
