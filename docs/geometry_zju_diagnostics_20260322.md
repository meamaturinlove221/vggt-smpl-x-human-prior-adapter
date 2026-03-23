# ZJU Geometry Diagnostics And Training Gate

This note closes the first geometry-first inference round by answering three concrete questions:

1. Does `alignment_rmse_after` explain the sparse-view failures?
2. Is `branch_dist_p90` consistently larger when `point_map` wins?
3. Do the source-ring coverage statistics separate the good and bad source-selection policies clearly enough to choose the next baseline?

Primary artifact:

- [diagnostics summary](/f:/vggt/vggt-main/output/geometry_diagnostics_zju/round1_round2_round3_v1/summary.md)
- [diagnostics json](/f:/vggt/vggt-main/output/geometry_diagnostics_zju/round1_round2_round3_v1/summary.json)
- [case diagnostics csv](/f:/vggt/vggt-main/output/geometry_diagnostics_zju/round1_round2_round3_v1/case_diagnostics.csv)
- [frame diagnostics csv](/f:/vggt/vggt-main/output/geometry_diagnostics_zju/round1_round2_round3_v1/frame_diagnostics.csv)

Scope:

- `1287` evaluated target-view cases
- round 1 fixed sparse + dense full-rig
- round 2 target-aware sparse rotation
- round 3 `12src` policy probe with `nearest_ring` and `uniform_ring`

## Group-Level Readout

Key rows from the diagnostics summary:

- `23cam_fullset / full_rig_excluding_target`
  - depth-win rate among decisive cases: `0.874`
  - avg geometry gain: `+0.001915`
  - avg coverage gain: `+0.021472`
  - avg align RMSE after: `0.088753`
  - avg branch distance p90: `0.086392`
- `6src_hist / rotate_template_offsets`
  - depth-win rate among decisive cases: `0.428`
  - avg geometry gain: `-0.000764`
  - avg coverage gain: `+0.021843`
  - avg align RMSE after: `0.135664`
  - avg branch distance p90: `0.124488`
- `12src_nested / uniform_ring`
  - depth-win rate among decisive cases: `0.310`
  - avg geometry gain: `-0.000248`
  - avg coverage gain: `-0.021101`
  - avg align RMSE after: `0.081406`
  - avg branch distance p90: `0.075110`
- `12src_nested / rotate_template_offsets`
  - depth-win rate among decisive cases: `0.089`
  - avg geometry gain: `-0.000894`
  - avg coverage gain: `-0.029728`
  - avg align RMSE after: `0.079400`
  - avg branch distance p90: `0.079664`
- `12src_nested / nearest_ring`
  - depth-win rate among decisive cases: `0.159`
  - avg geometry gain: `-0.000452`
  - avg coverage gain: `-0.021911`
  - avg align RMSE after: `0.101104`
  - avg branch distance p90: `0.099143`

## Question 1: Does Alignment Explain The Sparse Failures?

Short answer:

- yes for dense/full-rig
- only partially for `12src`
- no as the primary explanation for `6src`

Dense/full-rig is the clean case:

- in `23cam_fullset`, `depth_unproject` wins with lower average post-alignment error than `point_map`: `0.083680` vs `0.121642`
- the correlation between geometry gain and `alignment_rmse_after` is strongly negative: `-0.6577`

That means when the dense-view geometry path looks better, it usually also aligns better after the branch-to-branch Sim(3) fit.

Sparse `6src` does **not** follow that same pattern:

- in target-aware `6src_hist`, `depth_unproject` wins while having *higher* average post-alignment RMSE than `point_map`: `0.171774` vs `0.098493`
- the correlation between geometry gain and `alignment_rmse_after` is near zero: `+0.0442`
- some hard cameras in `6src_hist` still remain `point_map`-favored even after target-aware selection, but their average post-alignment RMSE is not catastrophically large

So the `6src` problem is not "alignment got worse, therefore geometry lost." It is better described as a sparse-view coverage and visibility tradeoff.

For `12src`, alignment is informative but not sufficient on its own:

- round-2 rotated `12src` has almost identical post-alignment RMSE in depth-win and point-win cases: `0.079386` vs `0.079910`
- round-3 `12src_uniform` becomes cleaner: depth-win cases do align better than point-win cases, `0.071167` vs `0.086333`

This means alignment becomes more diagnostic once the source policy is improved, but it still does not fully determine the result.

## Question 2: Is Branch Distance Larger When Point Map Wins?

Short answer:

- yes in the dense/full-rig regime
- yes in the improved `12src_uniform` regime
- not consistently in sparse `6src`

Clear positive evidence:

- `23cam_fullset`: point-win cases have higher `branch_dist_p90` than depth-win cases, `0.107202` vs `0.082928`
- `12src_uniform`: point-win cases also have higher `branch_dist_p90`, `0.078449` vs `0.073671`
- group-level correlation is negative in both regimes: `-0.6930` for dense full-rig and `-0.3744` for `12src_uniform`

Counterexample:

- target-aware `6src_hist`: depth-win cases actually show larger average branch distance than point-win cases, `0.117558` vs `0.096718`

So `branch_dist_p90` is a useful diagnostic, but not a universal decision rule. It becomes more trustworthy once the source coverage policy is already reasonable.

## Question 3: Do Ring-Coverage Statistics Separate Policies Clearly?

Yes. This is the strongest diagnostics result after the branch comparison itself.

The source-ring features separate the policies very clearly:

- `23cam_fullset`
  - source coverage ratio: `0.9545`
  - gap std: `0.2083`
  - target ring distance mean: `6.0000`
- `12src_uniform`
  - source coverage ratio: `0.9545`
  - gap std: `0.2764`
  - target ring distance mean: `6.0000`
- `12src_rotate`
  - source coverage ratio: `0.9091`
  - gap std: `0.8620`
  - target ring distance mean: `6.1667`
- `6src_rotate`
  - source coverage ratio: `0.7273`
  - gap std: `2.1148`
  - target ring distance mean: `6.3333`
- `12src_nearest`
  - source coverage ratio: `0.5455`
  - gap std: `2.7525`
  - target ring distance mean: `3.5000`

Interpretation:

- `nearest_ring` is too locally concentrated and leaves too much ring uncovered
- `uniform_ring` restores wide ring coverage and low gap variance
- `6src_rotate` still has sparse coverage, but its target-aware placement gives the best current sparse compromise for keeping subject coverage from collapsing
- `12src_uniform` is much better than `12src_rotate` and `nearest_ring`, but it still does not yet flip the overall branch decision in favor of `depth_unproject`

## Training Gate Recommendation

This is the current gate after the full inference-side diagnostics:

1. `23cam_fullset` passes the inference gate strongly.
2. `6src_hist + rotate_template_offsets` is the best current sparse baseline if a sparse regime is required immediately.
3. `12src_uniform` is worth keeping as a secondary sparse probe, but it is not strong enough yet to be the primary long-training sparse baseline.
4. `12src_nearest` should not be used as the main sparse baseline.
5. The next optimization target should still be source-policy / geometry-chain work, not restoration of the old `ghost` stack.

Concretely, the recommended next order remains:

1. keep original VGGT
2. keep `camera=True`, `depth=True`, `point=False`, `track=False`
3. use `depth + camera -> unproject -> render` as the geometry-first branch under evaluation
4. use dense/full-rig as the strongest reference setting
5. if sparse is needed now, prefer `6src_hist + rotate_template_offsets`
6. if a `12src` profile is specifically required, prefer `uniform_ring`
7. only after that keep the minimal extra term path, such as `loss_unproject_geometry`
8. do not restore `ghost`/mask/bbox/confidence-stack training objectives

## Bottom Line

The mentor's main direction remains supported:

- the current render path question has been answered
- `point/world_points` is the path that was effectively producing the rendered output we were inspecting
- switching the render input to `depth + camera -> unprojected points` is meaningful and already strongly supported in dense-view conditions

The new nuance from diagnostics is:

- dense-view evidence is already strong enough to justify continuing the geometry chain
- sparse-view behavior depends heavily on source policy
- the best immediate sparse baseline is not "restore ghost loss," but "choose the right source policy first"
