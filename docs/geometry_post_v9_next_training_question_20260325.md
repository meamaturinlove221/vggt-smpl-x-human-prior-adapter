# Geometry Post-v9 Next Training Question (2026-03-25)

## Summary

- The post-v9 residual line remains locally bounded.
- The project is still in local-only mode. No cloud launch is allowed.
- The current stable local lead is [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml).
- This lead preserves the `depth + camera -> unproject -> render` geometry chain and keeps the same `nearest_ring + geom_plus_raw` input path.
- It inherits the real `grad+conf` mask-respect fix and remains the last tested lead that is locally stable under validation.
- Two bounded `Camera_B1` conf-target normalization follow-ups were tried locally: the broader whole-frame variant failed the tighter short gate, and the narrower foreground-bottom20 variant later failed the `100 train / 20 val` long gate after only a marginal short-gate win.
- The line is still not baseline-clean on `conf_depth` and `reg_depth`, so cloud stays off.
- `active_view_mean` remains redundant under the current 4-image recipe, and the tested fixed-scale B1-conditioned variants are now frozen.
- A new quality-signal audit exposed a bounded quality-conditioned follow-up, and the first `Camera_B1 + quality>=3.0 + foreground-bottom20 + scale0.5` candidate has now been tested locally.
- That first quality-conditioned candidate kept `camera/T` flat but only improved `conf_depth` by `0.0001` and left `reg_depth` flat, so it did not clear promotion.
- A new postmortem shows the hard `q>=3.0` gate was also over-selective: it still excluded `4 / 10` bad `Camera_B1` audit rows and touched only `0.7476%` of conf-depth-supervised pixels on a 512-sample train slice.
- A follow-up quality-region boundary check shows the region is too narrow as well: all `10 / 10` bad `Camera_B1` audit rows still have positive non-bottom `conf_depth` deltas, while `q>=3.0 + whole_foreground` covers `22.6975%` of conf-depth-supervised pixels on a fresh 512-sample train slice versus only `0.7983%` for `q>=3.0 + bottom20`.
- A further rule-shape sweep shows the next question should not be another broader hard-threshold whole-foreground sibling either: `q>=3.0 + whole_foreground` still omits `4 / 10` bad audit rows, `q>=2.75 + whole_foreground` still omits `1 / 10`, while a continuous `qmin -> qmax` whole-foreground reference rule covers all `10 / 10` bad rows with a `5.1155%` effective conf-depth reduction fraction on the same fresh 512-sample train slice.
- The continuous loss knob is now implemented locally, but both tested whole-foreground continuous follow-ups are now frozen: the first exact `linear qmin -> qmax + whole_foreground + scale0.5` candidate regressed both `conf_depth` and `reg_depth`, and the softer `scale0.75` fallback recommended by the local soft sweep also regressed both depth terms.
- The line therefore returns to `steady_hold` until a fresh manual question justifies an even softer or reshaped quality-conditioned rule.

## Current Lead

- Gate note:
  - [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md)
- Compare versus previous local lead:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_lead_20260325_v1/summary.md)
- Compare versus baseline:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_baseline_20260325_v1/summary.md)
- Val deltas versus `confdepth_dropworst_only`:
  - `loss_camera: 0.0219 -> 0.0219`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2514 -> 0.2289`
  - `loss_reg_depth: 0.1799 -> 0.1760`
- Remaining blocker versus baseline:
  - `loss_conf_depth: 0.1181 -> 0.2289`
  - `loss_reg_depth: 0.1181 -> 0.1760`
- Long-gate demotion evidence for the attempted successor:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_current_lead_longgate_100t_20v_overnight_20260326_v1_vs_previous_lead/summary.md)
  - `loss_camera: 0.0498 -> 0.0499`
  - `loss_T: 0.0209 -> 0.0210`
  - `loss_conf_depth: -0.0577 -> -0.0496`
  - `loss_reg_depth: 0.0911 -> 0.0965`
- Diagnostic evidence that motivated the exhausted B1 family:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3/summary.md)
  - `conf_depth` was already anchor-only on the audited 32-sample val slice.
  - the only clearly positive camera-level delta remained concentrated in `Camera_B1`, especially in the foreground bottom band.
- Broader failed B1 follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
  - `loss_conf_depth: 0.2289 -> 0.3233`
  - `loss_reg_depth: 0.1760 -> 0.1901`
- Narrower B1-bottom20 follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1bottom20confscale05_vs_lead_20260325_v1/summary.md)
  - short gate: `loss_conf_depth: 0.2289 -> 0.2287`
  - short gate: `loss_reg_depth: 0.1760 -> 0.1759`
  - long gate status: failed to hold promotion
- Quality-signal follow-up audit:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_quality_signal_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - `all_anchor corr(quality_score, delta_conf_depth) = 0.4404`
  - `Camera_B1 corr(quality_score, delta_conf_depth) = 0.6555`
  - `non_B1 delta_conf_depth_mean = -0.5302`
- First quality-conditioned follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_vs_lead_20260326_v1/summary.md)
  - `loss_camera: 0.0219 -> 0.0219`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2288 -> 0.2287`
  - `loss_reg_depth: 0.1759 -> 0.1759`
  - `loss_objective: 0.2451 -> 0.2452`
- Quality-conditioned postmortem:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_conditioned_candidate_postmortem_anchorb1qge3bottom20_20260326_v1/summary.md)
  - bad `Camera_B1` audit rows below `quality_min=3.0`: `4 / 10`
  - affected conf-depth-supervised pixel fraction on 512 train samples: `0.7476%`
- Quality-region boundary check:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_region_boundary_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - positive non-bottom `conf_depth` rows on bad `Camera_B1`: `10 / 10`
  - `q>=3.0 + whole_foreground` conf-depth-supervised pixel fraction on a fresh 512 train-sample slice: `22.6975%`
  - `q>=3.0 + bottom20` conf-depth-supervised pixel fraction on the same slice: `0.7983%`
- Quality rule-shape sweep:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_rule_shape_sweep_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - `q>=3.0 + whole_foreground` omitted bad audit rows: `4 / 10`
  - `q>=2.75 + whole_foreground` omitted bad audit rows: `1 / 10`
  - continuous `qmin -> qmax + whole_foreground` effective reduction fraction on a fresh 512 train-sample slice: `5.1155%`
- First exact continuous whole-foreground follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale05_vs_lead_20260326_v1/summary.md)
  - `loss_camera: 0.0219 -> 0.0218`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2289 -> 0.2776`
  - `loss_reg_depth: 0.1760 -> 0.1848`
- Softer continuous-rule sweep:
  - [summary.md](/f:/vggt/vggt-main/output/zju_quality_continuous_soft_rule_sweep_confdepth_dropworst_gradconfmask_20260326_v1/summary.md)
  - recommended fallback rule: `linear_qmin_qmax_wholefg_scale075`
  - effective reduction fraction: `5.9433% -> 2.9716%`
  - bad `Camera_B1` audit rows still covered: `10 / 10`
- First softer continuous whole-foreground follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale075_vs_lead_20260326_v1/summary.md)
  - `loss_camera: 0.0219 -> 0.0218`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2289 -> 0.2483`
  - `loss_reg_depth: 0.1760 -> 0.1797`
- Redundant aggregation candidate:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
  - `active_view_mean` does not materially move `camera`, `T`, `conf_depth`, or `reg_depth` on the tighter gate.

## Next Question

- Repo-level manifest:
  - [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json)
- Recommended question:
  - after both the first exact and first softer continuous `Camera_B1` whole-foreground quality-conditioned rules (`linear qmin -> qmax`, `scale0.5` and `scale0.75`) regressed `conf_depth` and `reg_depth` versus the stable lead, is any even softer or reshaped quality-conditioned rule worth approving as a fresh manual question without harming `camera/T`?
- Implementation note:
  - [composed_dataset.py](/f:/vggt/vggt-main/training/data/composed_dataset.py) now forwards `selection_anchor_quality_score` into the training sample, and [loss.py](/f:/vggt/vggt-main/training/loss.py) now supports both `anchor_conditioned_conf_target_quality_min/max` and `anchor_conditioned_conf_target_quality_interp='linear'` with `quality_low/high`; the remaining blocker is no longer missing plumbing but that both the first exact and first softer whole-foreground rules already failed, so any future manual question must justify an even softer or reshaped rule that improves both `conf_depth` and `reg_depth`.
- Keep rejected or frozen:
  - [dropworst_supervised vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md)
  - [bestanchor local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md)
  - [min_supervised_views=2 local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md)
  - [trainmix50 vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_trainmix50_vs_lead_20260325_v1/summary.md)
  - [viewmean vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
  - [anchorb1confscale05 vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
  - [anchorb1bottom20 long gate vs stable lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_current_lead_longgate_100t_20v_overnight_20260326_v1_vs_previous_lead/summary.md)
  - [anchorb1qge3bottom20 quality-conditioned vs stable lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_vs_lead_20260326_v1/summary.md)
  - [anchorb1qge3bottom20 postmortem](/f:/vggt/vggt-main/output/zju_quality_conditioned_candidate_postmortem_anchorb1qge3bottom20_20260326_v1/summary.md)
  - [anchorb1qlinearwholefgconfscale05 vs stable lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale05_vs_lead_20260326_v1/summary.md)
  - [anchorb1qlinearwholefgconfscale075 vs stable lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale075_vs_lead_20260326_v1/summary.md)

## Cloud Gate

- `cloud_gate` stays `false`.
- No Modal launch is allowed from this state.
- Cloud can only be reconsidered after the remaining baseline gap is closed locally, consistency remains clean, and a fresh manual decision is made.
