# Geometry Post-v9 Next Training Question (2026-03-25)

## Summary

- The post-v9 residual line remains locally bounded.
- The project is still in local-only mode. No cloud launch is allowed.
- The current local lead is [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml).
- This lead preserves the `depth + camera -> unproject -> render` geometry chain and keeps the same `nearest_ring + geom_plus_raw` input path.
- It also fixes a real local bug in the depth loss path: `grad+conf` now respects the narrowed conf-depth mask.
- Even after that fix, the lead is still not baseline-clean on `conf_depth` and `reg_depth`, so cloud stays off.
- A fresh `conf_depth` aggregation test also showed that `active_view_mean` is redundant under the current 4-image recipe; it is numerically identical to the current lead on the tighter gate.
- The completed `conf_depth attribution v3` localized the remaining positive camera-level gap to anchor `Camera_B1`, but the single allowed train-only `anchorb1confscale05` follow-up failed the tighter local gate and is now frozen.

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
- Current semantics audit:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3/summary.md)
  - `conf_depth` is already anchor-only on the audited 32-sample val slice, with zero extra-supervised and zero source-only conf-valid pixels.
- Failed anchor-conditioned follow-up:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)
  - `loss_conf_depth: 0.2289 -> 0.3233`
  - `loss_reg_depth: 0.1760 -> 0.1901`
- Redundant aggregation candidate:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
  - `active_view_mean` does not materially move `camera`, `T`, `conf_depth`, or `reg_depth` versus the current lead.

## Next Question

- Repo-level manifest:
  - [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json)
- Recommended question:
  - what genuinely new local training question remains after the completed `Camera_B1`-conditioned conf-target normalization round failed the tighter gate?
- Keep rejected:
  - [dropworst_supervised vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md)
  - [bestanchor local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md)
  - [min_supervised_views=2 local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md)
  - [trainmix50 vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_trainmix50_vs_lead_20260325_v1/summary.md)
  - [viewmean vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
  - [anchorb1confscale05 vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_anchorb1confscale05_vs_lead_20260325_v1/summary.md)

## Cloud Gate

- `cloud_gate` stays `false`.
- No Modal launch is allowed from this state.
- Cloud can only be reconsidered after the remaining baseline gap is closed locally and a fresh manual decision is made.
