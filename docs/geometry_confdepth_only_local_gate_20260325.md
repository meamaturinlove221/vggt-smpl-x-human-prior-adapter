# Geometry Conf-Depth-Only Local Gate (2026-03-25)

## Summary

- `dropworst_supervised` remains rejected because it changes the full supervision route, not just the conf-depth branch.
- The current local lead is `confdepth_dropworst_gradconfmask`.
- It keeps the same `nearest_ring + geom_plus_raw` input path and the same `conf_depth-only` drop-worst rule.
- It also fixes a real local loss-routing issue: the `grad+conf` depth branch now respects the narrowed `conf_depth` mask instead of confidence-weighting pixels that were already removed from conf-depth supervision.
- This lead improves `val_loss_conf_depth` and `val_loss_reg_depth` versus the previous `confdepth_dropworst_only` lead while keeping `val_loss_camera` and `val_loss_T` flat.
- It is still not cloud-ready because `conf_depth` and `reg_depth` remain materially worse than baseline.

## Implemented

- Dataset and batch routing:
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
  - `point_masks` still drive `reg_depth`, `camera`, and `unproject`.
  - `conf_depth_point_masks` only narrow `loss_conf_depth`.
- Loss routing:
  - [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - `compute_depth_loss()` now accepts `respect_conf_mask_in_grad_conf`.
  - `regression_loss()` can keep plain gradient supervision while removing confidence weighting outside `conf_depth_point_masks`.
- Current local-lead config:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)

## Evidence

- Intermediate `confdepth_dropworst_only` gate:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_vs_lead_20260325_v1/summary.md)
  - versus `nearest_rawpool` on val:
  - `loss_camera: 0.0218 -> 0.0219`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2696 -> 0.2514`
  - `loss_reg_depth: 0.1831 -> 0.1799`
- Current `gradconfmask` gate:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_lead_20260325_v1/summary.md)
  - versus `confdepth_dropworst_only` on val:
  - `loss_camera: 0.0219 -> 0.0219`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2514 -> 0.2289`
  - `loss_reg_depth: 0.1799 -> 0.1760`
  - `loss_unproject_geometry: 0.1869 -> 0.1830`
- Current lead versus baseline:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_vs_baseline_20260325_v1/summary.md)
  - baseline vs current lead on val:
  - `loss_conf_depth: 0.1181 -> 0.2289`
  - `loss_reg_depth: 0.1181 -> 0.1760`
- Current lead attribution:
  - [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v1/summary.md)
  - on the audited 32-sample val slice, `conf_depth` is already anchor-only:
  - `anchor_supervised conf_valid_pixels = 450648`
  - `extra_supervised conf_valid_pixels = 0`
  - `source_only conf_valid_pixels = 0`
- Redundant aggregation candidate:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
  - switching `loss_conf_depth` from pixel-mean to active-view-mean is effectively identical on the tighter gate:
  - `loss_camera: 0.0219 -> 0.0219`
  - `loss_T: 0.0003 -> 0.0003`
  - `loss_conf_depth: 0.2289 -> 0.2289`
  - `loss_reg_depth: 0.1760 -> 0.1759`
  - conclusion: under the current 4-image recipe, this is not a new lead; it is a redundant variant because only one conf-depth-active view contributes per sample on the audited slice.

## Decision

- Promote `confdepth_dropworst_gradconfmask` to the current local lead.
- Do not promote to cloud.
- Keep `cloud_gate = false`.
- Keep rejected:
  - [dropworst_supervised vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md)
  - [bestanchor local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md)
  - [min_supervised_views=2 local gate](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md)
  - [trainmix50 vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_trainmix50_vs_lead_20260325_v1/summary.md)
  - [viewmean vs lead](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md)
