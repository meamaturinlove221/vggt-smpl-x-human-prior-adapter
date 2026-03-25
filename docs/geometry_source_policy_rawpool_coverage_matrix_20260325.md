# Geometry Source-Policy Rawpool Coverage Matrix (2026-03-25)

## Summary

- This matrix now tracks the current local-only lead:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml)
- The older `nearest_rawpool` config is retained only as a historical stage marker:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml)
- The current blocker remains unchanged:
  - `val_loss_camera` and `val_loss_T` are locally stable
  - `val_loss_conf_depth` and `val_loss_reg_depth` are still worse than baseline
  - therefore the line is still local-only and cloud stays closed

## Coverage Matrix

| Item | Status | Evidence | Decision |
| --- | --- | --- | --- |
| `post-v9` inference patch collection should stop by default | covered in nightly + consistency check | [nightly_decision.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/nightly_decision.json), [consistency_check.md](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/consistency_check.md) | fixed as `patch_collection_stop = true` |
| `cloud_gate` must stay closed by default | covered in state + manifests | [state.json](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/state.json), [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | fixed as `cloud_gate = false` |
| Current training lead should be `confdepth_dropworst_gradconfmask` | covered in config + gate doc + manifests | [zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml), [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md), [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json) | fixed as current lead |
| Older `nearest_rawpool` should be treated as historical rather than current | covered in gate progression docs | [geometry_source_policy_rawpool_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_local_gate_20260325.md), [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md) | historical stage only |
| `min_supervised_views = 2` should not re-enter nightly | covered in gate docs + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | frozen |
| `source_anchor_policy = max_depth_conf` should not re-enter nightly | covered in gate docs + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | frozen |
| `dropworst_supervised` should not re-enter nightly | covered in compare doc + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | frozen |
| `trainmix50` should not re-enter nightly | covered in compare doc + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_trainmix50_vs_lead_20260325_v1/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | frozen |
| `active_view_mean` should not re-enter nightly | covered in compare doc + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_confdepth_dropworst_gradconfmask_viewmean_vs_lead_20260325_v1/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | frozen as redundant |
| "exclude rawpool-only views from geometry supervision" should not be reopened as a standalone candidate | covered by code + audit | [loss.py](/f:/vggt/vggt-main/training/loss.py), [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py), [summary.md](/f:/vggt/vggt-main/output/zju_source_policy_supervision_audit_nearest_rawpool_20260325_v1/summary.md) | frozen as redundant on current code path |
| Default nightly should be `diagnostic_only` and run `conf_depth attribution v3` | covered in nightly plan + manifest | [geometry_source_policy_rawpool_nightly_plan_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_nightly_plan_20260325.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | fixed |
| Completed `conf_depth attribution v3` should constrain the next candidate family | covered by latest local audit + manifest | [summary.md](/f:/vggt/vggt-main/output/zju_conf_depth_attribution_confdepth_dropworst_gradconfmask_20260325_v3/summary.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | fixed as `anchor_conditioned_conf_target_normalization` |
| A single-candidate local gate is still allowed for one genuinely new `conf_depth semantics` candidate | covered in nightly plan + manifest | [geometry_source_policy_rawpool_nightly_plan_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_nightly_plan_20260325.md), [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json) | allowed under one-candidate rule |

## Still Open

- The remaining open problem is not "missing source-policy hook".
- The remaining open problem is not "rawpool-only views directly leak into geometry loss".
- The remaining open problem is not another sampler/exposure aggregation toggle.
- The remaining open problem is:
  - the current `confdepth_dropworst_gradconfmask` lead is already anchor-only on the audited val slice
  - there is still a `conf_depth / reg_depth` gap to baseline
  - the completed v3 diagnostic localizes the remaining positive camera-level gap to anchor `Camera_B1`
  - the next allowed move is one anchor-conditioned `conf_depth semantics` candidate, not another sampler/exposure branch

## Single Source Of Truth

- [consistency_check.md](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/consistency_check.md)
- [geometry_post_v9_next_training_question_20260325.md](/f:/vggt/vggt-main/docs/geometry_post_v9_next_training_question_20260325.md)
- [zju_next_training_question_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_training_question_v1.json)
- [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json)
