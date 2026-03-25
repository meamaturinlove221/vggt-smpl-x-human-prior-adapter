# Geometry Source-Policy Rawpool Local Gate (2026-03-25)

## Summary

- Late update:
  - the current local lead has now moved one step narrower, from the older `nearest_rawpool` route to the new `confdepth_dropworst_only` route
  - see [geometry_confdepth_only_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_confdepth_only_local_gate_20260325.md)
  - that candidate improves `val_loss_conf_depth` and `val_loss_reg_depth` versus the previous lead while keeping `val_loss_T` flat and `val_loss_camera` effectively flat
  - but it still does not close the remaining baseline gap on the depth-side terms, so cloud stays locked

- The training-side source-policy headroom blocker is no longer just "main vggt_geom has only 4 cached views".
- A new local-only route now exists on the real main cache:
  - keep one supervised geom anchor
  - fill the remaining source-policy slots from the raw camera ring
  - leave cloud closed
- This path is implemented, probed, smoke-tested, and short-paired locally.
- It has now also been re-gated after a frame-level camera-mask fix and one tighter local validation run.
- The current local lead is:
  - [zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml)
- But the local gate is still not fully passed:
  - the tighter gate now improves `val_loss_objective`, `val_loss_camera`, and `val_loss_T`
  - but depth-confidence / depth-regression terms are still mixed
  - and a tested `min_supervised_views=2` variant was worse, not better
  - so cloud remains locked

## Implemented

- Dataset rawpool support:
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
  - new `source_view_pool = geom_plus_raw`
  - one supervised geom anchor is kept first
  - remaining views may be raw source-only cameras
- Windows non-ASCII image-path fix:
  - [dataset_util.py](/f:/vggt/vggt-main/training/data/dataset_util.py)
  - `read_image_cv2` now falls back to `np.fromfile + cv2.imdecode`
- Launcher override fix:
  - [run_zju_vggt_geom_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_zju_vggt_geom_minimal_finetune.ps1)
  - `SourceViewPool` no longer silently overwrites config defaults
- Gradient-loss fix for source-only views:
  - [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - confidence regularization is now masked to valid gradient support
- Camera-loss frame-mask fix:
  - [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - source-only raw views no longer enter camera loss just because the first frame in the sample was supervised
- Optional supervised-view floor for source-policy sampling:
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
  - new `min_supervised_views` keeps a configurable number of geom-supervised views before filling with rawpool views

## Main-cache Probe

- Probe output:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_uniform_rawpool_maincache_num4_20260325/summary.json)
- Key readout:
  - `available_view_count = 4`
  - `available_candidate_view_count = 23`
  - `camera_names = [Camera_B13, Camera_B16, Camera_B20, Camera_B3]`
  - `source_only_camera_names = [Camera_B16, Camera_B20, Camera_B3]`
- Meaning:
  - headroom on the intended `num_images = 4` recipe is now real on the main cache
  - it no longer depends on a full-length multiview add-on cache just to activate selection

- Probe with `min_supervised_views = 2`:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_nearest_rawpool_maincache_minsup2_20260325/summary.json)
- Key readout:
  - `camera_names = [Camera_B9, Camera_B13, Camera_B18, Camera_B17]`
  - `supervised_camera_names = [Camera_B9, Camera_B13]`
  - `source_only_camera_names = [Camera_B18, Camera_B17]`
- Meaning:
  - the new supervised-view floor is wired correctly on the real main cache
  - so the next question is quality, not whether the sampler can express the mix

## Smoke

- Main-cache rawpool 1-train / 1-val smoke:
  - [zju_vggt_geom_unproject_source_policy_uniform_rawpool_maincache_smoke_v2_20260325](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_source_policy_uniform_rawpool_maincache_smoke_v2_20260325)
- Key readout:
  - train `loss_objective = -37.8653` before the gradient-loss fix path was corrected
  - after the loss fix, rawpool runs no longer collapse into the old large negative `loss_grad_depth` artifact
- Meaning:
  - the rawpool path is wired into the real training chain
  - the next question is quality, not wiring

## Local Pair Results

- Uniform rawpool pair after the loss fix:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_rawpool_probe_local_20260325_v2/summary.md)
- Key readout:
  - `val loss_objective: 2.0669 -> 1.2519`
  - `val loss_camera: 0.0897 -> 1.1211`
  - `val loss_T: 0.0653 -> 1.0830`
- Meaning:
  - objective-side numbers improve
  - camera branch regresses too much
  - uniform rawpool is not the right lead

- Nearest rawpool pair after the loss fix:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_probe_local_20260325_v1/summary.md)
- Key readout:
  - `val loss_objective: 2.0669 -> 0.7655`
  - `val loss_camera: 0.0897 -> 0.2737`
  - `val loss_T: 0.0653 -> 0.2454`
- Meaning:
  - nearest rawpool is materially better than uniform rawpool on the current local gate
  - but at that stage it still had a clear camera/translation regression versus baseline

- Nearest rawpool stricter gate after the frame-level camera-mask fix:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_local_gate_20260325_v1/summary.md)
- Key readout:
  - `val loss_objective: 0.6343 -> 0.2415`
  - `val loss_camera: 0.0644 -> 0.0218`
  - `val loss_T: 0.0475 -> 0.0003`
  - `val loss_conf_depth: 0.1181 -> 0.2696`
  - `val loss_reg_depth: 0.1181 -> 0.1831`
- Meaning:
  - the frame-mask fix removed the earlier camera-branch regression on the tighter local gate
  - nearest rawpool is still the current lead
  - but the axis is not fully fixed yet because depth-confidence / regression terms still worsen

- Nearest rawpool with `min_supervised_views = 2`:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_minsup2_local_gate_20260325_v1/summary.md)
- Key readout:
  - `val loss_objective: 0.6343 -> 0.5988`
  - `val loss_camera: 0.0644 -> 0.2335`
  - `val loss_T: 0.0475 -> 0.1543`
  - `val loss_unproject_geometry: 0.2038 -> 0.4702`
- Meaning:
  - keeping two supervised geom views did not stabilize the candidate
  - it is not the new lead and should not replace the current nearest rawpool candidate

- Nearest rawpool with `source_anchor_policy = max_depth_conf`:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_bestanchor_local_gate_20260325_v1/summary.md)
- Key readout:
  - `val loss_objective: 0.6343 -> 0.9155`
  - `val loss_camera: 0.0644 -> 0.2358`
  - `val loss_T: 0.0475 -> 0.1054`
  - `val loss_conf_depth: 0.1181 -> 0.2599`
  - `val loss_unproject_geometry: 0.2038 -> 0.5787`
- Meaning:
  - cached 4-view confidence ranking alone is not a good anchor rule for this training path
  - the max-depth-conf anchor variant is rejected locally and does not replace the current lead

- Nearest rawpool with `drop_worst_by_depth_conf_if_multi_supervised`:
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_baseline_20260325_v1/summary.md)
  - [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_source_policy_nearest_rawpool_dropworst_vs_lead_20260325_v1/summary.md)
  - [summary.md](/f:/vggt/vggt-main/output/zju_source_policy_supervision_audit_nearest_rawpool_dropworst_20260325_v1/summary.md)
- Key readout:
  - audit: `32` train samples become exactly `32` supervised views and `96` source-only views
  - audit: `rawpool_geometry_leak_detected = false`
  - vs current lead on val:
    - `loss_camera: 0.0218 -> 0.0177`
    - `loss_T: 0.0003 -> 0.0003`
    - `loss_reg_depth: 0.1831 -> 0.1548`
    - `loss_conf_depth: 0.2696 -> 0.2810`
    - `loss_objective: 0.2415 -> 0.1060`
- Meaning:
  - this candidate really does what it claims at the sampler level: when multiple cached supervised views appear, it demotes the lowest-confidence one to source-only
  - but it still does not satisfy the stricter promotion rule, because `loss_conf_depth` rebounds even though `loss_reg_depth`, `loss_camera`, and `loss_objective` improve
  - so it is a real local candidate, but it is not promoted over the current lead yet

## Decision

- Keep:
  - dataset rawpool support
  - non-ASCII image-path fallback
  - gradient-loss masking fix
  - nearest rawpool as the current local lead
- Also record:
  - rawpool-only views are already excluded from direct depth/conf/reg/unproject supervision in the current code path
  - this is now backed by a local supervision audit:
    - [summary.md](/f:/vggt/vggt-main/output/zju_source_policy_supervision_audit_nearest_rawpool_20260325_v1/summary.md)
  - the `dropworst_supervised` variant is also now audited and locally gated:
    - [summary.md](/f:/vggt/vggt-main/output/zju_source_policy_supervision_audit_nearest_rawpool_dropworst_20260325_v1/summary.md)
- Do not:
  - open cloud
  - revert to global depth-threshold / reliable-region / ghost directions
  - treat the current short pair as cloud-ready proof

## Current Gate

- Current status:
  - local wiring: passed
  - local headroom activation on main cache: passed
  - local short-pair quality gate: passed after the frame-mask fix
  - tighter local gate: still not fully passed
- Blocking issue before cloud:
  - the current nearest rawpool lead still shows mixed `loss_conf_depth / loss_reg_depth` behavior under the tighter local gate
- So the correct next local step is:
  - continue local source-policy refinement or gating around the remaining depth-confidence / regression tradeoff
  - but do not keep collecting rejected sampler variants that already failed the tighter local gate (`min_supervised_views=2`, `max_depth_conf` anchor)
  - keep `cloud_gate = false`
