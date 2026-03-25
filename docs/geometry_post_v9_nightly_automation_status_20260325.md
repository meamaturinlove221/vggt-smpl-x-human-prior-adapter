# ZJU post-v9 Nightly Automation Status (2026-03-25)

## Summary

- The post-v9 nightly decision protocol is implemented in local code and has completed one full `diagnose -> classify -> bounded-search gate -> stop/cloud-ready gate` cycle.
- The post-v9 nightly decision protocol now also survives a fresh auto-mode rerun after the latest source-policy local-gate updates.
- The current local main manifest remains `v9`. No `v10` manifest was generated.
- The single allowed bounded-search night has already been consumed, and it did not produce a shared donor pattern across at least two residual cases.
- The current terminal status is:
  - `patch_collection_stop = true`
  - `ready_for_new_training_question = true`
  - `cloud_gate = false`
  - `launch_cloud_now = false`
- Under the current project rule, the local residual line is not fully fixed yet, so cloud stays closed.
- The latest local training lead is now the rawpool training line:
  - [geometry_source_policy_rawpool_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_local_gate_20260325.md)
  - current lead remains `nearest_ring + geom_plus_raw` with `min_supervised_views = 1`
  - the follow-on nightly plan for that local training axis is now frozen in:
    - [geometry_source_policy_rawpool_nightly_plan_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_nightly_plan_20260325.md)
    - [zju_source_policy_rawpool_local_nightly_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_source_policy_rawpool_local_nightly_v1.json)
  - the explicit coverage matrix is now recorded in:
    - [geometry_source_policy_rawpool_coverage_matrix_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_rawpool_coverage_matrix_20260325.md)

## Implemented

- Orchestrator:
  - [run_zju_post_v9_residual_cluster_nightly.py](/f:/vggt/vggt-main/scripts/run_zju_post_v9_residual_cluster_nightly.py)
- Frozen residual manifest:
  - [zju_post_v9_residual_cluster_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_post_v9_residual_cluster_v1.json)
- Repo-level cloud pair template:
  - [zju_next_cloud_pair_v1.json](/f:/vggt/vggt-main/scripts/manifests/zju_next_cloud_pair_v1.json)
- Generic legacy backfill wrapper:
  - [run_legacy_backfill_from_current_manifest.ps1](/f:/vggt/vggt-main/scripts/run_legacy_backfill_from_current_manifest.ps1)
- Legacy backfill summarizer:
  - [summarize_legacy_backfill_manifest.py](/f:/vggt/vggt-main/scripts/summarize_legacy_backfill_manifest.py)
- Residual region compare update:
  - [compare_zju_region_batch_summaries.py](/f:/vggt/vggt-main/scripts/compare_zju_region_batch_summaries.py)
  - now supports `--ignore_view_profile` so `6src_hist` can be aligned with `12src_nested`
- Preflight self-filter fix:
  - [invoke_modal_zju_preflight.ps1](/f:/vggt/vggt-main/scripts/invoke_modal_zju_preflight.ps1)
  - now ignores the nightly orchestrator itself instead of misclassifying `run_zju_post_v9_residual_cluster_nightly.py` as a stale repo-scoped local process

## Decision Outputs

- Terminal decision after bounded-search consumption:
  - [nightly_decision.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_070838/nightly_decision.json)
  - [nightly_decision.md](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_070838/nightly_decision.md)
- Latest auto-mode stability check:
  - [nightly_decision.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/nightly_decision.json)
  - [nightly_decision.md](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/nightly_decision.md)
- Persistent state:
  - [state.json](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/state.json)
- Generated run-local cloud template:
  - [cloud_experiment_template.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/cloud_experiment_template.json)
- Generated next-training-question brief:
  - [next_training_question_brief.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/next_training_question_brief.json)
  - [next_training_question_brief.md](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_155739/next_training_question_brief.md)
- Consistency check:
  - [consistency_check.json](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/consistency_check.json)
  - [consistency_check.md](/f:/vggt/vggt-main/output/geometry_post_v9_nightly_state/consistency_check.md)
- Local source-policy hook gate:
  - [geometry_source_policy_training_hook_local_gate_20260325.md](/f:/vggt/vggt-main/docs/geometry_source_policy_training_hook_local_gate_20260325.md)
  - now includes the real-main-cache `num_images=3` smoke, the cache inventory result, and the multi-subdir union `num_images=4` smoke
- Cache-variant compare:
  - [compare_zju_geom_cache_variants.py](/f:/vggt/vggt-main/scripts/compare_zju_geom_cache_variants.py)
  - now confirms that the two current full-length cache variants are camera-identical and do not add union headroom

## Key Findings

- The 8 residual cases have been fully classified. See:
  - [cluster_summary.json](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_070838/cluster_summary.json)
  - [cluster_summary.md](/f:/vggt/vggt-main/output/geometry_post_v9_residual_cluster_nightly/20260325_070838/cluster_summary.md)
- The `repeated_cameras` group satisfied the headroom-level search gate, so nightly executed the single allowed bounded search.
- The best guard-pass search results did not share a donor pattern:
  - `B13@frame0`: `+B14 / -B1`
  - `B11@frame600`: `+B13 / -B6`
  - `B13@frame600`: `+B11 / -B1`
  - `B23@frame1080`: `+B2 / -B16`
- Because of that, the correct outcome is not `upgrade to v10`, but `stop patch collection and prepare a new training/ablation question`.
- The latest training-question brief now also records that:
  - `vggt_geom` and `vggt_geom_4v_backup` do not create any extra candidate-view headroom when combined
  - so the remaining blocker is a truly wider candidate-view pool, not a simple local cache merge
  - the local code path for multi-subdir union is already verified end-to-end, so the remaining blocker is coverage rather than local wiring
  - the frame-level camera-mask bug has been fixed, so the current local blocker is no longer the old camera regression from source-only raw views
  - a direct supervision-routing audit now shows that source-only raw views already have zero camera/depth/unproject eligibility on the current nearest_rawpool lead
  - the same direct-leak hypothesis is also false on the `dropworst_supervised` variant, which now audits down to exactly one active supervised view per sampled batch item
  - the latest tighter rawpool gate now improves objective and camera/T on the current lead, but still leaves a depth-confidence / depth-regression tradeoff
  - two simple sampler variants have already been rejected locally and should not silently re-enter the nightly lead:
    - `min_supervised_views = 2`
    - `source_anchor_policy = max_depth_conf`
  - a third sampler-side variant, `drop_worst_by_depth_conf_if_multi_supervised`, is now locally validated but still not promoted because `loss_conf_depth` worsens relative to the current lead
  - the current lead / blocker / cloud gate fields now also pass a machine-readable consistency check across `nightly_decision`, `state`, `next_training_question_brief`, and the repo-level manifests

## Current Gate

- Current automation state:
  - `latest_decision = ready_for_new_training_question`
  - `bounded_search_consumed = true`
  - `pending_bounded_search_group = null`
  - `cloud_gate = false`
- This means:
  - the local nightly automation path is implemented and reusable;
  - the inference-side residuals are not fully removed;
  - the next step, if any, should be a newly defined training/ablation problem rather than automatic cloud launch.

## Runtime Hygiene

- `modal app list` is empty.
- There are no repo-scoped `python / powershell / modal` processes left running.
- Cloud has no redundant app, and the local runtime is clean.
