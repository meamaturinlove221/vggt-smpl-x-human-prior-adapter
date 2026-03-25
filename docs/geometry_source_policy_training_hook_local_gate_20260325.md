# Geometry Source-Policy Training Hook Local Gate (2026-03-25)

## Summary

- A minimal training-side `source_policy` hook is now implemented for the ZJU pseudo-geometry dataset.
- Supported policies are:
  - `random`
  - `nearest_ring`
  - `uniform_ring`
- The hook is wired into the baseline training configs, and a local candidate scaffold now exists:
  - [zju_vggt_geom_unproject_source_policy_uniform_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_uniform_minimal.yaml)
- The local finetune launcher now resolves the real non-empty `ZJU_DIR` for a requested `geom_subdir`, so main-cache source-policy smokes no longer depend on manually passing the correct root.
- The current default `vggt_geom` cache on the real main root still exposes only `4` views per frame, so with `fix_img_num=4` the new policy hook remains inert on the intended default recipe.

## Implemented

- Dataset hook:
  - [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
- Probe updates:
  - [probe_zju_vggt_geom_dataset.py](/f:/vggt/vggt-main/scripts/probe_zju_vggt_geom_dataset.py)
  - now auto-detects a valid local ZJU root instead of assuming the old `F:` path
  - now defaults to `allow_duplicate_img = false` so source-policy probes do not silently fall back to duplicate random sampling
  - now reports `available_view_count`, `camera_names`, and `selection_anchor_camera`
- Launcher root resolution:
  - [run_zju_vggt_geom_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_zju_vggt_geom_minimal_finetune.ps1)
  - now resolves `ZjuDir` against `seq_names + geom_subdir + frame_*.npz`
  - this fixes the local case where `F:\...` existed but `CoreView_390/vggt_geom` itself was empty
- Cache inventory:
  - [audit_zju_geom_cache_inventory.py](/f:/vggt/vggt-main/scripts/audit_zju_geom_cache_inventory.py)
  - inventories local ZJU roots / geom_subdirs / sampled view counts and headroom
- Baseline config hooks:
  - [zju_vggt_geom_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_minimal.yaml)
  - [zju_vggt_geom_unproject_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_minimal.yaml)
- Candidate scaffold:
  - [zju_vggt_geom_unproject_source_policy_uniform_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_uniform_minimal.yaml)

## Cache Inventory

- Inventory summary:
  - [summary.md](/f:/vggt/vggt-main/output/zju_geom_cache_inventory_20260325_090442/summary.md)
- key readout:
  - real main `vggt_geom` root is the non-empty `G:` dataset root, not the empty `F:` copy
  - `vggt_geom`: `frame_count = 1171`, sampled views `4..4`, `headroom@4 = False`, `headroom@3 = True`
  - `vggt_geom_4v_backup`: also `4..4`
  - `vggt_geom_test6` / `vggt_geom_mvdebug_local`: `6..6`
- conclusion:
  - the current blocker is now narrowed to the intended `4-image` recipe on the real main cache, not to path resolution or to the source-policy hook itself

## Cache Union Check

- Variant compare summary:
  - [summary.md](/f:/vggt/vggt-main/output/zju_geom_cache_compare_CoreView_390_vggt_geom_vs_vggt_geom_4v_backup_20260325_092447/summary.md)
- key readout:
  - `common_frame_count = 1171`
  - `sampled_union_size_max = 4`
  - `adds_headroom_over_left_at_num_images_4 = False`
  - `adds_new_cameras_in_any_sample = False`
- conclusion:
  - the two currently available full-length cache variants are camera-identical
  - simply combining `vggt_geom` with `vggt_geom_4v_backup` does not create any new source-selection headroom
  - so the remaining blocker is not “wire multiple existing caches together”, but “obtain a truly wider candidate-view pool”

## Probes

- Default main cache probe:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_uniform_policy_20260325_v3/summary.json)
  - key readout:
    - `geom_subdir = vggt_geom`
    - `available_view_count = 4`
    - `num_images = 4`
    - `selection_anchor_camera = null`
  - conclusion:
    - the hook is present, but there is no selection headroom under the current main cache

- Main cache probe with headroom:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_uniform_policy_maincache_num3_autoroot_20260325/summary.json)
  - key readout:
    - `geom_subdir = vggt_geom`
    - `available_view_count = 4`
    - `num_images = 3`
    - `selection_anchor_camera = Camera_B5`
  - conclusion:
    - on the real main cache, the hook does activate as soon as the requested image count drops below the cached view count

- Multi-view debug cache probe:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_uniform_policy_test6_20260325/summary.json)
  - key readout:
    - `geom_subdir = vggt_geom_test6`
    - `available_view_count = 6`
    - `num_images = 4`
    - `selection_anchor_camera = Camera_B10`
    - selected cameras:
      - `Camera_B10, Camera_B23, Camera_B19, Camera_B5`
  - conclusion:
    - the new training-side `uniform_ring` path does activate correctly when the cache provides more views than the requested image count

## Training Smoke

- Local 1-train/1-val smoke:
  - log dir:
    - [zju_vggt_geom_unproject_source_policy_uniform_smoke_20260325](/f:/vggt/vggt-main/logs/zju_vggt_geom_unproject_source_policy_uniform_smoke_20260325)
  - runtime:
    - `geom_subdir = vggt_geom_test6`
    - `config = zju_vggt_geom_unproject_source_policy_uniform_minimal`
    - `limit_train_batches = 1`
    - `limit_val_batches = 1`
- key readout:
  - the training chain initialized, loaded checkpoint, ran one train batch, saved checkpoints, and completed one val batch locally
  - train `loss_objective = 3.9278`
  - val `loss_objective = 3.2882`
- conclusion:
  - the candidate config is no longer only a YAML scaffold
  - it is verified end-to-end on the multi-view debug cache
  - the remaining blocker is the main `vggt_geom` cache headroom, not the training entry itself

- Main cache 1-train/1-val smoke with headroom:
  - log dir:
    - [zju_vggt_geom_unproject_source_policy_uniform_maincache_num3_smoke_20260325](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_source_policy_uniform_maincache_num3_smoke_20260325)
  - runtime:
    - `geom_subdir = vggt_geom`
    - `num_images = 3`
    - launcher auto-resolved `zju_dir` to the real non-empty main root
  - key readout:
    - train `loss_objective = 3.3979`
    - val `loss_objective = 3.9662`
  - conclusion:
    - the source-policy candidate is now verified end-to-end on the real main cache when there is any selection headroom
    - the unresolved blocker is specifically the `4-view cache + num_images=4` combination used by the intended default recipe

- Multi-subdir union probe:
  - [summary.json](/f:/vggt/vggt-main/output/zju_vggt_geom_probe_uniform_policy_uniondebug_num4_20260325/summary.json)
  - key readout:
    - `geom_subdir = vggt_geom_4v_backup,vggt_geom_test6`
    - `available_view_count = 8`
    - `num_images = 4`
    - `geom_subdirs_present = ['vggt_geom_4v_backup', 'vggt_geom_test6']`
  - conclusion:
    - the dataset-side multi-subdir union path does enlarge the source pool under the intended `4-image` recipe

- Multi-subdir union 1-train/1-val smoke:
  - log dir:
    - [zju_vggt_geom_unproject_source_policy_uniform_uniondebug_smoke_20260325](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_source_policy_uniform_uniondebug_smoke_20260325)
  - config:
    - [zju_vggt_geom_unproject_source_policy_uniform_union_debug_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_source_policy_uniform_union_debug_minimal.yaml)
  - key readout:
    - train `loss_objective = 3.6586`
    - val `loss_objective = 3.5403`
  - conclusion:
    - the local code path is now verified end-to-end even for `num_images = 4` when a multiview add-on cache exists
    - the remaining blocker is no longer dataset/launcher/probe support, but the lack of a full-length multiview add-on cache on the real training split

## Full-length Add-on Plan

- Add-on coverage summary:
  - [summary.md](/f:/vggt/vggt-main/output/zju_multiview_addon_plan_CoreView_390_vggt_geom_4v_backup_20260325_094424/summary.md)
- Generated manifest:
  - [full_length_addon_manifest.json](/f:/vggt/vggt-main/output/zju_multiview_addon_plan_CoreView_390_vggt_geom_4v_backup_20260325_094424/full_length_addon_manifest.json)
- key readout:
  - `base_frame_count = 1171`
  - `addon_frame_union_count = 2`
  - `missing_addon_frame_count = 1169`
  - `recommended_extra_cameras = [Camera_B10, Camera_B14, Camera_B19, Camera_B23]`
  - `recommended_target_union_cameras = [Camera_B1, Camera_B10, Camera_B13, Camera_B14, Camera_B19, Camera_B23, Camera_B5, Camera_B9]`
- conclusion:
  - the current local blocker is now fully specified
  - we no longer just know that a full-length multiview add-on is missing; we know exactly how many frames are missing and which stable extra cameras the add-on needs to supply

## Decision

- Keep the dataset-side source-policy hook.
- Keep the local candidate config scaffold.
- Keep the local training smoke result as proof that the new source-policy path is wired into the actual training chain.
- Do **not** open cloud.
- Treat the current blocker as:
  - the real main `vggt_geom` cache is already `4`-view sparse
  - so the new policy hook cannot change the intended `num_images=4` training batch composition yet
  - root-resolution, training-entry, and multi-subdir union support are no longer the blocker
  - the unresolved piece is full-length multiview coverage, not local code wiring

## Next Local Requirement

- If this training question continues locally, the next required step is not another cloud run.
- It is one of:
  - make a multi-view cache variant available for the real training split
  - or add a raw-view sampler / higher-level source-selection path that is not capped by the current `4`-view cache
  - if the cache path is chosen, the generated full-length add-on manifest is now the concrete local gap spec
  - then rerun the same source-policy smoke under the intended `num_images=4` recipe before any cloud consideration
