# Geometry Direction Status 2026-03-22 Night

## What is now settled

- The local Windows crash-prone path has been materially reduced.
- The active detached Modal pair run completed cleanly.
- The pure geometry branch comparison already exists in this repo and is the better answer to the mentor's immediate question than tonight's `unproject_geometry` fine-tune pair.

## Local stability status

### Confirmed local risks

- The repo still contains many local checkpoints under `training/logs/**/ckpts/*.pt` around `6.168 GB` each.
- The local Anaconda PyTorch build does **not** support the RTX 5080 (`sm_120`), so local CUDA inference is currently blocked without upgrading PyTorch.

### Fixes now in place

- [invoke_modal_zju_preflight.ps1](/f:/vggt/vggt-main/scripts/invoke_modal_zju_preflight.ps1)
  - default free-memory floor raised from `8 GB` to `12 GB`
  - detects repo-scoped `powershell/python/modal` launch residue
  - can auto-stop stale detached launcher trees before the next cloud launch
- [run_modal_zju_geometry_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_geometry_minimal_finetune.ps1)
  - now requests stale local launcher cleanup during preflight
- [run_modal_zju_unproject_geometry_ablation_pair.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_unproject_geometry_ablation_pair.ps1)
  - now requests stale local launcher cleanup during preflight

### Validation

- The previously lingering local launcher tree for the detached Modal pair run disappeared without interrupting the remote app.
- After cleanup, the Modal app still completed successfully and no active redundant Modal apps remained.

## Tonight's cloud pair run

- App: `ap-1gh2quvjXOu2dL8YxeQAKt`
- Output root:
  - `vggt-out/geometry_pairs/20260322_201558_zju_geom_modal_pair_500step_remote_v1`
- Final pair status:
  - `state = completed`
  - both `baseline` and `unproject` stages completed

### Important caveat

This pair run is **not** the mentor's pure first-round ablation.

- `baseline` used:
  - `zju_vggt_geom_minimal`
- `unproject` used:
  - `zju_vggt_geom_unproject_minimal`

The second config adds a new loss term:

- [zju_vggt_geom_unproject_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_minimal.yaml)
  - `loss.unproject_geometry`

So this run answers:

- "What happens if we add the current `unproject_geometry` supervision?"

It does **not** answer:

- "What happens if we only switch the rendered point source from `world_points` to `depth + camera` without adding new loss?"

### Observed outcome

Pulled from the final validation log lines:

- baseline:
  - `val_loss_objective avg = 0.0995`
  - `val_loss_camera avg = 0.0123`
  - `val_loss_conf_depth avg = 0.0168`
- unproject-loss candidate:
  - `val_loss_objective avg = 0.1117`
  - `val_loss_camera avg = 0.0125`
  - `val_loss_conf_depth avg = 0.0183`
  - `val_loss_unproject_geometry avg = 0.0407`

Interpretation:

- this extra-loss variant did **not** beat the plain geometry baseline on the main validation objective in this 500-step run
- this makes it even more important not to skip the mentor's pure source-path comparison

## The mentor-aligned first-round result that already exists

The repo already contains source-only geometry branch comparisons:

- [batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_zju_batch/batch_summary.md)
- [coreview390_6src_hist summary](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_6src_hist/summary.md)
- [coreview390_12src_nested summary](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_12src_nested/summary.md)
- [coreview390_23cam_fullset summary](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_23cam_fullset/summary.md)

Batch readout:

- runs: `3`
- `depth_unproject` wins: `2`
- `point_map` wins: `0`
- ties: `1`

Per profile:

- `6src_hist`
  - decision: `depth_unproject`
  - point MAE: `0.0344`
  - depth MAE: `0.0331`
  - point coverage: `0.4535`
  - depth coverage: `0.5068`
- `12src_nested`
  - decision: `tie`
  - point MAE: `0.0397`
  - depth MAE: `0.0385`
  - point coverage: `0.3365`
  - depth coverage: `0.3162`
- `23cam_fullset`
  - decision: `depth_unproject`
  - point MAE: `0.0444`
  - depth MAE: `0.0415`
  - point coverage: `0.2210`
  - depth coverage: `0.2522`

Interpretation:

- for the mentor's current question, the repo already has evidence that `depth + camera -> unprojected points` is at least competitive and often better than `point_map`
- this supports staying on the geometry-first route and *not* reviving the old ghost stack as the main branch

## Practical next step

The next clean experiment should be:

1. Keep the original geometry baseline.
2. Run only the point-source comparison path.
3. Do **not** add `unproject_geometry` or ghost-related losses in that first confirmation round.
4. Only after that, decide whether a minimal new geometry loss is worth adding.
