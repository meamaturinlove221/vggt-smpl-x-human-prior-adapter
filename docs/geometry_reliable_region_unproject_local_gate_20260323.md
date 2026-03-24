# Reliable-Region Unproject Local Gate 2026-03-23

## Goal

This note records the first mentor-aligned implementation of a minimal
"reliable-region" treatment for the `unproject_geometry` auxiliary loss.

The motivation came from the mentor's concern that the ZJU cached depth target
may be less reliable near:

- foreground silhouette edges
- ground / bottom-band regions
- background-adjacent zones

The intended response was deliberately narrow:

- keep the original VGGT `camera + depth` baseline intact
- keep the geometry-first direction
- do not reopen the old `ghost` stack
- do not introduce another generic `confgate / threshold / pow` sweep
- only restrict the auxiliary `unproject_geometry` loss to a stricter target-side region

## Code Changes

The implementation was added to the current original-VGGT training path.

### Dataset / batch plumbing

- [zju_vggt_geom.py](/f:/vggt/vggt-main/training/data/datasets/zju_vggt_geom.py)
  - now preserves raw `foreground_masks` separately from `point_masks`
  - this keeps the existing `min_depth_conf` filtering semantics unchanged while still exposing the original foreground region
- [composed_dataset.py](/f:/vggt/vggt-main/training/data/composed_dataset.py)
  - now forwards `foreground_masks` into the training batch when present

### Loss-side treatment

- [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - `compute_unproject_geometry_loss` now supports:
    - `use_foreground_region_mask`
    - `foreground_erode_px`
    - `foreground_drop_bottom_ratio`
  - added helper `build_reliable_foreground_region_mask(...)`
  - the helper:
    - optionally erodes the foreground mask to avoid uncertain silhouette bands
    - optionally drops the bottom image band to suppress likely floor / ground-contact zones
  - this mask only affects the auxiliary `loss_unproject_geometry`
  - it does **not** rewrite the baseline `camera` or `depth` losses

### Config

- [zju_vggt_geom_unproject_reliable_region_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_reliable_region_w005_minimal.yaml)
  - starts from the current minimal ZJU geometry recipe
  - keeps:
    - `camera_source=gt`
    - `mask_source=mask`
    - `zju_min_depth_conf=0.0`
    - `enable_camera=True`
    - `enable_depth=True`
    - `enable_point=False`
    - `enable_track=False`
  - adds only:
    - `loss.unproject_geometry.weight = 0.05`
    - `use_foreground_region_mask = True`
    - `foreground_erode_px = 5`
    - `foreground_drop_bottom_ratio = 0.2`

## Local Smoke

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_zju_vggt_geom_minimal_finetune.ps1 `
  -Config zju_vggt_geom_unproject_reliable_region_w005_minimal `
  -ExpName zju_vggt_geom_unproject_reliable_region_smoke_local_v1 `
  -NumImages 4 `
  -MaxImgPerGpu 2 `
  -AccumSteps 1 `
  -MaxEpochs 1 `
  -LimitTrainBatches 2 `
  -LimitValBatches 1 `
  -NumWorkers 0
```

Result:

- run completed successfully on local Windows + RTX 5080
- no crash
- no dataloader failure
- no loss-shape failure
- no batch-key mismatch around `foreground_masks`

Key values from [log.txt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_reliable_region_smoke_local_v1/log.txt):

- `train_loss_objective = 3.9870`
- `train_loss_unproject_geometry = 0.6566`
- `val_loss_objective = 2.9088`
- `val_loss_unproject_geometry = 0.6731`

Interpretation:

- the new reliable-region path is numerically stable
- the mentor-aligned masking treatment is now genuinely implemented in code
- this was enough to justify one short local A/B gate before any cloud launch

## Local A/B Gate

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_zju_unproject_geometry_ablation_pair.ps1 `
  -BaselineConfig zju_vggt_geom_minimal `
  -CandidateConfig zju_vggt_geom_unproject_reliable_region_w005_minimal `
  -ExpPrefix zju_vggt_geom_reliable_region_probe_local_v1 `
  -NumImages 4 `
  -MaxImgPerGpu 2 `
  -AccumSteps 1 `
  -MaxEpochs 1 `
  -LimitTrainBatches 5 `
  -LimitValBatches 2 `
  -NumWorkers 0
```

Outputs:

- [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_reliable_region_probe_local_v1/summary.md)
- [summary.json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_reliable_region_probe_local_v1/summary.json)
- [baseline log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_reliable_region_probe_local_v1_baseline/log.txt)
- [candidate log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_reliable_region_probe_local_v1_unproject/log.txt)

Key readout:

- train objective:
  - baseline: `2.5827`
  - reliable-region candidate: `2.6161`
  - delta: `+0.0334`
- val objective:
  - baseline: `1.3937`
  - reliable-region candidate: `1.4268`
  - delta: `+0.0331`
- candidate-only auxiliary term:
  - `val_loss_unproject_geometry = 0.6439`

## Decision

The local gate is mixed but not good enough for cloud:

- implementation: **pass**
- local numerical stability: **pass**
- short local objective improvement: **fail**

So the current decision is:

- keep this code path in the repo
- do **not** launch a cloud pair from it yet
- do **not** promote it to the mainline training recipe yet

## Current Interpretation

This result is different from the earlier global `zju_min_depth_conf` threshold line:

- the old scalar target filtering was a target-side confidence threshold
- this new branch is a region-restricted auxiliary geometry treatment

So the negative local gate does **not** mean the mentor's concern was wrong.

It means:

- the concern now has a real implementation in the codebase
- but the first short local recipe does not yet show enough optimization benefit to justify cloud time

## Next Step

Until stronger local evidence appears, the project should keep the current priority order:

1. keep the geometry-first direction
2. keep `depth + camera -> unproject` as the main inference-side geometry hypothesis
3. do not reopen the old `ghost` stack
4. do not relaunch cloud for this reliable-region treatment yet
5. continue diagnosis where evidence is stronger:
   - sparse-view source-policy sensitivity
   - hard-camera current-vs-legacy gap

