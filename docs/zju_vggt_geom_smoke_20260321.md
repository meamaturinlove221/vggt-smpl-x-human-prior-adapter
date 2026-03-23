# ZJU VGGT Geometry Smoke 2026-03-21

This note records the first successful local Windows smoke fine-tune run for the geometry-first ZJU pseudo-supervision path.

## Goal

Validate that the original VGGT codebase can run a minimal local fine-tune loop on:

- local machine: `Windows + RTX 5080 16GB`
- dataset: `ZJU_MoCap / CoreView_390`
- supervision source: cached `vggt_geom`
- enabled heads: `camera + depth`
- disabled heads: `point + track`

The purpose of this run is infrastructure validation, not final training quality.

## Command

Executed from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_zju_vggt_geom_minimal_finetune.ps1 `
  -ExpName zju_vggt_geom_smoke_local_v3 `
  -LimitTrainBatches 1 `
  -LimitValBatches 1 `
  -MaxEpochs 1 `
  -NumImages 4 `
  -MaxImgPerGpu 4 `
  -AccumSteps 1 `
  -NumWorkers 0
```

The wrapper now resolves these defaults automatically:

- `ZjuDir`: `G:\数据集\datasets\ZJU_MoCap\data\zju_mocap`
- `Checkpoint`: `G:\项目备份\vggt_小感度不起作用\vggt\model.pt`

## Result

The smoke run completed successfully:

- model instantiated
- train dataset instantiated
- val dataset instantiated
- checkpoint loaded
- 1 training batch finished
- checkpoint save finished
- 1 validation batch finished

Key metrics from [log.txt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/log.txt):

- train objective: `4.3503`
- train camera loss: `0.0515`
- train depth conf loss: `3.5250`
- val objective: `3.4253`
- val camera loss: `0.0561`
- val depth conf loss: `2.5854`

## Outputs

- [log.txt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/log.txt)
- [model.txt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/model.txt)
- [checkpoint.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/ckpts/checkpoint.pt)
- [checkpoint_0.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/ckpts/checkpoint_0.pt)

## Code Changes That Made This Work

- [run_zju_vggt_geom_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_zju_vggt_geom_minimal_finetune.ps1)
  - switched to top-level `zju_*` Hydra overrides
  - fixed default local path resolution for Chinese Windows paths
- [trainer.py](/f:/vggt/vggt-main/training/trainer.py)
  - added single-process fallback when `WORLD_SIZE=1`
  - skipped DDP wrapping and barrier in local smoke mode
  - fixed checkpoint saving when model is not wrapped by DDP
  - fixed exact batch-limit semantics
  - fixed objective scalar logging for validation
- [dynamic_dataloader.py](/f:/vggt/vggt-main/training/data/dynamic_dataloader.py)
  - added single-replica sampler fallback when distributed is not initialized
- [zju_vggt_geom_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_minimal.yaml)
  - added `_self_` to defaults to avoid Hydra composition warning

## Remaining Non-Blocking Issues

- AMP API calls still emit future warnings; they do not block execution
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not supported on this platform; it is only a warning
- loading the original checkpoint reports unexpected `point_head` and `track_head` keys, which is expected because those heads are disabled in this minimal geometry config

## Interpretation

This smoke result is enough to support the current execution policy:

- local 5080 for quick geometry validation and minimal ablations
- Modal only after the geometry branch or new supervision term is worth a longer run
- do not restore the old ghost-heavy objective stack at this stage
