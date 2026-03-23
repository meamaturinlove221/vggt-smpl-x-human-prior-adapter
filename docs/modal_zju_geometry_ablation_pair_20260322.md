# Modal ZJU Geometry Ablation Pair 2026-03-22

This note defines the cloud path for the current mentor-approved direction:

- stay on original VGGT
- keep `camera + depth`
- compare baseline vs `unproject_geometry`
- do not revive the old ghost stack

The goal here is not to change the research question.
The goal is to stop wasting local Windows `RTX 5080 16GB` time on long paired runs when Modal is already available.

## What Changed

I aligned the current Modal launcher with the volume layout that already exists in your older project:

- data volume default: `vggt-zju-data`
- output volume default: `vggt-out`
- ZJU root inside the data volume: `zju_mocap`

This matches the volume state already present in the account instead of inventing a new parallel set of empty volumes.

I also changed the checkpoint strategy to reduce local-machine pressure:

- the cloud launcher no longer auto-uploads the local `model.pt`
- it first tries the requested data-volume checkpoint path
- if that path is missing, it now falls back to the existing split checkpoint under `vggt-out/weights/model.pt.part*`
- the split checkpoint is assembled remotely instead of re-uploading `4GB+` from Windows every run

I also added a new paired launcher:

- [run_modal_zju_unproject_geometry_ablation_pair.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_unproject_geometry_ablation_pair.ps1)

It does three things:

1. upload the checkpoint once if you explicitly ask it to
2. run baseline on Modal
3. run `unproject_geometry` on Modal with the same non-geometry knobs

## Which Knobs Are Safe To Change

To keep the experiment interpretable, separate the knobs into two groups.

Safe throughput knobs:

- Modal GPU type
- `max_img_per_gpu`
- `accum_steps`
- `num_workers`
- Modal CPU / memory / timeout

Experiment-definition knobs that should stay fixed unless you intentionally want a new setting:

- `NumImages`
- source-view policy
- target-view evaluation set
- `GeomSubdir`
- ZJU sequence

The important one is `NumImages`.
That controls how many views each sample contains, so changing it changes the experiment itself.

## Recommended Profiles

The paired launcher supports these profiles:

- `strict_compare`
  - `A100-40GB`
  - `max_img_per_gpu=4`
  - closest to the local batch-1 semantics
- `a10040_balanced`
  - `A100-40GB`
  - `max_img_per_gpu=8`
  - faster, but effective batch size changes
- `a10080_balanced`
  - `A100-80GB`
  - `max_img_per_gpu=8`
  - recommended default for the next serious paired run
- `a10080_fast`
  - `A100-80GB`
  - `max_img_per_gpu=12`
  - use only if you explicitly accept a more aggressive batch-size change

Current recommendation:

- keep `NumImages=4`
- move long paired runs to `a10080_balanced`
- use `strict_compare` only when you need the cleanest continuity with the local batch-1 baseline
- for overnight jobs, prefer `-Detach` on the single-run launcher so the remote app survives local network drops
- do not pass `-LocalCheckpoint` unless you explicitly want a fresh upload

## Example Commands

Single cloud run, baseline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_zju_geometry_minimal_finetune.ps1 `
  -ZjuSubdir "zju_mocap" `
  -SeqNames "CoreView_390" `
  -GeomSubdir "vggt_geom" `
  -CheckpointSubpath "checkpoints/model.pt" `
  -Config "zju_vggt_geom_minimal" `
  -ExpName "zju_geom_modal_baseline_smoke" `
  -ModalGpu "A100-40GB" `
  -NumImages 4 `
  -MaxImgPerGpu 4 `
  -Detach `
  -LimitTrainBatches 20 `
  -LimitValBatches 5
```

Paired cloud run, recommended next step:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_zju_unproject_geometry_ablation_pair.ps1 `
  -ZjuSubdir "zju_mocap" `
  -SeqNames "CoreView_390" `
  -GeomSubdir "vggt_geom" `
  -CheckpointSubpath "checkpoints/model.pt" `
  -ExpPrefix "zju_geom_modal_pair_500step_v1" `
  -ThroughputProfile "a10080_balanced" `
  -NumImages 4 `
  -LimitTrainBatches 500 `
  -LimitValBatches 20
```

## Interpretation Rule

If you increase `max_img_per_gpu`, do not compare that run numerically against the old local batch-1 run as if they were the same training condition.

Instead:

- compare baseline vs `unproject_geometry` within the same Modal paired schedule
- then compare whether the render-side branch preference still favors `depth + camera`

That keeps the research conclusion clean even when the cloud run uses a faster batch setup.

## Practical Note On Modal Stability

Recent smoke runs showed that local network interruption can stop a non-detached Modal app with:

- `Stopping app - local client disconnected. Use modal run --detach ...`

So for any real overnight run:

- use the single-run launcher with `-Detach`
- then poll `modal app list --json`
- only launch the next stage after the previous detached app has stopped cleanly
- avoid auto-uploading the huge local checkpoint unless you truly need to refresh it
