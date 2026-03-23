# Modal ZJU Geometry Progress 2026-03-22 Evening

## What is now fixed

The local crash-prone launch path is no longer the blocker.

The main fixes now in place are:

- hard local preflight before cloud launch
- direct remote-function launch instead of relying on the old `local_entrypoint` path
- base64 config payloads so Windows PowerShell does not corrupt JSON arguments
- reduced Modal code sync scope to `training/` and `vggt/`
- remote requirement resolution fallback for Modal container imports
- corrected recursive requirements parsing so `torch` and `torchvision` actually enter the Modal image
- single-process fallback for missing `LOCAL_RANK` / `RANK`
- background launch wrapper so long Modal launches can keep running without tying up the foreground shell

## Smoke result

The minimal single-run smoke finally completed successfully.

Successful run:

- app: `ap-pOxCwXjDHmG87IGs0EYsLP`
- exp: `zju_geom_modal_baseline_smoke_remote_v6`
- output root:
  - `vggt-out/geometry_smoke/zju_geom_modal_baseline_smoke_remote_v6`

Observed in logs:

- checkpoint fallback assembly worked
- model initialized and resumed from checkpoint
- one train step ran
- validation ran
- checkpoints were written
- app completed cleanly

## Current overnight run

The paired cloud run is active.

Current run:

- app: `ap-1gh2quvjXOu2dL8YxeQAKt`
- exp prefix: `zju_geom_modal_pair_500step_remote_v1`
- output root:
  - `vggt-out/geometry_pairs/20260322_201558_zju_geom_modal_pair_500step_remote_v1`

Current state when last checked:

- Modal app state: `ephemeral (detached)`
- tasks: `1`
- baseline stage had already entered training
- `pair_status.json` already exists under the output root

Known paths:

- pair status:
  - `vggt-out/geometry_pairs/20260322_201558_zju_geom_modal_pair_500step_remote_v1/pair_status.json`
- baseline output:
  - `vggt-out/geometry_pairs/20260322_201558_zju_geom_modal_pair_500step_remote_v1/baseline`
- local launcher log:
  - [zju_geom_modal_pair_500step_remote_v1.out.log](/f:/vggt/vggt-main/output/zju_geom_modal_pair_500step_remote_v1.out.log)

## Important note

Modal warned that the existing volumes are high on inode usage:

- `/mnt/out`: about `91%`
- `/mnt/data`: about `85%`

This is not blocking the current run yet, but it is now a real operational risk for later runs if many more files are created.
