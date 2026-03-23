# Modal ZJU Geometry Minimal Fine-tune

This is the cloud counterpart of the local ZJU pseudo-geometry workflow.

The PowerShell wrapper and Modal entrypoint were dry-run validated locally on `2026-03-21`, and the defaults are now aligned to the existing Modal volumes already used in the older project:

- data volume: `vggt-zju-data`
- output volume: `vggt-out`
- ZJU root inside the data volume: `zju_mocap`

The launcher now also supports detached execution for long runs:

- use `-Detach` when the job should survive local network interruption
- leave `-LocalCheckpoint` empty by default if the remote volume already has a usable checkpoint source

It keeps the scope narrow:

- stay on the original VGGT repository
- keep `camera + depth`
- keep `point` and `track` disabled
- do not revive the old ghost stack
- keep ZJU as the same human-domain validation target used locally
- move to Modal only after the local paired baseline is understood

## 1. Install What the Local Launcher Needs

If your local environment does not already have Modal installed, add it to the 5080 environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local_5080_env.ps1 -InstallModalDeps
```

## 2. Prepare Modal Volumes

The ZJU launcher assumes two existing volumes:

- data volume: `vggt-zju-data`
- output volume: `vggt-out`

Override them if your existing project already uses different names:

```powershell
$env:VGGT_ZJU_MODAL_DATA_VOLUME = "your-zju-data-volume"
$env:VGGT_ZJU_MODAL_OUTPUT_VOLUME = "your-zju-output-volume"
```

Inside the data volume, place:

- the ZJU root, for example `zju_mocap`
- the checkpoint, for example `checkpoints/model.pt`

The launcher can upload the checkpoint for you from a local path, but it no longer does that automatically.
This is deliberate, because the local `model.pt` is several GB and repeated uploads can destabilize the local machine.

If `checkpoints/model.pt` is missing in the data volume, the Modal entrypoint now also tries a remote fallback:

- `vggt-out/weights/model.pt.part*`

Those split parts are assembled remotely, which is much safer than re-uploading the same huge checkpoint from Windows each run.

## 3. Optional Checkpoint Upload Only

If you want to upload the checkpoint before launching training:

```powershell
modal run .\modal_zju_geometry_minimal_finetune.py::upload_checkpoint `
  --local-path "G:\path\to\model.pt" `
  --remote-subpath "checkpoints/model.pt"
```

## 4. Launch The Minimal ZJU Geometry Fine-tune

Baseline example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_zju_geometry_minimal_finetune.ps1 `
  -ZjuSubdir "zju_mocap" `
  -SeqNames "CoreView_390" `
  -CheckpointSubpath "checkpoints/model.pt" `
  -Config "zju_vggt_geom_minimal" `
  -ExpName "zju_geom_minimal_modal" `
  -Detach
```

Unproject-geometry example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_zju_geometry_minimal_finetune.ps1 `
  -ZjuSubdir "zju_mocap" `
  -SeqNames "CoreView_390" `
  -CheckpointSubpath "checkpoints/model.pt" `
  -Config "zju_vggt_geom_unproject_minimal" `
  -ExpName "zju_geom_unproject_modal" `
  -Detach
```

Direct Modal CLI:

```powershell
modal run .\modal_zju_geometry_minimal_finetune.py::run_zju_geometry_finetune `
  --zju-subdir "zju_mocap" `
  --seq-names "CoreView_390" `
  --local-checkpoint "G:\path\to\model.pt" `
  --checkpoint-subpath "checkpoints/model.pt" `
  --config "zju_vggt_geom_unproject_minimal" `
  --exp-name "zju_geom_unproject_modal"
```

## 5. What The ZJU Modal Launcher Forces

The cloud launcher mirrors the local ZJU wrapper:

- `model.enable_camera=True`
- `model.enable_depth=True`
- `model.enable_point=False`
- `model.enable_track=False`
- `loss.point=null`
- `loss.track=null`

It also pushes the runtime overrides needed by the ZJU pseudo-geometry dataset:

- `zju_dir`
- `zju_seq_names`
- `zju_geom_subdir`
- `zju_camera_source`
- `zju_mask_source`
- `zju_holdout_stride`

## 6. Recommended Usage

Use this script only after:

- the local paired baseline has been inspected
- the local ZJU smoke and short paired runs are stable
- the Modal volume already contains the ZJU root

For now, the intended sequence stays the same:

1. local 5080 baseline
2. local paired baseline vs `unproject_geometry`
3. Modal ZJU minimal fine-tune
4. only then consider longer or broader geometry supervision
