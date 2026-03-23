# Modal Geometry Minimal Fine-tune

This is the cloud counterpart of the local geometry-first fine-tune workflow.

It keeps the scope narrow:

- stay on the original VGGT repository
- keep `camera + depth`
- keep `point` and `track` disabled
- do not revive the old ghost stack
- move to Modal only after the local baseline supports the geometry direction

## 1. Install What the Local Launcher Needs

If your local environment does not already have Modal installed, add it to the 5080 environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local_5080_env.ps1 -InstallModalDeps
```

## 2. Prepare Modal Volumes

The new Modal script assumes two volumes:

- data volume: `vggt-geometry-data`
- output volume: `vggt-geometry-output`

Override them with environment variables if your existing project already uses different names:

```powershell
$env:VGGT_MODAL_DATA_VOLUME = "your-data-volume"
$env:VGGT_MODAL_OUTPUT_VOLUME = "your-output-volume"
```

Inside the data volume, place:

- the CO3D dataset directory, for example `co3d`
- the CO3D annotation directory, for example `co3d_annotations`
- the checkpoint, for example `checkpoints/model.pt`

The script can upload the checkpoint for you from a local path. It does not try to upload CO3D automatically because that dataset is too large for an ad hoc launcher.

## 3. Optional Checkpoint Upload Only

If you want to upload the checkpoint before launching training:

```powershell
modal run .\modal_geometry_minimal_finetune.py::upload_checkpoint `
  --local-path "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  --remote-subpath "checkpoints/model.pt"
```

## 4. Launch the Minimal Geometry Fine-tune

Windows wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_modal_geometry_minimal_finetune.ps1 `
  -Co3dSubdir "co3d" `
  -Co3dAnnotationSubdir "co3d_annotations" `
  -LocalCheckpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  -CheckpointSubpath "checkpoints/model.pt" `
  -ExpName "geometry_minimal_modal"
```

Direct Modal CLI:

```powershell
modal run .\modal_geometry_minimal_finetune.py::run_geometry_finetune `
  --co3d-subdir "co3d" `
  --co3d-annotation-subdir "co3d_annotations" `
  --local-checkpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  --checkpoint-subpath "checkpoints/model.pt" `
  --exp-name "geometry_minimal_modal"
```

## 5. What the Modal Launcher Forces

The cloud launcher mirrors the local minimal wrapper:

- `model.enable_camera=True`
- `model.enable_depth=True`
- `model.enable_point=False`
- `model.enable_track=False`
- `loss.point=null`
- `loss.track=null`

It also redirects logging and checkpoints into the Modal output volume instead of the repo directory.

## 6. Recommended Usage

Use this script only after:

- the local A/B baseline has been inspected
- `depth + camera` is at least competitive with `point map`
- your CO3D paths inside the Modal data volume are confirmed

For now, the intended sequence stays the same:

1. local 5080 baseline
2. local small-sample validation
3. Modal minimal fine-tune
4. only then consider one extra geometry/reconstruction supervision term
