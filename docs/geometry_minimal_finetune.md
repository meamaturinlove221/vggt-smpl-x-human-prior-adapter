# Geometry Minimal Fine-tune

This is the second-stage entry after the geometry baseline suggests that the `depth + camera` chain is worth pursuing.

Principles:

- stay on the original VGGT training stack
- keep `camera + depth`
- keep `point` and `track` disabled
- do not bring back the legacy ghost stack
- change one variable at a time

## 1. Prepare the 5080 Environment

Install the local 5080 environment plus training dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local_5080_env.ps1 -InstallTrainingDeps
```

## 2. Dry-run the Fine-tune Command

The wrapper below uses the original `training/launch.py` entry, with runtime overrides so you do not need to edit `training/config/default.yaml` by hand.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_geometry_minimal_finetune.ps1 `
  -Co3dDir "D:\path\to\co3d" `
  -Co3dAnnotationDir "D:\path\to\co3d_anno" `
  -Checkpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  -DryRun
```

If you are not sure where your CO3D data lives on this machine, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\find_co3d_candidates.ps1
```

This writes a quick scan result to:

- [co3d_candidates_targeted.md](/f:/vggt/vggt-main/output/co3d_candidates_targeted.md)
- [co3d_candidates_targeted.json](/f:/vggt/vggt-main/output/co3d_candidates_targeted.json)

## 3. Local 5080 Starter Settings

The wrapper defaults are conservative for a 16 GB Windows desktop GPU:

- `max_img_per_gpu=4`
- `accum_steps=4`
- `max_epochs=5`
- `limit_train_batches=100`
- `limit_val_batches=50`
- aggregator frozen

This is intended as a short validation run, not a final training recipe.

## 4. What the Wrapper Forces

The wrapper always keeps the training close to the geometry-first direction:

- `model.enable_camera=True`
- `model.enable_depth=True`
- `model.enable_point=False`
- `model.enable_track=False`
- `loss.point=null`
- `loss.track=null`

So the fine-tune remains focused on the original camera and depth chain.

## 5. When to Move to Modal

Use Modal only after:

- the local geometry baseline is understood
- the dataloader paths are verified
- the short local fine-tune command is syntactically correct

Then move the same minimal setup to Modal instead of reviving the older ghost-heavy training pipeline.

See:

- [modal_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_geometry_minimal_finetune.py)
- [modal_geometry_minimal_finetune.md](/f:/vggt/vggt-main/docs/modal_geometry_minimal_finetune.md)
