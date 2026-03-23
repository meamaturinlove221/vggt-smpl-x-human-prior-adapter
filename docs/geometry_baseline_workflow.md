# Geometry Baseline Workflow

This workflow implements the first step of the new VGGT direction:

- stay on the original VGGT codebase
- do not reintroduce the legacy ghost stack
- compare `point map` against `depth + camera -> unprojection` under the same inference pass

## What Runs Locally

Use the local baseline first on the Windows `RTX 5080 16GB` machine for:

- original inference smoke tests
- small-sample branch comparisons
- single-case visual inspection
- offline review of point clouds and target-view re-render results

If the local 5080 environment is not ready yet, bootstrap it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_local_5080_env.ps1
```

The local baseline script is:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_geometry_baseline.ps1 `
  -ImageFolder .\examples\kitchen\images `
  -MaxImages 8 `
  -TargetFrames 0
```

Or call Python directly:

```bash
python scripts/compare_geometry_branches.py \
  --image_folder examples/kitchen/images \
  --max_images 8 \
  --target_frames 0
```

If the script exits before inference with an `sm_120` or `no kernel image is available` style error, your local PyTorch build is too old for the `RTX 5080`. Upgrade to a PyTorch build that supports this GPU before treating local CUDA as available. Until then, use local runs only as CPU smoke tests or move the real baseline to Modal.

## Output Artifacts

The script writes a new run folder under `output/geometry_baseline/` by default.

Key files:

- `summary.md`: compact report for quick reading
- `summary.json`: structured metrics
- `render_metrics.csv`: per-target comparison metrics
- `ply/point_map.ply`: filtered point cloud from the point-map branch
- `ply/depth_unproject.ply`: filtered point cloud from the `depth + camera` branch
- `renders/target_XXX_compare.png`: target image, both branch re-renders, and difference panels

The re-render comparison uses the same target frame for both branches. By default it excludes the target frame's own points so the comparison is not dominated by trivial self-copy.

## Decision Rule

After the first local run, answer these questions before changing training:

- Is the `depth + camera` branch more complete?
- Is it less noisy or less ghosted in the re-rendered target view?
- Does it keep the subject intact without obvious collapse or cut-off?

If the answer is yes, keep the next step on the geometry chain. Only then consider the smallest possible extra geometry or reconstruction loss.

If the answer is no, first inspect camera prediction stability, depth quality, and input view consistency. Do not jump back to the previous ghost stack.

## What Stays On Modal

Keep Modal for later stages only:

- formal fine-tuning
- sweep jobs
- overnight runs
- larger-batch or longer-duration experiments

This repo now includes a minimal Modal launcher for the geometry-first path:

- [modal_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_geometry_minimal_finetune.py)
- [modal_geometry_minimal_finetune.md](/f:/vggt/vggt-main/docs/modal_geometry_minimal_finetune.md)

It is intentionally narrow and does not bring back the old ghost-heavy training pipeline.
