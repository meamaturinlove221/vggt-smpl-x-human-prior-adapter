# Modal Local Stability 2026-03-22

## What was causing the local machine to feel unstable

The repeatable high-risk path was the launch side, not the actual cloud training loop.

Two concrete local hazards were present:

1. Very large local artifacts existed under `training/logs/**/ckpts/*.pt`.
   - many of these files are about `6.168 GB` each
2. The Modal launch path could still become expensive or fragile if a huge local checkpoint was passed in again.

This combination made it easy to misread the situation as "cloud training is heavy on the local machine",
when in practice the local machine was getting stressed before the remote job even started cleanly.

## Fixes applied

### 1. Added a hard local preflight

New script:

- [invoke_modal_zju_preflight.ps1](/f:/vggt/vggt-main/scripts/invoke_modal_zju_preflight.ps1)

It now checks:

- free local RAM before launch
- unexpected repo-scoped local `python/modal` processes
- active matching Modal apps
- presence of large local artifacts
- dangerous large local checkpoint upload attempts

It refuses large local checkpoint uploads unless explicitly overridden.

### 2. Added long-run launch guards

Updated:

- [run_modal_zju_geometry_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_geometry_minimal_finetune.ps1)

Changes:

- runs preflight by default
- blocks long non-detached runs unless explicitly overridden
- keeps the safer "remote checkpoint fallback" path as the default

### 3. Reworked the paired Modal runner

Updated:

- [run_modal_zju_unproject_geometry_ablation_pair.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_unproject_geometry_ablation_pair.ps1)
- [modal_zju_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_zju_geometry_minimal_finetune.py)

Changes:

- the pair launcher now targets a single detached remote orchestration entrypoint instead of keeping the local machine attached through two sequential runs
- baseline and `unproject_geometry` are now orchestrated by one remote pair function
- pair status is designed to be written under the pair output root once the remote function actually starts

### 4. Shrunk the Modal code sync scope

Before this pass, the launch path mounted the whole repo and relied on ignore rules.

Now the training image only syncs:

- `training/`
- `vggt/`

That reduced the effective code payload for the training code itself to well under `1 MB`:

- `training/` excluding logs and ckpts: about `54` files, `0.0004 GB`
- `vggt/`: about `75` files, `0.0005 GB`

This is the most important local-stability improvement from today.

## Current cloud status

As of `2026-03-22` evening:

- all old dangling Modal apps were stopped
- there are no redundant active cloud apps left running
- the latest detached smoke attempts created apps but stayed at:
  - `State = ephemeral (detached)`
  - `Tasks = 0`

This means the remaining bottleneck is no longer the old local huge-artifact path.
The remaining issue is earlier in the Modal startup chain, likely before the remote training function body begins producing output.

## Practical interpretation

The local crash / memory-risk path has been materially reduced.

What is still unresolved is not "training OOM on the local 5080".
What is unresolved is "why the detached Modal launch is not advancing from app creation into real remote task execution quickly enough".

That should be debugged next from the Modal startup side, not by reviving the old local heavy workflow.
