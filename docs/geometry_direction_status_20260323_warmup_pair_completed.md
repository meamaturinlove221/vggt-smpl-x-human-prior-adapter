# Geometry Direction Status 2026-03-23 Warmup Pair Completed

## Executive status

- A third-round single-variable geometry experiment is now complete.
- Change tested:
  - keep the same `unproject_geometry` loss definition
  - keep the same final weight `0.2`
  - only add a linear warmup on the geometry-loss weight
- Result:
  - the warmup version did **not** improve the previous `4000-step` cloud result
  - it is worse than the fixed-weight `unproject_geometry` run on validation objective

## What changed in code

- [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - added an optional config-driven warmup schedule for auxiliary loss weight
- [trainer.py](/f:/vggt/vggt-main/training/trainer.py)
  - exposes per-batch normalized training progress as `train_progress`
- [zju_vggt_geom_unproject_warmup_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_warmup_minimal.yaml)
  - new candidate config with:
    - `weight: 0.2`
    - `warmup_start: 0.0`
    - `warmup_end: 0.25`
    - `warmup_init_factor: 0.0`

## Local validation before cloud

- A local smoke run succeeded with the new config:
  - [local smoke log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_warmup_smoke_local_v1/log.txt)
- The smoke also confirmed the warmup behavior:
  - first train batch objective matched `camera + depth` only
  - the geometry term was not injected at full weight from step 0

## Completed cloud pair

- Modal app:
  - `ap-IitJqSmVZdtfoeDQQ9LHjZ`
- Final state:
  - `stopped`
- Remote pair root:
  - `/geometry_pairs/20260323_022730_zju_geom_modal_pair_4000step_a10080fast_warmup_v1`
- Local summary:
  - [warmup pair summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_022730_zju_geom_modal_pair_4000step_a10080fast_warmup_v1/summary.md)
- Previous fixed-weight reference:
  - [fixed-weight 4000-step summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_005034_zju_geom_modal_pair_4000step_a10080fast_v1/summary.md)

## Main comparison

Validation objective against baseline:

- fixed-weight `unproject_geometry`
  - baseline: `0.0546`
  - candidate: `0.0589`
  - delta: `+0.0043`
- warmup `unproject_geometry`
  - baseline: `0.0545`
  - candidate: `0.0629`
  - delta: `+0.0084`

Train objective against baseline:

- fixed-weight `unproject_geometry`
  - baseline: `0.1260`
  - candidate: `0.1348`
  - delta: `+0.0088`
- warmup `unproject_geometry`
  - baseline: `0.1252`
  - candidate: `0.1338`
  - delta: `+0.0086`

## Interpretation

The warmup experiment gives a clean answer:

- warmup slightly reduced train-side damage
- but it made validation objective noticeably worse
- so the issue is probably not just “the geometry term turns on too early”

This means the current best reading is:

- the mentor's geometry-chain direction is still correct
- the first-round branch swap result still stands
- but the current auxiliary `unproject_geometry` formulation is not fixed by a simple weight warmup

## Recommended next step

Do not keep iterating on warmup schedules as the main fix.

The next single-variable candidate should be more conservative and simpler:

1. lower the maximum `unproject_geometry` weight itself
2. keep the rest of the recipe unchanged
3. compare again against the same `zju_vggt_geom_minimal` baseline

The most reasonable immediate follow-up is:

- test a smaller geometry-loss weight, for example `0.05` or `0.1`
- do not reintroduce `ghost` logic
- do not mix multiple new geometry tricks into the same run

## Cleanup status

- No active redundant Modal app remains.
- No repo-scoped local `powershell/python/modal/cmd` process remained after launch cleanup and completion.
