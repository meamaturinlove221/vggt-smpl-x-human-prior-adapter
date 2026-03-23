# Geometry Direction Status 2026-03-23 Confgate Completed

## Executive status

- A confidence-gated geometry-loss variant is now implemented and tested.
- This is the first auxiliary geometry-loss variant that gets close enough to baseline to be worth keeping.
- It still does **not** beat baseline yet, but it is clearly the strongest candidate in the current loss family.

## What changed

- [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - `compute_unproject_geometry_loss` now supports optional depth-confidence gating
  - the geometry loss can use `predictions["depth_conf"]` as detached per-pixel weights
  - quantile filtering now supports keeping aligned masks for weighted reduction
- New config:
  - [zju_vggt_geom_unproject_confgate_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_confgate_w005_minimal.yaml)
  - recipe:
    - `weight: 0.05`
    - `use_depth_conf_gate: True`
    - `detach_depth_conf: True`
    - `depth_conf_power: 1.0`

## Local smoke

- The new config passed local smoke:
  - [confgate smoke log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_confgate_w005_smoke_local_v1/log.txt)

## Completed cloud pair

- Modal app:
  - `ap-T50LU7DvxhjBP1NV4OTHHf`
- Final state:
  - `stopped`
- Remote pair root:
  - `/geometry_pairs/20260323_065320_zju_geom_modal_pair_4000step_a10080fast_confgate_w005_v1`
- Local summary:
  - [confgate summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_065320_zju_geom_modal_pair_4000step_a10080fast_confgate_w005_v1/summary.md)

## Main result

Validation objective:

- baseline: `0.0555`
- confgate `w=0.05`: `0.0569`
- delta: `+0.0014`

Train objective:

- baseline: `0.1263`
- confgate `w=0.05`: `0.1292`
- delta: `+0.0029`

Compared with earlier candidates:

- `confgate_w0.05`: `+0.0014`
- `fixed_w0.05`: `+0.0040`
- `fixed_w0.2`: `+0.0043`
- `fixed_w0.1`: `+0.0048`
- `warmup_w0.2`: `+0.0084`

Sweep table:

- [geometry_candidate_sweep_4000step_summary.md](/f:/vggt/vggt-main/output/geometry_pairs_cloud/geometry_candidate_sweep_4000step_summary.md)

## Interpretation

This is the cleanest readout so far:

- pure weight tuning helped only a little
- warmup hurt
- confidence-gating helps much more than either of those
- the model seems to benefit when geometry loss is applied more strongly to pixels that the depth branch itself already trusts

But the honest bottom line is still:

- it has not crossed baseline
- so it is not yet justified to replace the current baseline training recipe

## Recommendation

Current survivor hierarchy is now:

1. `confgate_w0.05`
2. `fixed_w0.05`
3. everything else in this auxiliary-loss family should be deprioritized

The most reasonable next move, if continuing this line, is:

- keep `confgate_w0.05` as the only active auxiliary geometry-loss candidate
- do not keep expanding generic weight sweeps
- if one more experiment is needed, make it narrowly targeted on the gate itself, for example:
  - a small confidence threshold
  - a different confidence exponent
  - or a replicate run to check whether the remaining `+0.0014` gap is stable

## Cleanup status

- No active redundant Modal app remains.
- No repo-scoped local `powershell/python/modal/cmd` process remained after final cleanup.
