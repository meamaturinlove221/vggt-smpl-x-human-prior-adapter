# Bottom-Only Unproject Local Gate 2026-03-23

## Goal

This note records a narrower follow-up to the earlier reliable-region
`unproject_geometry` treatment.

The mentor's concern was now more specific:

- the ZJU cached depth target may be unreliable near the ground / bottom band
- a full foreground-region restriction may still be too broad
- the failure pattern seen in sparse hard cameras keeps concentrating in
  `bg_bottom_band`

So this follow-up intentionally kept the same minimal geometry-first recipe but
reduced the auxiliary mask intervention to a smaller change:

- keep the original VGGT `camera + depth` baseline intact
- keep the geometry-first direction
- do not reopen the old `ghost` stack
- do not add another target-threshold sweep
- only test whether dropping the bottom foreground band is a cleaner auxiliary
  treatment than the earlier eroded-foreground version

## Config

- [zju_vggt_geom_unproject_bottom_only_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_bottom_only_w005_minimal.yaml)
  - starts from the same minimal ZJU geometry recipe
  - keeps:
    - `camera_source=gt`
    - `mask_source=mask`
    - `zju_min_depth_conf=0.0`
    - `enable_camera=True`
    - `enable_depth=True`
    - `enable_point=False`
    - `enable_track=False`
  - adds only:
    - `loss.unproject_geometry.weight = 0.05`
    - `use_foreground_region_mask = True`
    - `foreground_erode_px = 0`
    - `foreground_drop_bottom_ratio = 0.2`

Relative to the earlier reliable-region version, this removes the `5px`
foreground erosion and keeps only the bottom-band drop.

## Local A/B Gate

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_zju_unproject_geometry_ablation_pair.ps1 `
  -BaselineConfig zju_vggt_geom_minimal `
  -CandidateConfig zju_vggt_geom_unproject_bottom_only_w005_minimal `
  -ExpPrefix zju_vggt_geom_bottom_only_probe_local_v1 `
  -NumImages 4 `
  -MaxImgPerGpu 2 `
  -AccumSteps 1 `
  -MaxEpochs 1 `
  -LimitTrainBatches 5 `
  -LimitValBatches 2 `
  -NumWorkers 0
```

Outputs:

- [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_bottom_only_probe_local_v1/summary.md)
- [summary.json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_bottom_only_probe_local_v1/summary.json)
- [baseline log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_bottom_only_probe_local_v1_baseline/log.txt)
- [candidate log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_bottom_only_probe_local_v1_unproject/log.txt)

Key readout:

- train objective:
  - baseline: `2.5827`
  - bottom-only candidate: `2.6165`
  - delta: `+0.0338`
- val objective:
  - baseline: `1.3937`
  - bottom-only candidate: `1.4269`
  - delta: `+0.0332`
- candidate-only auxiliary term:
  - `val_loss_unproject_geometry = 0.6480`

## Comparison Against The Earlier Reliable-Region Gate

Reference:

- [geometry_reliable_region_unproject_local_gate_20260323.md](/f:/vggt/vggt-main/docs/geometry_reliable_region_unproject_local_gate_20260323.md)

Earlier reliable-region readout:

- train objective delta: `+0.0334`
- val objective delta: `+0.0331`
- `val_loss_unproject_geometry = 0.6439`

Bottom-only readout:

- train objective delta: `+0.0338`
- val objective delta: `+0.0332`
- `val_loss_unproject_geometry = 0.6480`

Interpretation:

- the narrower bottom-only treatment does **not** recover the local objective
- it is not better than the earlier reliable-region probe
- both variants currently fail the same short local gate by essentially the same
  margin

## Decision

The bottom-only follow-up does not pass the local gate:

- implementation: **pass**
- local numerical stability: **pass**
- short local objective improvement: **fail**

So the current decision is:

- keep the code path available in the repo
- do **not** move this branch to cloud
- do **not** promote it to the mainline training recipe
- do **not** spend more cloud time on this auxiliary-region family until a
  stronger local signal appears

## Current Interpretation

This result sharpens the earlier conclusion.

The mentor's concern about unreliable bottom / ground-adjacent depth targets is
still well-motivated, but the first two training-side treatments are not yet
showing enough optimization benefit:

- full reliable-region auxiliary masking did not beat baseline
- bottom-only auxiliary masking also did not beat baseline

So the strongest current evidence remains on the analysis side:

- `depth_unproject` often still helps the human region
- the hardest sparse failures are dominated by `bg_far`, especially
  `bg_bottom_band`
- source-policy changes improve `bg_far` more clearly than these first two
  auxiliary masking variants improve training

## Next Step

Until a better local gate appears, the project should keep the next-step
priority order as:

1. keep the geometry-first direction
2. keep `depth + camera -> unproject` as the main inference-side geometry path
3. keep cloud off for this auxiliary-region family
4. focus on stronger evidence lines:
   - sparse-view source-policy sensitivity
   - hard-camera current-vs-legacy render gap
   - bottom/background diagnostics rather than more generic loss sweeping
