# ZJU Geometry View Sweep Round 1

This note records the first large target-view sweep after the single-case ZJU baseline had already confirmed that `depth + camera` can beat `point map` on `CoreView_390 / frame 1080 / Camera_B5`.

The key difference here is scope:

- the earlier baseline fixed one target view
- this sweep changes the target camera and frame
- the goal is to answer the mentor's question more honestly at the `view` level

## Setup

Sequence:

- `CoreView_390`

Frames:

- `0`
- `150`
- `300`
- `450`
- `600`
- `750`
- `900`
- `1080`
- `1170`

Profiles:

- `6src_hist`
- `12src_nested`
- `23cam_fullset`

Target-camera policy:

- `6src_hist`: all cameras not already inside the fixed 6-view source subset
- `12src_nested`: all cameras not already inside the fixed 12-view source subset
- `23cam_fullset`: every target camera, with sources redefined as "all other cameras"

That produced:

- `459` cases
- `0` failures

Artifacts:

- [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.md)
- [summary.json](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.json)
- [summary.csv](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.csv)
- [sweep_manifest.json](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/sweep_manifest.json)

## Overall Readout

Across all `459` cases:

- `depth_unproject` wins: `199`
- `point_map` wins: `163`
- ties: `97`

So the answer is no longer "geometry always wins".

The real answer after a large view sweep is:

- the geometry chain is strongly supported in some regimes
- but it is not yet a universal winner once the target view changes broadly

## By Profile

### `23cam_fullset`

This is the strongest result for the mentor's geometry-chain idea:

- runs: `207`
- `depth_unproject` wins: `167`
- `point_map` wins: `24`
- ties: `16`
- average MAE gain: `+0.001915`
- average coverage gain: `+0.021472`

Frame-level pattern:

- `300`, `450`, `750`, `1080`: `23 / 23` depth wins
- `600`: `22 / 23` depth wins
- `150`: `20 / 23` depth wins
- `0`: `14 / 23` depth wins
- `900`: `17 / 23` depth wins
- `1170`: only `2 / 23` depth wins

Interpretation:

- when the source set is dense and always covers the rig except the target itself, `depth + camera -> unproject -> render` is strongly supported
- the geometry-chain direction is therefore real, not just a paper claim
- but even this dense setting degrades near the tail frame `1170`

### `6src_hist`

This sparse fixed subset does **not** generalize well across target views:

- runs: `153`
- `depth_unproject` wins: `28`
- `point_map` wins: `83`
- ties: `42`
- average MAE gain: `-0.001673`
- average coverage gain: `-0.003066`

Interpretation:

- the earlier `Camera_B5` result was good
- but once the target camera changes and the same 6-view subset is kept fixed, the advantage mostly disappears
- this suggests the sparse subset was at least partly target-specific

### `12src_nested`

This medium subset also does **not** generalize well across target views:

- runs: `99`
- `depth_unproject` wins: `4`
- `point_map` wins: `56`
- ties: `39`
- average MAE gain: `-0.000442`
- average coverage gain: `-0.026664`

Interpretation:

- even with more views than `6src_hist`, the fixed subset still favors `point_map` on most target views
- again, this means we should not generalize the original single-target success too broadly

## What This Means For The Mentor Question

The mentor's decision path was:

1. confirm the current rendering path
2. if rendering mainly depends on `point / world points`, test `camera + depth -> unprojected points -> rendering`
3. judge whether that geometry path is a better foundation

After this round, the answer is:

- **yes**, that geometry path is a strong and defensible direction
- but the strongest support is currently in the **dense / full-rig** setting
- for **sparse fixed source subsets**, the result is still target-view dependent and often favors `point_map`

So the correct conclusion is more precise than before:

- do not go back to the old ghost stack
- keep the geometry-chain direction
- but do not claim that "just switching the render point source" already solves every sparse-view case

## Why The New Result Does Not Contradict The Old Single-Case Baseline

The earlier baseline on `frame 1080 / Camera_B5` is still valid.

What changed is only the scope:

- before: one target view
- now: many target views and many frames

The large sweep shows that the single-case result was **real but local**:

- especially real for `23cam_fullset`
- less transferable for `6src_hist` and `12src_nested`

## Current Recommendation

1. Keep `23cam_fullset` as the strongest evidence for the geometry-chain direction.
2. Treat the sparse-profile failures as a view-selection problem or geometry-quality problem, not as a reason to restore ghost objectives.
3. The next most useful sweep is not "more ghost loss", but one of:
   - target-aware source subset design
   - re-derived `6src` / `12src` source subsets for each target camera
   - better sparse-view camera/depth calibration quality checks
4. If a render-path change is proposed for the real project, phrase it as:
   - "already well supported for dense multi-view input"
   - "still needs more work for sparse fixed-view subsets"

## Reproduction Command

This is the command used for the round-1 sweep:

```powershell
.\.venv5080\Scripts\python.exe .\scripts\run_zju_geometry_view_sweep.py `
  --template_reports `
    "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\6src_hist\CoreView_390\frame_001080_Camera_B5\run_20260316_110745\report.json" `
    "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\12src_nested\CoreView_390\frame_001080_Camera_B5\run_20260316_110256\report.json" `
    "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\23cam_fullset\CoreView_390\frame_001080_Camera_B5\run_20260316_111217\report.json" `
  --local_zju_root "G:\数据集\datasets\ZJU_MoCap\data\zju_mocap" `
  --checkpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  --output_root "output/geometry_view_sweep_zju/round1_coreview390_v1" `
  --frame_ids "0,150,300,450,600,750,900,1080,1170" `
  --target_cameras all `
  --skip_existing
```
