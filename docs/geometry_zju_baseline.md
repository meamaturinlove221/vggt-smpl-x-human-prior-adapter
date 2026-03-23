# ZJU Geometry Baseline

This is the more task-relevant follow-up to the generic `examples/` geometry baseline.

Instead of rendering back into one of the input frames, this workflow:

- takes an old `report.json` from the human reconstruction project
- extracts the original source-view setup
- runs the original VGGT on source views only
- aligns the predicted source cameras to the real ZJU cameras with Sim(3)
- re-renders both geometry branches into the real target camera

That makes it much closer to the actual human-domain ghosting discussion.

## Main Entry

Windows wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_zju_geometry_baseline_from_report.ps1 `
  -ReportJson "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\6src_hist\CoreView_390\frame_001080_Camera_B5\run_20260316_110745\report.json"
```

Geometry-first primary wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_zju_geometry_primary_from_report.ps1 `
  -ReportJson "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\6src_hist\CoreView_390\frame_001080_Camera_B5\run_20260316_110745\report.json" `
  -PrimaryBranch depth_unproject
```

Direct Python:

```powershell
.\.venv5080\Scripts\python.exe .\scripts\compare_geometry_branches_zju_report.py `
  --report_json "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\6src_hist\CoreView_390\frame_001080_Camera_B5\run_20260316_110745\report.json" `
  --local_zju_root "G:\数据集\datasets\ZJU_MoCap\data\zju_mocap" `
  --checkpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt"
```

Every run now produces both:

- the full A/B comparison outputs
- one canonical primary geometry output for downstream use:
  - `primary_summary.md`
  - `renders/primary_render.png`
  - `renders/primary_weight.png`
  - `ply/primary_geometry.ply`

## Current Human-Domain Results

Aggregate summary:

- [coreview390_batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_batch_summary.md)

Per-profile summaries:

- [6src_hist](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_6src_hist/summary.md)
- [12src_nested](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_12src_nested/summary.md)
- [23cam_fullset](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_23cam_fullset/summary.md)

Current readout:

- `6src_hist`: `depth + camera` wins on both MAE and coverage
- `12src_nested`: `depth + camera` wins MAE, while `point map` keeps slightly higher coverage
- `23cam_fullset`: `depth + camera` wins on both MAE and coverage

So on the actual CoreView_390 human-domain case, `depth + camera` is not just a paper claim or an examples-only effect. It already wins or stays competitive across all three tested source-view profiles.

## Large Target-View Sweep Update

That earlier statement was correct for the original single target view, but I have now expanded it into a much larger sweep:

- [geometry_zju_view_sweep_round1_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_view_sweep_round1_20260322.md)
- [round1 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.md)
- [geometry_zju_view_sweep_round2_targetaware_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_view_sweep_round2_targetaware_20260322.md)
- [round2 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.md)
- [round1 vs round2 comparison](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_vs_round2_targetaware_v1/comparison.md)

Round-1 sweep scope:

- `459` cases
- `9` frames
- multiple target cameras
- all three source-view profiles

Important nuance from that larger view-level sweep:

- `23cam_fullset` strongly supports `depth + camera`
- `6src_hist` and `12src_nested` do not generalize nearly as well once the target view changes

So the strongest current claim is now:

- the geometry chain is strongly supported for dense / full-rig input
- but sparse fixed-view subsets remain target-view dependent
- this means the mentor's direction is still right, but we should not oversell the sparse-view generalization yet

Round-2 sparse follow-up:

- I re-ran the sparse setting with a target-aware rotated sparse subset instead of the old fixed `Camera_B5` subset
- `6src_hist` improved materially under this change
- `12src_nested` improved only slightly and still remained mostly `point_map`-leaning

So the updated strongest claim is:

- dense/full-rig evidence still strongly supports `depth + camera`
- sparse failures were partly caused by fixed source subsets
- this explanation is strong for `6src_hist`
- but `12src_nested` still needs a better sparse policy or better sparse geometry quality

Batch summary after deduping repeated source-camera signatures:

- [batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_zju_batch/batch_summary.md)

## Primary Geometry Artifact

Verified primary-output run:

- [primary summary](/f:/vggt/vggt-main/output/geometry_primary_zju/coreview390_6src_hist_primary/primary_summary.md)
- [full summary](/f:/vggt/vggt-main/output/geometry_primary_zju/coreview390_6src_hist_primary/summary.md)

Current readout from that materialized run:

- selected primary branch: `Depth+Camera`
- coverage ratio: `0.5068`
- MAE: `0.0331`
- the canonical output files are ready for downstream inspection without reopening the branch-comparison logic

## Why This Matters

This is stronger evidence than the generic examples baseline because it matches the real project structure:

- same human sequence family
- same target-camera setting
- same multi-view source subsets that were already discussed in the previous project

That makes the geometry-first direction much easier to defend in the next advisor discussion.
