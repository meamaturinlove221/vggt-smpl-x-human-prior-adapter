# V15 SMPL-X Native Execution Rollup

Status: `v15_smplx_native_research_loop_complete_no_strict_pass`

This rollup covers only the SMPL-X native reset requested after external hand/hair routes were paused. No formal cloud train/infer/export was launched.

## Completed

- SMPL-X asset resolver: `v15_smplx_native_assets_ready`
- SMPL-X NumPy forward probe: `v15_smplx_forward_ready`
- SMPL-X native part mapper: `v15_smplx_native_part_map_ready`
- Real 4K4D camera prior export: `v15_smplx_real_camera_raster_ready`
- VGGT prior case builder: completed for `output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15`
- VGGT config: `training/config/4k4d_smplx_native_prior.yaml`
- SMPL-X native weak prior loss wrapper: `training/loss_smplx_native_prior.py`
- Bounded local overfit runner: `v15_smplx_native_overfit_observed_research_only`
- Fusion audit: `v15_smplx_fusion_effect_mixed_or_negative_research_only`
- Mentor report: `reports/20260508_v15_mentor_facing_smplx_native_report.md`

## Key Artifacts

- Asset report: `reports/20260508_v15_smplx_asset_scope_reset.json`
- Forward report: `reports/20260508_v15_smplx_forward_probe.json`
- Part report: `reports/20260508_v15_smplx_native_part_mapper.json`
- Real-camera raster report: `reports/20260508_v15_smplx_camera_raster_export.json`
- Prior export root: `output/surface_research_preflight_local/V15_SMPLX_native_camera_raster_export`
- Native training case: `output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15`
- Body/hand anchor source case: `output/training_cases/0012_11_frame0000_6views_v15_smplx_native_bodyhand`
- Overfit runner report: `reports/20260508_v15_smplx_native_overfit_runner.json`
- Fusion audit report: `reports/20260508_v15_smplx_fusion_effect_audit.json`

## Real-Camera Prior Metrics

- view count: `6`
- resolution: `518x518`
- mean silhouette vs raw mask IoU: `1.0`
- prior visible pixels: `25046`
- normal valid pixels: `25046`
- left hand pixels: `480`
- right hand pixels: `124`
- head heuristic pixels: `3305`
- face-front heuristic pixels: `3305`

The hand masks are SMPL-X native surface-raster weak anchors. The head/face maps are SMPL-X native image-space heuristic maps. Neither should be called a strict teacher.

## Fusion Audit Result

The bounded local overfit ran successfully, but the self-delta was not positive:

- IoU delta: `-0.007437705993652344`
- target recall delta: `-0.12675118446350098`
- loss delta: `-0.1262691468000412`

Scalar loss dropped, but IoU and target recall both got worse. The SMPL-X native route is wired and auditable, but it has not demonstrated mentor-pass improvement.

## D-Line State

- `strict_candidate_passes = 0`
- `strict_teacher_passes = 0`
- no strict registry write
- no candidate package
- no teacher package
- no predictions export
- formal cloud remains blocked

## Process Cleanup

All spawned Codex subagents were closed. The only recurring local process observed afterward was an external `git add -A`/`git add -u` command repeatedly restarted by a parent tool; individual instances were terminated, but the parent was not killed to avoid disrupting the editor/client.

