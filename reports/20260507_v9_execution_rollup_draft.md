# V9 Execution Rollup Draft

Created: 2026-05-07T23:45:15+08:00  
Repo: `D:\vggt\vggt-main`  
Status: draft fact index, research-only.

This file is a factual rollup for main-thread integration. It does not claim a formal pass, strict teacher, candidate, export, predictions, or registry update.

## Guard Contract

- Formal cloud remains red: `formal_cloud_unblocked=false`.
- Research-only preflight is allowed.
- Forbidden here and still not written: strict pass, teacher package/export, candidate package/export, `predictions.npz`, registry write.
- Strict counts remain `strict_candidate_passes=0` and `strict_teacher_passes=0`.

Evidence: `reports/20260507_v9_dline_final_guard_refresh.json`.

## Real Work Completed

### V9 cloud assets staged and verified

- Status: `v9_cloud_assets_verified`.
- Local and remote verification both completed.
- Verified assets include 60 RGB images, 60 masks, 60 cameras, prior maps, camera sidecar, query cache, template payload, and VGGT token cache.
- Query cache: 576 queries, feature dim 2048.
- Token cache shape: `[1, 6, 1374, 2048]`.
- Template payload: 39,962 hybrid vertices.
- Missing assets: none reported.

Evidence:
- `reports/20260507_v9_cloud_asset_staging_status.json`
- `reports/20260507_v9_cloud_asset_verification_status.json`

### A5-X2 MUSt3R true backend artifact audit

- Status: `completed_weak_pool_only`.
- Real backend family: MUSt3R.
- Checkpoint reported: `MUSt3R_224_cvpr.pth`.
- Input image count: 6.
- Best non-empty point cloud: 301,056 vertices, finite ratio 1.0.
- Ready only for weak teacher pool evidence.
- `strict_teacher_ready=false`.

Conclusion: A5-X2 MUSt3R is real and non-empty, but weak-pool only. It is not a strict teacher.

Blockers:
- Not known-camera aligned to the original 4K4D camera frame yet.
- Default confidence thresholds exported zero vertices; usable output requires low confidence threshold.
- No depth residual / original 6-view reprojection audit has passed.
- No full/head/face/hairline/hands Open3D visual gate has passed.

Evidence:
- `reports/20260507_v9_a5x2_must3r_artifact_audit.json`
- `output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend_audit/summary.json`

### Cloud-A B-Fus3D3 real asset preflight

- Status: `completed_research_only_not_candidate`.
- Remote training consumed staged query/template assets.
- `query_cache_exists=true`.
- `template_payload_exists=true`.
- `allow_procedural_fallback=false`.
- `procedural_fallback_used=false`.
- Train command returned 0 and wrote a train summary.
- Bounded run: max 5,000 steps, max 8 cases, max 1 hour.
- Real eval IoU: 0.9824561403508771.
- Real features beat controls:
  - vs shuffle IoU: +0.34983014565591697
  - vs random IoU: +0.4178963209377846
  - vs zero IoU: +0.3278076142737796

Conclusion: Cloud-A V9 B-Fus3D3 is a real-asset research progress signal with fallback disabled and unused. It remains research-only, not a candidate or teacher.

Evidence:
- `output/surface_research_cloud_preflight/Cloud_A_V9/b_fus3d3_real_asset_train_preflight/v9_cloud_a_real_asset_wrapper_summary.json`
- `output/surface_research_cloud_preflight/Cloud_A_V9/b_fus3d3_real_asset_train_preflight/summary.json`

### Backend dependency smokes

- Confirmed D-root repos: `D:\2d-gaussian-splatting-main`, `D:\MASt3R-SLAM-main`, `D:\must3r-main`.
- Dependency smoke monitoring ended with `live_modal_tasks_after_cleanup=0`.
- Results are mixed: 2DGS dependency import ready; MASt3R-SLAM and Hair-GS remain blocked.

Evidence: `reports/20260507_v9_backend_dependency_smoke_status.json`.

## 2DGS And COLMAP Scene State

2DGS dependency status:
- `dependency_import_ready_needs_colmap_scene`.
- Import probe returned 0 and reported `2.0.0+cu118 11.8 True`.
- Installed/imported dependency path evidence exists at `output/surface_research_cloud_preflight/V9_backend_dependency_smokes/2dgs_fixed/2dgs_summary.json`.

COLMAP scene smoke state:
- Current scene smoke report status: `loader_ready_train_smoke_blocked_or_skipped`.
- `summary.json` was not present in `output/surface_research_cloud_preflight/Cloud_B_V9/a5x2_2dgs_colmap_scene_smoke` at rollup time.
- Observed files:
  - `model_smoke/input.ply` (8,128,817 bytes)
  - `model_smoke/cameras.json` (2,434 bytes)
  - `model_smoke/cfg_args` (272 bytes)
  - `report.md`

Pending slot: do not claim 2DGS scene smoke success until a current scene smoke `summary.json` or equivalent completed summary appears.

## Current Blockers

### MASt3R-SLAM

Status: `blocked_dependency_build_or_import_failed`.

Blockers:
- `mast3r` import missing.
- `imgui` build requires `clang++`.
- `curope` build missed `torch` during setup.
- `lietorch` build hit CUDA 11.8 vs PyTorch 12.4 mismatch.

Evidence: `reports/20260507_v9_backend_dependency_smoke_status.json`.

### Hair-GS

Status: `blocked_dependency_build_or_import_failed`.

Blockers:
- `diff_gaussian_rasterization` build failed.
- Hair-GS import missing `pytorch3d`.
- Pip found no matching `pytorch3d` wheel.

Evidence: `reports/20260507_v9_backend_dependency_smoke_status.json`.

### B-hand11

Status: `blocked_missing_real_hand_token_decoder`.

Blockers:
- B-hand10 remains a synthetic/proxy HGGT-style smoke.
- No true B-hand11 VGGT-token/HGGT hand decoder asset was found.
- No hand-specific learned decoder checkpoint or MANO plus residual decoder asset was found.

Evidence: `reports/20260507_v9_hand_hair_real_module_status.json`.

### B-hair4

Status: `blocked_missing_real_hair_topology_module`.

Blockers:
- B-hair3 remains a synthetic/proxy HairGS-style topology smoke.
- No true B-hair4 HairGS/topology module asset was found.
- Missing learned root/strand topology network, checkpoint, differentiable multiview projection loss, and real hair/head segmentation merging.

Evidence: `reports/20260507_v9_hand_hair_real_module_status.json`.

## Unified Surface / D-Line

- Unified surface precheck remains blocked.
- Ready components: cloud asset staging, A5-X2 true dense backend evidence, B-Fus3D3 body/head/face preflight.
- Not ready: A5-X2 strict teacher intake, B-hand11 real decoder, B-hair4 real topology.
- D-line formal guard verdict: `FORMAL_CLOUD_STILL_BLOCKED_RESEARCH_ONLY_ALLOWED`.

Evidence:
- `reports/20260507_v9_unified_surface_precheck_status.json`
- `reports/20260507_v9_dline_final_guard_refresh.json`

## Pending Slots For Main Thread

- `2dgs_colmap_scene_smoke_summary`: pending; `summary.json` not present at rollup time.
- `strict_teacher_or_candidate_gate`: blocked; strict candidate/teacher counts are 0.
- `unified_surface_candidate_precheck`: blocked; A5-X2 strict teacher intake, B-hand11, and B-hair4 are not ready.

## Non-Claims

- No strict pass written.
- No teacher package/export written.
- No candidate package/export written.
- No `predictions.npz` written.
- No registry write performed.
- No formal cloud unblock claimed.
- No full V9 breakthrough claimed.
