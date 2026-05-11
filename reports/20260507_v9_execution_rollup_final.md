# V9 Execution Rollup Final

Research-only V9 execution was landed and run to current true blockers. No strict pass, teacher export, candidate export, predictions export, or registry write was produced.

## Completed

- V9 reality audit completed. Prior V8 positives were classified as proxy/smoke unless backed by later true-backend runs.
- Cloud asset staging completed and remote assets were verified: query cache, template payload, VGGT tokens, 60 images/masks, cameras, and priors. Procedural fallback is no longer silently accepted.
- Cloud-A B-Fus3D3 real-asset preflight completed with staged assets and no procedural fallback. It remains research-only.
- A5-X2 MUSt3R true backend ran and produced nonempty weak-pool pointmaps. It did not pass strict teacher intake because known-camera alignment, reprojection, and region gates are still missing.
- 2DGS read the staged known-camera COLMAP scene, loaded 6 cameras and 301056 weak-pool initial points, and completed a 30-iteration research-only train smoke.
- MASt3R-SLAM official `main.py` ran on staged 4K4D PNG input and returned 0. It wrote logs, trajectory, and PLY, but the reconstruction PLY has 0 vertices, so this is not usable geometry.
- Hair-GS dependencies now import on Modal with cu118/torch2.0/gcc11/sm86. CUDA rasterizer, simple-knn, PyTorch3D, and `HairGaussianModel` import passed. Next blocker is FLAME and hair dataset conversion.
- D-line guard/referee stayed red for formal cloud and clean for research-only artifacts.

## Blocked

- Unified surface candidate precheck is blocked: A5-X2 remains weak-pool only, B-hand11 real token decoder is unavailable, and B-hair4 real topology backend is unavailable.
- Formal cloud remains blocked: `strict_candidate_passes=0` and `strict_teacher_passes=0`.

## Key Paths

- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\a5x2_2dgs_colmap_scene_smoke\summary.json`
- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\summary.json`
- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V9_backend_dependency_smokes\hair_gs_fix2\hair_gs_fix2_summary.json`
- `D:\vggt\vggt-main\reports\20260507_v9_backend_dependency_smoke_status.md`
- `D:\vggt\vggt-main\reports\20260507_v9_unified_surface_precheck_status.json`
- `D:\vggt\vggt-main\reports\20260507_v9_dline_final_guard_refresh.md`

Final verdict: V9 executable work was run to the real backend/dependency/data blockers. It is not a teacher/candidate success, and it must not be promoted to formal cloud or strict gate.
