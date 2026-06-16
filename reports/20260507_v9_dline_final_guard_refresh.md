# V9 D-line Final Guard Refresh

Status: `FORMAL_CLOUD_STILL_BLOCKED_RESEARCH_ONLY_ALLOWED`

Refreshed at `2026-05-07T23:32:15+08:00` (`2026-05-07T15:32:15Z`) from `D:\vggt\vggt-main`.

## Guard Conclusion

- Formal cloud remains blocked.
- `strict_candidate_passes=0` and `strict_teacher_passes=0`.
- Research cloud gate is allowed only for bounded research jobs under `output/surface_research_cloud_preflight`.
- No `predictions.npz` was found under the research cloud output root.
- No teacher package, candidate package, registry write, or strict-pass artifact file was found in the research cloud output root.
- JSON scan found no research-flag violations and no `formal_cloud_unblocked: true`.

The path scan does see research names such as `teacher_intake` and `candidate_points`; these are diagnostic output labels, not formal promotion packages.

## Guard Runs

- `python tools/check_cloud_gate_status.py --json`: blocked because `strict_candidate_passes is 0`.
- `python tools/check_cloud_gate_status.py --teacher-supervised --json`: blocked because `strict_candidate_passes is 0` and `strict_teacher_passes is 0`.
- `python tools/check_research_cloud_gate_status.py ...`: `research_cloud_preflight_allowed`, with `formal_guard_still_blocked=true`.
- `python tools/unified_surface_candidate_precheck_v9.py ...`: `blocked_no_unified_surface_candidate_precheck`.

The artifact referee script was not rerun because its internal gate call writes the default manifest outside the D-line write scope. I used the existing referee/gate reports plus read-only artifact scans instead.

## Available Or Partly Available

| Backend / lane | Current state | Guard meaning |
| --- | --- | --- |
| V9 cloud asset staging | `v9_cloud_assets_verified` | Assets exist locally/remotely; still research-only. |
| Cloud-A V9 B-Fus3D3 real-asset preflight | completed, no procedural fallback | Body/head/face research preflight available; not teacher/candidate. |
| A5-X2 MUSt3R | real non-empty point cloud, `best_vertex_count=301056` | Weak-pool evidence only; strict teacher is blocked. |
| 2DGS dependency smoke | `dependency_import_ready_needs_colmap_scene` | Dependencies imported; needs COLMAP-format 4K4D scene. |

## Blocked

| Backend / lane | Blocker |
| --- | --- |
| A5-X2 MUSt3R strict teacher intake | Missing known-camera alignment, original 6-view reprojection/depth residual audit, and region visual gates. |
| MASt3R-SLAM | Dependency/import failures: missing `mast3r`, missing `clang++` for imgui, curope torch build issue, CUDA 11.8 vs PyTorch 12.4 mismatch for lietorch. |
| Hair-GS | Dependency/import failures: `diff_gaussian_rasterization` build failed, missing `pytorch3d`, no matching `pytorch3d` wheel. |
| NeuS2 | No runnable backend/checkpoint found. |
| B-hand11 | Real VGGT hand-token decoder absent; B-hand10 is still synthetic/proxy. |
| B-hair4 | Real HairGS/topology module absent; B-hair3 is still synthetic/proxy. |

## Unified Precheck

`can_merge_unified_surface=false`.

Ready components: cloud asset staging, A5-X2 true dense backend presence, Cloud-A V9 B-Fus3D3 body/head/face preflight.

Blocking components: A5-X2 strict teacher intake, B-hand11 real decoder, B-hair4 real topology.

Decision: do not merge proxy/synthetic V8 artifacts into a candidate, and do not write predictions, teacher, candidate, registry, or strict pass state.

## Written Files

- `D:\vggt\vggt-main\reports\20260507_v9_dline_final_guard_refresh.json`
- `D:\vggt\vggt-main\reports\20260507_v9_dline_final_guard_refresh.md`
- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V9_dline_final_guard_refresh\summary.json`
- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V9_dline_final_guard_refresh\report.md`
- Supporting command outputs under `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V9_dline_final_guard_refresh\research_gate_status.*` and `unified_precheck_status.*`
