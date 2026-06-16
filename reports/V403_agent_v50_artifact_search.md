# V403 Agent V50 Artifact Search

- Generated: `2026-05-09T05:39:40.0416137+08:00`
- Workspace: `D:\vggt\vggt-main`
- Mode: read-only artifact search
- Writes performed: this Markdown report and `reports/V403_agent_v50_artifact_search.json`
- Strict registry/pass writes: none
- Large Modal downloads: none

## Bottom Line

I did not find a filled original V50/V64/V138 candidate package, `package_files.zip`, `V64_candidate_pass_bundle.zip`, `V138_candidate_pass_bundle_v3.zip`, `strict_registry_entry_v50.json`, or `A_v50_v64_integrity_terminal.json` in the searched local roots or listed Modal paths.

What does exist:

- Empty/stub original-target remnants under `D:\vggt\vggt-main\output\frozen_candidates` and `D:\vggt\vggt-main\archive`.
- Rebuilt/equivalent local packages under `D:\vggt\vggt-main\output\V223_rebuilt_candidate_package` and `D:\vggt\vggt-main\output\frozen_candidates\V50R_rebuilt_after_artifact_loss`.
- Backup prior-case packages under `G:\项目备份\vggt_cam_depth大文件索引\output\...`.
- Modal source-equivalent payloads under `vggt-4k4d-output:/surface_research_cloud_preflight/V42_prior_enabled_predictions`, `V25_research_vggt_predictions`, and `V16_smplx_native_prior_case`.

## Local Original-Target Checks

These paths exist but are empty skeletons, not filled packages:

- `D:\vggt\vggt-main\output\frozen_candidates\V50_smplx_native_candidate_pass`
  - Direct children: 1
  - Recursive file count: 0
- `D:\vggt\vggt-main\output\frozen_candidates\V50_smplx_native_candidate_pass\package_files`
  - Direct children: 0
  - Recursive file count: 0
- `D:\vggt\vggt-main\archive\V64_candidate_pass_bundle\frozen_candidate\package_files`
  - Direct children: 0
  - Recursive file count: 0
- `D:\vggt\vggt-main\archive\V138_candidate_pass_bundle_v3\frozen_candidate\package_files`
  - Direct children: 0
  - Recursive file count: 0

These expected files were not found:

- `D:\vggt\vggt-main\archive\V64_candidate_pass_bundle.zip`
- `D:\vggt\vggt-main\archive\V138_candidate_pass_bundle_v3.zip`
- `D:\vggt\vggt-main\archive\package_files.zip`
- `D:\vggt\vggt-main\output\surface_research_preflight_local\V50_final_promotion_transaction\strict_registry_entry_v50.json`
- `D:\vggt\vggt-main\reports\A_v50_v64_integrity_terminal.json`

## Local Rebuilt/Equivalent Artifacts

`D:\vggt\vggt-main\output\V223_rebuilt_candidate_package` is present and contains a rebuilt candidate package from V42/V25/V16 source evidence. Important files include:

- `manifest.json` - 24,320 bytes
- `package_files\candidate_confidence_from_v42.npz` - 85,602,711 bytes
- `package_files\candidate_depths_from_v42.npz` - 30,775,206 bytes
- `package_files\candidate_normals_from_v42.npz` - 100,071,633 bytes
- `package_files\candidate_points_from_v42.npz` - 105,107,787 bytes
- `package_files\control_audit_from_v42.json` - 19,263 bytes
- `package_files\prior_effect_from_v42.json` - 35,089 bytes
- `package_files\v16_case_manifest.json` - 19,245 bytes
- `package_files\v16_prior_inputs.npz` - 4,029,306 bytes
- `package_files\v16_prior_targets.npz` - 42,090,945 bytes
- `package_files\v25_baseline_depths.npz` - 30,776,642 bytes
- `package_files\v25_baseline_points.npz` - 105,107,011 bytes
- `package_files\v42_guard.json` - 573 bytes

`D:\vggt\vggt-main\archive\V223_rebuilt_candidate_package.zip` is present at 504,782,072 bytes.

`D:\vggt\vggt-main\output\frozen_candidates\V50R_rebuilt_after_artifact_loss` is also present with `manifest.json`, `hash_manifest.json`, equivalent package files, and visual board images. Existing reports identify this as a rebuild, not original V50.

## G: Backup Artifacts

Found prior/equivalent packages:

- `G:\项目备份\vggt_cam_depth大文件索引\output\zju_vggt_geom_probe_smplprior_headhair_local_verify\sample_prior_case_package.npz` - 2,142,322 bytes
- `G:\项目备份\vggt_cam_depth大文件索引\output\zju_vggt_geom_probe_smplprior_headhair_hd_views\sample_prior_case_package.npz` - 2,142,322 bytes
- `G:\项目备份\vggt_cam_depth大文件索引\output\zju_vggt_geom_probe_smplprior_headhair_hd_views_23view\sample_prior_case_package.npz` - 10,819,761 bytes
- `G:\项目备份\vggt_cam_depth大文件索引\output\probe_smplxvertexfusion_local\sample_prior_case_package.npz` - 3,459,921 bytes
- `G:\项目备份\vggt_cam_depth大文件索引\output\teacher_fixed_visual_lift_benchmark\advisor_ready_package.20260407`

`G:\数据集` contains 4K4D dataset archives/metadata, but I did not find a V50/V64/V138 candidate package there.

## F:, Desktop, Downloads

- `F:\vggt`: no relevant original V50/V64/V138/package artifact found. Matches were noise such as `.venv5080` or unrelated 4K4D scripts/reports.
- `C:\Users\WINDOWS\Desktop`: no relevant artifact found.
- `C:\Users\WINDOWS\Downloads`: no relevant artifact found. Broad manifest hits were unrelated mod manifests.

## Modal

`modal app list` returned an empty app table.

Volumes listed:

- `vggt-lhm-cache`
- `vggt-pshuman-official-cache`
- `vggt-pifuhd-cache`
- `vggt-4k4d-output`
- `vggt-4k4d-data`
- `vggt-code`
- `vggt-out`
- `vggt-zju-data`

No literal V50/V64/V138 candidate package directory was found in the listed Modal paths. Equivalent source payloads exist:

- `vggt-4k4d-output:/surface_research_cloud_preflight/V42_prior_enabled_predictions`
  - Small JSON: `research_summary.json`, `research_prior_effect.json`, `control_real_zero_shuffle_random_dropout.json`, `v42_research_guard.json`
  - Large NPZs listed only: `research_confidence.npz` 81.6 MiB, `research_normals_geometric.npz` 95.4 MiB, `research_points_world.npz` 100.2 MiB, `research_depths.npz` 29.3 MiB
- `vggt-4k4d-output:/surface_research_cloud_preflight/V25_research_vggt_predictions`
  - Small files: `research_summary.json`, `research_report.md`, `v25_research_guard.json`
  - Large NPZs listed only: `research_confidence.npz` 58.1 MiB, `research_points_world.npz` 100.2 MiB, `research_depths.npz` 29.4 MiB
- `vggt-4k4d-output:/surface_research_cloud_preflight/V16_smplx_native_prior_case`
  - `case_manifest.json` 18.8 KiB
  - `v15_smplx_native_prior_summary.json` 2.9 KiB
  - `inputs.npz` 3.8 MiB
  - `targets.npz` 40.1 MiB

Related Modal 4K4D outputs also exist:

- `vggt-4k4d-output:/vggt_4k4d_infer/0012_11_frame0000_20views`
  - `summary.json` 11.2 KiB
  - `predictions.npz` 104.5 MiB, listed only
- `vggt-4k4d-output:/vggt_4k4d_infer/0012_11_frame0000_fullviews`
  - `summary.json` 37.3 KiB
  - `predictions.npz` 315.7 MiB, listed only
- `vggt-4k4d-output:/vggt_4k4d_infer/0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2`
  - `summary.json` 52.4 KiB
  - `predictions.npz` 224.2 MiB, listed only

Large Modal files noted but not downloaded:

- `vggt-zju-data:/checkpoints/model.pt` - 4.7 GiB
- `vggt-lhm-cache:/LHM_prior_model.tar` - 17.5 GiB

## Prior Local Reports Used As Breadcrumbs

- `D:\vggt\vggt-main\reports\V224_v50_frozen_candidate_recovery_plan.md`
- `D:\vggt\vggt-main\reports\V400_v50r_strict_promotion_transaction.md`
- `D:\vggt\vggt-main\reports\V402_codex_session_v50_provenance_search.md`

They are consistent with today’s search: original V50 was not restored from current local/Modal evidence; V223/V50R are rebuild/equivalent paths and should not be represented as original V50.

## Limitations

- One broad text-reference search over old reports/backups timed out after about 304 seconds. I replaced it with narrower path, directory, report, and Modal metadata checks.
- A targeted `rg --files` scan reported access denied for `D:\vggt\vggt-main\output\_tmp_tests\vggt_cloud_gate_test_as1qepx4`. This did not block the named V50/V64/V138/package checks.
- No Modal files were downloaded, including small JSON/manifests; all Modal evidence is from listings.
