# Geometry Direction Status 2026-03-22 Cloud Compare Completed

## Executive status

- The local Windows crash-prone launch path is now repaired enough for safe cloud launching.
- The mentor-aligned first-round experiment is now completed on Modal:
  - compare only `point_map` vs `depth + camera -> unprojected points`
  - no new geometry loss
  - no ghost stack
- The new cloud batch completed successfully on all 3 canonical ZJU report cases.

## What was fixed before going back to cloud

- [invoke_modal_zju_preflight.ps1](/f:/vggt/vggt-main/scripts/invoke_modal_zju_preflight.ps1)
  - default free-memory floor raised to `12 GB`
  - stale detached repo launchers can be detected and stopped
  - transient local launcher residue no longer blocks or destabilizes the next cloud launch
- [run_modal_zju_geometry_branch_compare.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_geometry_branch_compare.ps1)
  - now sends each old `report.json` as `report_json_b64`
  - avoids PowerShell `ConvertTo-Json` corrupting the raw report payload into `value + PSPath + PSDrive`
- [modal_zju_geometry_branch_compare.py](/f:/vggt/vggt-main/modal_zju_geometry_branch_compare.py)
  - now decodes `report_json_b64` back to the exact original report text
  - keeps backward-compatible recovery for the older broken `report_json.value` shape

## Local readiness status

- Local preflight now passes cleanly:
  - free memory observed around `18.6 GB`
  - no repo-scoped redundant `powershell/python/modal` processes remained
  - no active redundant Modal apps remained
- Local RTX 5080 path is usable through `.venv5080`:
  - PyTorch build: `2.10.0+cu128`
  - CUDA arch support includes `sm_120`
  - local pure geometry smoke already ran successfully on `cuda + bfloat16`

Important clarification:

- the old Anaconda PyTorch path is still unsuitable for the 5080
- the repaired local path is `.venv5080`, not the old Anaconda environment

## Completed cloud batch

- Modal app:
  - `ap-MOqSyP1pq2pqukdR08TzbA`
- App description:
  - `vggt-zju-geometry-branch-compare`
- Final state:
  - `completed`
- Remote output root:
  - `vggt-out/geometry_compare/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix`
- Local pulled summaries:
  - [batch_summary.md](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/batch_summary.md)
  - [batch_status.json](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/batch_status.json)
  - [6src summary](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/CoreView_390_frame_001080_Camera_B5_6src_hist_summary.md)
  - [12src summary](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/CoreView_390_frame_001080_Camera_B5_12src_nested_summary.md)
  - [23cam summary](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/CoreView_390_frame_001080_Camera_B5_23cam_fullset_summary.md)

## Cloud batch result

- runs: `3`
- `depth_unproject` wins: `2`
- `point_map` wins: `1`
- ties: `0`

Per case:

- `6src_hist`
  - decision: `point_map`
  - point MAE: `0.0369`
  - depth MAE: `0.0372`
  - point coverage: `0.3551`
  - depth coverage: `0.3543`
- `12src_nested`
  - decision: `depth_unproject`
  - point MAE: `0.0443`
  - depth MAE: `0.0405`
  - point coverage: `0.2365`
  - depth coverage: `0.2538`
- `23cam_fullset`
  - decision: `depth_unproject`
  - point MAE: `0.0485`
  - depth MAE: `0.0429`
  - point coverage: `0.1635`
  - depth coverage: `0.2214`

## Interpretation against the mentor's question

The mentor's immediate requirement was:

1. confirm what point-source path is actually producing the render
2. once confirmed that the current render mainly comes from `point/world points`, try the alternate path:
   - `camera + depth -> unprojected points -> render`
3. do this first without adding new loss

This cloud batch now answers that question directly.

What the result says:

- `depth + camera` is not universally better on every sparse-view case
- but it wins the majority of the current canonical cases: `2 / 3`
- and its strongest advantage appears as view count increases:
  - clear win on `12src_nested`
  - stronger win on `23cam_fullset`
- this is enough to keep the main direction on the geometry chain
- it is **not** a reason to revive the old ghost stack

## Relation to older local evidence

The repo already had an earlier local batch summary:

- [older local batch summary](/f:/vggt/vggt-main/output/geometry_baseline_zju_batch/batch_summary.md)

That older local batch reported:

- `depth_unproject` wins: `2`
- `point_map` wins: `0`
- ties: `1`

So the old local batch and the new cloud batch agree on the main conclusion:

- `depth_unproject` is the stronger overall next baseline direction

They differ on the hardest sparse case:

- local batch: `6src_hist` was slightly favorable to `depth_unproject`
- cloud batch: `6src_hist` is slightly favorable to `point_map`

This means the honest readout is:

- the geometry-chain direction is supported overall
- but the very sparse-view regime is still borderline and should be treated as a stress case, not as settled

## Current recommended next step

If continuing from here, the clean next action should be:

1. keep the source-path switch result as the new first-round evidence
2. treat `depth_unproject` as the primary geometry candidate branch
3. do **not** restore `ghost` losses or multi-weight image-side logic
4. if a next experiment is needed, add only one minimal geometry-side supervision term on top of this baseline
5. keep `6src_hist` as the regression case to watch

## Operational notes

- Modal volume inode usage is still high:
  - `/mnt/out` around `91%`
  - `/mnt/data` around `84.6%`
- This did not block the completed batch, but it remains a real future risk.
