# A5 + D-Line Four-Mainline Referee Status

Date: 2026-05-07 16:27 +08:00

Scope: side referee only for A5 artifact availability and D-line strict cloud
guard state. No tools, registry, cloud guard, predictions, checkpoints, or
existing reports were edited.

## Verdict

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher-supervised route = blocked
teacher export = blocked
candidate export = blocked
```

No new accepted external dense artifact was found beyond the known A5
COLMAP/MVS outputs and adapter-contract/failfast artifacts.

## A5 Artifact Re-Check

`external_models/` still contains only MediaPipe landmark assets:

```text
face_landmarker.task
hand_landmarker.task
```

Current A5-like local output directories are still the known COLMAP/MVS,
teacher-gate-failure, depth-range-audit, and external-adapter contract/failfast
directories:

```text
output/surface_research_preflight_local/A5_hybrid12_colmap_depth_range_audit_20260507
output/surface_research_preflight_local/A5_external_dense_adapter_guard_reject_python_20260507
output/surface_research_preflight_local/A5_external_dense_adapter_failfast_openmvs_contract_20260507
output/surface_research_preflight_local/A5_teacher_gate_failure_summary_20260507
output/surface_research_preflight_local/A5_known_direct_v1_hybrid12_teacher_gate_signfix_headshoulder
output/surface_research_preflight_local/A5_known_direct_v1_adj12_teacher_gate_signfix_headshoulder
output/surface_research_preflight_local/A5_known_direct_v1_adj6_teacher_gate_signfix_headshoulder
output/surface_research_preflight_local/A5_v8_colmapcuda_adj6_teacher_gate_signfix_headshoulder
output/surface_research_preflight_local/A5_triangulated_adj6_dryrun_after_db_camera_sync
output/surface_research_preflight_local/A5_known_direct_dense_adj6_dryrun
output/surface_research_preflight_local/A5_known_camera_colmap_workspace_smoke
output/surface_research_preflight_local/A5_external_dense_adapter_contract
```

The external dense adapter summaries remain blocked/contract-only:

```text
A5_external_dense_adapter_contract:
  status = blocked_no_backend_output
  backend_run.attempted = false
  input_mesh.provided = false
  input_depth_file_count = 0

A5_external_dense_adapter_failfast_openmvs_contract_20260507:
  status = blocked_no_backend_output
  backend_run.attempted = false
  input_mesh.provided = false
  input_depth_file_count = 0
```

The guard-reject directory contains only a backend workspace manifest. It is not
an accepted dense mesh or calibrated multi-view depth set.

The latest A5 depth-range audit is still explicitly research-only and records:

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal_cloud_train_infer_export = blocked
truthful_status = research_only_depth_range_audit_not_teacher_not_candidate
no_new_colmap_run = true
no_teacher_export = true
no_candidate_export = true
no_registry_write = true
no_predictions_write = true
```

Its decision remains that hybrid12 COLMAP depth maps have raw ROI depth pixels,
but the fused shared point cloud fails the same-protocol teacher gate; this does
not qualify as the required new external dense shared surface/depth artifact.

## Strict Cloud Guard

`python tools/check_cloud_gate_status.py --json` returned blocked JSON:

```text
cloud_allowed = false
teacher_supervised = false
reasons = strict_candidate_passes is 0
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-07T03:58:35.809940+00:00
registry_age_hours = 4.469
strict_candidate_passes = 0
strict_teacher_passes = 0
```

`python tools/check_cloud_gate_status.py --teacher-supervised --json` returned
blocked JSON:

```text
cloud_allowed = false
teacher_supervised = true
reasons = strict_candidate_passes is 0;
          teacher-supervised route requested but strict_teacher_passes is 0
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-07T03:58:35.809940+00:00
registry_age_hours = 4.469
strict_candidate_passes = 0
strict_teacher_passes = 0
```

The commands exited nonzero because the guard is red. That is the expected
blocked state, not an unblock signal.

## Commands Run

```text
rg -n "A5|cloud guard|check_cloud_gate_status|strict_candidate_passes|strict_teacher_passes|D-line|referee" C:\Users\WINDOWS\.codex\memories\MEMORY.md
git status --short
Get-ChildItem -LiteralPath reports -Filter '20260507_a5_dline_four_mainline_referee_*'
Select-String -LiteralPath C:\Users\WINDOWS\.codex\memories\MEMORY.md -Pattern A5,cloud guard,check_cloud_gate_status,strict_candidate_passes,strict_teacher_passes,D-line,referee
Get-ChildItem -LiteralPath reports -Filter '*a5*'
Get-ChildItem -LiteralPath reports -Filter '*dline*'
Test-Path -LiteralPath tools/check_cloud_gate_status.py
Get-Content -LiteralPath reports/20260507_a5_external_dense_backend_required_artifact.md -TotalCount 220
Get-Content -LiteralPath reports/20260507_a5_next_unblocker_decision.md -TotalCount 220
Get-Content -LiteralPath reports/20260507_dline_post_b16_guard_audit.md -TotalCount 260
Get-Content -LiteralPath tools/check_cloud_gate_status.py -TotalCount 260
python tools/check_cloud_gate_status.py --json
python tools/check_cloud_gate_status.py --teacher-supervised --json
Get-ChildItem -LiteralPath external_models -Force
Get-ChildItem -LiteralPath output\surface_research_preflight_local -Force | Where-Object { $_.Name -match 'A5|a5|colmap|dense|external' }
Get-ChildItem -LiteralPath output\surface_research_preflight_local -Directory -Filter 'A5*'
Get-ChildItem -LiteralPath output\surface_research_preflight_local -Recurse -File -Include *.ply,*.obj,*.npy,*.npz,*.exr
Get-ChildItem -LiteralPath output\surface_research_preflight_local -Directory -Filter 'A5_external*'
Get-Content -LiteralPath output\surface_research_preflight_local\A5_external_dense_adapter_contract\a5_external_dense_backend_preflight_summary.json
Get-Content -LiteralPath output\surface_research_preflight_local\A5_external_dense_adapter_failfast_openmvs_contract_20260507\a5_external_dense_backend_preflight_summary.json
Get-Content -LiteralPath output\surface_research_preflight_local\A5_external_dense_adapter_guard_reject_python_20260507\a5_external_dense_backend_preflight_summary.json
Get-ChildItem -LiteralPath output\surface_research_preflight_local\A5_hybrid12_colmap_depth_range_audit_20260507 -File
Get-ChildItem -LiteralPath output\surface_research_preflight_local\A5_external_dense_adapter_contract,output\surface_research_preflight_local\A5_external_dense_adapter_failfast_openmvs_contract_20260507,output\surface_research_preflight_local\A5_external_dense_adapter_guard_reject_python_20260507 -Recurse -File
Get-Content -Raw -LiteralPath output\surface_research_preflight_local\A5_hybrid12_colmap_depth_range_audit_20260507\a5_hybrid12_colmap_depth_range_audit_summary.json | ConvertFrom-Json
Get-ChildItem -LiteralPath output\surface_research_preflight_local -Directory -Filter 'A5*' | Where-Object { $_.LastWriteTime -gt [datetime]'2026-05-07T14:34:08' }
Get-Date -Format o
```

`rg` failed with Windows `Access is denied`, so PowerShell `Select-String` and
`Get-ChildItem` checks were used for the referee pass.
