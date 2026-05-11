# Modal Research-Preflight Runtime Audit

Status: `modal_research_preflight_ran_successfully_for_latest_a5_hybrid12_known_direct`

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
research-preflight cloud diagnostics = allowed
```

## Modal App State

`modal app list` showed recent ephemeral `vggt-surface-research-preflight`
apps in `stopped` state. `modal app logs` for the stopped app ids reported:

```text
Stopping app - local client disconnected.
Stopping app - local entrypoint completed.
```

No active server-side Python traceback was available from those stopped
ephemeral apps. The visible dashboard crash-loop state should therefore not be
treated as proof that the latest research job failed.

## Confirmed Remote Run

Downloaded from Modal volume:

```text
vggt-4k4d-output/A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/research_preflight_summary.json
vggt-4k4d-output/A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/research_preflight_launch_guard.json
vggt-4k4d-output/A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/A5_known_camera_colmap_workspace/a5_known_camera_colmap_preflight_summary.json
```

Key runtime result:

```text
lane = A5_known_camera_colmap_workspace
dense_mode = known_direct
gpu = A100-40GB
status = completed
returncode = 0
stderr_tail = ""
elapsed_seconds = 77.741
```

COLMAP internal steps:

```text
image_undistorter = succeeded
patch_match = succeeded
stereo_fusion = succeeded
fused_ply_exists = true
fused_points = 101420
```

## Interpretation

The latest A5 hybrid12 known-direct research-preflight job did run and produced
a fused PLY. The failure is not a Modal runtime failure for this job. The failure
is downstream strict teacher viability: the A5 fused output remains unsuitable as
a mentor-grade target-frame dense teacher because later strict audits show poor
head/face/hairline/hands compatibility and Open3D visual failure.

## Safety

The launch guard kept the run research-only:

```text
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
formal_cloud_train_infer_export = blocked unless local strict gate passes
```

No strict pass state was written.
