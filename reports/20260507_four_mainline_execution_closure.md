# 2026-05-07 Four-Mainline Execution Closure

Status: `executed_local_research_landing_strict_gate_red`

This is the closure report for the four-mainline landing request. It records
what was actually run locally, what was delegated to GPT-5.5 xhigh side agents,
and why no formal cloud, teacher export, candidate export, prediction export, or
strict pass was written.

## Final Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher-supervised route = blocked
teacher export = blocked
candidate export = blocked
research-preflight = diagnostic only
```

Final guard commands returned blocked:

```text
python tools/check_cloud_gate_status.py --json
  cloud_allowed = false
  reason = strict_candidate_passes is 0

python tools/check_cloud_gate_status.py --teacher-supervised --json
  cloud_allowed = false
  reasons = strict_candidate_passes is 0;
            teacher-supervised route requested but strict_teacher_passes is 0
```

## Agent Runtime

Side branches were launched as requested:

```text
model = GPT-5.5
reasoning = xhigh
```

The first B-hair worker disconnected, so it was relaunched. The completed
workers covered A5/D-line, B-hand7 readiness, B-Fus3D0-v2, and B-hair0. The
main thread implemented and ran B-GS0.

## A5 Dense Teacher

Owner: side agent A5/D-line.

Report:

```text
reports/20260507_a5_dline_four_mainline_referee_status.md
```

Decision:

```text
No new accepted external dense artifact exists beyond the known A5 COLMAP/MVS
and adapter-contract/failfast outputs.
A5 COLMAP/CUDA remains frozen as backend smoke, not teacher.
```

The required next A5 artifact is still exactly one same-frame dense shared
surface mesh or mutually consistent calibrated multi-view depth set. Without
that artifact, running more COLMAP view/threshold loops is blocked.

## B-Fus3D0-v2

Owner: GPT-5.5 xhigh side agent.

Implemented and ran:

```text
tools/b_fus3d0_v2_contract_preflight.py
```

Primary outputs:

```text
reports/20260507_b_fus3d0_v2_contract_preflight_status.md
output/surface_research_preflight_local/B_Fus3D0_v2_contract_preflight_hybrid6_layer23/
```

Local inputs validated:

```text
scene images/masks = 60 / 60
cameras = 60
VGGT token cache = [1, 6, 1374, 2048]
query evidence = 576 queries
latent grid seed = 5832 points
surface-template seed = 39962 vertices / 80569 faces
```

Decision:

```text
Contract/preflight complete and fail-closed.
Full B-Fus3D0-v2 smoke remains blocked until:
tools/b_fus3d0_v2_latent_grid_sdf_backend_smoke.py
```

This did not tune B19. The existing B19 remains frozen because real did not beat
shuffle/zero controls.

## B-GS0

Owner: main thread.

Implemented and ran:

```text
tools/b_gs0_smplx_anchored_free_gaussian_smoke.py
tools/b_gs0_open3d_contact_sheet.py
```

Primary outputs:

```text
reports/20260507_b_gs0_smplx_anchored_free_gaussian_status.md
output/surface_research_preflight_local/B_GS0_smplx_anchored_free_gaussian_smoke/
```

Open3D runtime verified:

```text
D:\anaconda\envs\g3splat\python.exe
open3d = 0.19.0
```

Open3D-readable point clouds:

```text
b_gs0_constrained_only_gaussians.ply = 9991 points
b_gs0_raw_free_gaussians.ply = 14270 points
b_gs0_free_gaussians.ply = 9000 points
b_gs0_anchored_plus_free_gaussians.ply = 18991 points
```

Contact sheets:

```text
output/surface_research_preflight_local/B_GS0_smplx_anchored_free_gaussian_smoke/open3d_contact_sheet/
```

Full 60-view mask-support-filtered result:

```text
free input = 14270
free kept = 9000
free support min = 24
free support max = 60

constrained mean_iou = 0.6300
anchored_plus_free mean_iou = 0.5841
combined_minus_constrained_iou = -0.0459
combined_minus_constrained_recall = 0.0026
combined_minus_constrained_overfill = 0.0472
combined_rgb_better_than_constrained = true
```

Decision:

```text
B-GS0 is now a real local representation smoke with constrained/free Gaussian
PLYs, 60-view mask-support filtering, raster diagnostics, and Open3D contact
sheets. It is not a strict pass: free Gaussians improve RGB proxy/recall but
decrease silhouette IoU and increase overfill, so geometry is not yet accepted
as non-template human surface.
```

## B-hand7

Owner: GPT-5.5 xhigh side agent.

Reports:

```text
reports/20260507_b_hand7_artifact_readiness_blocked.md
reports/20260507_b_hand7_artifact_readiness_blocked.json
```

Decision:

```text
B_hand7_continuous_connected_hand_surface_review cannot be produced from current
local artifacts.
```

Blocker:

```text
B-hand4 has a connected proxy, but its own summary is pass=false.
B-hand3 hand_gate.pass=false.
views_passing_raw_hand_anchor = 0
views_with_compact_3d_hand_boxes = 0
```

Required next artifact:

```text
same-frame dense connected hand/arm surface containing both hands, wrists, and
forearm/arm context, with palm/finger continuity reviewable in Open3D.
```

## B-hair0

Owner: GPT-5.5 xhigh side agent, with a main-thread fallback preflight also
written. The hybrid6 side-agent output is the richer authoritative artifact.

Implemented and ran:

```text
tools/b_hair0_contract_preflight.py
```

Primary outputs:

```text
reports/20260507_b_hair0_contract_preflight_status.md
output/surface_research_preflight_local/B_hair0_contract_preflight_hybrid6_layer23/
```

Support readout:

```text
hairline support_ge_2_ratio = 0.376
scalp support_ge_2_ratio = 0.161
head_top support_ge_2_ratio = 0.020
hair_ring support_ge_2_ratio = 0.583
```

Decision:

```text
B-hair0 local inputs and support metrics are recorded. Backend/export/cloud
remain blocked until a research-only backend exists:
tools/b_hair0_backend_smoke.py
```

The main thread also wrote a fallback support PLY package under:

```text
output/surface_research_preflight_local/B_hair0_contract_preflight/
```

It is diagnostic only and not a pass source.

## Verification

Compilation checks passed:

```text
python -m py_compile tools/b_gs0_smplx_anchored_free_gaussian_smoke.py
python -m py_compile tools/b_hair0_contract_preflight.py
python -m py_compile tools/b_fus3d0_v2_contract_preflight.py
D:\anaconda\envs\g3splat\python.exe -m py_compile tools/b_gs0_open3d_contact_sheet.py
```

Open3D read check passed for B-GS0 PLYs using the `g3splat` env.

## Bottom Line

The requested plan is locally landed as executable research artifacts and
fail-closed contracts:

```text
A5 = parked until real external dense artifact exists
B-Fus3D0-v2 = contract/preflight complete, backend smoke not implemented yet
B-GS0 = implemented and run; Open3D assets produced; geometry not accepted
B-hand7 = blocked, missing continuous connected hand/arm surface
B-hair0 = contract/preflight complete, backend smoke not implemented yet
D-line = red, cloud blocked
```

No route currently satisfies the mentor strict full/head/face/hairline/hands
surface gate. The next executable implementation targets are:

```text
tools/b_fus3d0_v2_latent_grid_sdf_backend_smoke.py
tools/b_hair0_backend_smoke.py
a stronger B-GS0 optimizer/render loop that improves geometry without overfill
```

Formal cloud remains correctly blocked until a strict candidate pass exists;
teacher-supervised cloud additionally requires a strict teacher pass.
