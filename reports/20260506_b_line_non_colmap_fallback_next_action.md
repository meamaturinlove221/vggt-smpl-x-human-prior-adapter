# 2026-05-06 B-Line Non-COLMAP Fallback Next Action

Status: `plan_only_no_pass`

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
formal train/infer/export = blocked
teacher/candidate export = blocked
```

This is a B-line research action note only. It does not write pass state, export
a teacher, export a candidate, or unblock cloud.

## Inspection Readout

- A5 known-camera COLMAP is no longer the smallest unblocker: triangulation and
  undistortion can work, but the available apt COLMAP reports no CUDA, so
  PatchMatch remains blocked.
- A4 and A4.1 already tested SDF/part-local SDF representations and remain
  negative as teacher/candidate routes: slim/template-like surface, no modeled
  face, hairline, hands, or clothing.
- B2 fixed the critical-family carrier/token support collapse with
  `hand=2,face=2,hair=3`, then failed a fair bounded smoke. The residual backend
  barely moved geometry and is frozen as a method.
- `tools/optimize_surface_token_backend_b2.py` currently parses
  `--critical-local-atlas` and defines `CriticalLocalAtlasResidual`, but that
  atlas branch is not wired into the optimization path. Running the existing flag
  as-is must not be counted as a representation test.

## Next Smallest B-Line Experiment

Run exactly one research-only B3 local-atlas surface-token probe after wiring the
existing atlas branch into a small helper or the B2 research tool:

```text
B3 critical-local-atlas surface-token probe
representation change = per-vertex local atlas residuals inside face/hand/hair
                        tokens, gated by B2 visibility and bounded by the same
                        carrier/support guards
not allowed = more B2 steps, larger token_hidden, scalar loss-weight loop
```

The point of the probe is to answer one question:

```text
Does adding local per-vertex atlas capacity inside supported critical tokens move
face/hand/hair geometry in a way the current one-delta-per-token B2 residual
could not?
```

## CLI

Do not run this until the output summary records
`critical_local_atlas_enabled=true` and local-atlas delta diagnostics. The
current checked script does not yet satisfy that guard.

```powershell
python tools\optimize_surface_token_backend_b2.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --template-payload output\normal_line_multiview_20260506\connected_surface_template_v28_semantic_detail_mouth_nose_fingers\connected_human_surface_template_payload.npz `
  --output-dir output\surface_research_preflight_local\B3_surface_tokens_t96_step4_critical_local_atlas_probe `
  --view-indices 0,10,20,30,40,50 `
  --target-size 96 `
  --token-grid 5 `
  --family-token-grid-overrides hand=2,face=2,hair=3 `
  --token-hidden 64 `
  --critical-local-atlas `
  --atlas-hidden 48 `
  --atlas-offset-scale 0.85 `
  --max-steps 4 `
  --overwrite
```

If local CUDA/nvdiffrast is unavailable, run the same command only through a
research-only Modal GPU path after the Modal wrapper can pass the atlas flag.
Do not use formal cloud train/infer/export.

## Stop Condition

Stop and freeze the branch if any of the following happens:

```text
summary missing critical_local_atlas_enabled=true
summary missing atlas local_delta stats for face/hand/hair
nonfinite loss
delta_guard or final_guard_failed
face/hand/hair local p90 delta remains below 0.001 m after 4 steps
max_vertex_delta remains below 0.002 m after 4 steps
Open3D/contact sheet remains connected-template face, cap-like hair, weak hands
```

Positive numeric IoU alone is not a pass condition. A useful research outcome is
only: local atlas changes critical-part geometry enough to justify a second
review run. It still cannot export a teacher/candidate or write a registry pass.
