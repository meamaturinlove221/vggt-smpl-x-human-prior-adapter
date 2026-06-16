# 2026-05-06 B2 Surface Token Backend Plan

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
teacher export          = blocked
candidate export        = blocked
formal VGGT train/infer/export = untouched
```

This is B-line research-preflight only. B2 is not a strict pass, not a teacher, not a candidate, and not a cloud unblock signal.

## Scope

Added `tools/optimize_surface_token_backend_b2.py` as a disjoint B-line research tool based on the existing B1 surface-token direction, but not as B1 hidden/step tuning and not as `image_mlp++`.

Files intentionally not touched:

- `modal_surface_research_preflight.py`
- A-line preflight tools
- formal train/infer/export entrypoints
- strict gate registries

## B2 Difference From B1

B1 already made token visibility and rendered diagnostics explicit. B2 changes the backend contract rather than tuning B1:

- token families are fixed as `body`, `hand`, `face`, and `hair`;
- integer template parts are mapped into those families:
  - `0 torso_limbs` and `5 lower_clothing_proxy` -> `body`
  - `1 left_hand` and `2 right_hand` -> `hand`
  - `3 head_face` -> `face`
  - `4 head_top_hairline` -> `hair`
- each family has its own head spec, parameter count, offset scale, normal residual scale, and visibility bias;
- projected RGB/mask aggregation is available before nvdiffrast, so the tool can still emit a useful blocked report;
- raster RGB/mask/depth/normal aggregation is first-class when nvdiffrast CUDA is available;
- stopping conditions are explicit and written into JSON/MD outputs.

## Implemented Script

`tools/optimize_surface_token_backend_b2.py` currently implements:

- part-aware quantized surface tokens;
- body/hand/face/hair token-family assignment and family-colored carrier mesh;
- per-family token heads via `PartSpecializedB2SurfaceTokenBackend`;
- projected multi-view RGB/mask token aggregation on CPU;
- nvdiffrast raster diagnostic path for mask, depth, normal, rendered RGB, and RGB residual maps;
- token diagnostic table as JSON and CSV;
- explicit stop reasons:
  - `blocked_no_nvdiffrast`
  - `blocked_no_cuda`
  - `initial_guard_failed`
  - `diagnostics_only`
  - `nonfinite_loss`
  - `delta_guard`
  - `plateau_guard`
  - `max_steps_reached`
  - `final_guard_failed`
- zero strict pass writes and no teacher/candidate exports.

## Minimal Smoke

Command run locally:

```powershell
python tools\optimize_surface_token_backend_b2.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --template-payload output\normal_line_multiview_20260506\connected_surface_template_v28_semantic_detail_mouth_nose_fingers\connected_human_surface_template_payload.npz `
  --output-dir output\surface_research_preflight_local\B2_surface_tokens_t48_diag_smoke `
  --view-indices 0,30 `
  --target-size 48 `
  --token-grid 4 `
  --token-hidden 32 `
  --max-steps 0 `
  --diagnostics-only `
  --overwrite
```

Result:

```text
status = blocked_no_nvdiffrast
strict_candidate_passes = 0
strict_teacher_passes = 0
formal train/infer/export = untouched
```

The blocked state is expected on this Windows environment because `nvdiffrast` is not importable, and PyTorch warns that the installed CUDA build does not support the local RTX 5080 `sm_120` device.

Smoke outputs:

```text
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_summary.json
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_summary.md
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_token_diagnostics.json
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_token_diagnostics.csv
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_carrier_mesh.ply
output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_family_carrier_mesh.ply
```

Projected smoke readout:

```text
token_count = 246
body projected_visible_token_fraction = 0.9772727273
hand projected_visible_token_fraction = 0.5454545455
face projected_visible_token_fraction = 1.0000000000
hair projected_visible_token_fraction = 0.7037037037
```

These are diagnostics only. They do not constitute a pass.

## Full Diagnostic Command

Use only in a compatible CUDA/nvdiffrast environment:

```powershell
python tools\optimize_surface_token_backend_b2.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --template-payload output\normal_line_multiview_20260506\connected_surface_template_v28_semantic_detail_mouth_nose_fingers\connected_human_surface_template_payload.npz `
  --output-dir output\surface_research_preflight_local\B2_surface_tokens_t96_diag `
  --view-indices 0,10,20,30,40,50 `
  --target-size 96 `
  --token-grid 5 `
  --token-hidden 64 `
  --max-steps 3 `
  --overwrite
```

Expected additional outputs when the raster backend is available:

- per-view initial/final target RGB;
- per-view target/render masks;
- per-view depth and normal PNGs;
- per-view rendered RGB and RGB residual maps;
- B2 research mesh, normals mesh, projected-RGB mesh, and visibility-gate mesh;
- final `stop_result` explaining why the run stopped.

## Stop Conditions

B2 must stop instead of claiming success when:

- nvdiffrast or CUDA is missing;
- any nonempty family fails minimum visible-token coverage;
- rendered mask IoU falls below the configured floor;
- RGB residual, depth variance, or normal dispersion exceeds configured diagnostic bounds;
- the loss becomes nonfinite;
- vertex deltas exceed the configured hard cap;
- the loss plateaus within the small research budget;
- `--max-steps` is reached.

Reaching `max_steps_reached` is a budget stop, not a success.

## Next Step

Run B2 in a Linux/WSL2/Docker/lab environment with compatible PyTorch CUDA and nvdiffrast. Review the rendered RGB/mask/depth/normal diagnostics per family before deciding whether any later training design is warranted. Keep strict passes at zero until a separate strict visual gate actually passes.
