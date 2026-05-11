# B-Fus3D19 Bounded Learned/Rendered Smoke Scope Decision

Status: `scope_contract_allowed_but_runtime_render_smoke_blocked_locally_no_train_no_export`

This is a local scope decision for the B-Fus3D19 side-line. It does not run a
smoke, train, infer, export, write predictions, write a strict pass, touch cloud
guards, or modify the formal B16/B18 lanes.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
teacher_export = blocked
candidate_export = blocked
formal_cloud_train_infer_export = blocked
cloud_guard_changes = none
B16 = frozen
B18 = audit-only frozen
```

## Decision

```text
Can define B19: yes, as a new bounded learned/rendered scope and implementation
contract only.

Can execute the full B19 learned/rendered smoke on this local Windows runtime:
no, because the actual rendered path requires nvdiffrast/CUDA and the local
B1/B2 summaries show nvdiffrast is not importable.
```

B19 is only non-duplicative if it is implemented as a new surface-token/SDF
bundle smoke with real/shuffle/zero controls and rendered closure. It must not
reuse the B16 latent-field mesh route, B5/B16-style tiny residual MLP route,
A4 tiny SDF route, visual-hull labels, or COLMAP-depth labels as standalone
supervision.

## Local Facts Read

- `tools/b_fus3d_surface_sdf_contract_preflight.py` exists and B17 passed schema.
- `reports/20260507_b_fus3d_surface_sdf_contract_status.md` reports
  `contract_preflight_complete = true`, matching token/query/control dimensions,
  and no train/mesh/prediction writes.
- `reports/20260507_b_fus3d_rendered_control_audit_status.md` reports B18
  rendered-control audit negative: B16 real does not beat shuffle/zero.
- `reports/20260507_b_fus3d_latent_field_visual_freeze.md` freezes B16 as a
  lightweight latent-field visual fail.
- `tools/b_fus3d_learned_decoder_smoke.py` is explicitly a tiny
  token-conditioned residual MLP, so it is excluded for B19.
- `tools/b_fus3d_renderable_decoder_smoke.py` is deterministic single-forward,
  not a learned B19 smoke.
- `tools/optimize_surface_token_backend_b2.py` contains useful part/family
  surface-token infrastructure, support guards, rendered diagnostics, and stop
  reasons, but current B2 residual backend is frozen as a geometry-negative
  method.
- Local summaries at
  `output/surface_research_preflight_local/B1_surface_tokens_t48_step1/surface_token_b1_summary.json`
  and
  `output/surface_research_preflight_local/B2_surface_tokens_t48_diag_smoke/surface_token_b2_summary.json`
  both show `blocked_no_nvdiffrast`.

## Exact Minimal Implementation Contract

Implement only after this scope decision, as a new script:

```text
tools/b_fus3d_b19_bounded_surface_sdf_render_smoke.py
```

The script must be research-only and must fail closed. It may run at tiny local
resolution and a tiny step budget, but it is still not formal train/infer/export.

Required representation:

```text
query-level SDF samples from B17/B6 query contract;
part/family surface-token carrier from the connected template;
family heads for body, face_core, hairline, left_hand, right_hand;
bounded query-to-carrier coupling, not unconstrained per-vertex offsets;
real/shuffle/zero token-control slots in one run;
support/risk output for every query family and rendered view.
```

Required learned unit:

```text
not a tiny generic MLP over vertices or latent-grid coordinates;
part/family heads may be small, but must operate on surface-token/query bundles;
must include visibility/support gates and output SDF, occupancy, normal,
rgb/proxy_color, confidence, and support.
```

Required rendered closure:

```text
render real/shuffle/zero surfaces or bounded carrier fields through the same
cameras;
write mask, depth, normal, rgb/residual diagnostics per selected view;
compare real-vs-shuffle and real-vs-zero in the same summary;
write explicit full/head/face/hairline/left_hand/right_hand review paths;
keep all outputs under output/surface_research_preflight_local/B_Fus3D19_*.
```

Allowed tiny defaults:

```text
target_size = 48 or 64
view_indices = 0,10,24,36,45,57 or a declared subset for first smoke
max_steps <= 3
query families = full_body, face_core, hairline, left_hand, right_hand
no checkpoint writes
no predictions.npz writes
no strict registry writes
```

## Inputs

Use only existing local research artifacts:

```text
--scene-dir output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop
--token-cache output/surface_research_preflight_local/B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/token_cache/aggregator_layer_23.npz
--surface-token-features output/surface_research_preflight_local/B_Fus3D1_surface_token_smoke_hybrid6_layer23/surface_token_features.npz
--query-evidence output/surface_research_preflight_local/B_Fus3D6_query_evidence_cache_hybrid6_layer23/b_fus3d_query_evidence_cache.npz
--b17-contract output/surface_research_preflight_local/B_Fus3D17_surface_sdf_contract_preflight_hybrid6_layer23/b_fus3d_surface_sdf_contract_summary.json
--latent-grid-real output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/b_fus3d_latent_grid_evidence_arrays.npz
--latent-grid-shuffle output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/b_fus3d_latent_grid_evidence_arrays.npz
--latent-grid-zero output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/b_fus3d_latent_grid_evidence_arrays.npz
--template-payload output/surface_research_preflight_local/connected_payload_self_describing/connected_human_surface_template_payload_self_describing.npz
--output-dir output/surface_research_preflight_local/B_Fus3D19_bounded_surface_sdf_render_smoke_hybrid6_layer23
--status-report reports/20260507_b_fus3d_b19_bounded_surface_sdf_render_status.md
```

The template payload is valid for this contract because it contains
`hybrid_vertices`, `hybrid_faces`, `part_ids`, and the required masks:
`face_front_vertex_mask`, `hairline_vertex_mask`, `left_hand_vertex_mask`,
`right_hand_vertex_mask`, and `head_vertex_mask`.

## Outputs

Minimum outputs for a future B19 implementation:

```text
b_fus3d_b19_summary.json
b_fus3d_b19_report.md
b_fus3d_b19_control_comparison.json
b_fus3d_b19_query_family_support.json
b_fus3d_b19_render_metrics.json
b_fus3d_b19_loss_curve.json
b_fus3d_b19_stop_result.json
per-control diagnostic meshes or fields marked research_only, if rendered
per-view mask/depth/normal/rgb_residual PNGs, if nvdiffrast is available
```

Forbidden outputs:

```text
predictions.npz
checkpoint.pt or trained weight export
teacher bundle
candidate bundle
strict pass registry update
cloud train/infer/export artifact
```

## Hard Stop

B19 must stop immediately with zero strict passes if any of these is true:

```text
nvdiffrast unavailable for rendered closure;
CUDA unavailable or incompatible;
B17 compatibility is false;
real/shuffle/zero controls do not share query/grid/view dimensions;
any required query family is absent;
hairline or either hand support is hidden instead of reported;
loss is NaN/Inf;
max query/carrier displacement exceeds a fixed cap;
real does not beat shuffle and zero on rendered metrics;
Open3D/render review remains shell/slab/template-like in full/head/face/hands;
the implementation starts to rely on visual-hull or COLMAP labels as the main
SDF supervision;
the implementation becomes hidden-size, step-count, threshold, smoothing, or
component-filter tuning.
```

Runtime hard stop on the current local machine:

```text
blocked_no_nvdiffrast
```

This is evidenced by the existing local B1 and B2 summaries. Therefore the next
local-safe action is script/contract implementation plus `--help` or static
compile only, not a rendered run.

## Blocked Loops

Do not continue these loops under B19:

```text
B16 latent-field rerun, hidden_dim tuning, step tuning, extraction-level tuning;
B18 repeated rendered audit of the same B16 real/shuffle/zero meshes;
B5 or b_fus3d_learned_decoder_smoke.py tiny residual MLP;
A4 tiny SDF / part-local SDF repetition;
visual-hull weak-label SDF training;
COLMAP-depth weak-label SDF training;
B14 sparse/raw-image offset proposal line search;
B2 more steps, hidden size, or scalar loss weights;
thresholding, smoothing, or connected-component filtering to manufacture a pass;
strict registry writes;
teacher/candidate export;
formal cloud train/infer/export.
```

## Final Scope Call

```text
B19 is allowed only as a bounded, research-only surface-token/query-SDF plus
rendered-control smoke contract. It is not currently runnable end-to-end on this
local Windows runtime because rendered closure is blocked by missing
nvdiffrast. If implemented later, it must be a new script with fail-closed
controls and must not recycle B16, B18, small-MLP, visual-hull, COLMAP-label, or
A4 tiny-SDF loops.
```
