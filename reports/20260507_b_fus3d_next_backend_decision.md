# 2026-05-07 B-Fus3D Next Backend Decision

Status: `research_only_next_backend_contract_no_train_no_export`

This is a local decision report for the B-Fus3D learned-backend side line. It is
not a strict pass, not a teacher, not a candidate, and not a cloud unblock.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal_cloud_train_infer_export = blocked
teacher_export = blocked
candidate_export = blocked
```

## Local Facts Read

The relevant local reports and artifacts already exist:

```text
reports/20260507_b_fus3d_latent_grid_evidence_status.md
reports/20260507_b_fus3d_latent_grid_evidence_control_comparison.md
reports/20260507_b_fus3d_latent_field_smoke_status.md
reports/20260507_b_fus3d_latent_field_visual_freeze.md
reports/20260507_b_fus3d_token_cache_status.md
reports/20260507_b_fus3d_query_sdf_status.md
reports/20260507_b_fus3d_visual_hull_sdf_labels_status.md
reports/20260507_b_fus3d_colmap_depth_sdf_labels_status.md
reports/20260507_b_fus3d_depth_label_learnability_status.md
reports/20260507_b_fus3d_b14_sparse_offset_decision.md
```

B15 found a measurable token signal:

```text
real token_cosine_mean    = 0.867266
shuffle token_cosine_mean = 0.765353
zero token term           = null
supported_ratio           = 0.502229
boundary_like_ratio       = 0.658779
```

B16 then proved that a tiny fixed latent field can fit an evidence-shaped field
and extract meshes, but the controls do not separate well enough:

```text
real:    vertices=2233, faces=4140, component_count=22, largest_component_ratio=0.9507
shuffle: vertices=2265, faces=4232, component_count=20, largest_component_ratio=0.9546
zero:    vertices=1430, faces=2752, component_count=13, largest_component_ratio=0.9637
```

The B16 Open3D review is already frozen as a visual fail:

```text
full body: fragmented slab / shell; not a normal human surface
head: hollow fragmented ring-like surface; no modeled face or hairline
face: no nose / mouth / central-face relief
hands: no connected wrist / palm / finger surface; only fragmented shell pieces
side/back/iso: patchy thin sheets, not a closed or plausible human body
```

## Frozen Lines

Do not continue any of these as the next B-Fus3D unblocker:

```text
B16 hidden_dim / step-count / extraction-level tuning
B16 smoothing, thresholding, or component filtering to manufacture a pass
B5/B16-style tiny MLP residual backend
B8 visual-hull SDF labels as training labels
B9/B10 COLMAP-depth weak labels as standalone SDF supervision
B14 sparse/raw-image offset proposal loop
B2/B5-style more steps, more hidden, or scalar loss-weight tuning
teacher export, candidate export, strict registry writes, or cloud unblock
```

These lines are frozen because they already tested the cheap hypotheses:

```text
small residual geometry barely moved the carrier;
visual-hull labels were mostly ambiguous;
COLMAP depth labels were learnable but weak as true surface supervision;
B14 produced sparse query proposals rather than a continuous surface;
B16 real and shuffle meshes were too similar and visually failed.
```

## Evidence That Remains Useful

The useful part of B15/B16 is not the mesh. The useful part is the input contract:
VGGT tokens can be queried by 3D locations, and real tokens carry more
cross-view consistency than shuffled tokens.

Keep:

```text
B0 token cache and layer-23 token arrays
B1 pooled part surface-token features
B6 query evidence cache with view-aware support
B6 query-support Open3D visualization for risk localization
B15 latent-grid evidence arrays and real/shuffle/zero controls
B10 learnability readout only as a signal that token evidence predicts weak depth labels
B2 balanced critical-family carrier support idea as infrastructure, not as geometry success
```

Useful support facts:

```text
face/head/hairline token coverage exists in B0;
face_core query support is strong in B6;
full_body query support is strong in B6;
left_hand support is usable in the selected hybrid6 query cache;
hairline and right_hand remain explicit low-support risk regions.
```

Not useful as pass evidence:

```text
B16 extracted mesh existence
B16 low fit loss
B16 largest component ratio
B15 occupancy/boundary-like fields by themselves
weak depth/visual-hull labels without rendered surface validation
```

## Next Architecture Contract

The next non-repeated B-Fus3D backend must change both representation and
supervision. It should be a surface-token/SDF backend with differentiable render
closure, not a lightweight field fitted to B15 evidence alone.

Required representation:

```text
query-level SDF or signed surface field, not vertex offset only
surface-token conditioning from B0/B1/B6/B15 evidence
visibility-aware multi-view aggregation per query
part/family-specific heads for body, face_core, hairline, left_hand, right_hand
explicit support/risk output per query and per rendered view
real/shuffle/zero token-control slots built into the contract
```

Required supervision:

```text
differentiable rendered mask/depth/normal/RGB diagnostics from the predicted surface
same-frame multi-view consistency losses
silhouette/depth/normal losses treated as diagnostics until visual review passes
part-local rendered diagnostics for head/face/hairline/hands
negative-control comparison against shuffle and zero tokens
```

Required stop rules:

```text
no strict pass write
no teacher/candidate export
no cloud formal train/infer/export
stop if real does not beat shuffle/zero on rendered diagnostics
stop if Open3D full/head/face/hands remains shell/slab/template-like
stop if low-support regions are hidden by confidence rather than reported
```

## Exact Local Minimal Artifact To Implement Next

Implement a contract-only preflight script:

```text
tools/b_fus3d_surface_sdf_contract_preflight.py
```

The script should be local-only and lightweight. It should not optimize, train,
extract a final mesh, call cloud, write predictions, or touch strict registries.

Inputs:

```text
--token-cache output/surface_research_preflight_local/B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/token_cache/aggregator_layer_23.npz
--query-evidence output/surface_research_preflight_local/B_Fus3D6_query_evidence_cache_hybrid6_layer23/b_fus3d_query_evidence_cache.npz
--latent-grid-real output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/b_fus3d_latent_grid_evidence_arrays.npz
--latent-grid-shuffle output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/b_fus3d_latent_grid_evidence_arrays.npz
--latent-grid-zero output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/b_fus3d_latent_grid_evidence_arrays.npz
--output-dir output/surface_research_preflight_local/B_Fus3D17_surface_sdf_contract_preflight_hybrid6_layer23
```

Outputs:

```text
b_fus3d_surface_sdf_contract_summary.json
b_fus3d_surface_sdf_contract_report.md
query_contract_schema.json
render_supervision_contract.json
control_comparison_contract.json
```

The minimal artifact should verify only:

```text
input files exist and have compatible query/view/feature dimensions;
query evidence can be partitioned into body, face_core, hairline, left_hand, right_hand;
real/shuffle/zero control arrays have matching grids;
the future SDF head API has named outputs: sdf, occupancy, normal, rgb, confidence, support;
the future renderer API has named diagnostics: mask, depth, normal, rgb, residuals;
all outputs are marked research_only and blocked_from_gate.
```

Success criterion for this next artifact:

```text
contract_preflight_complete = true
trained_weights_written = false
mesh_written = false
predictions_written = false
strict_candidate_passes = 0
strict_teacher_passes = 0
formal_cloud_train_infer_export = blocked
```

Only after this contract preflight exists should a later bounded learned smoke be
considered. That later smoke must compare real/shuffle/zero rendered diagnostics
and must fail closed if the Open3D full/head/face/hands visual review remains
shell-like.

## Decision

The next non-repeated B-Fus3D unblocker is:

```text
Build B-Fus3D17 as a local surface-token/SDF contract preflight with built-in
render-supervision and control-comparison schema.
```

It is intentionally not:

```text
B16 tuning
small MLP tuning
visual-hull label training
COLMAP-label training
B14 offset continuation
strict pass writing
cloud training/inference/export
```

