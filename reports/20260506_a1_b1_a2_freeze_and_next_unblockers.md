# A1/B1/A2 Research-Preflight Freeze And Next Unblockers

Status: `research_preflight_negative_not_mentor_pass`

This report freezes the first surface-research preflight matrix after the
Modal research-only entrypoint was made reliable. It does not claim mentor
success, does not export a teacher, does not export a candidate, and does not
write strict pass state.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
research-preflight Modal = allowed only as isolated artifact generation
```

Formal VGGT cloud entrypoints remain guarded. The A-line/B-line runs below are
research-only unblocker checks.

## A1: Visual-Hull Mesh Refinement

Output:

```text
output/surface_research_preflight/A1_refine_visual_hull_mesh_t96_step20_sdf_gpu
```

Key numbers:

```text
views = 0,10,20,30,40,50
avg_initial_iou = 0.6613060806
avg_final_iou = 0.6734070187
avg_iou_delta = +0.0121009381
avg_initial_precision = 0.6615026433
avg_final_precision = 0.6739816907
avg_precision_delta = +0.0124790474
avg_recall_delta = -0.0007005853
strict_candidate_passes = 0
strict_teacher_passes = 0
```

Decision:

```text
freeze_as_silhouette_shrink_preflight
```

Reason:

```text
A1 confirms nvdiffrast/raw-mask gradients can shrink an over-covering visual
hull. It remains silhouette-driven and has no evidence of modeled head, face,
hairline, hands, clothing, or full-body free-view surface quality. It must not
be used as a teacher or candidate.
```

## B1: Surface-Token Backend Smoke

Output:

```text
output/surface_research_preflight/B1_surface_tokens_t96_step10_gpu
```

Key numbers:

```text
views = 0,10,20,30,40,50
avg_initial_iou = 0.7594788682
avg_final_iou = 0.7596625728
avg_iou_delta = +0.0001837047
tokens_with_min_view_support = 342
raster_tokens_with_min_view_support = 273
strict_candidate_passes = 0
strict_teacher_passes = 0
```

Decision:

```text
freeze_current_b1_as_plumbing_smoke
```

Reason:

```text
B1 proves the cloud nvdiffrast/token path runs and exports token visibility,
mask/depth/normal/support/photometric diagnostics. The learned geometry signal
is too weak to move the surface. It should not become a hidden-size, step-count,
or scalar-weight tuning loop unless the surface source/supervision changes.
```

## A2: Tiny Raw Neural Occupancy Field

A2 is the first lane in this matrix that does not recycle VGGT depth, VGGT
point maps, or VGGT normals. It optimizes a small occupancy/radiance field from
raw RGB, raw masks, known cameras, and the A3 visual-hull bbox.

### A2 6-View Training Run

Output:

```text
output/surface_research_preflight/A2_neural_field_t64_step120_6v_gpu
```

Key numbers:

```text
views = 0,10,20,30,40,50
avg_render_iou = 0.8431617105
avg_render_precision = 0.8682773392
avg_render_recall = 0.9671884969
avg_foreground_rgb_residual_mean = 0.2681049605
occupied_points = 38530
occupied_fraction = 0.1469802856
mesh_status = extracted
mesh_component_count = 88
mesh_largest_component_ratio = 0.9549
contact_sheet = output/surface_research_preflight/A2_neural_field_t64_step120_6v_gpu/A2_mesh_review/contact_sheet.png
```

Interpretation:

```text
The raw neural field can fit training-view masks and RGB partially, which is a
real signal. The extracted mesh is still a thick occupancy volume: side/top
views are slab-like, and face, hairline, and hands have no mentor-reviewable
surface detail.
```

### A2 Thin Regularized Run With Earlier Eval Bug

Output:

```text
output/surface_research_preflight/A2_neural_field_t64_step160_train4_eval2_thin_gpu
```

Key numbers:

```text
intended_train_views = 0,20,40,50
intended_eval_views = 10,30
actual_eval_views = 0,20,40,50
avg_render_iou = 0.8856387507
avg_render_precision = 0.8967116537
avg_render_recall = 0.9860241571
occupied_points = 32992
occupied_fraction = 0.1258544922
mesh_component_count = 91
mesh_largest_component_ratio = 0.9536
```

Interpretation:

```text
This run is useful only as a train-view fit. The Modal wrapper did not pass
held-out eval views in that revision, so it must not be cited as held-out
evidence.
```

### A2 Fixed Held-Out Run

Output:

```text
output/surface_research_preflight/A2_neural_field_t64_step40_train4_eval2_thin_fixed_gpu
```

Launch verification:

```text
cmd includes --eval-view-indices 10,30
actual_gpu = NVIDIA A100-SXM4-40GB
status = completed
returncode = 0
formal VGGT cloud train/infer/export = blocked at launch
```

Key numbers:

```text
train_views = 0,20,40,50
eval_views = 10,30
train_avg_render_iou = 0.7301804157
train_avg_render_precision = 0.7494503314
train_avg_render_recall = 0.9655343688
train_avg_foreground_rgb_residual_mean = 0.2543994188
eval_avg_render_iou = 0.6454713535
eval_avg_render_precision = 0.6589585395
eval_avg_render_recall = 0.9693406593
eval_avg_foreground_rgb_residual_mean = 0.2418985143
occupied_points = 79544
occupied_fraction = 0.3034362793
mesh_status = extracted
mesh_vertices = 21896
mesh_faces = 42970
mesh_component_count = 75
mesh_largest_component_ratio = 0.9731001096
contact_sheet = output/surface_research_preflight/A2_neural_field_t64_step40_train4_eval2_thin_fixed_gpu/A2_mesh_review/contact_sheet.png
```

Visual decision:

```text
fail
```

Reason:

```text
Held-out recall is high but held-out precision is low. The model covers the
human mask by over-filling a thick volume rather than learning a thin,
continuous, person-specific surface. The contact sheet shows a thick side/top
occupancy slab with no modeled face, hairline, or connected hand detail.
```

## Frozen Loops

The following are now explicitly disallowed for this matrix:

```text
A1 plus more silhouette steps
B1 plus hidden-size/step/weight tuning
A2 plus occupancy-threshold/render-threshold tuning
A2 plus more steps without changing representation
A2 train-view-only metrics as held-out evidence
any teacher export from A1/B1/A2
any strict pass write from A1/B1/A2
any formal cloud train/infer/export while strict passes are 0
```

## Next Non-Redundant Unblockers

The next A-line should change the representation, not tune the current one:

```text
A2.2 SDF/NeuS-style surface field:
  - signed-distance or density-derived surface representation
  - eikonal regularization
  - near-surface opacity concentration
  - multi-view RGB/mask supervision
  - rendered depth/normal outputs
  - explicit held-out view metrics
  - mesh Open3D/contact-sheet review before any teacher gate

or external same-frame dense reconstruction backend:
  - known-camera neural surface / Gaussian surface extraction / MVS only as init
  - must rasterize to original 6-view protocol
  - must pass Open3D full/head/face/hairline/hands before teacher export
```

The next B-line should also change the carrier/supervision:

```text
B2 local surface-token backend:
  - not image_mlp++ and not B1 scalar tuning
  - part-specialized face/hair/hand/body surface tokens
  - visibility-aware multi-view feature aggregation
  - nvdiffrast mask/RGB/depth/normal losses
  - only enter teacher/candidate flow after visible Open3D improvement
```

## Bottom Line

```text
A1/B1/A2 prove the Modal research-preflight environment works and that raw
RGB/mask/camera signals are usable. They do not produce a mentor-level teacher
or candidate. The current tiny occupancy field should be frozen as a useful
negative: it fits masks by making a thick volume, not a normal human surface.
```
