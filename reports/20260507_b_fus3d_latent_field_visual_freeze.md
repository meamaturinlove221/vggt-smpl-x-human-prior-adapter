# B-Fus3D16 Latent Field Visual Freeze

Status: `frozen_as_lightweight_latent_field_smoke_visual_fail`

This report freezes the fixed B-Fus3D16 latent-field smoke after Open3D review.
It is not a teacher, not a candidate, not a strict pass, and not a cloud
unblock.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal_cloud_train_infer_export = blocked
```

## Reviewed Artifacts

```text
output/surface_research_preflight_local/B_Fus3D16_latent_field_smoke_fixed_hybrid6_layer23/
output/surface_research_preflight_local/B_Fus3D16_real_open3d_review_full/
output/surface_research_preflight_local/B_Fus3D16_real_open3d_review_head/
output/surface_research_preflight_local/B_Fus3D16_real_open3d_review_hands/
```

The fixed smoke extracted meshes for real, shuffle, and zero controls:

```text
real:    vertices=2233, faces=4140, component_count=22, largest_component_ratio=0.9507
shuffle: vertices=2265, faces=4232, component_count=20, largest_component_ratio=0.9546
zero:    vertices=1430, faces=2752, component_count=13, largest_component_ratio=0.9637
```

The real and shuffle controls are too similar. This means the fixed B16 field is
mostly learning a weak occupancy / visual-hull-like scaffold rather than a
token-driven human surface with personal head, face, hairline, or hand detail.

## Open3D Visual Decision

The real mesh fails all mentor-relevant visual checks:

```text
full body: fragmented slab / shell; not a normal human surface
head: hollow fragmented ring-like surface; no modeled face or hairline
face: no nose / mouth / central-face relief
hands: no connected wrist / palm / finger surface; only fragmented shell pieces
side/back/iso: patchy thin sheets, not a closed or plausible human body
```

The reviewed hand views confirm that the mesh is not simply weak in the hand ROI;
the underlying surface itself is broken into thin sheets and fragments. This
cannot be promoted into a teacher or candidate by thresholding.

## Frozen Actions

Do not continue with:

```text
B16 hidden_dim tuning
B16 step-count tuning
B16 extraction-level tuning
B16 smoothing / threshold / component filtering to create a pass
B16 teacher export
B16 candidate export
B16 cloud unblock
```

## Allowed Next Direction

B16 can only be cited as evidence that latent-grid routing has a measurable
signal but the lightweight field is under-specified. The next B-Fus3D route must
be a genuinely different backend, such as a surface-token / SDF architecture with
visibility-aware multi-view aggregation, differentiable rendering supervision,
and part-specific heads. It must not be another small MLP or visual-hull label
loop.

