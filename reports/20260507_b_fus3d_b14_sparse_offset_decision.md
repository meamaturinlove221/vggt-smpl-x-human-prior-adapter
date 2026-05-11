# B-Fus3D14 Sparse Offset Decision

Status: `frozen_as_sparse_raw_image_photometric_offset_readout`

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher_export = blocked
candidate_export = blocked
```

## Inputs Reviewed

```text
output/surface_research_preflight_local/B_Fus3D14_raw_image_offset_proposal_precheck_hybrid6_layer23/
output/surface_research_preflight_local/B_Fus3D14_raw_image_offset_proposal_open3d_review_full/
output/surface_research_preflight_local/B_Fus3D14_raw_image_offset_proposal_open3d_review_head/
output/surface_research_preflight_local/B_Fus3D14_raw_image_offset_proposal_open3d_review_hands/
reports/20260507_b_fus3d_raw_image_offset_proposal_status.md
reports/20260507_b_fus3d_raw_image_offset_proposal_visual_full_status.md
reports/20260507_b_fus3d_raw_image_offset_proposal_visual_head_status.md
reports/20260507_b_fus3d_raw_image_offset_proposal_visual_hands_status.md
```

## Numeric Readout

```text
query_count = 576
selected_count = 297
selected_ratio = 0.515625
negative_offset_count = 169
positive_offset_count = 128
fixed_delta = 0.012
```

## Visual Review

The full/head/face/hands Open3D screenshots show sparse query points rather
than a continuous surface. The head and face regions do not contain modeled
facial geometry or a continuous hairline/head surface. The hand regions appear
as small point clusters without wrist connectivity, fingers, or a continuous
hand surface.

This is therefore not a normal-human Open3D surface and not eligible for a
teacher/candidate gate.

## Decision

```text
B-Fus3D14 is useful only as evidence that a raw-image photometric offset signal
exists at sparse query points.

It is not:
- a mesh
- a dense teacher
- a VGGT candidate
- a strict pass
- a cloud/export trigger
```

## Allowed Next Action

Only one bounded rendered-mesh proposal precheck may be designed, and only as a
research diagnostic. It must keep the B14 query set, selected mask, offset
delta, cameras, render settings, and random seed fixed. It must output
fragmentation, largest connected component, continuity, full/head/face/hairline
/ hands Open3D review, and an explicit fail/pass-for-next-discussion JSON.

It must not tune offset scale, thresholds, smoothing, hole fill, query count,
confidence, hidden size, steps, or render parameters into another loop. It must
not export a teacher/candidate or write any strict pass.

## No-Go Criteria

If the bounded precheck still renders as sparse fragments, shell, detached hand
clusters, template face, or non-human topology, the B-line must move from sparse
offset readouts to a real learned surface-token/SDF backend instead of adding
another B14 variant.
