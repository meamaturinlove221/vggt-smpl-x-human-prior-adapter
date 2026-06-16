# Surface Upper-Bound Preflight Status

Date: 2026-05-04

## Current Truth

This branch adds a local-only surface upper-bound preflight:

```text
tools/optimize_human_surface_upperbound.py
```

The run does **not** produce a mentor-level result. It is also not true
gradient-based surface optimization, because the current default environment has
no differentiable renderer:

```text
pytorch3d: false
nvdiffrast: false
kaolin: false
true_gradient_surface_optimization_available: false
```

The script therefore truthfully labels its outputs as:

```text
dense_observation_upperbound_preflight_not_true_differentiable_optimization
```

It consolidates dense VGGT observations into shared surfels through SMPL-X
canonical bins, then rasterizes those surfels back to the original sparse-view
protocol. SMPL-X is used only for canonical correspondence / visible body-part
structure, not as a face or hair geometry teacher.

Cloud remains blocked:

```text
reports/20260504_strict_gate_registry.json
strict_candidate_passes = 0
strict_teacher_passes = 0
```

## What Was Tested

Three dense-to-sparse preflight runs were created under:

```text
output/normal_line_multiview_20260504/surface_upperbound_preflight_60v_to_6v_headshoulder
output/normal_line_multiview_20260504/surface_upperbound_preflight_60v_to_6v_headshoulder_loose
output/normal_line_multiview_20260504/surface_upperbound_preflight_60v_to_6v_fullbody_loose
```

The first run used strict raster distance. The loose runs intentionally allowed
a much larger raster distance only to diagnose whether the previous rejection
was caused by too-tight distance gating. The loose runs are diagnostic and cannot
be treated as a pass without the strict candidate gate.

## Headshoulder Strict Result

Input:

```text
60v targetcam30 raw head/face hardmask source
6v original headshoulder target
max_raster_distance = 0.08
```

Key results:

```text
accepted surfels: 1221
head surfels: 863
face surfels: 492
hands surfels: 424
changed target pixels: 1629
full changed fraction: 0.0058
head changed fraction: 0.0126
face changed fraction: 0.0245
hands changed fraction: 0.0000
```

Risk flags:

```text
face_surfel_support_sparse = true
hand_surfel_support_sparse = true
face_rasterization_low_change = true
head_rasterization_low_change = true
hands_rasterization_low_change = true
```

Conclusion: the dense same-frame observations barely transfer back into the
original 6-view headshoulder protocol. This cannot be a candidate.

## Loose Headshoulder Diagnostic

Input:

```text
same source/target as above
max_raster_distance = 0.30
```

Key results:

```text
accepted surfels: 1221
changed target pixels: 71161
full changed fraction: 0.2547
head changed fraction: 0.4833
face changed fraction: 0.4864
hands changed fraction: 0.1068
```

This confirms that loosening raster distance can force many pixels to change,
but the face and hand surfel support remains sparse. This is a geometry-warning
diagnostic, not a successful upper bound.

## Loose Full-Body Diagnostic

Input:

```text
60v human-crop source
6v human-crop full-body target
max_raster_distance = 0.30
```

Key results:

```text
accepted surfels: 8802
head surfels: 5610
face surfels: 3810
hands surfels: 1844
changed target pixels: 134158
full changed fraction: 0.8128
head changed fraction: 0.7962
face changed fraction: 0.7851
hands changed fraction: 0.6932
```

This looks more active numerically, but the strict candidate gate and Open3D
visual review reject it. High changed-pixel coverage is not evidence of normal
human geometry.

## Full Candidate Gate

Candidate-specific full-body input was used and provenance now passes:

```text
candidate: surface_upperbound_preflight_60v_to_6v_fullbody_loose
gate output:
output/normal_line_multiview_20260504/candidate_gate_surface_upperbound_preflight_60v_to_6v_fullbody_loose
fullbody_provenance_gate.pass = true
```

The real gates fail:

```text
numeric_gate.pass = false
fullbody_gate.pass = false
normal_gate.pass = false
shape_gate.pass = false
visual_gate.pass = false
cloud_upload_blocked = true
```

Numeric face margin:

```text
world_points p40 face delta: +352
required face margin: +500
depth_unprojection p40 face delta: 0
```

Normal-depth-point consistency regresses on the geometry relations that matter:

```text
head pred_vs_depth: 11.73 deg -> 25.68 deg
head depth_vs_point: 6.50 deg -> 23.97 deg
face pred_vs_depth: 12.68 deg -> 25.79 deg
face depth_vs_point: 8.06 deg -> 24.16 deg
```

`face pred_vs_point` improves numerically, but that isolated improvement does
not rescue the candidate because depth/point agreement, shape, full-body, hands,
and Open3D visual all fail.

Full-body / hand gate:

```text
world_points p40: fail, hand fail
world_points fixed: fail, hand fail
depth_unprojection p40: fail, hand fail
depth_unprojection fixed: fail, hand fail
```

Explicit visual review:

```text
output/normal_line_multiview_20260504/candidate_gate_surface_upperbound_preflight_60v_to_6v_fullbody_loose/visual_review_codex_fail.json
```

The contact sheet shows shell-like head/face surfaces, non-human full-body slabs
and flying sheets from side/right/back/iso views, and detached sparse hand
fragments. It does not look like a normal human in Open3D.

## Conclusion

The surface upper-bound preflight is a useful non-redundant diagnostic, but it
does not unblock the mentor task. It proves that simply consolidating existing
VGGT dense observations through SMPL-X canonical bins and rasterizing them back
to the sparse protocol still reorganizes shell geometry rather than producing a
modeled head/face/hairline, attached hands, and normal full-body surface.

This route should now be frozen unless a genuinely new ingredient is added.
Do not continue by tuning only:

```text
canonical_bin_size
min_surfel_observations
min_surfel_views
max_surfel_spread
max_raster_distance
alpha
confidence percentile
```

Those would be threshold/support loops and are already covered by the strict vs
loose diagnostics above.

## Remaining Non-Wall Unblockers

Only two routes remain technically meaningful:

1. Provide or construct a strict-passing target-frame dense human surface teacher
   that already passes teacher gate and Open3D visual review.
2. Add a true learned / differentiable human surface backend with usable
   silhouette, photometric, depth, normal, and visibility objectives. This first
   requires a local differentiable renderer or an equivalent render-and-optimize
   module. The current preflight is not that module.

Until one of these exists and then passes the full local strict candidate gate,
cloud upload/run remains blocked.
