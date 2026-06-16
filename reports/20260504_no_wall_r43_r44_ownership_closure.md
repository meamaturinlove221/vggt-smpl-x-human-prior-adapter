# No-Wall Closure: r43/r44, Ownership, Gradient, And Surface Backend

Date: 2026-05-04

## Current Truth

- Strict candidate passes: `0`
- Strict teacher passes: `0`
- Cloud upload/run: blocked
- Registry: `reports/20260504_strict_gate_registry.json`
- Current schema: `20260504_visual_fullbody_hands_v2`

This is not a mentor-final state. The local outputs still do not pass the
combined head/face/hairline, full-body, hand, shape, normal, and explicit
Open3D visual gates.

## r40 Visual Recheck

Visual sheet:

```text
output/normal_line_multiview_20260504/candidate_gate_r40_partaware_softweight_bodyhand_smoke1/candidate_gate_visual_sheet.png
```

Direct visual read:

- head/face are still shell-like with large central holes;
- the apparent fixed-threshold or p40 count changes do not create eyes, nose,
  mouth, hairline, or a modeled facial surface;
- full-body side/back views remain sheet-like rather than a solid normal human;
- hand crops are detached sheet fragments, not attached hand geometry.

Therefore r40 remains negative even where normal consistency or some fixed
threshold point counts look better.

## r43 Shared Surfel Backend Is Already A Negative Minimal Probe

Gate package:

```text
output/normal_line_multiview_20260504/candidate_gate_r43_rayconsistent_shared_surfel_backend_diag/report.md
```

The r43 probe already tested the local version of a shared human point/surfel
backend:

- SMPL-X canonical bins were used as cross-view correspondence keys only;
- output positions came from VGGT observations, not SMPL-X vertices;
- both world_points and depth_unprojection were rasterized back into the same
  protocol;
- full/head/face/hands Open3D packages were generated.

It failed:

- world_points face p40: `15669`, below signfix `16825`;
- depth_unprojection face p40: `16031`, below signfix `16764`;
- head/face depth-normal and depth-point consistency regressed strongly;
- full-body/hands visual gate failed;
- explicit Open3D visual review failed.

The visual sheet still shows head/face holes and fragmented hands. Do not rerun
this by only changing canonical bin size, support thresholds, raster distance,
or point confidence. That would be another support/threshold loop.

## r44 Real-Camera Oracle Is Negative

Gate package:

```text
output/normal_line_multiview_20260504/candidate_gate_r44_real_camera_oracle_keepworld/report.md
```

r44 replaced the depth-unprojection camera chain with crop-corrected 4K4D RGB
camera geometry aligned into the VGGT gauge, while keeping world_points
unchanged. This was a camera/crop diagnostic, not HART/PnP and not a teacher.

It failed:

- world_points face p40 did not improve: `16825 -> 16825`;
- depth_unprojection face p40 regressed: `16764 -> 16162`;
- fixed depth face also regressed: `17214 -> 16584`;
- head/face normal consistency regressed in all listed comparisons;
- full-body/hands visual gate failed.

This closes the local hypothesis that the current failure is mainly caused by
the VGGT camera head or crop-intrinsics convention. Camera replacement,
HART-style PnP, and real-camera oracle variants are not current unblockers.

## p40 Ownership Diagnosis

Ownership audit:

```text
output/normal_line_multiview_20260504/p40_ownership_geometry_r40_vs_signfix_headshoulder/p40_ownership_geometry_summary.json
```

The ownership audit explains why fixed-threshold and p40 point-count changes
can look promising but still fail visually.

Target view `0`, face ROI, p40:

| Source | Group | Pixels | Coverage | LCC |
|---|---:|---:|---:|---:|
| world_points | candidate_only | `7296` | `0.586` | `1.000` |
| world_points | baseline_only | `8758` | `0.704` | `0.971` |
| depth_unprojection | candidate_only | `2961` | `0.238` | `0.808` |
| depth_unprojection | baseline_only | `12446` | `1.000` | `1.000` |

Target view `0`, head ROI, p40:

| Source | Group | Pixels | Coverage | LCC |
|---|---:|---:|---:|---:|
| world_points | candidate_only | `18785` | `0.749` | `1.000` |
| world_points | baseline_only | `19295` | `0.769` | `0.971` |
| depth_unprojection | candidate_only | `9123` | `0.364` | `0.820` |
| depth_unprojection | baseline_only | `25090` | `1.000` | `1.000` |

The candidate does not create a new coherent target-view depth surface. In the
face/head depth_unprojection path it loses most of the baseline support. This
matches the Open3D read: the model is changing confidence ownership and visible
sheets, not producing a modeled face/head surface.

## Gradient Path Is Not The Main Blocker

Gradient audits:

```text
output/normal_line_multiview_20260504/selfgeom_grad_audit_r40_partaware_softweight_bodyhand_v2/grad_audit.md
output/normal_line_multiview_20260504/selfgeom_grad_audit_r41_headshoulder_case_only/grad_audit.md
```

The self-geometry and weak-SMPL-X anchor losses do reach the world/depth/point
heads. For example, r40 reports non-zero gradients from:

- `loss_prior_depth_point`;
- `loss_prior_depth_point_normal`;
- `loss_prior_smplx_weak_anchor`.

So the current blocker is not simply a disconnected loss. The losses push the
geometry heads, but the available targets/objectives still converge toward
self-consistent sheets or shell surfaces instead of real head/face/hairline
geometry.

## Confidence Is A Symptom, Not An Unblocker

The r40 face ROI diagnosis shows a confidence/ranking split:

- baseline p40 threshold: `38.5067`;
- r40 p40 threshold: `45.5556`;
- fixed-threshold face counts are higher than p40 counts.

However fixed-threshold Open3D still looks shell-like and does not pass the
shape, full-body, hand, or visual gates. Therefore confidence boosting,
threshold tuning, or p40/fixed reweighting must remain frozen unless tied to a
new surface mechanism that passes visual geometry.

## Local HART-Like Surface Backend Availability

Local dependency check in the `g3splat` environment:

```text
torch: available
open3d: available
skimage: available
pytorch3d: unavailable
kaolin: unavailable
nvdiffrast: unavailable
mcubes: unavailable
trimesh: unavailable
point_cloud_utils: unavailable
DPSR/dpsr: unavailable
```

The repo currently has Open3D TSDF/Poisson-style scripts and shared
point/surfel aggregation scripts, but it does not contain a HART-style DPSR /
indicator-grid / learned continuous surface backend. The available local
surface tools have already produced shell-like or strict-gate-failed results.

Therefore "HART-like surface representation" remains a research-level
unblocker, not an already implemented local candidate.

## Raw Asset And Teacher Status

The raw 4K4D assets are local, but the SMC inventory found no ready dense
head/face mesh or aligned surface. Kinect TSDF forms a coarse raw sensor mesh,
but the strict original 6-view teacher gate fails:

```text
output/normal_line_multiview_20260504/teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews/teacher_gate_summary.json
```

Key failure:

- max face_core coverage: `0.0008`;
- max head_face coverage: `0.1885`;
- hairline coverage: `0.0`.

Kinect remains a coordinate/asset diagnostic only unless a future convention
fix produces a strict teacher gate pass with explicit Open3D visual approval.

## Frozen After This Closure

Do not continue by:

- adding epochs or small weight changes to r16/r18/r19/r37/r40-r44/r57-r66;
- tuning p40/fixed thresholds or confidence boosts;
- rerunning shared surfel with only bin/support/raster-distance changes;
- replacing cameras via HART-style PnP or r44-style real-camera oracle;
- recycling TSDF, Poisson, visual hull, Kinect, COLMAP/MVS, or SMPL-X face
  scaffold as a teacher without a current strict teacher pass.

## Remaining Honest Unblockers

The remaining local unblockers are narrow:

1. Bring in or implement a genuinely continuous surface/field backend that is
   not just point/surfel aggregation or TSDF from the current shell. It must
   locally pass the full candidate gate before any cloud work.
2. Find a new same-subject, same-frame dense head/face/hairline surface asset
   that can be projected into the original 6-view headshoulder protocol and
   pass strict teacher gate before any teacher-supervised training.
3. If Kinect is revisited, first prove a coordinate-convention fix under strict
   teacher gate and Open3D visual review. No Kinect patching or training before
   that.

Until one of these happens, the truthful state remains: no mentor-level pass
and no cloud upload/run.
