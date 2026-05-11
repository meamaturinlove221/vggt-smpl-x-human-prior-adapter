# A-Pose Kinect And HART Surface Preflight Closure

Date: 2026-05-04

## Current Truth

The current strict registry is still red:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- strict gate schema: `20260504_visual_fullbody_hands_v2`
- cloud upload/run: blocked by local guard

This note closes the latest A-pose Kinect and HART-style surface preflight so
that they are not accidentally treated as new passing teacher routes.

## A-Pose Kinect Asset Status

Readable same-subject A-pose Kinect assets exist locally:

```text
G:\data-used-in-4K4D-compatible-path\apose\apose_kinect\0012_apose03_kinect.smc
```

PowerShell renders the non-ASCII root with mojibake in some logs, but the raw
asset is present and was fused in two TSDF asset-quality diagnostics:

```text
output/local_teacher_probes/apose_kinect_tsdf_asset_v02/kinect_smc_tsdf_summary.json
output/local_teacher_probes/apose_kinect_tsdf_asset_v03_allcomponents/kinect_smc_tsdf_summary.json
```

Observed asset facts:

- v02 mesh: `43435` vertices, `87460` triangles, extent
  `[0.3426, 0.7567, 0.1818]`, `1328` connected components before filtering;
- v03 all-components mesh: `62847` vertices, `116191` triangles, extent
  `[0.6623, 1.5297, 0.6601]`;
- both summaries explicitly mark the result as
  `asset_quality_diagnostic_not_teacher_pass`;
- both summaries state that the mesh is not aligned to the target VGGT world
  and cannot be used for training unless later pose/deformation and strict
  teacher gates pass.

## Why This Does Not Unblock Training

The A-pose asset is not the target action frame:

```text
target protocol: 0012_11_frame0000_6views_sparseproto_headshoulder_crop
asset pose:      0012_apose03 Kinect A-pose
```

Therefore it cannot be used as a head/face/hairline teacher by direct
projection, TSDF patching, or mesh transfer. To become a valid teacher source,
it would first need a separate nonrigid transfer stage:

1. fit or recover the A-pose surface in its own Kinect/RGB camera convention;
2. establish reliable SMPL-X/body correspondences without making SMPL-X a
   face, hair, or clothing teacher;
3. pose/deform the clothed surface into the target frame;
4. project it back into the original 6-view headshoulder crop protocol;
5. pass `tools/audit_headface_teacher_surface.py` with explicit Open3D visual
   review;
6. only then run a one-frame local overfit smoke and the full candidate gate.

No such strict-passing transferred teacher currently exists locally.

## HART-Style Surface Preflight Status

HART remains useful as a design reference, not as a simple local switch:

- HART is not camera-free; it estimates cameras from predicted point maps.
  The local HART-style PnP/camera-replacement ablation was already negative.
- The repo has local TSDF/Poisson, shared surfel/canonical-bin, unlearned
  oriented indicator, and Sapiens-normal oriented indicator probes.
- Those local substitutes have failed the current strict gate, including
  full-body, hands, head/face shape, signed normal convention, and explicit
  Open3D visual review.
- The repo does not currently contain a ready HART-quality learned surface
  backend with oriented point maps, residual human normals, body
  correspondences/tightness, DPSR/indicator-grid fusion, and learned 3D grid
  refinement.

The latest HART-adjacent local probes are already closed:

```text
reports/20260504_r45_oriented_indicator_surface_closure.md
reports/20260504_r46_sapiens_oriented_indicator_closure.md
```

## Decision

Do not proceed with:

- direct A-pose Kinect TSDF as head/face teacher;
- A-pose mesh patching into the target VGGT predictions;
- A-pose-to-target training without a strict transferred-teacher gate pass;
- HART-style PnP replacement as a camera-free route;
- Sapiens-normal, Poisson, TSDF, shared-surfel, or unlearned indicator reruns
  that only change parameters.

Allowed future work from this branch is limited to a genuinely new
pose-transfer or learned visibility-aware surface implementation. It must first
produce either:

- a strict-passing teacher under the original 6-view headshoulder protocol; or
- a strict-passing teacherless/self-supervised candidate package with
  full-body and hands passing the visual gate.

Until that happens, the truthful status remains:

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
cloud                   = blocked
```
