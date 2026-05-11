# No Unreviewed Local Surface Unblocker Inventory

Date: 2026-05-04

## Current Truth

The current strict registry remains red:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud upload/run: blocked

This inventory records a read-only sweep for any local surface backend or dense
asset that has not already been covered by the 2026-05-04 strict registry and
closure reports.

## Result

No new, unreviewed local unblocker was found.

The workspace does not currently contain a ready local learned surface backend
equivalent to the HART stack:

- oriented point maps plus residual human normals;
- body correspondence/tightness;
- DPSR or indicator-grid fusion;
- learned 3D grid refinement;
- usable local weights/code to run that full pipeline.

The repo does contain diagnostic or external-adapter code, but those are not
strict-passing unblockers.

## Local Code / Asset Findings

LHM-related files exist:

```text
D:\vggt\vggt-main\__tmp_external_repos\LHM
D:\vggt\vggt-main\modal_lhm_mesh_teacher.py
```

However, they are not a ready local strict-passing teacher route:

- no directly usable local LHM weights were found in this workspace sweep;
- existing LHM-derived teacher gates have already failed strict teacher review;
- Modal/download-oriented wrappers cannot authorize local cloud training while
  the strict registry is red.

Sapiens-related wrappers and outputs exist:

```text
D:\vggt\vggt-main\modal_sapiens_normal_teacher.py
D:\vggt\vggt-main\modal_sapiens_depth_teacher.py
D:\vggt\vggt-main\output\detail_normal_refiner_20260425\external_sapiens_normal_teacher\original6v_headshoulder_sapiens03b\sapiens_normals.npz
```

But the Sapiens-normal local variant was already tested in r46 and failed:

```text
reports/20260504_r46_sapiens_oriented_indicator_closure.md
```

MediaPipe assets exist:

```text
D:\vggt\vggt-main\external_models\face_landmarker.task
D:\vggt\vggt-main\external_models\hand_landmarker.task
```

They are landmark detectors, not dense head/face/hairline/full-body surface
teachers. Existing facelandmark shared-surface attempts have already failed
strict teacher gate.

Raw 4K4D assets exist and are useful for input/camera/SMPL-X bridge work, but
the local inventory found no hidden same-frame dense clothed mesh, dense head
surface, or ready head/face/hairline teacher not already audited.

## Already Covered / Frozen

The sweep found no reason to reopen:

- r43 shared surfel / canonical-bin backend;
- r44 real-camera/crop-corrected camera oracle;
- r45 unlearned oriented indicator surface;
- r46 Sapiens-normal oriented indicator surface;
- LHM mesh teacher gates;
- MediaPipe/facelandmark TSDF or patch routes;
- Kinect/TSDF direct teacher routes;
- COLMAP/MVS/60v dense-only teacher routes;
- SMPL-X face scaffold or strong-prior routes.

Relevant closure reports:

```text
reports/20260504_no_wall_r43_r44_ownership_closure.md
reports/20260504_r45_oriented_indicator_surface_closure.md
reports/20260504_r46_sapiens_oriented_indicator_closure.md
reports/20260504_apose_kinect_and_hart_surface_preflight_closure.md
reports/20260504_r40_r41_p40_ownership_and_fullbody_visual_closure.md
```

## Decision

Do not continue by re-running old external model names, local mesh-looking
assets, or non-learned surface reconstruction parameter sweeps.

The only local actions that would be non-redundant are:

1. introduce a genuinely local learned/optimized visibility-aware surface
   backend with the missing HART-like components and run it through the current
   strict candidate gate; or
2. add a new same-subject/same-frame dense calibrated teacher asset that
   includes head/face/hairline/full-body/hands, then run exactly one strict
   teacher gate before any training.

Until one of those exists, the truthful status is still:

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
cloud                   = blocked
```
