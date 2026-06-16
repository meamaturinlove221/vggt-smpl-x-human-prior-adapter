# Required Unblockers For Mentor Pass

Date: 2026-05-04

## Current State

Local strict gate status:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud upload/run: blocked by code guard

The current blocker is not a missing report. It is the absence of a local geometry source or mechanism that can produce a continuous, aligned, non-shell, normal-human Open3D result under the original sparse-view human protocol.

## What Cannot Be Used As The Unblocker

The following have already been closed as non-passing or diagnostic-only:

- r16/r18/r19/r57-r66 epoch extensions;
- r37/r40-r42 loss/ROI/confidence variants;
- HART-style PnP replacement by itself;
- TSDF/signfix shell teachers;
- direct 60-view VGGT fused surface or Poisson/visual-hull reconstructions;
- Kinect projection/patching without a strict teacher pass;
- external mesh/depth model name chasing;
- fixed-threshold or ROI-count-only positives;
- orphan `visible_surface_teacher_audit_summary.json` passes without current strict teacher gate and explicit Open3D visual approval.

## Acceptable Unblocker A: Strict-Passing Local Candidate

A teacherless/self-supervised route can proceed only if it locally produces a candidate package with:

- original 6-view headshoulder same-protocol pass;
- full-body candidate-specific NPZ, not inherited fullbody evidence;
- p40 and fixed-threshold full/head/face outputs;
- both `world_points` and `depth_unprojection`;
- normal-depth-point consistency not regressing on head/face;
- face/head/hairline shape metrics passing;
- full-body front/side/right-side/back/iso Open3D visual pass;
- hands attached and not reduced to detached sheets, phone fragments, or forearm-only blobs;
- explicit visual-review JSON with all required checks true.

Only after this strict candidate pass may a cloud run be considered.

## Acceptable Unblocker B: Strict-Passing Teacher For Teacher-Supervised Route

If the next route uses teacher supervision, the teacher must first pass:

```text
tools/audit_headface_teacher_surface.py
```

Minimum requirements:

- one shared 3D surface, not per-view 2.5D patches;
- projectable to `0012_11_frame0000_6views_sparseproto_headshoulder_crop`;
- face_core and head_face coverage without large holes;
- largest connected component not fragmented;
- depth residual compatible with the same protocol;
- explicit Open3D teacher visual review showing continuous head/face/hairline surface;
- no shell, floating patch, blue-hole center face, or template-only SMPL-X face.

Only after teacher pass is one-frame ROI overfit allowed. The overfit still must pass the full candidate gate, including full-body and hands.

## Acceptable Unblocker C: Restored Raw Dataset Asset

The raw dataset path is currently unavailable locally:

```text
G:\<Chinese raw dataset folder provided in earlier runs>
```

If the path becomes available, the first action is not training. The first action must be a read-only asset audit:

1. locate same-subject/same-frame 4K4D/DNA raw cameras, masks, RGB, Kinect/depth, and SMPL-X annotation;
2. verify camera convention and crop intrinsics against the existing original 6-view headshoulder protocol;
3. build a candidate 60-view or sensor-derived head/face surface only if it can be projected back to the original 6-view protocol;
4. run strict teacher gate with explicit Open3D visual review;
5. only if the teacher passes, run one-frame local overfit smoke;
6. only if the candidate passes full strict gate, consider cloud.

## Acceptable Unblocker D: New Surface Representation Mechanism

HART is useful as a design reference but not as a simple camera-head swap. A genuinely new mechanism would need to go beyond the existing VGGT per-view sheet outputs, for example:

- oriented point map plus normal fusion into a surface representation;
- explicit body-part/correspondence support that does not make SMPL-X a face/hair/clothing teacher;
- visibility-aware surface completion that preserves full-body and hands;
- local Open3D validation under the same full/head/face/full-body/hands gate.

This mechanism must first run locally. It must not be presented as mentor success unless the Open3D result looks like a normal human from free views, not only camera-view screenshots.

## Cloud Guard

Cloud entrypoints now block by default while the registry is red:

```text
modal_4k4d_vggt_train.py
modal_4k4d_vggt_infer.py
```

The guard requires a fresh registry and:

```text
strict_candidate_passes > 0
```

Teacher-supervised cloud work additionally requires a strict teacher pass. This prevents a stale, numeric-only, or legacy visible-surface pass from authorizing cloud compute.

