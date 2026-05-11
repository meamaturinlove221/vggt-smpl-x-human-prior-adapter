# A5 Next Unblocker Decision

Status: `blocked_until_external_dense_artifact`

This note is read-only/small-edit guidance for the A5 dense-teacher lane. It
does not run reconstruction, tune COLMAP, export a teacher/candidate, write
predictions, write strict pass state, or change cloud guard behavior.

## Current Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
```

Local checks confirm:

```text
external_models/ = face_landmarker.task, hand_landmarker.task only
PATH dense backends = no colmap, OpenMVS, instant-ngp, ns-train, gsplat, NeuS
A5 external adapter summaries = blocked_no_backend_output
A5 hybrid12 known-direct fused_points = 101420, strict teacher fail
```

The existing A5 COLMAP/CUDA research smokes are already negative under the
strict same-protocol teacher gate. The failure pattern is not missing raw
PatchMatch pixels; it is incompatible/fragmented shared surface support:

```text
face_core depth-compatible = 47
head_face depth-compatible = 1203
hairline depth-compatible = 20
```

## Allowed

The only A5 action worth executing next is a bounded artifact-intake preflight,
and only after a new artifact is present on disk:

```text
run tools/a5_external_dense_backend_preflight.py on exactly one externally
produced same-frame dense shared mesh or calibrated consistent depth set, then
run the existing strict Open3D full/head/face/hairline/hands teacher review.
```

## Blocked

Until that artifact exists, A5 has no non-redundant local execution step.

```text
blocked: more COLMAP view-count/source-pair/fusion-threshold attempts
blocked: known-direct/adjacent/hybrid view gambling
blocked: visual hull, SMPL-X/template, landmark, LHM, VGGT depth/point recycling
blocked: strict pass writes, teacher export, candidate export
blocked: formal cloud train/infer/export
```

## Required Artifact

Provide exactly one of:

```text
Option A: shared human surface mesh
  format: .ply or .obj
  frame: same subject/frame as 0012_11 frame0000
  coordinate: A5 calibrated world, or documented transform into it
  content: full body, head, face, hairline, hands

Option B: consistent multi-view depth set
  format: per-view depths plus camera/view mapping
  frame: same subject/frame and calibrated 4K4D cameras
  condition: mutually consistent shared surface, not independent per-view shells
```

Decision:

```text
A5 next bounded action = wait for/import one real external dense artifact.
If no such artifact is available, A5 is explicitly blocked; move effort to
B-Fus3D/B-hand/D-line rather than repeating COLMAP parameter searches.
```
