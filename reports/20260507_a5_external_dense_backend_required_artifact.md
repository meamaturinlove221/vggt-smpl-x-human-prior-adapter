# A5 External Dense Backend Required Artifact

Status: `blocked_waiting_for_external_backend_artifact`

This note freezes the local A5 COLMAP/MVS loop and defines the minimum external
dense-backend artifact needed before A5 can make non-redundant progress.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher export = blocked
candidate export = blocked
```

## Local Backend Inventory

The local `external_models/` directory currently contains only:

```text
face_landmarker.task
hand_landmarker.task
```

Those are MediaPipe landmark assets. They are not OpenMVS, 3DGS, 2DGS, SuGaR,
NeuS, Instant-NGP, PatchMatchNet, CasMVSNet, or any other dense shared-surface
backend.

No local runnable entrypoint was found for:

```text
InterfaceCOLMAP
DensifyPointCloud
ReconstructMesh
TextureMesh
instant-ngp
OpenMVS
2DGS / SuGaR / 3DGS
NeuS / NeuS2 / VolSDF
PatchMatchNet / CasMVSNet
```

## Why Local A5 COLMAP Is Frozen

The existing A5 known-camera COLMAP research-preflight is not empty, but it is
teacher-negative under the strict protocol:

```text
known_direct hybrid12 fused points = 101420
face_core raw_hit_pixels = 11699, depth_compatible = 47
head_face raw_hit_pixels = 37760, depth_compatible = 1203
hairline raw_hit_pixels = 6496, depth_compatible = 20
```

The depth-range audit shows PatchMatch depth maps have valid pixels in
face/head/hairline ROIs, but the fused shared point cloud becomes spatially
incompatible with the original 6-view strict protocol. Therefore additional
COLMAP view-count/source-pair/fusion-threshold attempts are now considered a
repeat loop.

## Required External Artifact

A5 can resume only when an external backend provides one of the following
same-frame, calibrated artifacts:

```text
Option A: one shared dense human surface mesh
  format: .ply or .obj
  coordinate: same calibrated world or documented transform into the A5 scene world
  content: full body, head, face, hairline, hands

Option B: one consistent multi-view depth set
  format: per-view depth maps plus camera/view mapping
  coordinate: same calibrated camera set
  condition: depths must be mutually consistent, not independent per-view shells
```

The artifact must come from the same subject/frame and known-camera image/mask
set. It cannot be a visual hull shell, SMPL-X template, landmark patch, VGGT
depth/point recycling, or per-view unaligned depth hallucination.

## Adapter Entry Contract

Once such an artifact exists, import it through:

```powershell
python tools\a5_external_dense_backend_preflight.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --output-dir output\surface_research_preflight_local\A5_external_dense_import_<backend>_<date> `
  --backend custom `
  --input-mesh <external_shared_mesh.ply> `
  --view-indices 0,10,24,36,45,57 `
  --eval-view-indices 0,30 `
  --target-size 96 `
  --overwrite
```

If the backend produces depth maps instead of a mesh, use the adapter depth
import mode for that backend workspace. The adapter remains research-only and
must not write `predictions.npz`, teacher export, candidate export, registry
state, or a strict pass.

## Forbidden Loops

```text
no adjacent-20/30 view-count loop
no PatchMatch source-pair loop
no stereo_fusion threshold loop
no visual-hull-only success claim
no SMPL-X/template/landmark scaffold as dense teacher
no Python/PowerShell/cmd wrapper around backend-command to bypass guard
no formal Modal train/infer/export
no strict registry writes
```

## Decision

A5 is blocked on a genuinely new dense-backend artifact. Until such an artifact
is available, the active work should remain on B-Fus3D/B-hand learned backend
design and D-line strict guard/reporting, not on more local COLMAP tuning.
