# 2026-05-07 Plan v4 Parallel Unblocker Status

## Current Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
formal cloud train/infer/export = blocked
teacher-supervised cloud route = blocked
research-preflight = allowed only for diagnostic artifacts
```

The strict registry was refreshed on 2026-05-07:

```text
schema_version = 20260504_visual_fullbody_hands_v2
strict_candidate_passes = 0
strict_teacher_passes = 0
registry_age_hours = 0.0 after refresh
```

No A/B/C/D lane wrote a pass, teacher export, candidate export, or formal cloud
run.

## A5 Same-Frame Dense Teacher Sprint

Goal:

```text
raw RGB + masks + known cameras + same frame
  -> dense human surface
  -> project back to original 6-view protocol
  -> strict teacher gate
```

### A5 v8 triangulated adj6

```text
views = 0,1,2,3,4,5
dense_mode = triangulated
COLMAP CUDA = true
sparse reconstruction = 6 images / 23 points
fused points = 2206
teacher gate = fail
```

Strict audit:

```text
face_core raw_sum=50 depth_compatible_sum=0
head_face raw_sum=100 depth_compatible_sum=0
hairline raw_sum=25 depth_compatible_sum=0
```

Decision: backend smoke only. It is not a teacher and not worth surface
extraction.

### A5 known-direct adj6

```text
views = 0,1,2,3,4,5
dense_mode = known_direct
fused points = 93692
teacher gate = fail
```

Strict audit:

```text
face_core raw_sum=2831 depth_compatible_sum=2
head_face raw_sum=10958 depth_compatible_sum=62
hairline raw_sum=2046 depth_compatible_sum=3
```

Decision: known-pose direct dense increases point count, but same-protocol
depth-compatible coverage is almost zero.

### A5 known-direct adj12

```text
views = 0..11
dense_mode = known_direct
fused points = 153062
teacher gate = fail
```

Strict audit:

```text
face_core raw_sum=6942 depth_compatible_sum=4
head_face raw_sum=16795 depth_compatible_sum=193
hairline raw_sum=6210 depth_compatible_sum=25
```

Decision: adjacent view count helps raw coverage but does not produce a
depth-compatible, continuous teacher surface.

### A5 known-direct hybrid12

```text
views = 0,1,13,15,24,25,26,28,36,45,57,59
dense_mode = known_direct
fused points = 101420
teacher gate = fail
```

Strict audit:

```text
face_core raw_sum=11699 depth_compatible_sum=47
head_face raw_sum=37760 depth_compatible_sum=1203
hairline raw_sum=6496 depth_compatible_sum=20
```

The hybrid view set can make raw ROI coverage look encouraging, but the
depth-compatible coverage remains tiny and fragmented. The best head-face
compatible render contains only 1203 points; face_core has 47 and hairline has
20. This is not a normal head/face/hairline surface.

### A5 Decision

The A5 COLMAP CUDA adapter is now functional. The blocker is not Modal, CUDA,
or DB/text identity. The blocker is teacher quality:

```text
known-camera COLMAP/MVS output is too sparse/misaligned after same-protocol
depth compatibility, especially in face_core and hairline.
```

Do not run adjacent-20/30 or threshold/fusion parameter loops as the next step.
The next non-redundant A-line action, if A5 continues, must be a different dense
backend class or a coordinate/depth-range audit, not more view-count gambling.

## B-Fus3D Skeleton

Agent B added:

```text
tools/b_fus3d_token_cache.py
reports/20260507_b_fus3d_token_cache_status.md
```

Dry-run cache:

```text
output/surface_research_preflight_local/B_Fus3D0_token_cache_dryrun/
```

Key result:

```text
VGGT token entrypoint = model.aggregator(images)
expected dry-run token shape = [1, 2, 1374, 2048]
local checkpoint candidate = C:\Users\WINDOWS\.cache\torch\hub\checkpoints\model.pt
status = metadata_only
```

The cache is safe: no cloud, no training, no prediction patching, no
depth/point/normal hard teacher use, no strict pass.

Next B-Fus3D action:

```text
run local --extract with explicit checkpoint on 2 views
then implement ROI-to-token coverage for face/hair/left-hand/right-hand/fullbody
```

## B-Hand Skeleton

Agent C added:

```text
tools/b_hand_evidence_cache.py
reports/20260507_b_hand_evidence_cache_status.md
```

Stable cache with predictions:

```text
output/surface_research_preflight_local/B_hand0_evidence_cache_6v_humancrop_softmatte_with_r34pred/
```

Key result:

```text
left hand: 5 visible ROI views, 4450 ROI pixels, camera rays available
right hand: 4 visible ROI views, 3061 ROI pixels, camera rays available
SMPL-X visible prior available for both hands
prediction support available for both hands
```

Remaining blockers:

```text
side-aware VGGT hand-token hook not implemented
side-specific SMPL-X hand anchor remains weak prior only
Open3D hand connected-to-arm visual gate not implemented/passed
```

Next B-hand action:

```text
local HGGT-style B0 skeleton with side tokens, wrist-arm anchor tokens,
hand ROI patch-token masks, weak SMPL-X hand prior, and Open3D hand precheck
```

## D-Line Guard

Agent D added:

```text
reports/20260507_dline_guard_status.md
```

Guard finding:

```text
formal cloud guard works
research-preflight summaries checked = 40
bad research flags = 0
no guard bug found
```

Registry was later refreshed:

```text
cloud_allowed = false
reason = strict_candidate_passes is 0
strict_teacher_passes = 0
```

## Current Plan

1. Freeze A5 COLMAP/CUDA decision smoke as teacher-negative unless a new dense
   backend class is introduced.
2. Promote B-Fus3D from metadata cache to local token extraction on 2 views.
3. Promote B-hand from evidence cache to local HGGT-style hand-token B0
   skeleton.
4. Keep D-line strict registry and cloud guard intact.

No mentor pass, no teacher pass, no candidate pass, and no formal cloud run has
been achieved.
