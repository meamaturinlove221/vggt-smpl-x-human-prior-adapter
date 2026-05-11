# 2026-05-04 Mentor-Facing Closure Pack

## One-line truth

The current local repo has **no mentor-level passing normal-line candidate** and **no strict-passing dense head/face teacher**. Cloud upload/run remains blocked.

Latest registry:

```text
reports/20260504_strict_gate_registry.json
```

- candidates scanned: `104`
- teachers scanned: `84`
- strict candidate passes: `0`
- strict teacher passes: `0`
- schema: `20260504_visual_fullbody_hands_v2`

This means no current result can be presented as solved. Normal metrics, p40/fixed point counts, or teacher coverage are not accepted unless the Open3D review shows a normal human with modeled head/face/hairline and acceptable full-body/hands.

## Frozen route summary

### 1. Hard teacher search

What was tried:

- 60-view VGGT fused surfaces;
- TSDF/Poisson/visual-hull surfaces;
- COLMAP/MVS/depth/mesh-ray teachers;
- external mesh/depth/model candidates;
- keypoint/MediaPipe/face relief patches;
- aligned SMPL-X or hybrid teacher targets.

Why it was not a duplicate:

- the audits covered mesh, point cloud, depth/NPZ, visible-surface, and strict teacher-gate variants;
- the current registry scans `84` teacher packages and `47` visible-surface audits.

Strict gate result:

- strict teacher passes: `0`;
- teacher numeric pass but visual fail: `5`;
- other strict teacher non-passes: `79`.

Open3D failure shape:

- shell-like surfaces;
- incomplete head/face coverage;
- large face_core/head_face holes;
- floating patches;
- depth-compatible but not visually modeled surfaces.

Why not continue by tuning:

- a teacher that is not continuous/aligned/visual-pass cannot become valid by changing the training loss;
- old numeric/coverage green subsets are explicitly blocked by the current schema.

### 2. Kinect / TSDF / Poisson / A-pose

What was tried:

- raw Kinect target-frame TSDF and projection routes;
- Kinect coordinate audits;
- 60-view Kinect-derived surface attempts;
- same-subject A-pose Kinect TSDF asset diagnostics;
- Poisson/TSDF variants from existing VGGT/signfix geometry.

Why it was not a duplicate:

- target-frame Kinect, 60-view Kinect, and A-pose Kinect were separated;
- A-pose was treated only as an asset diagnostic, not directly as a target-frame teacher.

Strict gate result:

- strict teacher pass: `0`;
- raw sensor fullbody/hand passes: `0`;
- A-pose Kinect assets are diagnostic-only.

Open3D failure shape:

- target-frame projections are not aligned enough to original 6-view protocol;
- face/head/hairline coverage is too low;
- A-pose mesh is not in the target action frame.

Why not continue by tuning:

- scale/camera-axis sweeps do not solve pose mismatch or missing target-frame dense head/face surface;
- direct A-pose-to-target use would violate the same-frame teacher requirement.

### 3. SMPL-X body/hand weak anchor

What was tried:

- raw SMPL-X body/hand anchor audits;
- SMPL-X prior maps and weak conditioning;
- SMPL-X/hybrid teacher targets;
- weak body/hand training/candidate variants.

Why it was not a duplicate:

- SMPL-X was tested both as weak prior and as stronger teacher-like target;
- full-body/hands gates were added because mentor explicitly requires whole-body sanity and hand checks.

Strict gate result:

- SMPL-X weak-anchor audit can provide a body/hand topology signal;
- it does not produce a full mentor candidate pass.

Open3D failure shape:

- SMPL-X can stabilize coarse body/hand structure;
- it cannot produce realistic personal face/hair/clothing geometry;
- stronger SMPL-X use risks template-like faces and missing loose geometry.

Why not continue by tuning:

- mentor already warned that SMPL-X prior cannot be too strong;
- face/head/hairline must come from image/multiview evidence or a real dense surface, not SMPL-X template geometry.

### 4. Normal-depth-point coupling

What was tried:

- r16/r18/r19 self-geometry and point-branch variants;
- signed normal audits and gradient checks;
- normal-depth-point consistency summaries;
- depth/point/normal consistency losses.

Why it was not a duplicate:

- the work checked both metric response and whether gradients reach geometry heads;
- r16 showed measurable consistency changes before final point-cloud failure.

Strict gate result:

- local consistency metrics can improve;
- no full candidate strict pass was produced.

Open3D failure shape:

- normal consistency improves while face/head remains shell-like;
- full-body and hands still fail hard gate in later candidates.

Why not continue by adding epochs:

- the problem is not disconnected gradients or too few epochs;
- normal consistency alone is necessary but not sufficient for modeled human surface.

### 5. True-highres / crop / token-grid variants

What was tried:

- human crop / headshoulder / softmatte input variants;
- true-highres and target-view crop probes;
- token-grid and high-res local refinements.

Why it was not a duplicate:

- input-side resolution and foreground occupancy were tested separately from loss/model changes.

Strict gate result:

- input cleanup can improve local evidence and normal/visibility metrics;
- it did not produce strict candidate pass.

Open3D failure shape:

- camera-view closeups may look cleaner;
- free-view Open3D still shows shells, holes, or slab-like full-body geometry.

Why not continue by replacing protocol:

- same-protocol original 6-view headshoulder gate is mandatory;
- high-res crop cannot replace original protocol unless it also passes full/head/face/fullbody/hands gate.

### 6. Canonical shared points / shared surface

What was tried:

- r43 shared human point/surfel backend with SMPL-X canonical-bin correspondence;
- r47 graph-optimized visibility-aware shared surface with critical face/hand support gating.

Why it was not a duplicate:

- r43 tested canonical aggregation;
- r47 added graph optimization, surface smoothing, visibility/support gating, rasterization back to `predictions.npz`, and explicit surface debug outputs.

Strict gate result:

- r47 numeric gate: `False`;
- r47 normal gate: `True`;
- r47 shape gate: `False`;
- r47 fullbody/hands gate: `False`;
- r47 visual gate: `False`.

Key r47 numbers:

- world_points face p40 delta: `+5` over signfix;
- depth_unprojection face p40 delta: `+93` over signfix;
- required meaningful margin: `+500`;
- headshoulder face surfels: `949`;
- headshoulder hands surfels: `216`;
- fullbody hands raster changed fraction: `0.0298`.

Open3D failure shape:

- head/face remains projected shell geometry;
- full-body side/back/iso remains slab-like;
- hands are detached sheet fragments, not attached hand geometry.

Why not continue by tuning:

- r47 already tests the non-threshold version of this idea;
- changing bin size, support thresholds, alpha, smoothing, raster radius, or confidence percentile would only rearrange VGGT-observed shell geometry.

### 7. HART PnP replacement and HART-like surface probes

What was tried:

- HART-style pointmap-to-PnP camera ablation;
- HART-adjacent local oriented indicator field probes;
- Sapiens-normal oriented indicator probe;
- local HART PDF and asset review.

Why it was not a duplicate:

- PnP/camera replacement was tested separately from surface representation;
- HART was re-read as a learned backend, not just a camera trick.

Strict gate result:

- HART PnP did not fix face/head Open3D;
- local unlearned indicator/Sapiens-normal probes failed strict gates.

Open3D failure shape:

- oriented/implicit surfaces from current VGGT observations still form incomplete or shell-like geometry;
- Sapiens normals alone do not produce the learned residual completion HART uses.

Why not continue by name chasing:

- HART's core is trained residual normal + tightness/body-part heads + DPSR + 3D U-Net residual indicator completion;
- local repo has no HART code/weights for those modules;
- local `G:\权重\zju_vggt` checkpoints are 2D view-decoder/debug assets, not HART surface backend.

## Current blocker

The blocker is not missing scripts, missing reports, or disconnected losses. The blocker is:

```text
No local mechanism currently produces a continuous, aligned, non-shell, normal-human Open3D surface under the original sparse-view protocol.
```

The existing VGGT dense map outputs tend to preserve or reorganize a 2.5D shell. Normal metrics can improve, but the surface does not become a modeled human head/face/hair/fullbody/hands geometry.

## Unblocker options for mentor decision

### Option A: strict-passing target-frame dense teacher

Needed properties:

- same subject, same frame, target action frame;
- dense head/face/hairline/fullbody/hands surface;
- projectable to `0012_11_frame0000_6views_sparseproto_headshoulder_crop`;
- compatible with original 6-view intrinsics/extrinsics/crops;
- Open3D looks like a normal human;
- passes strict teacher gate before training.

Then and only then:

1. run one-frame local ROI overfit;
2. run full candidate gate with fullbody/hands;
3. consider cloud only after strict candidate pass.

### Option B: learned human surface backend

Needed properties:

- a model-level learned surface representation, not post-hoc thresholding;
- visibility-aware multiview aggregation;
- body-part/correspondence awareness without using SMPL-X as face/hair/clothing truth;
- residual normal/refinement module;
- surface-to-view rasterization back to depth/world_points/normal/confidence;
- local strict candidate gate as final judge.

This is a new module-development direction. It is not a small r-number continuation.

## Cloud policy

Cloud remains blocked until:

```text
strict_candidate_passes > 0
```

Teacher-supervised cloud work additionally requires:

```text
strict_teacher_passes > 0
```

No current package satisfies either condition.
