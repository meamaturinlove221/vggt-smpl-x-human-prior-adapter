# 2026-05-04 Learned Human Surface Backend v0 Design

## Purpose

This design is the next non-wall method direction after r47. It is not a claim of success and not an authorization for cloud. The goal is to define what a real learned human surface backend would need to do, so future work does not fall back into threshold/support/teacher-name loops.

Current status:

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
cloud                   = blocked
```

## Design principle

The backend must go beyond per-view dense map patching.

Current failed pattern:

```text
per-view depth / point / normal
  -> local smoothing / confidence / shared-bin fusion
  -> rasterize back
  -> Open3D still shell-like
```

Required pattern:

```text
VGGT multiview evidence
  -> learned human surface representation
  -> visibility/body-part-aware refinement
  -> surface-to-view rasterization
  -> strict full/head/face/fullbody/hands Open3D gate
```

The key word is **learned**. A hand-written graph, Poisson, TSDF, indicator field, or confidence rule can be useful for diagnostics, but it has already failed as the final mechanism.

## Inputs

The v0 backend should consume:

- RGB images after human crop / matting;
- VGGT tokens or decoder features;
- VGGT `world_points`;
- VGGT `depth`;
- VGGT `normal`;
- confidence maps:
  - `world_points_conf`;
  - `depth_conf`;
  - `normal_conf`;
- human mask / matting mask;
- SMPL-X prior maps as weak structure:
  - canonical coordinates;
  - body part / marker embedding;
  - visibility mask;
  - posed coarse body surface;
  - optional hands/body topology channels.

SMPL-X is not an input face/hair/clothing teacher. Its role is correspondence, visibility, and coarse body/hand topology.

## Intermediate representation

Use shared human surface tokens rather than per-view pixels.

Each surface token should carry:

- canonical coordinate or body-part bin;
- current 3D position;
- current normal;
- feature vector from multiview VGGT tokens;
- view support count;
- source-view visibility distribution;
- body part label or soft part probability;
- confidence / uncertainty;
- local surface scale;
- optional residual offset from coarse SMPL-X body.

Important: tokens for face, hairline, hands, loose clothing, and torso should not be treated with the same prior strength. SMPL-X can be medium for torso/limbs, weaker for hands, low for face, and near-zero for hair/clothing detail.

## Modules

### 1. Visibility-aware multiview aggregation

Purpose:

- collect evidence from all views into each surface token;
- avoid false confidence from a single projected sheet;
- track support explicitly for face/hairline/hands.

Inputs:

- VGGT per-view features;
- `world_points`, `depth`, `normal`;
- camera/extrinsic/intrinsic;
- SMPL-X canonical/part maps;
- mask/matting.

Outputs:

- token feature;
- view-support vector;
- visibility logits;
- uncertainty.

Must avoid:

- treating low-support face/hand tokens as high-confidence pass;
- using SMPL-X template surface as final face/hair geometry.

### 2. Body-part-aware surface refinement

Purpose:

- refine tokens using part-specific priors and local neighborhoods;
- preserve torso/limb/full-body continuity without template-locking the face.

Suggested behavior:

- torso/limbs: stronger smoothness and SMPL-X topology anchor;
- hands: weak-to-medium topology anchor plus strict attachment/support checks;
- face/head: image/multiview evidence dominates; low SMPL-X template weight;
- hairline/hair/clothing: no hard SMPL-X surface target.

Outputs:

- refined token positions;
- refined token normals;
- token confidence.

### 3. Residual normal head

Purpose:

- avoid standalone normal maps that look good but do not shape geometry;
- learn a residual correction tied to the shared surface.

Possible formulation:

```text
N_surface = normalize(N_base + N_residual)
```

Where `N_base` can come from:

- VGGT normal;
- depth-derived normal;
- point-derived normal;
- optional human normal estimator as a weak base.

The residual must be trained/evaluated together with surface geometry. Sapiens or any monocular normal is not enough by itself.

### 4. Learned surface completion / refinement block

Purpose:

- fill sparse/self-occluded human regions in a learned way;
- avoid Poisson/TSDF-only shell completion.

Possible variants:

- token graph transformer over surface tokens;
- sparse 3D U-Net over a human-aligned grid;
- low-resolution indicator grid residual like HART;
- hybrid surfel-to-grid module.

Required property:

- the completion must be trained with real surface supervision or a strict-passing teacher;
- otherwise it will reduce to r45/r46/r47-style unlearned shell rearrangement.

### 5. Surface-to-view rasterizer

Purpose:

- output the same protocol that the current gate already understands.

Outputs:

- `depth`;
- `world_points`;
- `normal`;
- `depth_conf`;
- `world_points_conf`;
- `normal_conf`;
- optional support maps.

The rasterizer must preserve:

- same 518x518 protocol;
- same camera conventions;
- both `world_points` and `depth_unprojection` paths;
- p40 and fixed-threshold evaluation.

## Training / supervision requirements

This backend cannot be trained honestly without at least one of:

1. strict-passing target-frame dense teacher;
2. dataset-level dense clothed human meshes;
3. real HART-like pretrained backend weights;
4. a synthetic scan dataset with SMPL-X alignment, body-part labels, tightness vectors, and indicator-grid supervision.

If none are available, only local preflight and architecture scaffolding are allowed. Cloud training is blocked.

Minimum supervision targets for a HART-like route:

- oriented point maps;
- reliable human normals or residual-normal targets;
- body-part labels / marker labels;
- tightness vectors or equivalent body correspondence;
- clothed mesh / indicator grid target for surface completion;
- visibility/mask targets.

## Evaluation gate

The v0 backend must use the existing strict candidate gate without bypass:

```text
tools/package_normal_candidate_gate.py
```

Required outputs:

- same-protocol 6-view headshoulder candidate NPZ;
- candidate-specific fullbody NPZ;
- full/head/face/hands Open3D renders;
- p40 and fixed-threshold metrics;
- normal-depth-point consistency;
- shape metrics;
- explicit visual-review JSON.

Pass only if all are true:

- face/head/hairline Open3D shows modeled geometry;
- full-body front/side/right/back/iso looks like a normal human;
- no large body holes, broken limbs, severe ghost slabs, or ghost layers;
- hands are attached, compact, and not detached sheets or scattered noise;
- both `world_points` and `depth_unprojection` are acceptable;
- no numeric-only or normal-only pseudo-pass.

## Minimal local preflight before implementation

Before coding a large model:

1. Confirm the input contract:
   - current VGGT predictions can provide oriented points/normals;
   - normal sign and camera/world conventions are explicit.
2. Confirm available supervision:
   - if no dense teacher/scan/HART weights exist, do not start cloud training.
3. Write a small module spec:
   - token schema;
   - rasterizer interface;
   - gate output format.
4. Run any generated candidate locally:
   - no cloud;
   - no pass claim;
   - full strict gate only.

## Stop conditions

Immediately stop and freeze the v0 attempt if:

- it only changes confidence, thresholds, support, smoothing, or raster radius;
- it lacks learned surface completion or real dense supervision;
- it improves normal metrics but Open3D remains shell-like;
- hands remain detached sheets;
- fullbody side/back/iso views remain slab-like;
- the full candidate gate fails.

## Decision

This design is the only method route still compatible with the mentor's guidance if no strict-passing dense teacher is provided. It must be treated as a new research module, not as another r-number patch.

No cloud work is allowed until local strict gate passes.
