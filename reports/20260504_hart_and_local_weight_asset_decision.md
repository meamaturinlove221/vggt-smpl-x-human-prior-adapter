# 2026-05-04 HART And Local Weight Asset Decision

## Current truth

The local registry remains red:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud: blocked by local guard

This note records the decision after reading the local HART PDF snippets, checking the repo, and scanning local dataset/weight assets. It is meant to prevent another loop of HART-name, external-model-name, or view-decoder-name chasing.

## What HART actually contributes

HART should not be reduced to PnP. The useful geometry part is a learned human surface backend:

1. VGGT-like transformer with human-specific dense heads:
   - point map + confidence;
   - residual normal head;
   - SMPL-X tightness direction/magnitude;
   - body-part label/confidence.
2. Residual normals:
   - HART does not just paste Sapiens normals;
   - it learns `normal = normalize(Sapiens_normal + learned_residual)`.
3. Oriented point maps into DPSR:
   - point maps and world normals form oriented points;
   - DPSR produces an initial indicator grid.
4. Learned 3D indicator-grid residual:
   - a 3D U-Net predicts a residual grid over the DPSR output;
   - this learned completion handles self-occlusion and missing human surface regions.
5. SMPL-X correspondence:
   - tightness vectors and body-part labels fit/align SMPL-X;
   - SMPL-X is body correspondence and structure, not face/hair/clothing truth.

This is materially different from the local routes already closed:

- r43 shared surfel/canonical-bin aggregation;
- r45 unlearned oriented indicator field;
- r46 Sapiens-normal oriented indicator;
- r47 graph-optimized visibility-aware shared surface;
- TSDF/Poisson/visual-hull/Kinect patching.

Those local routes lack the learned residual normal, tightness/body-label heads, and learned 3D indicator-grid residual.

## Local HART availability check

Local HART PDF:

```text
C:\Users\WINDOWS\WPSDrive\681698644\WPS云盘\硕士\主项目\参考论文\HART_Human Aligned Reconstruction Transformer.pdf
```

Existing local snippets:

```text
output/normal_line_multiview_20260504/hart_relevant_snippets.txt
```

Local repo / external repo status:

- `__tmp_external_repos\LHM` exists, but it is LHM code/wrappers, not HART.
- Modal/Sapiens/MediaPipe wrappers exist, but they are not HART learned surface backend weights.
- No local HART code, HART checkpoint, DPSR+3D-UNet checkpoint, HART tightness/body-part heads, or HART residual-normal head was found.

Web check as of 2026-05-04 also indicates HART code/models are not locally available here. Therefore HART cannot be invoked as a local candidate or used to authorize cloud work.

## Local raw 4K4D asset check

Raw dataset root:

```text
G:\数据集\datasets\data_used_in_4K4D
```

Available categories:

- `main/*.smc`
- `kinect/*.smc`
- `annotations/*.smc`
- `apose/apose_main/*.smc`
- `apose/apose_kinect/*.smc`
- `data_used_in_4K4D_rgb_cams.zip`

Extension inventory under this root:

- `.smc`: `38`
- `.mp4`: `8`
- `.json`: `2`
- `.txt`: `1`
- `.zip`: `1`

No ready `.ply`, `.obj`, `.off`, `.npz`, or strict-passing same-frame dense clothed surface was found under the raw 4K4D root. The raw data is useful input, but it is not itself a target-frame head/face/fullbody teacher.

Existing repo tools already cover the obvious raw routes:

- Kinect projection/depth teacher tools;
- Kinect TSDF tools;
- visual hull probes;
- SMPL-X raw body/hand anchor tools.

Those routes have been strict-gated or closed as non-passing. The blocker is not simply that the raw root was missing.

## Local weight asset check

Local weight root:

```text
G:\权重\zju_vggt
```

Observed assets:

- many `viewdec_*` checkpoints;
- `GeomViewDecoder_*`;
- `SimpleViewDecoder`;
- debug images for view decoder / visual tests.

Decision:

- These are not HART weights.
- They do not provide a learned 3D indicator-grid residual, tightness/body-part heads, or clothed human mesh backend.
- They may be separately audited as view-decoder assets, but they cannot be treated as a surface-backend unblocker or cloud authorization while the strict registry is red.

## What can still be done locally

Allowed local preflight only:

1. HART contract audit:
   - verify whether current VGGT predictions provide sufficiently reliable oriented points and normals.
   - Existing gate packages already show normal consistency can improve without visual pass.
2. External mesh teacher audit:
   - if a new mesh appears, run `tools/audit_visible_surface_teacher.py` or `tools/audit_headface_teacher_surface.py` before any training or fusion.
3. Raw target-frame surface construction:
   - only useful if it creates a strict-passing target-frame dense surface;
   - must pass teacher gate before one-frame overfit.
4. Learned backend implementation:
   - only meaningful if new supervision/weights/data are available for residual normal, body/tightness heads, and 3D indicator-grid residual completion.

## What must not be done next

Do not proceed with:

- HART-style PnP replacement as a geometry solution;
- HART-name reruns without HART weights or learned backend;
- Sapiens-normal-only or Poisson/DPSR-only reruns;
- r47 parameter tuning;
- view-decoder checkpoint chasing as if it were a 3D human surface backend;
- cloud upload/run while `strict_candidate_passes=0`.

## Decision

Current local assets do not contain a ready HART-quality learned surface backend or a strict-passing target-frame dense teacher. Therefore the project remains blocked for cloud.

The only non-wall next actions are:

- acquire or build a real learned human surface backend with appropriate supervision;
- produce a strict-passing target-frame dense teacher from raw data;
- or obtain actual HART/LHM-style mesh outputs/weights and run them through strict teacher/candidate gates before any training.
