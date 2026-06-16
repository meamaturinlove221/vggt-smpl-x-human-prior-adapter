# Cloud Guard And No-Wall Status

Date: 2026-05-04

## Current Gate Truth

The latest strict registry is:

```text
reports/20260504_strict_gate_registry.json
```

Current counts:

- strict candidate passes: `0`
- strict teacher passes: `0`
- candidate gate packages scanned: `98`
- teacher gate packages scanned: `83`

Therefore cloud upload, cloud inference, and cloud training remain blocked. This is not a performance optimization decision; it is a truthfulness guard. The local Open3D/full-body/hand/head/face gate has not produced a normal human-looking 3D result.

## Code Guard Added

The Modal training and inference entrypoints now check the strict gate registry before any cloud upload or cloud run:

```text
modal_4k4d_vggt_train.py
modal_4k4d_vggt_infer.py
```

Blocked entrypoints:

- `upload_cases`
- `run_cases`
- `run_cases_from_local`
- `upload_scene`
- `run_scene`
- `run_scene_from_local`

Download-only entrypoints are intentionally not blocked because downloading existing artifacts does not create a new cloud run or upload a failed local state.

The guard requires both:

```text
strict_teacher_passes > 0
strict_candidate_passes > 0
```

In the current state this raises:

```text
Cloud <purpose> blocked by mentor strict gate: strict_candidate_passes=0, strict_teacher_passes=0.
```

An override exists only for an explicitly approved diagnostic:

```text
VGGT_ALLOW_CLOUD_WITHOUT_STRICT_PASS=1
```

This override must not be used for mentor-final claims or routine training.

## Verification

Syntax check passed:

```text
D:\anaconda\envs\g3splat\python.exe -m py_compile modal_4k4d_vggt_train.py modal_4k4d_vggt_infer.py
```

Behavior check used a fake local Modal module because the local environment does not install the real `modal` package. Both guards correctly blocked under the current red registry:

```text
Cloud unit-test blocked by mentor strict gate: strict_candidate_passes=0, strict_teacher_passes=0.
```

## HART Interpretation

The HART PDF supports the current decision not to replace VGGT's camera head blindly. HART does not become camera-free; it predicts viewpoint-invariant point maps, then estimates cameras from point maps with RANSAC/PnP because foreground-centered human crops violate the centered-principal-point assumption. It also relies on a larger system: oriented point maps, residual human normals, SMPL-X tightness/body-part heads, DPSR/indicator-grid refinement, and marching cubes.

Therefore a local `camera_head -> PnP` swap is not a mentor pass route by itself. The earlier HART-style PnP ablation remains a diagnostic/frozen negative replacement route unless a new full surface-representation mechanism is implemented and then passes the same full strict gate.

## Data Availability

The raw dataset path is currently unavailable locally:

```text
G:\方象鹿
```

`Test-Path` returns `False`. This blocks any truthful claim that a new internal raw 4K4D/DNA/Kinect teacher can be mined right now. Existing local teacher probes have already been audited by the strict registry and do not pass.

## No-Wall Decision

The work should not continue by repeating:

- r16/r18/r19/r57-r66 epoch extensions;
- HART-style PnP camera replacement;
- TSDF/signfix shell teachers;
- Kinect projection/patching without a new coordinate-plus-visual teacher pass;
- external mesh name chasing;
- confidence threshold, fixed-threshold, or ROI-count-only tricks.

Allowed local directions remain:

1. keep SMPL-X as a weak body/hand topology and real-data bridge, not as a face/hair/clothing teacher;
2. keep normal-depth-point coupling as the geometry principle, but only with a genuinely new point/surface mechanism rather than another loss-weight repeat;
3. keep human crop/matting as an input ablation, while preserving full-body and hands as hard gates;
4. require explicit Open3D review showing a normal human-looking full body, head, face, hairline, and attached hands before any pass or cloud upload.

## Current Blocker

The local repo still lacks a strict-gate-passing result. The blocker is not merely missing scripts; the blocker is missing a local geometry source or mechanism that produces continuous, aligned, non-shell, normal-human Open3D point clouds under the original 6-view headshoulder/full-body protocol.

