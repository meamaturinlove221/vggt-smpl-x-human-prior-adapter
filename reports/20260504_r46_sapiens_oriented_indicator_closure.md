# r46 Sapiens-Oriented Indicator Surface Closure

Date: 2026-05-04

## Current Truth

- Strict candidate passes: `0`
- Strict teacher passes: `0`
- Cloud upload/run: blocked by local guard
- Registry: `reports/20260504_strict_gate_registry.json`
- Registry generated at: `2026-05-04T05:24:53.523164+00:00`
- Candidate gates scanned after r46: `103`
- Teacher gates scanned after r46: `84`

r46 is not mentor-final and must not be uploaded to cloud.

## Why r46 Was Run

HART emphasizes that high-quality human normals, such as Sapiens normals, are
important for oriented point-map surface reconstruction. r45 had already shown
that an unlearned oriented indicator field built from VGGT/derived normals does
not pass. r46 tested one narrow non-cloud diagnostic question:

> If the same unlearned oriented field uses Sapiens normals as its normal base,
> does that alone turn current VGGT observations into mentor-level surface
> geometry?

This was not training, not teacher supervision, not HART/DPSR, and not a cloud
run.

## What r46 Tested

Script:

```text
tools/build_oriented_indicator_surface_predictions.py
```

New diagnostic options:

```text
--external-normal-npz output/detail_normal_refiner_20260425/external_sapiens_normal_teacher/original6v_headshoulder_sapiens03b/sapiens_normals.npz
--external-normal-transform flip-yz
--normal-source predicted
```

Outputs:

```text
output/normal_line_multiview_20260504/r46_sapiens_oriented_indicator_surface_diag_on6v_headshoulder/predictions.npz
output/normal_line_multiview_20260504/r46_sapiens_oriented_indicator_surface_diag_on6v_humancrop_fullbody/predictions.npz
output/normal_line_multiview_20260504/candidate_gate_r46_sapiens_oriented_indicator_surface_diag/report.md
output/normal_line_multiview_20260504/candidate_gate_r46_sapiens_oriented_indicator_surface_diag/candidate_gate_visual_sheet.png
```

The external Sapiens normal base had valid normal pixels per view:

```text
58698, 53675, 46434, 48040, 56178, 43996
```

## Strict Gate Result

r46 failed the full strict candidate gate.

### Numeric Gate

| Source | Ref face p40 | r46 face p40 | Delta | Ref fixed face | r46 fixed face | Face margin ok |
|---|---:|---:|---:|---:|---:|---|
| world_points | `16825` | `15847` | `-978` | `16825` | `15847` | `False` |
| depth_unprojection | `16764` | `15837` | `-927` | `17214` | `16273` | `False` |

Sapiens normals as the oriented-field base regressed same-protocol face counts
relative to signfix, so it is not even a numeric near-pass.

### Normal Gate

The normal gate failed for the same structural reason seen in r45: the generated
surface can make absolute angles look artificially good, while signed normal
polarity becomes incompatible with the reference protocol.

Examples:

| ROI/comparison | Candidate angle deg | Signed polarity ok | Candidate neg-frac |
|---|---:|---|---:|
| head_pred_vs_depth | `0.3652` | `False` | `0.0029` |
| head_pred_vs_point | `0.0001` | `False` | `0.0000` |
| face_pred_vs_depth | `0.2017` | `False` | `0.0030` |
| face_pred_vs_point | `0.0001` | `False` | `0.0000` |

The reference convention has predicted normals mostly opposite to derived
normals under the signed audit. r46 flips that distribution, so the abs-angle
numbers are not valid pass evidence.

### Shape, Full-Body, And Hands

Shape gate failed, including head/face fixed target-view coverage checks.

Full-body/hands also failed:

- per-view body coverage failed;
- hand gate failed for all p40/fixed, world/depth variants;
- only `1` of `2` eligible hand views passed hand kept-ratio;
- compact 3D hand boxes: `1`;
- implausible hand boxes: `1-2` depending on source/gate.

### Visual Review

Contact sheet:

```text
output/normal_line_multiview_20260504/candidate_gate_r46_sapiens_oriented_indicator_surface_diag/candidate_gate_visual_sheet.png
```

Direct visual read:

- head/face remain broken shells and patchy sheets, not a modeled face;
- full-body front/back contain large holes;
- side/right-side views are slab-like and ghosted;
- hands are detached floating fragments, not attached hand geometry.

The explicit visual review remains fail:

```text
output/normal_line_multiview_20260504/candidate_gate_r46_sapiens_oriented_indicator_surface_diag/visual_review_codex_fail.json
```

## Closure

r46 closes the narrow hypothesis that HART's Sapiens-normal idea can be reused
locally by only replacing the normal base inside an unlearned oriented field.
It cannot. Sapiens normals alone can orient a smoother/closed shell, but they do
not create the missing face/head/hairline surface, normal-human full body, or
attached hands.

Do not continue by changing only:

- Sapiens normal transform (`identity`, `flip-yz`, sign flips);
- grid resolution;
- truncation/radius;
- confidence percentile;
- field smoothing;
- marching-cubes threshold;
- visual-review wording.

Those would be another normal-base/field-parameter loop. If HART is used as a
technical reference, the missing piece is the full learned mechanism:

- per-pixel oriented point maps in a common frame;
- residual high-quality normal prediction, not 2D normal patching alone;
- body correspondence/tightness heads for human alignment;
- DPSR / indicator-grid construction;
- learned 3D grid refinement for occlusions;
- local strict gate over full/head/face/fullbody/hands before cloud.

The current repo does not contain HART code/weights or a ready DPSR/3D-UNet
indicator refinement stack. Until such a real implementation or a strict-passing
dense teacher asset exists, the truthful state remains no mentor-level pass.
