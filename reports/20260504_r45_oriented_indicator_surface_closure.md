# r45 Oriented Indicator Surface Closure

Date: 2026-05-04

## Current Truth

- Strict candidate passes: `0`
- Strict teacher passes: `0`
- Cloud upload/run: blocked by local guard
- Registry: `reports/20260504_strict_gate_registry.json`
- Registry generated at: `2026-05-04T05:01:08.748196+00:00`
- Candidate gates scanned after r45: `102`
- Teacher gates scanned after r45: `84`

r45 is not mentor-final and must not be uploaded to cloud. It is a local
diagnostic probe of whether an unlearned oriented implicit/indicator surface can
turn the current VGGT observations into a continuous human surface.

## What r45 Tested

Script:

```text
tools/build_oriented_indicator_surface_predictions.py
```

Inputs:

```text
output/modal_results/20260424_signfix_ckpt4_on6v_headshoulder/predictions.npz
output/modal_results/20260427_humancrop6v_ckpt0_on6v_humancrop_fullbody/predictions.npz
```

Outputs:

```text
output/normal_line_multiview_20260504/r45_oriented_indicator_surface_diag_on6v_headshoulder/predictions.npz
output/normal_line_multiview_20260504/r45_oriented_indicator_surface_diag_on6v_humancrop_fullbody/predictions.npz
output/normal_line_multiview_20260504/candidate_gate_r45_oriented_indicator_surface_diag/report.md
output/normal_line_multiview_20260504/candidate_gate_r45_oriented_indicator_surface_diag/candidate_gate_visual_sheet.png
```

The probe used current VGGT `world_points` / `depth_unprojection` observations,
derived/predicted/mixed normals, a nearest-oriented field, marching cubes, and
raycasting back to the original 6-view protocol. It did not use a new teacher,
training target, cloud compute, HART/DPSR code, or external dense surface.

## Numeric Gate Result

Strict candidate gate report:

```text
output/normal_line_multiview_20260504/candidate_gate_r45_oriented_indicator_surface_diag/report.md
```

r45 failed the same-protocol headshoulder numeric gate:

| Source | Ref face p40 | r45 face p40 | Delta | Ref fixed face | r45 fixed face | Face margin ok |
|---|---:|---:|---:|---:|---:|---|
| world_points | `16825` | `16452` | `-373` | `16825` | `16452` | `False` |
| depth_unprojection | `16764` | `16279` | `-485` | `17214` | `16761` | `False` |

The fixed-threshold face counts also remain below the reference, so this is not
a p40-only filtering artifact.

## Normal Gate Result

r45 also failed normal consistency because the raycasted field created a
self-consistent but convention-mismatched normal polarity relative to the
reference protocol.

Examples:

| ROI/comparison | Angle delta deg | Signed polarity ok | Neg-frac delta | OK |
|---|---:|---|---:|---|
| head_pred_vs_depth | `-10.8991` | `False` | `0.9927` | `False` |
| head_pred_vs_point | `-8.8711` | `False` | `1.0000` | `False` |
| face_pred_vs_depth | `-11.8677` | `False` | `0.9903` | `False` |
| face_pred_vs_point | `-8.3753` | `False` | `1.0000` | `False` |

This is a useful diagnostic: a continuous field can make absolute normal angles
look better, but signed normal convention and final Open3D geometry still fail.
Therefore r45 cannot be counted as normal-depth-point success.

## Shape, Full-Body, And Hand Gate Result

Shape gate failed:

```text
head_world_points_fixed
head_depth_unprojection_fixed
face_world_points_fixed
face_depth_unprojection_p40
face_depth_unprojection_fixed
```

Full-body/hands gate failed even with candidate-specific fullbody input:

- full-body front/side/right/back/iso visual checks: failed;
- body proportions: failed;
- hands attached/not missing/not scattered: failed;
- hand eligible views: `2`;
- passing hand kept-ratio views: `1`;
- compact 3D hand boxes: `1`;
- implausible hand boxes: `1`.

The contact sheet confirms the numeric failure. Head/face views remain broken
shells with holes, not a modeled face. Full-body side/back/iso views contain
sheet-like floating structures and the hands are detached fragments rather than
attached hand geometry.

## Visual Review Result

Explicit visual review file:

```text
output/normal_line_multiview_20260504/candidate_gate_r45_oriented_indicator_surface_diag/visual_review_codex_fail.json
```

It is intentionally marked fail. The Open3D sheet does not look like a normal
human under the mentor criteria:

- face/head still lack a continuous facial surface;
- hairline/head are broken from free views;
- full body contains ghost sheets and slab-like side views;
- hands are fragmented and detached.

## Closure

r45 closes the local hypothesis that an unlearned nearest-normal oriented
indicator field can rescue the current VGGT observations. It can close or
raycast the existing shell, but it does not create the missing face/head/hairline
geometry or valid full-body/hand structure.

Do not continue by changing only:

- grid resolution;
- truncation/radius;
- marching-cubes threshold;
- normal source (`pred`, `derived`, `mixed`);
- confidence percentile;
- raycast distance;
- mesh smoothing or component size.

Those would be another support/threshold/smoothing loop around the same failed
surface. A future surface-representation route must be genuinely new: learned
or explicitly multi-view optimized, visibility-aware, and able to pass the same
full candidate gate locally before cloud.

## Remaining Honest Unblockers

Only these remain consistent with the mentor constraints:

1. A true learned or optimized continuous human surface/field backend that is
   more than unlearned TSDF/Poisson/surfel/oriented-field aggregation, and that
   locally passes full/head/face/full-body/hands strict gate.
2. A new same-subject, same-frame dense head/face/hairline asset that can be
   projected into the original 6-view headshoulder protocol and passes strict
   teacher gate before any teacher-supervised training.
3. A Kinect coordinate/asset fix only if it first produces a strict teacher gate
   pass with explicit Open3D visual approval. No Kinect patching or training is
   allowed before that.

Until one of these exists, the truthful state remains: no mentor-level pass and
no cloud upload/run.
