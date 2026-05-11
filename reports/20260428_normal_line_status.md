# 2026-04-28 normal line status

## Truthful status

The normal line is active and partially validated, but it is not a mentor-final pass yet.

Current evidence supports this narrow claim:

- VGGT now has a coupled depth / point / normal supervision path instead of treating the normal map as a standalone visualization target.
- The r2 depth-normal-depthpoint run improves normal-depth-point consistency metrics on the same 6-view headshoulder protocol.
- The improvement does not yet convert into a clearly better same-protocol Open3D face/head point cloud.
- The r3 stop-gradient attempt improves some coupling metrics but regresses same-protocol face ROI and does not produce a cleaner targetcam30 face/head visualization.

Therefore, the correct current wording is:

> The normal-depth-point coupling direction is implemented and measurable, but the sparse-view face/head point cloud is still not mentor-final quality.

## Mentor requirements tracked here

- Normal is not only a pretty normal map; it must couple back to depth and point geometry.
- Depth-derived normal and point-derived normal must be compared against the predicted normal.
- Head / face / hairline are the primary quality target, but full-body point cloud cannot have obvious holes, broken limbs, or severe missing structure.
- Human crop is treated as an input preprocessing base, not as the main structural innovation.
- SMPL-X prior is useful but must not dominate personal geometry; template-like artifacts remain a risk.
- Multi-view deployment must be checked instead of assuming more views always help.

## Implemented normal coupling

Code/config changes:

- `training/loss.py`
  - Adds depth-to-normal consistency.
  - Adds depth-to-point consistency.
  - Adds point-to-normal consistency with optional normal stop-gradient.
  - Adds helpers for depth-to-camera point maps and world-to-camera point maps.
- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r2_depthpoint.yaml`
  - Enables depth-normal and depth-point losses with face/head/hairline/boundary weighting.
- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r3_normalstopgrad.yaml`
  - Keeps r2 coupling but treats the predicted normal as a stop-gradient target for point/depth consistency.
- `tools/normal_line_multiview_eval.py`
  - Computes pred-normal vs depth-normal, pred-normal vs point-normal, and depth-normal vs point-normal metrics for full/head/face ROI.

Validation performed:

- `python -m py_compile training\loss.py`
- tensor smoke test for stop-gradient behavior
- cloud training and same-protocol inference for r2 and r3
- Open3D full/head/face ROI rendering

## Same-protocol 6-view headshoulder result

Reference scene:

- `output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop`

Reference baseline:

- `signfix ckpt4`
- face ROI: `16825`
- head ROI: `40527`
- full ROI: `184213`
- conf p40: `38.5067`

r2 depth-normal-depthpoint:

- ckpt0 face ROI: `16800`
- ckpt1 face ROI: `16520`
- ckpt2 / inference face ROI: `16548`
- conclusion: consistency improves, but same-protocol face ROI does not beat signfix and Open3D face/head quality is not clearly better.

r3 normal-stopgrad:

- r3-from-signfix ckpt1 face ROI: `16624`
- r3-from-r2ckpt2 ckpt1 face ROI: `16596`
- conclusion: not a candidate for mentor-final delivery.

## Normal consistency evidence

Same-protocol 6-view headshoulder face ROI:

| entry | pred vs depth mean angle | pred vs point mean angle | depth vs point mean angle |
|---|---:|---:|---:|
| signfix ckpt4 | 12.6836 | 8.3753 | 8.0617 |
| r2 ckpt0 | 8.6187 | 6.2342 | 6.8938 |
| r2 ckpt2 | 7.6985 | 4.3526 | 6.4768 |
| r3 from signfix | 9.0624 | 5.2943 | 7.5306 |
| r3 from r2ckpt2 | 7.3152 | 4.5289 | 6.2328 |

Targetcam30 6-view headface face ROI:

| entry | pred vs depth mean angle | pred vs point mean angle | depth vs point mean angle |
|---|---:|---:|---:|
| signfix ckpt4 | 11.1447 | 9.8765 | 5.6897 |
| r2 ckpt2 | 8.8427 | 8.2036 | 4.7534 |
| r3 from signfix | 8.9725 | 8.5866 | 4.4790 |
| r3 from r2ckpt2 | 8.5398 | 7.9676 | 4.4586 |

Metric conclusion:

- r2/r3 make the predicted normal more consistent with depth/point local geometry.
- Metric gains alone are insufficient because the Open3D face/head geometry remains torn or template-like.

## Open3D evidence

Review sheets:

- `output\normal_line_multiview_20260428\review_sheets\r3_stopgrad_targetcam30_face_head_sheet_v2.png`
- `output\normal_line_multiview_20260428\review_sheets\r3_stopgrad_targetcam30_side_sheet.png`
- `output\normal_line_multiview_20260428\review_sheets\r2_targetcam30_fullbody_front_sheet_v2.png`
- `output\normal_line_multiview_20260428\review_sheets\r2_targetcam30_fullbody_side_sheet.png`

Full-body point cloud package for mentor inspection:

- `output\normal_line_delivery_20260428\targetcam30_fullbody_r2ckpt2.zip`
- includes `r2ckpt2_targetcam30_6v_fullbody_depthunproj_open3d.ply`
- includes front/side/head/face renders and comparison sheets

Visual conclusion:

- r2 targetcam30 full-body point cloud is better for inspection than the old signfix front-view render.
- r2/r3 targetcam30 face can look more face-like from a favorable front crop, but side view still shows depth tearing and the face/head surface is not clean enough.
- r3 stop-gradient is not a breakthrough.

## Multi-view status

Already evaluated:

- r2 ckpt0 humancrop `3v / 6v / 16v / 60v`
- r2 targetcam30 `6v`
- r3 targetcam30 `6v`

Known r2 humancrop Open3D point counts:

| views | full | head | face |
|---:|---:|---:|---:|
| 3 | 57041 | 12549 | 5500 |
| 6 | 111078 | 24437 | 9656 |
| 16 | 273512 | 60173 | 23227 |
| 60 | 800000 capped | 176000 | 63088 |

Multi-view conclusion:

- More views increase point coverage and full-body stability.
- More views do not automatically solve face/head surface clarity.
- A 60-view targetcam30 r2 ckpt2 run is being used to separate view-count bottleneck from normal-coupling bottleneck.

## Current blocker

The blocker is no longer the existence of the normal branch or evaluation scripts.

The blocker is that the coupled normal signal has not yet forced a sufficiently clean target-view head/face surface in sparse-view reconstruction.

Likely next technical directions inside the normal line:

1. Keep depth-normal-point coupling, but use a stronger local ROI objective focused on visible head/face pixels rather than global consistency alone.
2. Compare 6v vs 60v targetcam30 with the same r2 checkpoint to test whether the current failure is view-count limited.
3. If 60v is still weak, the normal teacher / normal target is not strong enough and must be replaced or refined before more sparse-view training.
4. If 60v is clearly better, distill only the local head/face normal-depth relationship back to sparse-view, not the whole point cloud.

## Do not claim

Do not claim:

- “normal line has passed”
- “r3 is better”
- “face/head is mentor-final”
- “higher point count alone is success”
- “front-only crop proves geometry quality”

Acceptable current claim:

> The normal line now provides measurable geometry coupling and a full-body point cloud package for inspection, but sparse-view face/head quality still needs a stronger local normal-depth objective or teacher before it can be called complete.

