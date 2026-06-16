# 2026-04-28 normal line r18/r19 local/cloud gate report

## Verdict

Not mentor-final.

I stopped further cloud iteration because the local and same-protocol evidence now shows the current 6-view target/teacher geometry is not good enough to drive a passing result. Continuing by adding more epochs or larger weights would violate the current rule: only go cloud after the local problem is actually fixed.

## Reference gate

Same-protocol original 6-view headshoulder signfix reference:

| Entry | Full | Head | Face | p40 threshold |
|---|---:|---:|---:|---:|
| signfix ckpt4 | 184213 | 40527 | 16825 | 38.5067 |

Pass still requires a meaningful face ROI gain over 16825 plus Open3D face/head visual improvement.

## r16 frozen as negative

r16 cross-view self-geometry was already evaluated before r18/r19:

| Entry | Full | Head | Face | p40 threshold |
|---|---:|---:|---:|---:|
| r16 xview selfgeom | 184213 | 40527 | 14981 | 58.3877 |

Why negative:

- face ROI is below signfix by 1844 points;
- face pred-vs-point consistency worsened in the original r16 comparison;
- Open3D face/head remained shell-like;
- fixed-threshold point count did not collapse, so this is not merely a p40 confidence artifact.

## Local fixes added before r18/r19

Added diagnostic tools:

- `tools/audit_normal_sign_convention.py`
- `tools/diagnose_face_roi_failure.py`
- `tools/measure_face_shape_metrics.py`

Patched logging:

- `training/config/4k4d_prior_case.yaml`
  - added cross-view scalar keys for train/val logs.

Prepared configs:

- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r17_signed_xview_audit.yaml`
- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r18_pointbranch_signed_roi.yaml`
- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r19_roi_pseudotarget_overfit.yaml`

Patched main depth/point losses:

- `training/loss.py`
  - base `compute_depth_loss` and `compute_point_loss` now accept ROI weight kwargs and pass a weight map into `regression_loss`.
  - This only changes behavior for configs that explicitly set `head_roi_weight`, `face_roi_weight`, `hairline_roi_weight`, `ear_band_roi_weight`, or `boundary_boost` under `loss.depth` / `loss.point`.

## Normal sign convention audit

Output:

- `output/normal_line_multiview_20260428/normal_sign_audit_r16_vs_signfix_on6v_headshoulder_allviews`
- `output/normal_line_multiview_20260428/normal_sign_audit_r18_vs_signfix_on6v_headshoulder_allviews`

Finding:

- predicted normals are opposite raw depth/point-map winding;
- `-N_pred` matches raw depth/point normals with positive signed cosine;
- SMPL-X coarse prior normal follows the raw derived normal direction, not predicted-normal direction.

This justified using:

- `flip_depth_normal: true`
- `orientation: oriented`

for depth-derived normal consistency.

## r18 result

Intent:

- freeze `normal_head`;
- use predicted normal as a fixed local surface-direction target;
- focus gradients on depth/point instead of making normal maps prettier.

Training:

- first attempt failed because normal head was unused by DDP;
- fixed by adding `normal_head` to `optim.frozen_module_names`;
- successful run:
  - `output/modal_training_results/20260428_normal_r18_pointbranch_signed_roi_smoke20_fixfreeze_from_ckpt4`
  - inference:
    - `output/modal_results/20260428_normal_r18_pointbranch_signed_roi_fixfreeze_ckpt0_on6v_headshoulder`

Same-protocol ROI:

| Entry | Full | Head | Face | p40 threshold |
|---|---:|---:|---:|---:|
| r18 pointbranch signed ROI | 184213 | 40527 | 14982 | 58.4267 |

Normal consistency:

| ROI | Comparison | signfix angle | r18 angle | Verdict |
|---|---|---:|---:|---|
| face | pred_vs_depth | 12.6836 | 10.6200 | improves |
| face | pred_vs_point | 8.3753 | 8.5977 | still worse |
| face | depth_vs_point | 8.0617 | 5.9028 | improves |

Conclusion:

r18 improves depth/point consistency but does not improve final face/head point cloud gate. It proves the same failure mode as r16: consistency can improve while the modeled face does not.

## r19 result

Intent:

- test one-frame ROI overfit;
- add ROI-weighted main pseudo-target depth/point supervision;
- verify whether the current 6-view teacher target can directly push face/head geometry.

Local config issue:

- first run with `repeat_batch: true` failed before training because prediction batch shape and teacher masks diverged;
- fixed by setting `repeat_batch: false`.

Successful training:

- `output/modal_training_results/20260428_normal_r19_roi_pseudotarget_overfit_b100_norepeat_from_ckpt4`
- inference:
  - `output/modal_results/20260428_normal_r19_roi_pseudotarget_overfit_b100_on6v_headshoulder`

Training signal:

- `loss_reg_point` reduced from about 0.0065 to about 0.0012;
- `loss_reg_depth` reduced from about 0.0062 to about 0.0016;
- point/depth gradients were active.

Same-protocol ROI:

| Entry | Full | Head | Face | p40 threshold |
|---|---:|---:|---:|---:|
| r19 ROI pseudotarget overfit | 184213 | 40527 | 12964 | 110.5681 |

Conclusion:

r19 is negative. It overfit the pseudo-target/confidence distribution but regressed the final face gate badly. Do not continue r19 by adding steps.

## Teacher target gate

I checked the teacher target that r19 was trying to learn:

- `output/teacher_normal_case_builds/20260424_headshoulder_teachernormal_r1/subset_predictions/0012_11_frame0000_6views_sparseproto_headshoulder_crop_teacher60subset.npz`

Outputs:

- `output/normal_line_multiview_20260428/teacher60subset_face_roi_diagnosis_on6v_headshoulder`
- `output/normal_line_multiview_20260428/teacher60subset_shape_metrics_vs_signfix_on6v_headshoulder`
- `output/normal_line_multiview_20260428/teacher60subset_consistency_vs_signfix_on6v_headshoulder`

Same-protocol 3D ROI:

| Entry | Full | Head | Face | p40 threshold |
|---|---:|---:|---:|---:|
| signfix | 184213 | 40527 | 16825 | 38.5067 |
| teacher60subset | 184213 | 40527 | 16645 | 21.8173 |

Teacher consistency:

| ROI | Comparison | signfix angle | teacher angle | Verdict |
|---|---|---:|---:|---|
| face | pred_vs_depth | 12.6836 | 40.3096 | much worse |
| face | pred_vs_point | 8.3753 | 39.3707 | much worse |
| face | depth_vs_point | 8.0617 | 7.2201 | slightly better |

Interpretation:

- teacher depth/point are internally a bit more consistent;
- teacher normal is not aligned with its own depth/point geometry;
- teacher hard face ROI is still below signfix;
- therefore it is not a valid mentor-level target for more cloud training.

## Multi-view note

Existing targetcam30 16-view signfix result has higher same-render face count:

- `output/modal_results/20260428_signfix_ckpt4_targetcam30_16v_headface_multiview`
- rendered check:
  - `output/normal_line_multiview_20260428/visual_check_signfix_targetcam30_16v_face_world_p40`
- face p40 under that targetcam30 16-view protocol: `22461`

This does not satisfy the original 6-view headshoulder pass gate, and the projection-only close-up still looks shell-like rather than modeled face geometry. It should be documented as multi-view behavior, not used as a replacement pass claim.

## Current blocker

The blocker is no longer a missing script or a missing cloud run.

Current blocker:

> The available 6-view teacher/target geometry does not provide a continuous, aligned, face/head surface that beats signfix. Normal-depth-point consistency can reduce internal angular errors, but it still optimizes a shell rather than generating eyes/nose/mouth/head surface geometry.

## Required next technical move

Do not launch more r16/r18/r19 cloud runs.

Only resume cloud training after one of these local gates is fixed:

1. Build or find a teacher target that itself passes the same original 6-view headshoulder face/head visual gate.
2. Or build a new target-view face/head surface source from real multi-view geometry that is continuous and aligned, then verify it locally before training.
3. Or change the research claim/protocol truthfully to multi-view/static-human reconstruction instead of claiming original 6-view sparse-view mentor pass.

Minimum local gate before the next cloud run:

- teacher/candidate face ROI must exceed signfix `16825` by a meaningful margin;
- Open3D face/head close-up must look like modeled geometry, not a texture shell;
- teacher predicted/derived normal convention must be audited and fixed;
- fixed-threshold count must not collapse.

## One-line handoff

r18 and r19 are negative: r18 improves consistency without improving face ROI, r19 proves ROI-weighted pseudo-target overfit regresses the face gate, and the current teacher60subset target itself is below signfix; stop cloud iteration until a locally verified, continuous, aligned head/face geometry teacher or a revised multi-view protocol exists.
