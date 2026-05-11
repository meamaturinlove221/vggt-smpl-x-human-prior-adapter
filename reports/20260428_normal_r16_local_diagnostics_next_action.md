# 2026-04-28 r16 local diagnostics and next action

## Current Verdict

r16 remains **negative** and must stay frozen:

- same-protocol p40 face ROI: `14981`, below signfix `16825`;
- full/head counts match signfix, but face gate fails;
- fixed threshold does not collapse, but Open3D face/head remains shell-like;
- face `pred_vs_point` worsens from `8.3753 deg` to `10.1420 deg`.

Do not continue with r16 by adding epochs or rerunning the same config.

## Local Diagnostics Completed

### 1. Normal Sign Convention Audit

Output:

- `output/normal_line_multiview_20260428/normal_sign_audit_r16_vs_signfix_on6v_headshoulder`
- `output/normal_line_multiview_20260428/normal_sign_audit_r16_vs_signfix_on6v_headshoulder_allviews`

Result:

- Predicted normals are globally opposite to raw depth-derived normals.
- Predicted normals are globally opposite to raw point-derived normals.
- Predicted normals are also opposite to raw SMPL-X coarse prior normals.
- `-N_pred` matches all three much better under signed cosine.

All 6 views show the same pattern. This means future signed losses must be
explicit about the derived-normal winding instead of reporting only abs cosine.

### 2. r16 Face ROI Failure Diagnosis

Outputs:

- `output/normal_line_multiview_20260428/r16_face_roi_failure_diagnosis_on6v_headshoulder`
- `output/normal_line_multiview_20260428/r16_face_roi_failure_diagnosis_on6v_headshoulder_view3`

Key facts:

- signfix p40 threshold: `38.5067`;
- r16 p40 threshold: `58.3877`;
- 2D face ROI p40 kept pixels: signfix `44963`, r16 `43322`;
- 2D face ROI fixed-threshold kept pixels: signfix `44963`, r16 `66767`;
- view03 p40 overlay shows r16 loses a top/hair/head band while keeping the central face;
- fixed-threshold overlay mostly keeps both and adds many candidate pixels.

Conclusion:

r16 is not a simple confidence collapse. It shifts confidence distribution and
spatial kept regions, while the 3D face ROI still drops. Point count and fixed
threshold alone are therefore not reliable pass signals.

### 3. Shape Metrics Added

Output:

- `output/normal_line_multiview_20260428/r16_shape_metrics_vs_signfix_on6v_headshoulder`

New local metrics:

- 2D ROI coverage;
- largest connected component ratio;
- camera-space z range;
- central-face protrusion proxy;
- PCA thinness / planarity.

These are not final mentor metrics, but they prevent future candidates from
being judged only by ROI count.

## Local Fixes Applied

### 1. Cross-view loss logging

Updated:

- `training/config/4k4d_prior_case.yaml`

Added train/val scalar logging for:

- `loss_prior_cross_view`;
- `loss_prior_cross_view_depth`;
- `loss_prior_cross_view_point`;
- `loss_prior_cross_view_normal`.

This is required before any new cloud run, otherwise r16/r18 cannot be diagnosed
by sub-loss behavior.

### 2. Normal sign audit tool

Added:

- `tools/audit_normal_sign_convention.py`

Purpose:

- compare `N_pred` and `-N_pred` against depth-derived, point-derived, and SMPL-X
  coarse prior normals;
- report signed cosine, negative fraction, and camera-ray orientation.

### 3. Face ROI failure diagnostic

Added:

- `tools/diagnose_face_roi_failure.py`

Purpose:

- compare baseline vs candidate p40/fixed kept/lost/new masks;
- summarize face/head/full confidence distribution;
- measure camera-space `world_points` vs `depth_unprojection` difference.

### 4. Shape metric tool

Added:

- `tools/measure_face_shape_metrics.py`

Purpose:

- add simple full/head/face shape gates beyond point count.

## Candidate Configs Prepared

### r17 signed audit

Added:

- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r17_signed_xview_audit.yaml`

Purpose:

- make depth-normal sign convention explicit;
- use `orientation: oriented`;
- use `flip_depth_normal: true`;
- keep cross-view normal unoriented only for audit continuity.

Important:

r17 alone is not expected to be enough, because the prior r5 signed-depthnormal
line was already negative. r17 is mainly the minimal sign-convention correction.

### r18 point-branch signed ROI

Added:

- `training/config/4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r18_pointbranch_signed_roi.yaml`

Purpose:

- keep signed derived-normal convention;
- detach predicted normal in depth-normal consistency so normal acts as a local
  surface-direction target;
- strengthen depth-point and cross-view point terms;
- set cross-view `normal_weight: 0.0` so the smoke focuses on geometry, not
  normal-only agreement;
- increase head/face/hairline ROI weights moderately without adding a new
  SMPL-X teacher.

## Next Action Gate

Local side is now fixed enough to allow a **small r18 smoke cloud run**, but not
enough to claim any method success.

The next cloud run, if executed, must be:

- r18 only, not r16 retry;
- short smoke from signfix ckpt4;
- same original 6-view headshoulder evaluation immediately afterward;
- ROI + Open3D + fixed threshold + signed normal audit + shape metrics;
- fail fast if face ROI and Open3D do not improve.

Pass still requires:

- same-protocol face ROI meaningfully above `16825`;
- Open3D face/head visibly more modeled, not only more points;
- full-body sanity preserved;
- both `world_points` and `depth_unprojection` checked;
- fixed threshold not collapsing;
- no truthful-reporting shortcut.
