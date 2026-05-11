# 2026-05-03 Self-Geometry / Crop / Weak-Prior Pivot Status

## Current Truth

No local result is mentor-final. Do not upload to cloud.

The hard-teacher route is now frozen as a defensive gate / blocker record, not
the active optimization path. The active branch is:

```text
codex/normal-selfgeom-crop-weakprior
```

The new active line is:

```text
human crop / soft matting input base
+ normal-depth-point self-geometry coupling
+ SMPL-X pose-aligned weak prior
+ real-data SMPL-X regressor/fitter bridge
+ strict full/head/face/body/hands Open3D gate
```

SMPL-X must remain weak topology / pose conditioning. It is not a face, hair,
clothing, or skirt hard teacher.

## Why Hard Teacher Search Is Frozen

The expanded strict registry remains at zero strict pass:

- `reports/20260502_expanded_strict_gate_registry.json`
- `reports/20260502_expanded_teacher_gate_blocker_status.md`

The hard teacher candidates that were rechecked include internal 60v TSDF,
Kinect projection/coordinate audits, SMPL-X scaffold/anchors, COLMAP/MVS
meshray, visual hull, keypoint/face relief, external pointcloud/mesh patches,
TSDF/Poisson, and historical r42/r57-r68 packages. They did not produce a
continuous, aligned, visually valid head/face surface that can be projected back
to the original 6-view headshoulder protocol. Numeric positives without visual
pass remain blocked.

## Normal Sign Audit

Audit output:

```text
output/normal_line_multiview_20260503/selfgeom_sign_audit_signfix_r16_r18_r20
```

Face ROI summary:

| Entry | pred vs derived signed cos | -pred vs derived signed cos | conclusion |
|---|---:|---:|---|
| signfix | about -0.97 to -0.98 | about +0.97 to +0.98 | predicted normal is opposite to raw depth/point winding |
| r16 | about -0.97 to -0.98 | about +0.97 to +0.98 | same convention issue |
| r18 | about -0.97 to -0.98 | about +0.97 to +0.98 | same convention issue |
| r20 | about -0.97 to -0.98 | about +0.97 to +0.98 | same convention issue |

This confirms that unoriented / abs-cos metrics can look good while signed
geometry convention is still ambiguous. New configs must use one explicit
normal convention, not just abs cosine.

## Gradient Audit

New tool:

```text
tools/audit_selfgeom_loss_gradients.py
```

Outputs:

```text
output/normal_line_multiview_20260503/selfgeom_grad_audit_r16_humancrop6v
output/normal_line_multiview_20260503/selfgeom_grad_audit_r20_humancrop6v
```

Key local result:

- r16: `loss_prior_cross_view_normal` strongly pushes `normal_head`, while
  `cross_view_point` also reaches `world_points/point_head`.
- r20: with normal head frozen and point-targeted terms enabled,
  `depth_point`, `cross_view_point`, and `point_normal` still produce non-zero
  `world_points/point_head` gradients.

Therefore the r16/r20 failure is not simply "the point branch received no
gradient." Continuing the same self-consistency setup with more epochs is not a
good next move. The likely issue is that self-consistency stabilizes an existing
shell unless the input/crop, sign convention, and ROI/fullbody gates are fixed.

## Input Base Matrix

Same checkpoint:

```text
output/local_training_results/r32_confstable_geomonly5/inference_model.pt
```

ABCD inference outputs:

| Group | Scene | Predictions |
|---|---|---|
| A full image | `output/4k4d_scenes/0012_11_frame0000_6views_sparseproto` | `output/local_inference_results/r32_abcd_A_full_6v/predictions.npz` |
| B human crop | `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_human_crop` | `output/local_inference_results/r32_confstable_geomonly1_on6v_fullbody/predictions.npz` |
| C softmatte | `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_human_crop_softmatte` | `output/local_inference_results/r32_abcd_C_softmatte_6v/predictions.npz` |
| D headshoulder | `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop` | `output/local_inference_results/r32_confstable_geomonly1_on6v_headshoulder/predictions.npz` |
| E headface | `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headface_crop` | `output/local_inference_results/r32_abcd_D_headface_6v/predictions.npz` |

Consistency matrix:

```text
output/normal_line_multiview_20260503/abcd_input_matrix_r32
```

Main finding:

- Human crop and softmatte improve normal-depth-point consistency numbers.
- Headface-only crop increases valid face pixels but worsens geometric
  consistency, so it cannot replace headshoulder/fullbody review.
- Crop is a useful input base, not a standalone structural innovation.

## Strict Gate / Visual Review

Existing strict package:

```text
output/normal_line_multiview_20260430/candidate_gate_r32_confstable_geomonly1
```

Explicit local visual fail record:

```text
output/normal_line_multiview_20260503/visual_review_r32_confstable_geomonly1_fail.json
```

Observed failure modes:

- Face/head closeups remain shell-like, not modeled face geometry.
- Hairline/head cap is fragmented or missing.
- Full-body side/back views show implausible shell/ghost volume.
- Hands are fragmented support/noise, not stable hand geometry.

This candidate remains negative despite normal consistency and some shape
metrics. Full-body and hands are hard bottom-line gates.

## Real-Data SMPL-X Bridge Update

Files:

```text
tools/run_realdata_smplx_driver.py
tools/import_external_smplx_params.py
tools/build_scene_prior_from_external_bundle.py
docs/realdata_smplx_driver.md
```

The repo already supports:

- external regressor JSON/NPZ import;
- external fitting-result import;
- estimator-command launch/import;
- normalized SMPL-X/camera bundle output;
- scene-local `prior_maps.npz` generation.

Added on this branch:

- `build_scene_prior_from_external_bundle.py --use-external-assets`
  materializes real-data image/mask assets validated by the driver into the
  output scene before prior generation.
- `docs/realdata_smplx_driver.md` now documents the inference/training bridge
  and weak-prior boundary.

## Next Allowed Work

1. Build a new local-only config that uses an explicit signed convention after
   flipping derived depth/point normals, with SMPL-X only as weak prior.
2. Keep human crop/softmatte as input base, but evaluate on headshoulder and
   fullbody gates, not only headface crop.
3. Run only local smoke/inference/gate first.
4. Cloud upload is allowed only after same-protocol head/face, full-body,
   hands, normal-depth-point, fixed threshold, Open3D visual review, and visual
   JSON all pass.

## Forbidden Claims / Routes

- Do not claim r16/r18/r20/r32/r57-r68 mentor-final.
- Do not upload any candidate from this state to cloud.
- Do not continue old hard-teacher search as the main route.
- Do not use point count, fixed-threshold point count, or normal abs-angle as
  the pass criterion.
- Do not use headface-only crop as a substitute for full-body human evaluation.
