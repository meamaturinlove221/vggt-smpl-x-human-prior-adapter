# 2026-04-28 r16 xview self-geometry eval on original 6v headshoulder

## Verdict

**Not mentor-final. Do not claim pass.**

r16 cross-view self-geometry training completed, but the same-protocol original
6-view headshoulder evaluation does not beat the signfix reference gate. The
main hard failure is face ROI: r16 p40 face ROI is **14981**, below the signfix
reference **16825**.

## Inputs

- Scene: `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop`
- r16 checkpoint: `vggt_4k4d_train/20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4/logs/ckpts/checkpoint_0.pt`
- r16 prediction: `output/modal_results/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder/predictions.npz`
- signfix reference prediction: `output/modal_results/20260424_signfix_ckpt4_on6v_headshoulder/predictions.npz`

Modal inference completed and downloaded 32 files to:

- `output/modal_results/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder`

## Same-Protocol ROI, p40

Reference signfix gate from handoff:

| metric | signfix reference |
|---|---:|
| full ROI | 184213 |
| head ROI | 40527 |
| face ROI | 16825 |
| p40 threshold | 38.5067 |

r16 same-protocol ROI summary:

| metric | r16 |
|---|---:|
| full ROI | 184213 |
| head ROI | 40527 |
| face ROI | 14981 |
| p40 threshold | 58.3877 |

Delta vs signfix:

| metric | delta |
|---|---:|
| full ROI | 0 |
| head ROI | 0 |
| face ROI | -1844 |

This fails the required numeric minimum because face ROI does not exceed
`16825`, let alone by a meaningful margin.

## Open3D Outputs

p40 Open3D outputs:

- `output/normal_line_multiview_20260428/r16_xview_selfgeom_open3d_on6v_headshoulder`

Fixed signfix-threshold outputs:

- `output/normal_line_multiview_20260428/r16_xview_selfgeom_fixedthr_on6v_headshoulder`

Open3D p40, `roi-source 2d`:

| ROI | point source | threshold | points written |
|---|---|---:|---:|
| full | world_points | 58.3877 | 184213 |
| full | depth_unprojection | 96.0520 | 184213 |
| head | world_points | 65.3767 | 81983 |
| head | depth_unprojection | 101.7758 | 81982 |
| face | world_points | 58.1033 | 43841 |
| face | depth_unprojection | 97.1725 | 43841 |

Open3D fixed threshold `38.5067`, `roi-source 2d`:

| ROI | point source | points written |
|---|---|---:|
| full | world_points | 273859 |
| full | depth_unprojection | 288847 |
| head | world_points | 127398 |
| head | depth_unprojection | 124858 |
| face | world_points | 66767 |
| face | depth_unprojection | 64757 |

Fixed threshold does not collapse by point count, but this is not enough for a
pass because the same-protocol 3D face ROI fails and the Open3D visual evidence
still looks shell-like rather than a modeled face.

Representative visual evidence:

- `output/normal_line_multiview_20260428/r16_xview_selfgeom_open3d_on6v_headshoulder/face_world_points/camera_view_03_crop.png`
- `output/normal_line_multiview_20260428/r16_xview_selfgeom_open3d_on6v_headshoulder/face_depth_unprojection/camera_view_03_crop.png`
- `output/normal_line_multiview_20260428/r16_xview_selfgeom_fixedthr_on6v_headshoulder/face_world_points_sfthr385067/camera_view_03_crop.png`
- `output/normal_line_multiview_20260428/r16_xview_selfgeom_fixedthr_on6v_headshoulder/face_depth_unprojection_sfthr385067/camera_view_03_crop.png`

Visual read: face/head still mainly show projected texture and broken side/hair
regions. Eyes, nose, mouth, face contour, and head surface are not clearly more
coherent than the existing signfix reference.

## Normal Consistency

Output:

- `output/normal_line_multiview_20260428/r16_xview_selfgeom_consistency_on6v_headshoulder`

Target view: `0`.

Selected metrics:

| ROI | comparison | signfix mean angle | r16 mean angle | result |
|---|---|---:|---:|---|
| full | pred_vs_depth | 12.7225 | 11.8296 | improved |
| full | pred_vs_point | 10.6692 | 9.9516 | improved |
| full | depth_vs_point | 6.0607 | 5.6155 | improved |
| head | pred_vs_depth | 11.7250 | 10.1918 | improved |
| head | pred_vs_point | 8.8711 | 8.1635 | improved |
| head | depth_vs_point | 6.5045 | 5.1713 | improved |
| face | pred_vs_depth | 12.6836 | 12.3519 | slight improvement |
| face | pred_vs_point | 8.3753 | 10.1420 | worse |
| face | depth_vs_point | 8.0617 | 5.9158 | improved |

The consistency signal is mixed. It does not rescue the candidate because point
cloud quality and face ROI are the decisive gates.

## Final State

r16 is evaluated and currently negative for mentor-final status:

- p40 same-protocol face ROI is below signfix.
- Open3D visuals do not show a clearly modeled face/head.
- Fixed threshold does not collapse by count, but count alone is not a pass.
- Consistency metrics improve in several rows but face `pred_vs_point` worsens,
  and the improvements do not translate into target-view face geometry.

Next normal-only work should diagnose why cross-view self-geometry improves some
depth/point agreement without improving the actual face/head point-cloud gate.
