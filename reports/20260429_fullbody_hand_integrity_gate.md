# 2026-04-29 full-body / hand integrity gate

## Mentor requirement

The normal-line gate cannot only inspect face/head. Face, head, and hairline remain the main target, but full-body geometry is the bottom line. A candidate cannot pass if the full body has large holes, broken limbs, severe ghosting, or obviously missing / fragmented hands.

## Implemented local gate

New audit tool:

- `tools/audit_fullbody_hand_integrity.py`

For each candidate and point source, it outputs:

- p40 or fixed-threshold full-body point count;
- 3D largest connected component ratio after voxel downsampling;
- 3D vertical body-band coverage to catch large missing body sections;
- per-view full-body kept-pixel overlays;
- RGB skin-extremity hand-risk masks after removing head/face;
- hand-risk kept ratios and overlays.

Open3D renderer update:

- `tools/render_open3d_pointcloud.py` now supports `--roi hands --roi-source 2d`.
- Every candidate package should include `full/head/face/hands` for both `world_points` and `depth_unprojection`.

## Current audit result

Reference and current local depth-offset probes pass the numerical full-body bottom-line screen:

| entry | source | threshold | points | largest component | min vertical bin | hand views passing |
|---|---|---:|---:|---:|---:|---:|
| signfix | world_points | p40 / fixed 38.5067 | 184213 | 0.9936 | 0.0499 | 4 |
| signfix | depth_unprojection | p40 | 184213 | 1.0000 | 0.0481 | 4 |
| signfix | depth_unprojection | fixed 38.5067 | 189605 | 1.0000 | 0.0521 | 4 |
| depth_roi_v2_o035 | world_points | p40 / fixed 38.5067 | 184213 | 0.9936 | 0.0493 | 4 |
| depth_roi_v2_o035 | depth_unprojection | p40 | 184213 | 1.0000 | 0.0477 | 4 |
| depth_roi_v2_o035 | depth_unprojection | fixed 38.5067 | 189605 | 1.0000 | 0.0518 | 4 |
| depth_roi_all_o025 | world_points | p40 / fixed 38.5067 | 184213 | 0.9916 | 0.0492 | 4 |
| depth_roi_all_o025 | depth_unprojection | p40 | 184213 | 1.0000 | 0.0478 | 4 |
| depth_roi_all_o025 | depth_unprojection | fixed 38.5067 | 189605 | 1.0000 | 0.0519 | 4 |

These numbers do not mean mentor-final pass. They only show the current probes did not create a new full-body collapse.

## Current visual status

The candidate still fails visual quality:

- face/head side views remain shell-like;
- face lacks reliable nose / eye / mouth geometry;
- hands ROI renders show fragmented arm / hand / phone surfaces rather than stable hand structure;
- therefore the normal line is still not mentor-final.

Current hand-risk evidence:

- `output\normal_line_multiview_20260428\depth_roi_offset_v2_o035_open3d_p40\hands_world_points`
- `output\normal_line_multiview_20260428\depth_roi_offset_v2_o035_open3d_p40\hands_depth_unprojection`
- `output\normal_line_multiview_20260428\fullbody_hand_audit_depth_roi_v2_o035_world_points_p40`
- `output\normal_line_multiview_20260428\fullbody_hand_audit_depth_roi_v2_o035_depth_unprojection_p40`

## Updated pass rule

A candidate may only be considered for cloud training or mentor delivery if all of the following hold locally:

- face/head main gate passes under original 6-view headshoulder protocol;
- `world_points` and `depth_unprojection` both beat signfix face ROI meaningfully;
- fixed threshold `38.5067` does not collapse;
- full-body largest connected component and vertical body-band coverage pass;
- Open3D full-body views show no large body holes, broken limbs, severe ghosting, or missing hands;
- Open3D hands ROI is not obviously fragmented or amputated;
- Open3D face/head views show modeled geometry, not only a texture shell.
