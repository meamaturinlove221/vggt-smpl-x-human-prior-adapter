# V16 SMPL-X Native Region ROI Builder

Status: `v16_smplx_native_region_roi_builder_ready_with_documented_fallbacks`

Research-only ROI builder. It writes no predictions, teacher/candidate package, registry, or strict-pass state.

## Decision

Runnable V16 SMPL-X-native ROI maps were produced without MANO/FLAME; sparse subregions use documented spatial fallbacks and no strict pass is claimed.

## Conditions

- left_right_hand_nonempty: `True`
- wrist_bridge_nonempty: `True`
- head_nonempty: `True`
- face_front_nonempty: `True`
- strict_pass_claimed: `False`

## Key Metrics

- view_count: `6`
- height: `518`
- width: `518`
- roi_count: `19`
- left_hand_pixels: `762`
- right_hand_pixels: `547`
- wrist_bridge_left_pixels: `4875`
- wrist_bridge_right_pixels: `6996`
- head_pixels: `6782`
- face_front_pixels: `2535`
- canonical_query_coverage: `1.0`
- native_lmk_fields: `['dynamic_lmk_bary_coords', 'dynamic_lmk_faces_idx', 'lmk_bary_coords', 'lmk_faces_idx']`

## View Support

| Region | Source | Total Pixels | Nonempty Views | Per View Pixels |
|---|---|---:|---:|---|
| body_visible | v15_prior_mask_or_silhouette | 68130 | 6 | [11490, 14720, 9515, 9819, 11538, 11048] |
| left_hand | smplx_native_joint2num_dominant_skinning_plus_v15_anchor | 762 | 6 | [28, 55, 424, 73, 48, 134] |
| right_hand | smplx_native_joint2num_dominant_skinning_plus_v15_anchor | 547 | 4 | [0, 0, 223, 248, 19, 57] |
| wrist_bridge_left | smplx_native_wrist_elbow_skinning_with_dilated_hand_bridge | 4875 | 6 | [575, 743, 2384, 257, 423, 493] |
| wrist_bridge_right | smplx_native_wrist_elbow_skinning_with_dilated_hand_bridge | 6996 | 6 | [852, 977, 730, 1900, 1206, 1331] |
| thumb_left | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 184 | 6 | [15, 36, 48, 17, 20, 48] |
| index_left | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 152 | 6 | [14, 24, 33, 20, 9, 52] |
| middle_left | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 114 | 6 | [13, 24, 31, 17, 6, 23] |
| ring_left | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 83 | 6 | [6, 12, 28, 20, 6, 11] |
| pinky_left | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 61 | 6 | [5, 10, 18, 16, 3, 9] |
| thumb_right | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 112 | 4 | [0, 0, 5, 47, 9, 51] |
| index_right | mixed_smplx_native_finger_and_spatial_fallback_for_sparse_views | 76 | 4 | [0, 0, 54, 9, 4, 9] |
| middle_right | documented_spatial_fallback_within_smplx_native_hand_roi | 187 | 4 | [0, 0, 71, 93, 9, 14] |
| ring_right | documented_spatial_fallback_within_smplx_native_hand_roi | 169 | 4 | [0, 0, 61, 84, 9, 15] |
| pinky_right | documented_spatial_fallback_within_smplx_native_hand_roi | 150 | 4 | [0, 0, 50, 73, 9, 18] |
| head | smplx_native_head_jaw_eye_neck_skinning_plus_spatial_fallback_when_sparse | 6782 | 6 | [1148, 1091, 1345, 1135, 1243, 820] |
| face_front | smplx_native_lmk_faces_dynamic_lmk_plus_v15_or_spatial_front_fallback | 2535 | 6 | [253, 267, 527, 702, 410, 376] |
| face_lmk_static | smplx_native_lmk_faces_idx_projected_via_nearest_canonical_vertex | 849 | 4 | [0, 0, 227, 431, 112, 79] |
| face_lmk_dynamic | smplx_native_dynamic_lmk_faces_idx_projected_via_nearest_canonical_vertex | 2361 | 6 | [253, 267, 517, 546, 402, 376] |

## Fallback Policy

- hand_and_wrist: Primary labels use SMPL-X joint2num/dominant skinning and V15 native hand anchors; wrist bridges dilate hand ROIs onto wrist/elbow support.
- fingers: If native per-finger skinning support is too sparse, split the SMPL-X-native hand ROI into thumb/index/middle/ring/pinky spatial bands and mark the source.
- head: Primary label uses SMPL-X Head/Jaw/Eye/Neck skinning and V15 head map; sparse views fall back to top silhouette support excluding hands.
- face_front: Primary label uses SMPL-X lmk_faces_idx and dynamic_lmk_faces_idx vertices; sparse views fall back to a centered crop inside the head ROI.

## Outputs

- roi_maps_npz: `D:\vggt\vggt-main\output\surface_research_preflight_local\V16_smplx_native_region_roi_builder\v16_smplx_native_region_roi_maps.npz`
- summary_json: `D:\vggt\vggt-main\output\surface_research_preflight_local\V16_smplx_native_region_roi_builder\summary.json`
- view_support_table_json: `D:\vggt\vggt-main\output\surface_research_preflight_local\V16_smplx_native_region_roi_builder\view_support_table.json`
- view_support_table_md: `D:\vggt\vggt-main\output\surface_research_preflight_local\V16_smplx_native_region_roi_builder\view_support_table.md`
- roi_contact_sheet: `D:\vggt\vggt-main\output\surface_research_preflight_local\V16_smplx_native_region_roi_builder\roi_contact_sheet.png`

## Blockers

- none
