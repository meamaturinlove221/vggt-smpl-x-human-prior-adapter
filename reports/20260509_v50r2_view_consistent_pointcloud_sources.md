# V50R2 View-Consistent Point Cloud Sources

This replaces the previous V15/V16-misaligned point-cloud exports.

## Source Decision

- V50R2 candidate points are exactly `v42_prior_enabled_payload__research_points_world.npz/frame0000[:6]`.
- Correct view order: `00, 01, 06, 11, 16, 21`.
- The old cam15/cam30/cam45/cam59 V50R2 figures were invalid because they used V15 6-view images and masks.

## Mentor Images
- full_body: `D:\vggt\vggt-main\output\mentor_report_v50r2\images\01_V223_full_body_pointcloud_v42_consistent.png`
- upper_body: `D:\vggt\vggt-main\output\mentor_report_v50r2\images\02_V223_upper_body_pointcloud_v42_consistent.png`
- head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\images\03_V223_head_face_pointcloud_v42_consistent.png`
- hands: `D:\vggt\vggt-main\output\mentor_report_v50r2\images\04_V223_hands_pointcloud_v42_consistent.png`

## Point Cloud PLYs
- view00_cam00_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_full_v42_consistent.ply` (11240 points)
- view00_cam00_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_full_v42_consistent_visual_upright.ply` (11158 points)
- view00_cam00_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_head_face_v42_consistent.ply` (1027 points)
- view00_cam00_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_head_face_v42_consistent_visual_upright.ply` (1015 points)
- view00_cam00_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_left_hand_v42_consistent.ply` (513 points)
- view00_cam00_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_left_hand_v42_consistent_visual_upright.ply` (504 points)
- view00_cam00_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_right_hand_v42_consistent.ply` (817 points)
- view00_cam00_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view00_cam00_right_hand_v42_consistent_visual_upright.ply` (816 points)
- view01_cam01_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_full_v42_consistent.ply` (14315 points)
- view01_cam01_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_full_v42_consistent_visual_upright.ply` (14147 points)
- view01_cam01_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_head_face_v42_consistent.ply` (982 points)
- view01_cam01_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_head_face_v42_consistent_visual_upright.ply` (982 points)
- view01_cam01_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_left_hand_v42_consistent.ply` (655 points)
- view01_cam01_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_left_hand_v42_consistent_visual_upright.ply` (646 points)
- view01_cam01_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_right_hand_v42_consistent.ply` (904 points)
- view01_cam01_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view01_cam01_right_hand_v42_consistent_visual_upright.ply` (904 points)
- view02_cam06_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_full_v42_consistent.ply` (9293 points)
- view02_cam06_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_full_v42_consistent_visual_upright.ply` (9272 points)
- view02_cam06_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_head_face_v42_consistent.ply` (387 points)
- view02_cam06_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_head_face_v42_consistent_visual_upright.ply` (383 points)
- view02_cam06_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_left_hand_v42_consistent.ply` (987 points)
- view02_cam06_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_left_hand_v42_consistent_visual_upright.ply` (987 points)
- view02_cam06_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_right_hand_v42_consistent.ply` (462 points)
- view02_cam06_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view02_cam06_right_hand_v42_consistent_visual_upright.ply` (462 points)
- view03_cam11_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_full_v42_consistent.ply` (12067 points)
- view03_cam11_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_full_v42_consistent_visual_upright.ply` (12063 points)
- view03_cam11_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_head_face_v42_consistent.ply` (4294 points)
- view03_cam11_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_head_face_v42_consistent_visual_upright.ply` (4292 points)
- view03_cam11_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_left_hand_v42_consistent.ply` (129 points)
- view03_cam11_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_left_hand_v42_consistent_visual_upright.ply` (129 points)
- view03_cam11_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_right_hand_v42_consistent.ply` (640 points)
- view03_cam11_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view03_cam11_right_hand_v42_consistent_visual_upright.ply` (640 points)
- view04_cam16_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_full_v42_consistent.ply` (11274 points)
- view04_cam16_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_full_v42_consistent_visual_upright.ply` (11260 points)
- view04_cam16_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_head_face_v42_consistent.ply` (4022 points)
- view04_cam16_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_head_face_v42_consistent_visual_upright.ply` (3999 points)
- view04_cam16_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_left_hand_v42_consistent.ply` (897 points)
- view04_cam16_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_left_hand_v42_consistent_visual_upright.ply` (890 points)
- view04_cam16_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_right_hand_v42_consistent.ply` (1024 points)
- view04_cam16_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view04_cam16_right_hand_v42_consistent_visual_upright.ply` (1024 points)
- view05_cam21_full: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_full_v42_consistent.ply` (11264 points)
- view05_cam21_full_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_full_v42_consistent_visual_upright.ply` (11256 points)
- view05_cam21_head_face: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_head_face_v42_consistent.ply` (3993 points)
- view05_cam21_head_face_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_head_face_v42_consistent_visual_upright.ply` (3991 points)
- view05_cam21_left_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_left_hand_v42_consistent.ply` (4312 points)
- view05_cam21_left_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_left_hand_v42_consistent_visual_upright.ply` (4305 points)
- view05_cam21_right_hand: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_right_hand_v42_consistent.ply` (1210 points)
- view05_cam21_right_hand_visual: `D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud\ply\v50r2_view05_cam21_right_hand_v42_consistent_visual_upright.ply` (1209 points)
