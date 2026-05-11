# V50R2 Point Cloud PLY Validity Audit

Conclusion: the multi-body / flat-sheet MeshLab view came from an invalid export pattern, not from a mentor-facing point cloud source. V50R2 stores per-view visible-surface point maps. Concatenating six views directly creates overlapping shells and apparent multiple bodies.

Do not send these files:
- `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\invalid_do_not_send\v50r2_human_only_candidate_world_merged6v.ply`
  - reason: This file is a stale/legacy 6-view concatenation of per-camera visible-surface point maps. It is not a calibrated fused point cloud. In MeshLab it can look like multiple bodies, flat sheets, or severe tearing. Do not send it to the mentor; use the per-view files under send_to_mentor/ instead.
- `D:\vggt\vggt-main\output\surface_research_preflight_local\V32_candidate_inference_research\v32_candidate_open3d_review_points.ply`
  - reason: Legacy full-frame/debug review PLY. It includes non-human/full-frame point-map pixels and is not a mentor-facing human point cloud.

Open these MeshLab-safe files first:
- full_body / camera_depth_unprojected / full_body_cam30_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_full_body_cam30_depth.ply`
  - points: 9819
  - reason: front/three-quarter full body view with one visible surface
- full_body / camera_depth_unprojected / full_body_cam15_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_full_body_cam15_depth.ply`
  - points: 9515
  - reason: side full body view with one visible surface
- full_body / camera_depth_unprojected / full_body_cam59_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_full_body_cam59_depth.ply`
  - points: 11048
  - reason: back/side full body view with one visible surface
- head_face / camera_depth_unprojected / head_face_cam30_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_head_face_cam30_depth.ply`
  - points: 1135
  - reason: head/face ROI from the readable front/three-quarter view
- head_face / refined_world_points / head_face_cam30_refined_world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_head_face_cam30_refined_world.ply`
  - points: 1135
  - reason: head/face refined candidate ROI, one view only
- left_hand / camera_depth_unprojected / left_hand_cam15_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_left_hand_cam15_depth.ply`
  - points: 2378
  - reason: left-hand visible view
- left_hand / refined_world_points / left_hand_cam15_refined_world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_left_hand_cam15_refined_world.ply`
  - points: 2378
  - reason: left-hand SMPL-X native local patch, one view only
- right_hand / camera_depth_unprojected / right_hand_cam30_depth: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_right_hand_cam30_depth.ply`
  - points: 1907
  - reason: right-hand visible view
- right_hand / refined_world_points / right_hand_cam30_refined_world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\send_to_mentor\v50r2_SEND_right_hand_cam30_refined_world.ply`
  - points: 1907
  - reason: right-hand SMPL-X native local patch, one view only

All per-view human-only exports:
- cam00 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam00.ply`
- cam00 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam00.ply`
- cam01 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam01.ply`
- cam01 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam01.ply`
- cam15 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam15.ply`
- cam15 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam15.ply`
- cam30 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam30.ply`
- cam30 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam30.ply`
- cam45 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam45.ply`
- cam45 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam45.ply`
- cam59 candidate world: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_candidate_world_cam59.ply`
- cam59 depth unprojected: `D:\vggt\vggt-main\output\mentor_report_v50r2\pointcloud_sources\v50r2_human_only_depth_unprojected_cam59.ply`
