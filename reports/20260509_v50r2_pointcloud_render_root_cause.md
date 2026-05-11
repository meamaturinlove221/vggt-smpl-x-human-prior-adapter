# V50R2 点云图渲染根因说明

旧图的问题不是 MeshLab 单独显示设置，而是底层图源协议错配。

- V50R2 candidate point map 实际对应 V42 `frame0000` 的前 6 个 view。
- 正确相机顺序是 `00, 01, 06, 11, 16, 21`。
- 旧 V223 脚本使用了 V15/V16 case 的图像、mask 和 camera id：`00, 01, 15, 30, 45, 59`。
- 这种错配会导致点云贴错 RGB、视角错标、MeshLab 中出现撕裂、残影、多人体或薄片感。
- 新脚本 `tools/v223_v50r2_view_consistent_sources.py` 已统一使用 V42/V50R2 一致协议。

## 新版导师图
- full_body: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/01_full_body_v42_consistent_pointcloud.png`
- upper_body: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/02_upper_body_v42_consistent_pointcloud.png`
- head_face: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/03_head_face_v42_consistent_pointcloud.png`
- hands: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/04_hands_v42_consistent_pointcloud.png`
- architecture: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/05_vggt_smplx_native_architecture.png`
- vertical_full_body_rgb: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/09_vertical_full_body_rgb.png`
- vertical_head_face_rgb: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/10_vertical_head_face_rgb.png`
- vertical_hands_rgb: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/11_vertical_hands_rgb.png`
- vertical_full_body_depth: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/12_vertical_full_body_depth.png`
- vertical_head_face_depth: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/13_vertical_head_face_depth.png`
- vertical_hands_depth: `D:/vggt/vggt-main/output/mentor_report_v50r2/images/14_vertical_hands_depth.png`
