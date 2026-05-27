# 4K4D Full-View Yuque Asset Index

## Case Summary

- Dataset: `4K4D`
- Sequence: `0012_11`
- Frame: `0`
- Target camera: `00`
- Full original view count: `60-view`
- View composition: `1 target + 59 source`
- Run summary: [summary.json](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/modal_results/0012_11_frame0000_fullviews/summary.json)
- Full report: [4k4d_fullview_vggt_0012_11_20260416_rerun.md](/f:/vggt/vggt-main/reports/4k4d_fullview_vggt_0012_11_20260416_rerun.md)

## Recommended Yuque Figures

### 1. Full-view RGB inputs

- Use this to show the complete original 60-view input set.
- File: [01_rgb_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/01_rgb_contact_sheet_60views.png)
- Suggested caption: `4K4D case 0012_11, frame 0, target cam 00 的完整 60 视角 RGB 输入（1 个 target + 59 个 source）。`

### 2. Full-view masks

- Use this if you want to explain foreground support or masked point-cloud rendering.
- File: [02_mask_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/02_mask_contact_sheet_60views.png)
- Suggested caption: `对应 60 视角的人体前景 mask 集合。`

### 3. Full-view depth predictions

- This is the best full-view summary of per-view depth outputs.
- File: [03_depth_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/03_depth_contact_sheet_60views.png)
- Suggested caption: `VGGT 在完整 60 视角上的深度预测总览。`

### 4. Full-view depth confidence

- This is useful when discussing which geometry is more reliable.
- File: [04_depth_conf_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/04_depth_conf_contact_sheet_60views.png)
- Suggested caption: `完整 60 视角的深度置信度分布总览。`

### 5. Full-view point confidence

- This complements the depth-confidence figure and is useful for point-cloud filtering discussion.
- File: [05_point_conf_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/05_point_conf_contact_sheet_60views.png)
- Suggested caption: `完整 60 视角的点图置信度分布总览。`

### 6. Fused point cloud from world-point branch, raw

- This is one of the most important final visualization figures.
- File: [06_world_point_fused_raw_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/06_world_point_fused_raw_views.png)
- Suggested caption: `基于 point-map / world-point 分支融合得到的高清点云总览（raw）。`

### 7. Fused point cloud from world-point branch, masked

- Use this with the raw figure as a pair.
- File: [07_world_point_fused_masked_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/07_world_point_fused_masked_views.png)
- Suggested caption: `基于 point-map / world-point 分支融合得到的高清点云总览（masked）。`

### 8. Fused point cloud from depth-unprojection branch, raw

- This is the depth-based full-view 3D reconstruction figure.
- File: [08_depth_unprojection_fused_raw_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/08_depth_unprojection_fused_raw_views.png)
- Suggested caption: `基于 depth + camera 反投影融合得到的高清点云总览（raw）。`

### 9. Fused point cloud from depth-unprojection branch, masked

- Use this with the raw depth-unprojection figure as a pair.
- File: [09_depth_unprojection_fused_masked_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/09_depth_unprojection_fused_masked_views.png)
- Suggested caption: `基于 depth + camera 反投影融合得到的高清点云总览（masked）。`

## Recommended Figure Order for Yuque

1. [01_rgb_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/01_rgb_contact_sheet_60views.png)
2. [03_depth_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/03_depth_contact_sheet_60views.png)
3. [04_depth_conf_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/04_depth_conf_contact_sheet_60views.png)
4. [05_point_conf_contact_sheet_60views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/05_point_conf_contact_sheet_60views.png)
5. [06_world_point_fused_raw_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/06_world_point_fused_raw_views.png)
6. [07_world_point_fused_masked_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/07_world_point_fused_masked_views.png)
7. [08_depth_unprojection_fused_raw_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/08_depth_unprojection_fused_raw_views.png)
8. [09_depth_unprojection_fused_masked_views.png](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets/09_depth_unprojection_fused_masked_views.png)

## Quantitative Notes You Can Reuse

- This rerun is the full original-view version rather than a reduced-view variant.
- `num_images = 60`
- `elapsed_seconds = 116.484`
- GPU used: `NVIDIA A100-SXM4-40GB`

## Point-cloud Summary Notes

- World-point branch summary: [pointcloud_summary.json](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/modal_results/0012_11_frame0000_fullviews/pointcloud/pointcloud_summary.json)
- Depth-unprojection branch summary: [pointcloud_summary.json](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/modal_results/0012_11_frame0000_fullviews/pointcloud_depth_unprojection/pointcloud_summary.json)

You can quote these numbers if needed:

- World-point branch:
  - raw valid points before confidence filtering: `16,099,440`
  - raw points after confidence filtering: `4,829,833`
  - raw points written: `180,000`
  - masked valid points before confidence filtering: `707,520`
  - masked points after confidence filtering: `212,256`
  - masked points written: `180,000`

- Depth-unprojection branch:
  - raw valid points before confidence filtering: `16,099,440`
  - raw points after confidence filtering: `4,829,834`
  - raw points written: `180,000`
  - masked valid points before confidence filtering: `707,520`
  - masked points after confidence filtering: `212,256`
  - masked points written: `180,000`

## Source Directories

- Consolidated Yuque-ready assets: [yuque_assets](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/yuque_assets)
- Original full-view scene package: [0012_11_frame0000_fullviews](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/4k4d_scenes/0012_11_frame0000_fullviews)
- Original full-view inference results: [0012_11_frame0000_fullviews](/f:/vggt/vggt-main/output/4k4d_fullview_rerun_20260416/modal_results/0012_11_frame0000_fullviews)
