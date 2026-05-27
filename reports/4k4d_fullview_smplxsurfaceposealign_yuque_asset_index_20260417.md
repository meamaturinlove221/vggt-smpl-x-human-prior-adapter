# 4K4D Full-View SMPL Surface Pose-Align Asset Index

## Case Summary

- Dataset: `4K4D`
- Dataset root: `G:\数据集\datasets\data_used_in_4K4D`
- Sequence: `0012_11`
- Frame: `0`
- Target camera: `00`
- Full original view count: `60-view`
- View composition: `1 target + 59 source`
- Training checkpoint used: `vggt-out:/20260417_smplxsurfaceposealign_headhair_resume1_bs6/ckpts/checkpoint_11.pt`
- Inference app: `ap-aotTw2V14pWItw1tQHtFmB`
- GPU: `NVIDIA A100 80GB PCIe`
- Elapsed time: `124.721 s`
- Main result summary: [summary.json](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417/summary.json)

## Recommended Yuque Figures

### 1. High-res fused point cloud, world-point branch, raw

- File: [pointcloud__fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_raw_views.png)
- Original location: [fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417/pointcloud/fused_pointcloud_raw_views.png)
- Suggested caption: `4K4D 序列 0012_11、frame 0 在完整 60 视角输入下，由 world-point 分支融合得到的高清点云总览（raw）。`

### 2. High-res fused point cloud, world-point branch, masked

- File: [pointcloud__fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_masked_views.png)
- Original location: [fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417/pointcloud/fused_pointcloud_masked_views.png)
- Suggested caption: `4K4D 序列 0012_11、frame 0 在完整 60 视角输入下，由 world-point 分支融合得到的人体区域高清点云总览（masked）。`

### 3. High-res depth-unprojection point cloud, raw

- File: [pointcloud_depth_unprojection__fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_raw_views.png)
- Original location: [fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection/fused_pointcloud_raw_views.png)
- Suggested caption: `4K4D 序列 0012_11、frame 0 在完整 60 视角输入下，由 depth + camera 反投影融合得到的高清点云总览（raw）。`

### 4. High-res depth-unprojection point cloud, masked

- File: [pointcloud_depth_unprojection__fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_masked_views.png)
- Original location: [fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection/fused_pointcloud_masked_views.png)
- Suggested caption: `4K4D 序列 0012_11、frame 0 在完整 60 视角输入下，由 depth + camera 反投影融合得到的人体区域高清点云总览（masked）。`

## PLY Deliverables

- World-point raw: [pointcloud__fused_pointcloud_raw.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_raw.ply)
- World-point masked: [pointcloud__fused_pointcloud_masked.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_masked.ply)
- Depth-unprojection raw: [pointcloud_depth_unprojection__fused_pointcloud_raw.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_raw.ply)
- Depth-unprojection masked: [pointcloud_depth_unprojection__fused_pointcloud_masked.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_masked.ply)

All four PLY files have `element vertex 180000` in the header and `180010` total lines including header.

## Quantitative Notes

- `num_images = 60`
- Human-prior feature map shape: `[60, 11, 518, 518]`
- Human-prior summary token shape: `[12, 8]`

World-point branch:

- raw valid points before confidence filtering: `16,099,440`
- raw points after confidence filtering: `4,830,185`
- raw points written: `180,000`
- masked valid points before confidence filtering: `707,520`
- masked points after confidence filtering: `212,273`
- masked points written: `180,000`

Depth-unprojection branch:

- raw valid points before confidence filtering: `16,099,440`
- raw points after confidence filtering: `16,099,440`
- raw points written: `180,000`
- masked valid points before confidence filtering: `707,520`
- masked points after confidence filtering: `212,256`
- masked points written: `180,000`

## Source Directories

- Consolidated Yuque assets: [4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417)
- Full inference results: [0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417](/f:/vggt/vggt-main/output/modal_results/0012_11_frame0000_fullviews_smplxsurfaceposealign_20260417)
- Full 60-view scene bundle: [4k4d_scene_fullviews_smplxsurfaceposealign_20260417](/f:/vggt/vggt-main/output/4k4d_scene_fullviews_smplxsurfaceposealign_20260417)

## Guard Status

- `modal app list` shows no active running app for this task.
- Redundant local `modal run` and `volume get` watcher processes from older runs were stopped after verification.
