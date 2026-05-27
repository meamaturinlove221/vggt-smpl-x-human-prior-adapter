# 4K4D Full-View VGGT Rerun

- date: `2026-04-16`
- dataset_root: `G:\数据集\datasets\data_used_in_4K4D`
- seq: `0012_11`
- frame: `0`
- target_camera: `00`
- total_views: `60`
- source_views: `59`
- local_scene_dir: `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\4k4d_scenes\0012_11_frame0000_fullviews`
- local_results_dir: `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews`
- modal_output_subdir: `vggt_4k4d_infer/0012_11_frame0000_fullviews`
- modal_app_id: `ap-RXN6tofW1y9SUrqaVoF2xm`
- modal_app_stopped_at: `2026-04-16 18:34:52 +08:00`

## Run Summary

- code state: rerun executed from the current repo state after the SMPL-related code landing
- num_images: `60`
- device: `cuda`
- gpu: `NVIDIA A100-SXM4-40GB`
- dtype: `torch.bfloat16`
- elapsed_seconds: `116.484`
- summary_json: `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews\summary.json`

## Point Cloud Outputs

- world_points raw views png:
  - `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews\pointcloud\fused_pointcloud_raw_views.png`
- world_points masked views png:
  - `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews\pointcloud\fused_pointcloud_masked_views.png`
- depth_unprojection raw views png:
  - `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews\pointcloud_depth_unprojection\fused_pointcloud_raw_views.png`
- depth_unprojection masked views png:
  - `F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews\pointcloud_depth_unprojection\fused_pointcloud_masked_views.png`

## PLY Validation

- validated:
  - `pointcloud/fused_pointcloud_raw.ply`
  - `pointcloud/fused_pointcloud_masked.ply`
  - `pointcloud_depth_unprojection/fused_pointcloud_raw.ply`
  - `pointcloud_depth_unprojection/fused_pointcloud_masked.ply`
- validator:
  - `tools/validate_ascii_ply.py`

## Note

- The cloud inference itself finished normally in a few minutes.
- Local result pulling briefly hit Windows-side `modal volume get` transfer truncation on ASCII `PLY` files; the hanging app/process chain was cleaned up and the files were re-pulled and revalidated successfully.
