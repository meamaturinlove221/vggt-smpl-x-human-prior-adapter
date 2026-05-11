# 《SMPL / SMPL-X + VGGT 语雀插图与附件清单（V2.1 口径）》

日期：`2026-04-19`

## 一、使用口径先说明白

- **方法说明口径**：按当前 `V2.1` 代码写，也就是 `pose-aligned SMPL-X driver + surface maps + summary tokens + layer-wise fusion + 顶点身份增强 + pose-noise + 多尺度融合`。
- **现有完整 60-view 结果图口径**：当前仓库里最完整、最适合直接发语雀的 4K4D 结果图，主要还是 `2026-04-17` 那一版 `smplx surface pose-align` 全视角产物。
- 因此，语雀正文里可以同时成立两件事：
  - 方法图与方法文案按 `V2.1` 写。
  - 结果图先使用现有完整 60-view pose-align 产物，并在正文或图注中说明“当前代码还比这批结果更进一步”。

## 二、正文必须放的核心插图

| 编号 | 建议标题 | 建议文件名 / 类型 | 当前来源 | 用来说明什么 | 建议放在语雀哪一节 | 图注方向 | 是否必须 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 原版 VGGT 方法示意图 | `fig_original_vggt_baseline.svg` | 仓库内自绘版本：[current_vggt_architecture_20260404.svg](/f:/vggt/vggt-main/docs/current_vggt_architecture_20260404.svg) | 先交代原版 VGGT 的通用多视角几何定位，给导师建立对照基线。 | “原版 VGGT 的定位与局限” | 图注里说明“based on the original VGGT architecture, used here only as baseline context”。如果最终采用论文原图，也要注明来源。 | `必须` |
| 2 | 新版 SMPL-X + VGGT 方法架构图 | `fig_smplx_vggt_v2_1_architecture.png` | 待生成，使用更新后的 prompt 文档：[smpl_smplx_vggt_architecture_prompts_20260417.md](/f:/vggt/vggt-main/docs/smpl_smplx_vggt_architecture_prompts_20260417.md) | 这是全文最核心的图，负责回答“SMPL-X 到底怎么接进 VGGT”。 | “新版融合设计 / 方法总览” | 图注重点写：原版主干保留，新增 pose-aligned surface condition、summary tokens、layer-wise fusion 和 training-time-only supervision。 | `必须` |
| 3 | 4K4D 完整 60 视角 RGB 输入总览 | `fig_4k4d_60view_rgb_contact_sheet.png` | [rgb_contact_sheet.png](/f:/vggt/vggt-main/output/4k4d_scene_fullviews_smplxsurfaceposealign_20260417/rgb_contact_sheet.png) | 说明这次展示的不是减视角 case，而是原始完整 60-view 输入。 | “实验设置 / 结果说明前的输入条件” | 图注写清楚：`0012_11`, `frame 0`, `target cam 00`, `1 target + 59 source`, full original views。 | `必须` |
| 4 | 4K4D 完整 60 视角人体 mask 总览 | `fig_4k4d_60view_mask_contact_sheet.png` | [mask_contact_sheet.png](/f:/vggt/vggt-main/output/4k4d_scene_fullviews_smplxsurfaceposealign_20260417/mask_contact_sheet.png) | 说明人体前景覆盖范围，以及后续 masked 点云展示到底裁的是什么区域。 | “实验设置 / 人体区域定义” | 图注强调这是结果可视化时的人体前景区域参考，不要写成方法主创新。 | `必须` |
| 5 | 60-view world-point 人体区域高清点云 | `fig_60view_world_point_masked.png` | [pointcloud__fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_masked_views.png) | 直接展示人体区域点云完整性，是正文最重要的结果图之一。 | “实验结果” | 图注突出人体区域完整性、头部局部质量、masked 视图更适合回答导师关心的人体区域问题。 | `必须` |
| 6 | 60-view depth 反投影人体区域高清点云 | `fig_60view_depth_unprojection_masked.png` | [pointcloud_depth_unprojection__fused_pointcloud_masked_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_masked_views.png) | 从 `depth + camera` 反投影分支再给一张人体区域结果，补足和 world-point 分支的互证。 | “实验结果” | 图注可写“从 depth branch 反投影得到的人体区域高清点云，与 world-point 分支形成互补证据”。 | `必须` |
| 7 | 版本演进 / 方法差异表 | `table_vggt_v1_v2_v2_1_diff.md` 或直接语雀表格 | 建议直接在语雀正文里手工表格化 | 导师更关心“原版、V1、V2、V2.1 到底差在哪”，一张紧凑表格比长段文字更高效。 | “方法演进”或“当前版本与原版差异” | 表格列建议：原版 VGGT / V1 / V2 / V2.1；行建议：输入人体条件、layer-wise fusion、summary tokens、输出侧 supervision、顶点身份增强、pose-noise、多尺度融合。 | `必须` |

## 三、建议放进正文但可根据版面压缩的图

| 编号 | 建议标题 | 建议文件名 / 类型 | 当前来源 | 用来说明什么 | 建议放在语雀哪一节 | 图注方向 | 是否建议 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 60-view world-point 全场景高清点云 | `fig_60view_world_point_raw.png` | [pointcloud__fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_raw_views.png) | 给出全场景上下文，避免导师误以为结果只在人体 crop 上成立。 | “实验结果” | 图注写“保留背景用于控制变量，说明方法没有把非人体区域全部丢掉”。 | `建议` |
| 9 | 60-view depth 反投影全场景高清点云 | `fig_60view_depth_unprojection_raw.png` | [pointcloud_depth_unprojection__fused_pointcloud_raw_views.png](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_raw_views.png) | 与上一张配对，展示反投影分支的全场景几何。 | “实验结果” | 图注可写“raw 视图用于补充整体场景结构，masked 视图用于强调人体区域改善”。 | `建议` |
| 10 | Pose-aligned surface condition 中间示意图 | `fig_pose_aligned_surface_condition_cam00.png` | 当前原始中间材料在：[smplx_vertex_feature_maps.npz](/f:/vggt/vggt-main/output/4k4d_scene_fullviews_smplxsurfaceposealign_20260417/human_prior/smplx_vertex_feature_maps.npz) | 这张图最能回答“为什么不是 keypoint，而是表面级条件”。建议从已有 `human_prior` 原始数据导出 1 张可视化图。 | “为什么强调 pose-aligned / 人体条件如何构造” | 图注建议写“示意图”，重点说明它是与真实视角对齐的人体 surface condition，而不是最终结果图。 | `强烈建议` |
| 11 | 输出侧 prior 监督示意图 | `fig_output_side_depth_point_prior.png` | 当前可以基于同一 `human_prior` 原始数据或方法图局部裁出示意 | 帮助解释 `depth prior / point prior / valid mask` 的角色，但不能让正文重心偏成 prior 工程细节。 | “输出侧 supervision” | 图注写成“training-time-only supervision illustration”，不要写成新的预测输出。 | `可放` |

## 四、适合放在附图 / 折叠部分 / 补充材料的内容

| 编号 | 建议标题 | 建议文件名 / 类型 | 当前来源 | 用来说明什么 | 建议位置 | 图注方向 | 是否可后放 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 4K4D 60-view PLY 附件包 | `.ply` 附件 | [pointcloud__fused_pointcloud_raw.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_raw.ply), [pointcloud__fused_pointcloud_masked.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__fused_pointcloud_masked.ply), [pointcloud_depth_unprojection__fused_pointcloud_raw.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_raw.ply), [pointcloud_depth_unprojection__fused_pointcloud_masked.ply](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__fused_pointcloud_masked.ply) | 给导师或组内同学保留可下载的 3D 结果，不一定放正文。 | 语雀文末附件 / 折叠区 | 图注不用太长，说明对应 world-point / depth-unprojection 的 raw 与 masked 即可。 | `可后放` |
| 13 | 结果数值摘要 JSON | `summary.json` / `pointcloud_summary.json` | [summary.json](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/summary.json), [pointcloud__pointcloud_summary.json](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud__pointcloud_summary.json), [pointcloud_depth_unprojection__pointcloud_summary.json](/f:/vggt/vggt-main/output/yuque_assets/4k4d_0012_11_fullviews_smplxsurfaceposealign_20260417/pointcloud_depth_unprojection__pointcloud_summary.json) | 给正文中的运行信息、点数统计、过滤后点数提供证据来源。 | 折叠说明 / 附件 | 适合在文中引用关键数字，不需要大段展开。 | `可后放` |
| 14 | pose-align 人体条件原始包 | `smplx_vertex_feature_maps.npz` | [smplx_vertex_feature_maps.npz](/f:/vggt/vggt-main/output/4k4d_scene_fullviews_smplxsurfaceposealign_20260417/human_prior/smplx_vertex_feature_maps.npz) | 作为“已有中间产物”的补充附件，后续可从这里再导出可视化。 | 折叠材料 | 说明这是一份中间特征归档，不建议直接拿它当正文插图。 | `可后放` |

## 五、如果时间紧张，哪些可以先不放

下面这些内容不是当前第一优先级，可以等正文主线稳定后再补：

1. `20-view vs 60-view` 单独对比图  
当前对话中它很有价值，但仓库里还没有看到统一整理好的 Yuque-ready 单图路径。可以等你们把已有 compare 图统一导出后，再决定放正文还是附录。

2. output-side prior 的细粒度可视化  
它对方法解释有帮助，但容易把文章重心带偏到 prior 工程实现，优先级低于方法总图和 60-view 最终点云结果。

3. 更细的 summary token 热图、channel-by-channel surface map 可视化  
这些适合组会或补充材料，不是导师第一次看语雀时的主阅读路径。

## 六、推荐的正文排版顺序

如果要尽快拼出一版逻辑顺的语雀正文，推荐顺序如下：

1. 原版 VGGT 方法示意图  
2. 新版 `V2.1` 方法架构图  
3. 版本演进 / 差异表  
4. 60-view RGB 输入总览  
5. 60-view mask 总览  
6. 60-view world-point masked 点云  
7. 60-view depth-unprojection masked 点云  
8. 60-view raw 点云两张  
9. pose-aligned surface condition 中间示意图  
10. PLY / JSON / human_prior 原始包作为文末附件

## 七、建议直接放进图注或正文的小提醒

建议在正文里保留一段非常明确的版本边界说明：

- 当前方法说明按 `V2.1` 代码口径整理。
- 当前完整 60-view 结果图主要来自 `2026-04-17` 的 `smplx surface pose-align` 全视角跑次。
- 现有结果已经能支撑“人体先验增强路线有效”，但若要让方法图、文案和结果图完全同一版本，后续仍需用最新 `V2.1` 代码重跑完整 60-view 产物。
