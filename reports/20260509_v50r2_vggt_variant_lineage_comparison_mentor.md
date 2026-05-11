# VGGT 系列人体点云路线纵向比较补充

- 生成时间 UTC：`2026-05-09T12:33:59Z`
- 比较范围：只比较“基于 VGGT 输出 depth / point map / normal 的人体点云路线”。
- Kinect、COLMAP/MVS、2DGS、PSHuman、DepthPro、Sapiens 等外部 teacher 或传感器路线不放进主表；它们只能作为失败路线或参考证据。
- 证据边界：4 月底很多 VGGT 变体的目录还在，但原始 `predictions.npz` 已经不在当前本地工作树里。因此这些路线只能按历史报告记录比较，不能伪装成这次重新复算过的点云结果。

## 这部分回答导师什么问题

导师要求先看纵向比较：baseline VGGT 的点云效果怎样，基于 VGGT 的其他人体点云估计方式有没有改善，数值和点云视觉都要看，再判断改进空间。

这次对比后可以比较明确地说：

1. 原始 full-image VGGT baseline 能恢复粗全身，但 head / face ROI 很弱。
2. crop / softmatte 是早期最明显的输入侧提升，主要改善人体在输入里的占比和 ROI 点数。
3. normal / depth / point 自一致路线落实了导师说的几何耦合，但没有单独把 6-view face/head 点云变成清晰连续的人脸几何。
4. SMPL-X prior-enabled V42 / V50R2 补齐了 prior、normal、region evidence 和 candidate package 闭环，但主 point-map 坐标相对 base VGGT 的改变很小。
5. 所以当前不能说“整体细节已经明显超过所有 VGGT baseline”。更稳妥的说法是：工程闭环已完成，候选包可提交；真正的视觉细节改进仍集中在 head/face/hairline 和 right hand。

## 纵向比较表

| 路线 | 类型 | 证据状态 | full 点数 | head 点数 | face 点数 | 结论 |
|---|---|---|---:|---:|---:|---|
| base_full_6v_vggt_preprocess_full | VGGT baseline / original full image | historical_report_only | 40882 | 8994 | 4177 | baseline weak; head/face ROI sparse |
| human_crop_6v_vggt | VGGT + human crop input preprocessing | historical_report_only | 111094 | 24441 | 11523 | large occupancy gain; not sufficient as final head/face quality |
| human_crop_hardmask_6v_vggt | VGGT + hard human mask/crop | historical_report_only | 111078 | 24437 | 10712 | occupancy improves, global geometry shifts more than plain crop |
| human_crop_softmatte_6v_vggt | VGGT + soft matte crop | historical_report_only | 151734 | 33382 | 15127 | densest ROI, but less stable than plain crop |
| normal_r16_xview_selfgeom | VGGT + normal/depth/point self-geometry | historical_report_only | 184213 | 40527 | 14981 | normal consistency partly improves, but face ROI is below signfix and Open3D remains shell-like |
| r32_selfgeom_crop_weakprior | VGGT + crop + weak SMPL-X prior + self-geometry | historical_report_only |  |  |  | negative strict visual review; shell/ghost head-face and fragmented hands |
| v25_base_vggt_research_prediction | VGGT base model research prediction | recomputed_current | mean valid 11575.5 per view in V50R2 comparison | mean valid 4019.17 per view | included in head_face region | basic full-body point-map output exists; no normal route |
| v42_prior_enabled_vggt | VGGT + SMPL-X prior-enabled prediction | recomputed_current | mean valid 11575.5 per view | mean valid 4019.17 per view | included in head_face region | normal evidence available; main point-map coordinate change from V25 is small |
| v50r2_candidate | VGGT candidate package with SMPL-X native evidence | recomputed_current | same main point map as V42 | tighter packaged head/face evidence, mean valid 2450.83 per view in comparison | packaged region evidence, not a larger raw point count | formal candidate closure and region evidence; not a new visibly sharper full-body point field |

## 逐路线说明

### 1. 原始 full-image VGGT baseline

早期记录中，原始 full-image VGGT 在 6-view 下 full ROI 约 `40.9k`，head ROI 约 `9.0k`，face ROI 约 `4.2k`。这说明 baseline 不是完全失败，它能给出粗的人体点云；但对导师关心的头脸细节来说，点数和视觉连续性都不足。

### 2. human crop / hardmask / softmatte

crop 线是最明确的输入侧收益。`human_crop` 把 full ROI 从约 `40.9k` 提到约 `111.1k`，head 从约 `9.0k` 提到约 `24.4k`，face 从约 `3.7k-4.2k` 提到约 `9.6k-11.5k`。这和导师录音里说的“人占比太小，看不清细节，crop 有道理”一致。

但 crop 的收益主要是 occupancy 和输入占比，不等价于真正的 3D 面部几何变清晰。`softmatte` 点数最高，但全局几何扰动也更大，所以不能只按点数说它最好。

### 3. normal / depth / point 自一致 VGGT

`r16_xview_selfgeom` 和后面的 `r32_selfgeom_crop_weakprior` 对应导师说的 depth、point、normal 要耦合，而不是三个分支各学各的。历史结果显示，一些 normal-depth-point consistency 指标确实改善；但同协议 face ROI 没有超过 signfix reference，Open3D 视觉仍然偏 shell-like，不能作为最终头脸点云质量通过。

这说明导师的方向是对的：normal 是必要的几何约束；但当前 normal/self-geometry 还不是单独充分条件。它能让输出之间更自洽，但缺少高质量、连续、对齐的局部 head/face teacher 或直接 point-map 优化时，点云视觉仍然不够。

### 4. V42 prior-enabled VGGT 与 V50R2 candidate

当前可复算的 V25/V42/V50R2 对比显示：V42 prior-enabled 相比 V25 base VGGT 的全像素 point-map mean L2 为 `0.00053544`；V50R2 candidate 主点图与 V42 的 max abs 差异为 `0.0`。这说明 V50R2 的主全身 point-map 不是在 V42 后又大幅变形出一套新几何。

V50R2 的主要价值在于：补齐 SMPL-X native prior-enabled 路线、normal availability、region evidence、head/hand package、formal replay、D-line candidate package 和 registry 闭环。它是 candidate 交付闭环，而不是“视觉细节全面碾压 baseline”的证据。

## 当前可复算文件审计

| 路径 | 是否存在 | 大小 | 修改时间 |
|---|---:|---:|---|
| `output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_points_world.npz` | True | 105107011 | 2026-05-08T20:27:57Z |
| `output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_points_world.npz` | True | 105107787 | 2026-05-08T20:27:58Z |
| `output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__candidate_points.npz` | True | 17459703 | 2026-05-08T21:44:09Z |
| `output/modal_results/20260421_6views_preprocess_full_b40/predictions.npz` | False | None | None |
| `output/modal_results/20260421_6views_preprocess_crop_b40/predictions.npz` | False | None | None |
| `output/modal_results/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder/predictions.npz` | False | None | None |
| `output/local_inference_results/r32_confstable_geomonly1_on6v_fullbody/predictions.npz` | False | None | None |

## 给导师的结论口径

如果导师问“其它基于 VGGT 做点云估计的结果有没有明显更好”，目前应回答：有局部路线收益，但没有一个旧路线真正达到最终要求。crop 对点数和输入占比帮助最大；normal/self-geometry 对一致性有帮助；SMPL-X prior-enabled V50R2 完成了候选包和几何证据闭环。但从最终点云视觉看，head/face/hairline/right hand 的细节仍然不够，下一步需要直接改善 target-view point map 或引入更可靠的独立局部几何证据。
