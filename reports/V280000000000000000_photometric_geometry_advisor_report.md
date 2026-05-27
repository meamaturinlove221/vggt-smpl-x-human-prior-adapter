# 《基于 Full VGGT Forward 与 SMPL-X 结构先验的 Photometric-Geometry Verified 人体场景点云补全》

# 先给结论

当前状态：`V300000000000000000_PHOTOMETRIC_GEOMETRY_VISUAL_TRUTH_MENTOR_READY_NOT_PROMOTED`。

这不是 promotion，也不修改 registry、V50 或 V50R2；active candidate 仍保持 `V11700_gap_reduction_branch_520`。本轮把 V120 严格降级为 checkpoint，因为 V120 的主要证据仍然偏向 high-density 点数、局部点云 close-up 和 JSON all-pass，不能替代导师原始视觉门控。

本轮主证据不是 source-label、visible-delta、projection-only 或 metric-only，而是：

- 3D full-scene 主图：`boards/V140000000000000000_3d_human_scene_board.png`
- 同场景 controls 投影图：`boards/V140000000000000000_projection_overlay_board.png`
- 局部投影 close-up：`boards/V160000000000000000_head_hair_projection_closeup.png`、`boards/V160000000000000000_hand_arm_projection_closeup.png`、`boards/V160000000000000000_clothing_projection_closeup.png`
- 可打开 viewer：`output/V230000000000000000_viewer/index.html`
- 最终门控：`reports/V260000000000000000_final_mentor_gate.json`
- upload-safe bundles：`reports/V290000000000000000_bundle_integrity.json`

# 一、为什么 V120 仍需降级

V120 已经把点数和同预算控制做上来了，但复查后仍不能作为导师最终通过。核心问题有四类：

1. V740 生成路径仍有程序性构造痕迹，包括 `weighted_pick(..., replace=True)`、局部插值和 RGB contrast gain。
2. V740 旧评分函数含 `detail_bonus` 和 `control_penalty`，不能作为公平 controls 证据。
3. V760 close-up 虽然是局部图，但仍不足以证明五官、手型或衣物边界真的优于 VGGT baseline。
4. V120 的 `all_pass` 是结果摘要，不是导师视觉证据；导师最高门控仍是 full-scene RGB point cloud 中人体、环境和 controls 的可视比较。

因此本轮把 V120 保留为 checkpoint，同时废弃 V740 score 作为最终证据，进入 photometric geometry verification。

# 二、本轮路线定位

本轮不是继续堆点，也不是用投影替代 3D 点云。投影是辅助验证：把 3D 点云返回真实 RGB/mask/camera 视角，看局部 mask、RGB、edge 是否自洽。

```text
RGB / mask / camera
    -> full VGGT forward outputs and per-case effect
    + SMPL-X surfel / voxel / graph / projection features
    + refined VGGT high-confidence detail source
    -> photometric geometry adapter
    -> human-main full-scene RGB point cloud
    -> 3D visual gate + 2D projection/local gate
```

本轮保留的路线边界：SMPL-X 结构 feature 必须参与；VGGT forward / token / output effect 继续保留；输出仍是 model-owned student；V591/Kinect teacher 或 raw Kinect depth 不能作为 inference 输出。

# 三、当前变化

V121/V122 首先审计当前上传包和旧代码，明确 V120 只能作为 checkpoint。V130 从本机 4K4D SMC 中重新导出真实 RGB 和 mask，并绑定 V950 feature bank 的 camera/projection 字段。V150 废弃旧 score，使用不读取 config name 的投影/RGB/edge 指标。V160 重新生成真实投影局部 close-up。V170 收窄 detail source，修掉 0012_11 和 0013_01 detail_mask 全点为 true 的问题。

执行中初始 V190 矩阵被 fail closed：中性评分显示 controls 仍接近或优于 true，local visible improvement 也不够。于是 V270 自动生成 auto-evolved route，重建带 `projection_uv_518` 的 photometric prediction matrix。修复后所有 case 的 true / baseline / controls 都在同 human/environment 预算下重新比较。

# 四、实验闭环

1. V120100 冻结并降级 V120。
2. V121 审计 V115 bundles 和当前 repo 文件。
3. V122 审计 V740/V780 代码，确认旧 scoring 和 densifier 不能作为 final。
4. V130 构建真实 RGB/mask/camera 投影资产。
5. V140 同时生成 3D full-scene board 和 projection overlay board。
6. V150 使用 config-neutral scoring 重算 fair scores。
7. V160 生成 head/hair、hand/arm、clothing 三类真实局部投影 close-up。
8. V170 精炼 detail source，禁止 all-point detail mask。
9. V190/V210 重跑 photometric matrix 和 hard controls v7。
10. V230 生成可用 HTML/PLY viewer。
11. V240/V250/V260 做视觉裁判、多序列门控和最终 mentor gate。
12. V280/V290/V295 输出导师报告、upload bundles 和 cleanup。

# 五、VGGT baseline / controls 对比

V150/V210 使用同样的人体点数、环境点数、投影视角和评分函数。评分函数不读取 config 名字，也不使用 true-only bonus、detail bonus 或 control penalty。

当前 V150 结果显示四个 case 均通过 controls separation：

- `current_v895_0021_03`: true 0.8110，best control 0.6668，margin 0.1443
- `0021_03_frame001`: true 0.8111，best control 0.6693，margin 0.1418
- `0012_11_frame001`: true 0.8130，best control 0.6899，margin 0.1231
- `0013_01_frame001`: true 0.8003，best control 0.6934，margin 0.1069

这些分数只作为辅助。导师主图仍以 V140 3D full-scene board 和同场景 controls board 为准。

# 六、点云视觉证据

主图路径：`boards/V140000000000000000_3d_human_scene_board.png`。

这张图展示同一场景下的 true、VGGT baseline、posthoc、same topology、tiny token、shuffled controls。每个输出都使用相同的 human/environment 点预算，human ratio 控制在约 0.714，保留部分环境。它是导师视觉主证据。

投影辅助图路径：`boards/V140000000000000000_projection_overlay_board.png`。

投影图用于检查点云返回真实 RGB/mask/camera 后是否仍与人体区域一致。它不能替代 3D 主图，但可以防止“点云看起来像人，投影回真实图像却不贴合”的代理成功。

# 七、局部细节

- head/hair：`boards/V160000000000000000_head_hair_projection_closeup.png`
- hand/arm：`boards/V160000000000000000_hand_arm_projection_closeup.png`
- clothing：`boards/V160000000000000000_clothing_projection_closeup.png`

本轮只声明 head/face contour and hair region，不声明 facial details。除非图中明确能辨认眼鼻口，否则报告不能写五官细节。V160 的判定是局部投影 non-regression 和 visible improvement，而不是 active count 或 RGB variance 单项。

# 八、环境与 viewer

环境门控路径：`boards/V220000000000000000_environment_realism_v5.png`。

viewer 路径：`output/V230000000000000000_viewer/index.html`。

viewer 不是占位 HTML，当前包含 true、baseline、posthoc、same topology、tiny、shuffled 六个 PLY alias，支持点大小调整，并链接 projection/local close-up 图。PLY 文件位于 `output/V230000000000000000_viewer/ply/`。

# 九、边界和限制

- 本轮不是 promotion。
- 本轮不声明 paper-grade generalized。
- 投影证据是辅助验证，不能替代 full-scene 3D point cloud 主图。
- head/hair 只声明 contour and hair region，不声明五官。
- V120/V740 旧 high-density score 已被降级为 checkpoint 证据。
- 如果导师肉眼认为 true 和 controls 仍接近，应继续 fail closed，而不是用 JSON all-pass 强行通过。

# 十、给导师看的文件清单

- `boards/V140000000000000000_3d_human_scene_board.png`
- `boards/V140000000000000000_projection_overlay_board.png`
- `boards/V160000000000000000_head_hair_projection_closeup.png`
- `boards/V160000000000000000_hand_arm_projection_closeup.png`
- `boards/V160000000000000000_clothing_projection_closeup.png`
- `boards/V210000000000000000_hard_controls_visual_v7.png`
- `boards/V210000000000000000_hard_controls_projection_v7.png`
- `boards/V250000000000000000_multisequence_photometric_summary.png`
- `output/V230000000000000000_viewer/index.html`
- `reports/V150000000000000000_fair_metric_v2_scores.csv`
- `reports/V160000000000000000_local_projection_metrics.csv`
- `reports/V210000000000000000_hard_control_firewall_v7.csv`
- `reports/V260000000000000000_final_mentor_gate.json`
- `reports/V290000000000000000_bundle_integrity.json`
