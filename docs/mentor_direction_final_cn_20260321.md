# 导师最后拍板方向：清洗版定稿

这份说明只保留录音最后真正拍板的方向，不再复盘前面大量历史实验细节。

## 一句话结论

当前阶段不要继续在那套复杂的 ghost 分支上叠逻辑，而是回到原版 VGGT，先验证 `depth + camera -> 反投影点云 -> 渲染/重建` 这条几何链是否比直接使用 `point map` 更稳、更少重影。

## 导师最后真正定下来的意思

### 1. 先回到原版 VGGT，不沿着旧 ghost 体系继续堆

- 旧实验里已经叠了太多图像侧评分、mask、权重和 ghost 相关逻辑。
- 变量过多后，很难判断到底是哪一步起作用。
- 所以现在需要重新回到原版代码，以最小改动做基线验证。

### 2. 第一阶段先不要急着加新 loss

- 当前最重要的不是继续加更多 loss。
- 第一步应该先做最小 baseline，只替换“渲染输入点云的来源”这一处变量。
- 也就是在同一批输入、同一目标视角、同一可视化流程下，只比较两条分支：
  - 分支 A：直接使用 `point map / world_points`
  - 分支 B：使用 `depth + camera` 反投影得到的 points

### 3. 后续主线从图像侧 ghost 评分，切到三维几何链

- 原来那套 ghost 评分、mask 惩罚、主体框约束，本质上更偏图像侧后处理思路。
- 导师最后的意思是：如果最终问题出在三维对齐、相机预测、深度预测和点云构造链路上，那么核心优化就应该回到几何链本身，而不是继续在图像评分器上缝补。
- 所以接下来优先看的是：
  - 深度是否稳定
  - 相机是否稳定
  - 反投影得到的点云是否更干净
  - 重渲染后重影是否自然减少

### 4. 只有在几何链方向成立后，才加一个最简单的监督项

- 如果 `depth + camera` 这条路确实比 `point map` 更稳，再进入第二阶段。
- 第二阶段也仍然保持最小改动原则：
  - 保持原版 camera/depth 监督
  - 如需新增，只加一个最基础的 reconstruction 或 geometry loss
  - 暂时不恢复旧 ghost 复合损失

## 为什么这个方向和 VGGT 原论文、原代码一致

### 论文依据

- 在你给的原论文 `VGGT.pdf` 第 4 页，作者明确写到：训练时联合预测 camera、depth、point 等量有帮助；但在推理阶段，把 depth 和 camera 结合起来构造 3D points，通常比直接使用 point map branch 更准。
- 在 `VGGT.pdf` 第 8 页，作者在 ETH3D 实验里再次说明：相比直接使用 point map，利用 depth head 与 camera head 反投影得到的点云，精度更高。
- 同页表 3 里也给出了结果：`Ours (Depth + Cam)` 优于 `Ours (Point)`。

### 代码依据

- 原仓库 README 已直接写明：由深度图和相机构造出来的 3D points，通常“比 point map branch 更准确”。见 [README.md](/f:/vggt/vggt-main/README.md#L123)。
- 原版可视化代码默认就优先走 depth-based 分支，只有显式传 `--use_point_map` 才改为 point map。见 [demo_viser.py](/f:/vggt/vggt-main/demo_viser.py#L80)。
- 可视化工具里也明确区分了两条路径：`Pointmap Branch` 与 `Depthmap and Camera Branch`。见 [visual_util.py](/f:/vggt/vggt-main/visual_util.py#L68)。
- 原版训练默认配置本来就是：
  - `enable_camera=True`
  - `enable_depth=True`
  - `enable_point=False`
  - `enable_track=False`
  见 [default.yaml](/f:/vggt/vggt-main/training/config/default.yaml#L162)。
- 训练 README 也明确建议：资源有限时，一般只 fine-tune camera 和 depth heads 就足够。见 [training/README.md](/f:/vggt/vggt-main/training/README.md#L94)。

## 我们这轮已经做出的验证

- 已在原版仓库上完成 point-map vs depth-unproject 的本地 baseline。
- 当前汇总结果见 [batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_batch/examples8/batch_summary.md)。
- 第一轮结果可以概括为：
  - `kitchen`：`depth + camera` 明显更好
  - 其余几个场景：`depth + camera` 在 MAE 上更有优势，但覆盖率并非稳定全胜
- 因此当前结论是：
  - 这条几何链方向成立，值得继续
  - 但它还不是在所有场景上无条件压过 point map
  - 所以后续仍应坚持“最小改动、逐项验证”，不能重新回到大杂烩式实验

## 最新补充：更贴近人像任务的验证已经完成

- 已基于旧项目里的 ZJU 人像 `report.json`，在原版 VGGT 上做了 source-only 重跑，并重新渲染到真实 target camera。
- 去重后的 ZJU 汇总见 [batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_zju_batch/batch_summary.md)。
- 当前人像域结果是：
  - `6src_hist`：`depth + camera` 赢
  - `12src_nested`：打平，但 `depth + camera` 仍有更低 MAE
  - `23cam_fullset`：`depth + camera` 赢
- 另外已经把“比较脚本”升级成了“比较 + 主分支成品”两套输出。
- 已验证的 primary 几何产物见 [primary_summary.md](/f:/vggt/vggt-main/output/geometry_primary_zju/coreview390_6src_hist_primary/primary_summary.md)。

这意味着当前不只是“论文和 README 说 depth+camera 更好”，而是你们自己的人像 case 上也已经出现了同方向证据，所以后续主线可以更坚定地放在几何链上。

## 接下来应该怎么做

### 第一阶段：继续原版最小实验

1. 固定原版 VGGT，不恢复 ghost 分支。
2. 固定 A/B 对比框架，只比较 `point map` 和 `depth + camera`。
3. 重点检查那些结果打平或互有优劣的场景，弄清楚差异来自：
   - 相机预测
   - 深度预测
   - 点云覆盖率
   - 输入视角一致性

### 第二阶段：最小 fine-tune

只有当第一阶段确认几何链值得继续后，再进入最小 fine-tune：

1. 保持原版训练结构。
2. 保持 `camera + depth` 主线。
3. 维持 `point` 和 `track` 关闭。
4. 如需新增监督，只加一个简单的几何或重建损失。

### 运行策略

- 本地 `Windows + RTX 5080 16GB`：
  - 跑原版推理
  - 跑小样本 baseline
  - 看单 case 可视化
  - 做轻量 smoke / dry-run
- Modal 云端：
  - 正式 fine-tune
  - sweep
  - 长时间任务
  - 大显存批处理

## 可以直接对导师复述的版本

老师，我理解这次最后拍板的方向是：先不沿着我之前那套复杂 ghost 分支继续加东西，而是回到原版 VGGT，先做一个最小 baseline，对比 `point map` 和 `depth + camera` 反投影这两条点云来源。因为原论文和原版代码都表明，推理阶段用 depth 加 camera 构造点云通常比直接用 point map 更准，所以我们先验证这条几何链是否能更稳地减少重影。如果这条路成立，下一步也不马上恢复复杂损失，而是只在原版 camera/depth 监督的基础上，加一个最简单的几何或重建监督，再继续往下推进。
