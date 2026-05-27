# SMPL / SMPL-X 融合 VGGT 汇报提纲

日期：`2026-04-16`

## 第 1 页：问题背景

- `VGGT` 在一般多视角几何上有效
- 但人体区域仍然存在遮挡强、细节弱、深度不稳的问题
- 研究问题：如何利用参数化人体先验提升人体区域几何质量

## 第 2 页：SMPL 原理

- `SMPL` 是参数化人体网格模型
- 用 shape 参数、pose 参数、pose corrective blendshape 和 linear blend skinning 表达人体
- 价值：可微、紧凑、符合人体结构、适合视觉与图形联合使用

## 第 3 页：SMPL-X 原理

- `SMPL-X` 在 `SMPL` 基础上加入手、脸、表情
- 相比 body-only 模型，更适合细粒度的人体表达
- 对我们的启发：人体先验不能只停留在粗 body support

## 第 4 页：为什么 VGGT 需要人体先验

- 仅靠图像和普通几何监督，人体容易出现缺失和模糊
- 人体不是普通背景几何，而是有强结构先验的对象
- 参数化人体模型可以提供稳定的人体区域支撑和 completion 起点

## 第 5 页：当前融合思路

- 不是让 `VGGT` 直接回归 `SMPL` 参数
- 而是把 `SMPL` 当作外部几何先验
- 核心思想：`soft prior + pseudo supervision`

## 第 6 页：当前已落地的数据流

- 读入 `SMPL` 顶点
- 投影到各训练视角
- 生成：
  - `smpl_prior_masks`
  - `smpl_prior_feature_maps`
  - `human_prior_completion_*`
  - `head_hair_*`
- 再送入 loss 分支

## 第 7 页：loss 融合方式

- depth 分支：
  - 强化 `head / hair` 细节区域
  - 引入 pseudo depth supervision
- unproject 分支：
  - 强化人体 completion 区域
  - 引入 pseudo world-point supervision
  - 约束 depth confidence 和 depth presence

## 第 8 页：为什么这不是模板硬贴

- 没有直接用 `SMPL` 覆盖 `VGGT` 输出
- `SMPL` 的作用是先验和伪监督，不是最终答案
- 这样可以保留服装、头发等非模板细节

## 第 9 页：当前已经落地到哪里

- 已落地在 `ZJU` 几何训练链路
- 关键文件：
  - `training/data/datasets/zju_vggt_geom.py`
  - `training/loss.py`
  - `training/trainer.py`
  - `training/config/...smplprior_headhair_longrun.yaml`
  - `scripts/probe_zju_vggt_geom_dataset.py`
- 当前 `4K4D` 推理脚本里还没有直接融合这套 prior

## 第 10 页：证据链

- 可导出的 prior 可视化
- completion 点云
- completed 点云
- summary / aggregate 统计
- 可以从“效果图”上升到“中间张量和统计证据”

## 第 11 页：局限

- 当前主要还是 `SMPL` prior
- 还不是完整 `SMPL-X` part-aware prior
- hand / face 还没有独立建模
- head / hair 仍有部分启发式成分

## 第 12 页：下一步 SMPL-X 方案

- 保持当前接口不变，先升级先验来源
- 引入 hand / face / torso / limbs 的 part-aware prior
- 做更细粒度的 pseudo supervision
- 进一步探索 surface consistency loss

## 第 13 页：结论

- `SMPL` 已经真实进入 `VGGT` 训练链路，而不是停留在展示图层面
- 当前方案的本质是“结构化人体先验驱动的人体区域几何增强”
- 下一阶段的研究重点是从 `SMPL prior` 升级到 `SMPL-X` 语义化人体先验
