# VGGT-SMPL-X Human Prior Adapter 中文说明

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter 架构图" width="100%" />
</p>

<p align="center">
  <a href="README.md">English README</a> ·
  <a href="#项目一句话">项目一句话</a> ·
  <a href="#和原版-vggt-的关系">和原版 VGGT 的关系</a> ·
  <a href="#这个仓库增加了什么">我的改动</a> ·
  <a href="#工程推进记录">工程推进记录</a> ·
  <a href="#可视化记录">可视化记录</a> ·
  <a href="#仓库定位">仓库定位</a>
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_research_route-2563eb" />
  <img alt="baseline" src="https://img.shields.io/badge/baseline-VGGT-7c3aed" />
  <img alt="prior" src="https://img.shields.io/badge/prior-SMPL--X-d97706" />
  <img alt="evidence" src="https://img.shields.io/badge/evidence-full--scene_point_cloud-0f766e" />
</p>

## 项目一句话

这个仓库记录的是一条 **VGGT + SMPL-X 人体结构先验** 的模型侧实验路线：在保留 VGGT 原本相机、深度、点图、轨迹预测能力的基础上，把 SMPL-X 提供的人体拓扑结构变成可以和真实相机对齐、可以参与监督、可以被对照实验审计的人体先验信号。

核心目标不是直接把 SMPL-X 模板替换成最终点云，而是让 VGGT student route 在人体区域获得更清楚的结构参考。

## 和原版 VGGT 的关系

原版 VGGT 是一个 feed-forward visual geometry baseline。它可以从单张或多张 RGB 图像中直接预测相机、深度图、point map 和 track。这个能力适合通用场景几何，但人体区域有额外的拓扑要求：头、躯干、手臂、腿、手脚之间的结构关系需要在三维点云中保持稳定。

本仓库是在这个 baseline 基础上做的增量：

| VGGT baseline | 本仓库增加的内容 |
| --- | --- |
| 主要依赖 RGB 输入 | RGB + SMPL-X 人体结构先验 |
| 预测 camera / depth / point map / track | 增加人体区域 prior maps、prior depth、prior points、prior normals、prior mask |
| 通用场景几何 | 加入人体拓扑约束 |
| 指标和可视化分散判断 | 明确 full-scene RGB point cloud 作为主要视觉证据 |
| point map 可能在人体区域散开 | 用 pose-aligned SMPL-X prior 提供结构参考 |

## 这个仓库增加了什么

### 1. Pose-aligned SMPL-X prior 构造

项目读取 SMPL-X pose / shape / expression / translation / scale 等参数，把参数化人体放入当前姿态和场景坐标系中，生成一个和当前样本对齐的人体结构参考。

### 2. 真实相机下的 prior rendering

SMPL-X mesh 会被投影到真实相机视角下，形成几类先验：

- `prior_maps`：图像空间的人体提示；
- `prior_depths`：人体区域的深度参考；
- `prior_points`：通过相机几何得到的人体点参考；
- `prior_normals`：局部表面方向；
- `prior_mask`：人体先验监督区域。

这些 prior 只有在 RGB、mask、camera intrinsic / extrinsic、VGGT 输出坐标都一致时才有意义。

### 3. HumanPriorAdapter / supervision path

项目保留 VGGT 主干，让人体先验作为轻量 adapter 或 supervision route 进入模型，而不是把 VGGT 变成 SMPL-X 参数回归器。这样可以保留 VGGT 对场景几何的建模能力，同时在人体区域引入结构约束。

### 4. baseline / control 组织

项目从一开始就按对照实验组织：

- vanilla VGGT；
- no-prior route；
- human-prior route；
- teacher / reference route；
- ROI / crop diagnostic；
- full-scene human-main visual evidence。

这样能避免把 teacher、projection overlay、局部 ROI 截图误写成最终模型成果。

## 工程推进记录

并行实验中排查过多条路线：

- projected target patch / summary-token patch；
- point-normal / human-crop finetuning；
- TeacherGeom / ROI combo；
- confidence-collapse pseudo-positive：ROI 点数增加，但 Open3D 或 confidence-based 检查反而更差；
- NormalBae、Sapiens、DepthAnything、DepthPro 等外部 reference；
- 6-view face/head ROI 复核；
- full-scene human-main visual evidence 检查。

这些实验说明，只增加 loss 或 ROI 点数不等于几何质量提升。人体 teacher 如果不连续、不对齐、可见面不完整，student 也会继承这些问题。

## 可视化记录

<p align="center">
  <img src="docs/figures/parallel_engineering_result_snapshot.svg" alt="parallel engineering result snapshot" width="100%" />
</p>

<p align="center"><sub>6-view face/head ROI 复核：局部面部结构已经可见，但连续性和稳定性还需要更强的几何支持。</sub></p>

<p align="center">
  <img src="docs/figures/external_reference_control.svg" alt="external reference control" width="100%" />
</p>

<p align="center"><sub>外部 reference route 用于相机、mask、teacher 质量审计，是参考控制组，不是 student 输出。</sub></p>

## 仓库定位

| 仓库 | 作用 |
| --- | --- |
| `VGGT-SMPL-X-Human-Prior-Adapter` | 模型侧 SMPL-X prior 注入与监督路线 |
| `VGGT-ZJU-Mocap-Adapter` | 数据适配、相机对齐、mask 审计和 trusted case export |
| `vggt-human-prior-builder` | release-safe 公开预处理 recipe 与 schema 边界 |
| `TuringResearch_plus` | 用于研究规划、证据管理和 workflow dry-run 的 MCP-first 工具链 |

## 这个项目体现的工作

这个项目体现了我在以下方面的工作：

- 理解并复用 VGGT 这类 visual geometry baseline；
- 设计 SMPL-X prior 接入方式，而不是直接替换模型输出；
- 处理真实相机下 prior depth / prior points / prior normals / prior mask 的对齐；
- 将人体结构先验、multi-view geometry、point cloud evidence 连接起来；
- 用 baseline、control、teacher/student 区分来组织实验和结论；
- 把阶段性结果整理为可复盘的研究工程记录。

## 成果展示页面

- https://www.yuque.com/maturinlove221/gqr279/emwf87ku108nzvez
- https://www.yuque.com/maturinlove221/gqr279/fg8lq33tgbwiagtt
