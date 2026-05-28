# VGGT-SMPL-X Human Prior Adapter 中文说明

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter 架构图" width="100%" />
</p>

<p align="center">
  <a href="README.md">English README</a> ·
  <a href="#这个仓库解决什么问题">项目定位</a> ·
  <a href="#核心路线">核心路线</a> ·
  <a href="#并行工程实验记录">并行工程实验</a> ·
  <a href="#成果怎么判断">成果判断</a> ·
  <a href="#成果展示页面">成果展示页面</a>
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-研究路线整理中-2563eb" />
  <img alt="backbone" src="https://img.shields.io/badge/backbone-VGGT-7c3aed" />
  <img alt="prior" src="https://img.shields.io/badge/prior-SMPL--X-d97706" />
</p>

这个仓库记录的是 VGGT + SMPL-X 人体先验路线里的模型侧部分。

它的目标很明确：在不改掉 VGGT 主干定位的前提下，把 SMPL-X 提供的人体拓扑先验接进多视角几何重建流程里，让模型在人体区域有更清楚的结构参考，尤其是头部、躯干、四肢、手部这类容易散掉或变成一团的区域。

这不是一个“套壳 SMPL-X”的仓库。SMPL-X 在这里提供人体结构参考，最后的几何输出仍然应该来自 VGGT 路线。

---

## 这个仓库解决什么问题

VGGT 本身擅长从多视角 RGB 中估计相机、深度、点图和轨迹。问题在于，人体不是普通物体。人体有比较强的拓扑结构，头、躯干、手脚之间的关系不能只靠点云指标来判断。

项目推进中最容易出现的问题是：

- 指标看起来有提升，但点云里看不出完整人体；
- 投影到图片上似乎还可以，换成 3D 点云后人形散掉；
- 单独截出人体区域能看，放回场景后比例、位置或结构不对；
- SMPL-X 本身像人，但模型输出没有真正学到人体结构；
- teacher、baseline、student 的边界混在一起，最后很难判断到底是哪条路线起作用。

这个仓库把问题收窄到一个模型侧路线：如何把 SMPL-X 的结构信息变成 VGGT 能使用的先验输入和监督信号，并且用清楚的证据标准判断它有没有真正帮助到点云重建。

---

## 核心路线

整体流程可以概括成四步。

### 1. 从真实相机和人体参数出发

输入侧保留 VGGT 原本需要的多视角 RGB 和相机信息，同时读取人体相关参数，用 SMPL-X 构造一个姿态对齐的人体结构参考。

这里强调“真实相机对齐”。如果人体先验只是停留在模板坐标系里，它对多视角重建帮助有限；只有把先验投到同一组相机视角下，才能和 VGGT 的输入、深度、点图监督接上。

### 2. 渲染视角对齐的人体先验证据

SMPL-X 人体结构会被转换成和当前视角对应的先验信号，例如：

- `prior_maps`：给模型看的图像空间先验；
- `prior_depths`：人体区域的深度参考；
- `prior_points`：由相机几何反投影得到的点参考；
- `prior_mask`：标记哪些区域可以参与人体先验监督。

这些信号的价值在于，它们和 RGB、相机、深度、点图处在同一套坐标和视角关系里。

### 3. 用 HumanPriorAdapter 接入模型路线

这个仓库采用轻量 prior adapter 的思路，把人体先验接进 VGGT 路线，而不是推翻原来的主干。

这样做的好处是边界清楚：

- VGGT 仍然负责最终几何预测；
- SMPL-X 负责提供人体结构先验；
- prior adapter 负责把先验转换成模型能使用的中间信号；
- 训练和评估阶段可以保留 baseline、no-prior、random-prior、shuffled-prior 等对照。

### 4. 用 full-scene 点云判断结果

最终判断不看单张诊断图，也不只看 loss。导师真正要看的，是人体为主体、同时保留一定环境上下文的 full-scene RGB point cloud。

也就是说，点云里要能看出人体结构，最好能和原始 VGGT baseline、控制组、adapter 输出放在同一视角、同一边界下比较。

---

## 并行工程实验记录

这条路线后来被放进更大的并行工程里复盘。任务从“把 SMPL-X 接入 VGGT”推进到了“sparse-view 人体高质量几何恢复的工程闭环”。这一阶段跑通了多条链路，也明确暴露出 6-view head / face 点云质量的上限。

主链路可以概括成四层：

1. **Pose-aligned SMPL-X driver**：读取 pose / shape / expression / translation / scale，把参数化人体放到当前姿态和场景坐标里。
2. **Dense prior maps**：把 posed mesh 投到真实相机下，生成逐视角对齐的 dense prior，包括 depth、camera/world points、normal、visibility、canonical coordinates、body-part features 等。
3. **Input-side / layer-wise fusion**：RGB 提供真实外观和背景，prior maps 提供 pose-aligned 几何位置，mask 限制人体先验的作用范围。先验不只在输入端拼一次，而是在多层特征演化过程中持续参与。
4. **Output-side supervision**：训练侧支持 depth / point / normal / point-normal 等几何监督，也支持 ROI 和 boundary 加权。

SMPL / SMPL-X 在这里的角色不是最终结果，而是 pose-aligned geometry prior。它提供人体大体位置、深度、表面方向和区域约束；真正要证明的仍然是下游模型是否能在 sparse-view 条件下生成更清晰、连续、稳定的 3D 人体点云。

这一阶段的经验也说明：只增加 loss 或者让 ROI 点数变多，并不等于几何质量提升。如果 teacher 本身不够连续、对齐，可见面也不完整，就容易出现“点数增加但 Open3D 更差”的伪阳性。

---

## 已排查路线和失败边界

并行实验里排查过多条方向：

- projected targetpatch / summary-token patch；
- 从同一 checkpoint 继续做 point-normal / humancrop 微调；
- TeacherGeom / ROI combo；
- confidence-collapse pseudo-positive，也就是 face ROI 点数看起来暴涨，但 confidence threshold 或 Open3D 评估反而说明质量更差；
- NormalBae、Sapiens、DepthAnything、DepthPro 等外部 teacher 路线。

这些排查得到的结论比较明确：当前瓶颈不是缺少脚本，而是缺少足够高质量、连续、对齐的 head / face geometry teacher，或者缺少一种能直接改善 sparse-view target-view surface 的局部几何优化方法。

所以后续路线必须转向更硬的几件事：

- real 3D learned residual；
- multi-view detail supervision；
- baseline high-confidence detail preservation；
- SMPL feature-conditioned local geometry branch；
- human-main full-scene visual gate。

---

## 当前成果快照

<p align="center">
  <img src="docs/figures/parallel_engineering_result_snapshot.svg" alt="parallel engineering result snapshot" width="100%" />
</p>

<p align="center"><sub>6-view face/head ROI 复核结果：已经能看到局部面部结构，但仍然存在连续性和稳定性问题。</sub></p>

<p align="center">
  <img src="docs/figures/external_reference_control.svg" alt="external reference control" width="100%" />
</p>

<p align="center"><sub>外部几何参考路线只作为相机、mask、teacher 质量排查记录，不作为 student 输出。</sub></p>

目前较安全的结论是：6 视角下已经取得了不错的局部面部结果，但仍然有瑕疵；同协议 6-view face / head 点云还没有达到足够清晰、连续、稳定的最终要求。

---

## 架构示意

```text
多视角 RGB + 真实相机
        │
        ├── VGGT 主干
        │        └── 相机 / 深度 / 点图 / 轨迹
        │
SMPL-X 人体参数
        │
        └── 真实相机下的人体先验渲染
                 ├── prior_maps
                 ├── prior_depths
                 ├── prior_points
                 └── prior_mask
                          │
                          └── HumanPriorAdapter
                                   │
                                   └── VGGT 路线产生的人体感知场景几何
                                            │
                                            └── full-scene RGB point cloud 证据
```

仓库首页上方的 SVG 图对应的就是这条路线。

---

## 成果怎么判断

这个项目不能只看 metric pass。更准确的判断顺序应该是：

| 层级 | 含义 | 是否足够 |
| --- | --- | --- |
| metric pass | loss 或几何指标变好 | 不够 |
| visual pass | 3D 输出能看出更清楚的人体结构 | 有价值，但还要对照 |
| advisor pass | 同视角、同边界、同点大小下，human-main full-scene RGB point cloud 明显更好 | 最终目标 |

projection overlay、isolated human scatter、SMPL-X-only、teacher/reference 图都只能作为辅助证据。真正能作为主证据的，必须是 VGGT student 路线产生的 full-scene RGB point cloud。

---

## 当前状态

这个仓库处在持续实验阶段，目前最重要的价值是把 SMPL-X 人体先验接入 VGGT 的模型路线，并且把训练、对照、证据判断的边界固定下来。

后续实验如果要继续推进，重点不是单纯调 viewer 或截图角度，而是继续检查：

- 先验是否真的进入 VGGT student 路线；
- true prior 是否明显优于 random / shuffled control；
- full-scene 点云里人体结构是否比 vanilla VGGT 更清楚；
- 环境上下文是否仍然保留；
- 结论是否能被文件、图像、manifest 和对照实验支撑。

---

## 成果展示页面

下面两个页面用于展示阶段性成果和项目说明：

- 成果展示页面：https://www.yuque.com/maturinlove221/gqr279/emwf87ku108nzvez
- 成果展示页面：https://www.yuque.com/maturinlove221/gqr279/fg8lq33tgbwiagtt

---

## 配图文件

```text
docs/figures/vggt_smplx_human_prior_adapter_architecture.svg
docs/figures/parallel_engineering_result_snapshot.svg
docs/figures/external_reference_control.svg
```
