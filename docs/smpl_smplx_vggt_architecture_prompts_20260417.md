# 《SMPL / SMPL-X + VGGT 新版架构图生成词（V2.1）》

日期：`2026-04-19`

适用对象：语雀方法图、导师汇报方法图、Gemini / 图像生成模型 / 画图助手

## 一、画图前必须统一的口径

- 当前**代码主线**已经进入 `V2.1`：`pose-aligned SMPL-X driver + 17 通道 surface feature maps + 12 维 summary features + layer-wise fusion + 顶点身份增强 + pose-noise + 多尺度 human prior fusion`。
- 当前**已经整理好的完整 60-view 4K4D 结果图**，主要来自更早一版 `smplx surface pose-align` 全视角前向，可以支撑“路线有效”，但它们还不是最新 `V2.1` 代码完全重跑后的最终展示版。
- 因此，**方法图应该画当前代码的 V2.1 设计**，而**结果图可以继续使用现有 60-view pose-align 产物**，但要在正文或图注中说明“方法图口径更新到了 V2.1，现有完整 60-view 结果仍以较早 pose-align 跑次为主”。

## 二、新版架构图必须在视觉上回答的问题

1. 原版 VGGT 主干基本保留，没有被改写成纯 SMPL 参数回归器。
2. SMPL-X 不再只是输入端的稀疏 keypoint / silhouette 提示，也不再只是输出端监督器，而是一个 **pose-driven、surface-level、layer-wise** 的人体条件源。
3. 这组人体条件来自**与输入图像人体姿态对齐**的 SMPL-X pose，而不是静态 canonical 人体模板。
4. 条件信息分成两类：
   - 逐视角 dense surface feature maps：更偏局部、逐视角、逐 patch 的人体表面条件。
   - pooled human summary tokens：更偏跨视角、全局人体结构摘要。
5. layer-wise fusion 要明确体现在**多个 backbone block**里，而不是只在输入口 fuse 一次。
6. 输出侧 `depth prior / point prior / valid mask` 是 **training-time-only** 的几何监督，只作用于 depth / point，不应画成推理时新增主输出。
7. `cameras / tracks` 不是被 SMPL-X prior 直接监督的对象。
8. `valid mask` 是 supervision gating / reliability mask，不是独立预测终端。
9. 图中可以加一个小型 `V2.1 Enhancements` 标注，说明当前代码相对早期 pose-align 版本进一步加入了：
   - vertex identity / body-part / skinning embedding
   - pose-noise robustness on the input-condition branch
   - multi-scale human prior fusion

## 三、最容易画错的地方

- 不要再把左侧主分支画成 `silhouette + keypoint heatmap`。如果要出现，也只能作为早期 V1 的对比或小注释，不能再作为当前主方法图的核心输入分支。
- 不要把 SMPL-X prior 画成直接监督 `camera head` 或 `tracks`。
- 不要把 `valid mask` 画成一个新的主输出框。
- 不要把 posed mesh 画成由 VGGT 预测出来的结果。这里应明确是“外部对齐得到的 SMPL-X 参数驱动的 posed mesh”。
- 不要把 dense surface maps 和 summary tokens 混成一种东西。
- 不要把方法图画成“SMPL-X 取代 VGGT”；正确叙事是“VGGT backbone largely unchanged + human prior enhancement”。
- 不要把 `pose-noise` 画成破坏监督项的噪声；它只应该作为输入条件分支的鲁棒性增强小注释存在。

## 四、完整版中文 Prompt

请绘制一张**学术论文风格、CVPR / ICCV 论文 figure 风格**的横向方法结构图，主题是：

**Pose-aligned SMPL-X Surface Conditioning for VGGT Human Geometry Enhancement**

整张图从左到右展开，要求是**“原版 VGGT 主干基本保留，在其上新增显式的人体表面条件分支与训练期几何监督分支”**。整体视觉参考原版 VGGT 架构图风格：白色背景、serif 或简洁学术字体、细灰箭头、柔和但区分明确的配色、整洁的论文排版，不要画成海报风，不要用大面积渐变背景。

### 画面布局要求

请把整张图分成四个横向区域：

1. 左侧：多视角输入与 pose-aligned SMPL-X driver  
2. 中左：SMPL-X 顶点特征构造、逐视角 surface maps、summary tokens  
3. 中央：原版 VGGT backbone（大体不变）与 layer-wise fusion  
4. 右侧：原版输出头与 training-time-only 输出侧监督  

### 区域 1：多视角输入与 pose-aligned SMPL-X driver

最左侧放 3 到 5 张堆叠的人体多视角 RGB 小图，明确是**多视角人体**而不是建筑或普通场景，标注：

- `Multi-view RGB Inputs`
- `60-view example shown as multi-view human observations`

在 RGB 输入下方或左下方，新增一个醒目的高亮模块：

- `Pose-aligned SMPL-X Driver`

这个模块内部要画出：

- `aligned SMPL-X parameters`
- `pose / shape / expression`
- `calibrated real cameras`
- `posed SMPL-X mesh`

请明确表达：这里的 SMPL-X 不是静态模板，而是**与当前输入人体姿态对齐的可驱动人体表面**。可以用一条小注释写：

- `pose must align with the input human pose`

### 区域 2：顶点特征构造与人体条件表示

在 `posed SMPL-X mesh` 之后，画一个中间模块：

- `Vertex Feature Construction`

这个模块请明确分成两类信息，最好在同一个框内做两个小分栏：

左边分栏：

- `surface geometry cues`
- `density / visibility`
- `body-local xyz`
- `camera-space xyz`
- `uv / radius`

右边分栏：

- `vertex identity enhancement`
- `vertex-id encoding`
- `body-part embedding`
- `skinning-chain embedding`

这个模块后面分成两路输出：

#### 路径 A：逐视角 dense surface maps

画成一摞 feature-map 小图，标注：

- `Per-view Surface Feature Maps`
- `17-channel surface condition`
- `projected / rasterized under real cameras`

请让这些小图明显是**与图像视角对齐的 dense maps**，而不是抽象 token。可在图中小字列出几个代表通道：

- `visibility`
- `body-local xyz`
- `cam xyz`
- `uv`
- `vertex-id`
- `body-part`
- `skinning`

如果版面允许，可以在这里加一个小标签：

- `multi-scale maps: x1 / x2 / x4`

#### 路径 B：pooled human summary tokens

画成一小排 token 矩形，标注：

- `Human Summary Tokens`
- `pooled human structure summary`
- `12 bins x 12-d features`

可用小注释说明它们来自人体表面统计聚合，例如：

- `center / spread / occupancy + part / skinning summaries`

### 区域 3：VGGT backbone 与 layer-wise fusion

中央请保留原版主干风格，并用一个大的灰色或淡蓝色框包住，标注：

- `VGGT Backbone (largely unchanged)`

框内沿用原版 VGGT 的核心结构表达：

- `Visual / Patch Tokens`
- `Add Camera Token`
- `Frame Attention`
- `Global Attention`
- `x L Blocks`

请明确画出**两类人体条件在 backbone 内的多层融合**，不能只在输入前 fuse 一次：

1. 把 `Per-view Surface Feature Maps` 用橙色或暖色路径接到多个 `Frame Attention` 或局部几何相关 block  
   标注：
   - `dense surface adapter`
   - `residual / gated fusion`
   - `layer-wise local human conditioning`

2. 把 `Human Summary Tokens` 用青绿色或蓝绿色路径接到多个 `Global Attention` 或跨视角 block  
   标注：
   - `summary-token adapter`
   - `global human context`
   - `layer-wise structural conditioning`

请让图中能一眼看出：**dense maps 更偏逐视角局部条件，summary tokens 更偏跨视角全局人体结构条件**。

建议在主干内部画出多处小的 adapter / gate 模块，显示这是**持续层内融合**。  
请不要把它画成简单的一次性 concat。

如果版面允许，可在 backbone 上方或旁边加一个小框：

- `V2.1 Enhancements`

框内写：

- `vertex identity / body-part / skinning`
- `pose-noise robustness`
- `multi-scale human prior fusion`

这个小框只是说明当前代码比早期 pose-align 版更进一步，不要让它喧宾夺主。

### 区域 4：输出头与 training-time-only supervision

在右侧保留原版 VGGT 的输出头与输出：

- `Camera Head`
- `DPT Depth Head`
- `DPT Point Head`

输出包括：

- `Cameras`
- `Depth Maps`
- `World Point Maps`
- `Tracks`

其中 `Tracks` 可适当弱化，标注为：

- `unchanged branch`

在右下方再单独画一条**训练阶段专用**的人体几何监督路径，使用绿色或青绿色虚线框突出，标题为：

- `Training-time Output-side SMPL-X Geometry Supervision`

路径内容为：

- `posed SMPL-X mesh`
- `real-camera projection`
- `depth prior`
- `point prior`
- `valid mask`

请明确：

- `real-camera aligned`
- `training-time only`
- `human-region supervision`

这条监督路径只用**虚线箭头**连接到：

- `Depth Maps`
- `World Point Maps`

不要连接到 `Cameras`，也不要连接到 `Tracks`。

请把 `valid mask` 画成一个 supervision gating 模块或 reliability mask，小字标明：

- `supervision gating, not a standalone output`

### 图中必须出现的全局含义

请让整张图能一眼传达以下结论：

- 原版 VGGT 主干仍然是主体。
- SMPL-X 被提升为 pose-driven、surface-level、layer-wise 的人体条件源。
- 条件源不是静态模板，而是由与输入姿态对齐的 SMPL-X mesh 驱动。
- dense surface maps 与 summary tokens 分工不同但互补。
- 输出侧 prior 仍然保留，但它只是训练时的人体几何监督。
- 整个方法不是“RGB 直接回归 SMPL-X 参数”，而是“VGGT + human geometry prior enhancement”。

### 颜色与视觉风格建议

- 原版 VGGT 主干：灰蓝、浅黄、浅绿等柔和学术配色。
- dense surface map 路径：暖橙色或琥珀色高亮。
- summary token 路径：蓝绿色或青色高亮。
- output-side supervision 路径：绿色虚线框或绿色高亮。
- 所有新增模块用虚线边框或高亮边框，原版模块保持实线和更克制的颜色。
- 白底、细灰箭头、少量阴影、整洁排布。

### 最后的禁止项

- 不要画成“silhouette + keypoint heatmap 是主角”的旧版方法图。
- 不要把 SMPL-X prior 画成直接生成最终点云。
- 不要把输出侧 prior 画成 inference output。
- 不要画成抽象的大橙框和几条模糊箭头；要具体表现 `pose-aligned driver`、`vertex feature construction`、`surface maps`、`summary tokens`、`layer-wise fusion`、`training-time supervision`。

## 五、简洁英文 Prompt

Draw a clean CVPR-style method figure for a human-enhanced VGGT model. The central message is: **the original VGGT backbone is largely preserved, while SMPL-X is upgraded into a pose-aligned, surface-level, layer-wise human conditioning branch**.

Use a left-to-right academic layout with white background, thin gray arrows, serif-like paper labels, and soft pastel colors.

The figure must include:

1. `Multi-view RGB Inputs` with stacked multi-view human images.
2. A highlighted `Pose-aligned SMPL-X Driver` showing aligned SMPL-X parameters, pose/shape/expression, calibrated cameras, and a posed SMPL-X mesh.
3. A `Vertex Feature Construction` module that combines:
   - surface geometry cues: density, visibility, body-local xyz, camera-space xyz, uv, radius
   - identity enhancement cues: vertex-id encoding, body-part embedding, skinning-chain embedding
4. Two human-condition outputs:
   - `Per-view Surface Feature Maps` labeled as `17-channel surface condition`, projected/rasterized under real cameras
   - `Human Summary Tokens` labeled as pooled human structure summaries, e.g. `12 bins x 12-d`
5. A central `VGGT Backbone (largely unchanged)` with patch/visual tokens, camera token, alternating `Frame Attention` and `Global Attention`, repeated for multiple blocks.
6. Explicit `layer-wise fusion` into multiple backbone blocks:
   - dense surface-map adapters feeding several frame-attention stages
   - summary-token adapters feeding several global-attention stages
   - residual / gated fusion rather than one-time concatenation
7. Original output heads on the right:
   - `Camera Head`
   - `DPT Depth Head`
   - `DPT Point Head`
   with outputs `Cameras`, `Depth Maps`, `World Point Maps`, and a visually de-emphasized `Tracks`.
8. A dashed `Training-time Output-side SMPL-X Geometry Supervision` branch from the posed SMPL-X mesh through `real-camera projection` to `depth prior`, `point prior`, and `valid mask`, connected only to depth-map and point-map outputs.

Clearly show that:

- SMPL-X is not a direct replacement for VGGT.
- The method is not direct RGB-to-SMPL-X regression.
- `valid mask` is supervision gating, not a final output.
- `Cameras` and `Tracks` are not directly supervised by SMPL-X priors.

Optionally add a small `V2.1 Enhancements` callout with:

- vertex identity / body-part / skinning
- pose-noise robustness
- multi-scale human prior fusion

Avoid the old V1 story of `silhouette + keypoint heatmap` as the main human branch.

## 六、保守简化版 Prompt

请画一张简化但逻辑绝对正确的论文风格方法图，重点只表达下面六件事：

1. 左侧是多视角人体 RGB 输入。
2. 左下新增 `Pose-aligned SMPL-X Driver`，由对齐后的 SMPL-X 参数和真实相机驱动出 `posed SMPL-X mesh`。
3. mesh 被投影成两类人体条件：
   - `Per-view Surface Feature Maps`
   - `Human Summary Tokens`
4. 中央是 `VGGT Backbone (largely unchanged)`，并在多个 block 中用小 adapter 表示 `layer-wise fusion`。
5. 右侧仍然输出 `Cameras / Depth Maps / World Point Maps / Tracks`。
6. 右下是 `training-time-only` 的 `depth prior / point prior / valid mask` 监督路径，只连到 `Depth Maps` 和 `World Point Maps`。

图中不要再出现 `silhouette + keypoint heatmap` 主分支，不要把 SMPL-X 画成直接预测最终点云，也不要把 `valid mask` 画成单独输出。
