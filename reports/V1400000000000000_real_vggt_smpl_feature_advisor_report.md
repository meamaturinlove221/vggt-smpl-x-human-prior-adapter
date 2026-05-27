# 基于 Real VGGT Token Binding 的 SMPL-X 人体结构先验：面向导师视觉门控的 Full-Scene RGB Point Cloud 补全

# 先给结论

当前状态：`V1800000000000000_REAL_VGGT_SMPL_FEATURE_DETAIL_MENTOR_READY_NOT_PROMOTED`。

不 promotion，不改 registry，不改 V50/V50R2，active candidate 仍为 `V11700_gap_reduction_branch_520`。

主图：`boards\V970000000000000_real_vggt_advisor_main_board.png`。

# 一、为什么 V900 还不够

V900 有人体主形和 full-scene board，但 TinyV330/synthetic token 风险没有消除，source-label/visible-delta 只能辅助，不能替代导师看的 RGB point cloud。

# 二、路线定位

本轮从真实 4K4D SMC RGB 解码输入，执行当前 repo 的 `VGGT.forward` / `Aggregator.forward`，再把 V940 的 SMPL-X surfel/voxel/graph/body-part/visibility/projection feature 绑定到 real VGGT token path。

# 三、架构图

```text
RGB / mask / camera
        ->
Real VGGT tokens
        +
SMPL-X 3D feature tokens
        ->
token-bound adapter
        ->
human-main full-scene RGB point cloud
```

# 四、主图证据

- full-scene main: `boards\V970000000000000_real_vggt_advisor_main_board.png`
- same-scene controls: `boards\V970000000000000_same_scene_controls_board.png`
- multi-sequence summary: `boards\V970000000000000_cloudcompare_style_board.png`

# 五、局部细节

- head/face/hair: `boards\V980000000000000_head_face_hair_detail_board.png`
- hand/arm: `boards\V980000000000000_hand_arm_detail_board.png`
- clothing boundary: `boards\V980000000000000_clothing_boundary_board.png`

# 六、Controls 和 claim 边界

True beats real VGGT baseline, posthoc surfel, same-topology/no-semantic, tiny synthetic token control, and source-label-only control in the V960 Modal matrix. If a future stronger topology-only control catches up, the claim must be downgraded to representation/topology contribution.

# 七、边界

Not promotion. Not paper-grade generalized. Local face detail is claimed as head/face contour and hair-region non-regression, not photo-level facial features.

# 八、给导师看的文件

See V160 bundle sidecar and viewer `viewer\V1600000000000000_real_vggt_smpl_feature_viewer.html`.
