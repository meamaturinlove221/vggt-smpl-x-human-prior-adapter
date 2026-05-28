# V10180 Depth-Cue And Geometry Repair Decision

结论：用户指出的二维化问题成立。

## 渲染问题

V10150 的主图渲染只使用 `points[:, :2]`，再按 `z` 排序，没有斜视透视、深度着色或侧向厚度提示。因此原图天然会像二维贴片，不能作为最终导师视觉板。

## 几何问题

V10170 的斜视深度板已经补了深度线索：

`D:\vggt\vggt-canonical-surfel-adapter\boards\V10170000000000000000_0012_11_frame001_oblique_depth_pointcloud_audit.png`

但候选本身没有比 baseline 明显更 3D：

- baseline PCA thickness ratio: `0.358275`
- candidate PCA thickness ratio: `0.360287`
- thickness gain: `0.002012`
- baseline z range: `0.203594`
- candidate z range: `0.200056`
- z range gain: `-0.003538`

所以不能只继续调 viewer/截图角度。渲染需要修，但主路线必须回到表示和几何重构。

## 下一步

进入 V10180/V10190：把所有导师板换成 oblique/depth-cued 3D 渲染作为检查工具，同时重构 candidate 的 3D morphology：canonical SMPL-X surfel/graph 支撑可见弱区，增加 thickness / side-view / limb-continuity 目标，保留真实 VGGT 环境。Projection 仍然只能是辅助。

Face detail 仍不适用；只允许写 `head/face contour and hair region only`。
