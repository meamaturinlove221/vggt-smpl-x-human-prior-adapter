# 基于 Full VGGT Forward 与 SMPL-X 三维结构先验的真实 3D 人体场景点云补全

# 先给结论

当前状态：`V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION`。

这不是 mentor-ready，也不是 promotion。V300 已降级为 checkpoint。当前可访问证据不能支撑“导师最终通过”，原因不是 zip 损坏，而是导师主视觉门控没有过：3D 主图缺少可信 full-scene 环境，V190 仍是旧预测重评分，local close-up 不能证明真实五官/手型/衣物边界细节。

# 一、为什么 V300 仍需降级

- 上传包内曾出现 final status / requirement audit / advisor report 口径冲突。
- V140/V420 的 3D 主图仍不能稳定证明“人体为主体且保留真实环境”。
- Projection overlay 只能辅助，不能替代 3D morphology。
- V160/V500 local close-up 还不能包装成 facial detail、hand detail 或 clothing boundary detail。
- V190 matrix 是 V740 predictions 的重评分，不是 fresh true 3D morphology student。

# 二、本轮路线定位

本轮把门控重新拉回导师原始要求：

```text
Full VGGT outputs
        +
SMPL-X surfel / voxel / graph
        ->
true 3D morphology student
        ->
human-main full-scene RGB point cloud
```

Projection、mask/RGB/edge score、source-label 全部只能作为辅助。

# 三、当前变化

- 已保存 V300100-V900 目标文件和 manifest。
- 已冻结并降级 V300。
- 已生成 V310/V320/V330/V340/V350 审计，明确当前 evidence 不足。
- 已生成 V850 auto-evolved next route：`D:\vggt\vggt-canonical-surfel-adapter\docs\goals\V850000000000000000_auto_evolved_true_3d_route.md`。

# 四、导师主图边界

当前 `boards/V420000000000000000_advisor_true_3d_main.png` 是诊断图，不是最终导师主图。它比旧 V140 更直立，但环境仍不足，所以 fail closed。

# 五、局部细节边界

当前只能写 head/hair/body contour，不得写 facial detail。hand/arm 与 clothing 也不能包装成真实手型或衣物边界提升。

# 六、Controls

Controls 框架保留，但当前仍不能用 V190 copied-prediction rescoring 证明 fresh model 优势。下一轮必须用 V850/V860 新矩阵。

# 七、下一步

执行 V850 auto-evolved route：canonical SMPL-X surfel/graph + real VGGT full-scene environment + true 3D local closeups。只有新的 3D 主图过导师视觉门控，才能返回 mentor-ready。

# 八、给导师看的文件

- `reports/V800000000000000000_final_mentor_gate.json`
- `reports/V800000000000000000_failed_gate_router.json`
- `docs/goals/V850000000000000000_auto_evolved_true_3d_route.md`
- `reports/V900000000000000000_final_status.json`
