# V880 Auto-Evolved Canonical Surfel Residual Route

当前结论：

V870 证明 baseline-preserving 比 V860 的 global SMPL remap 更合理，但仍不能作为导师最终通过。

核心失败：

1. true 与 VGGT baseline 在 3D 主图中仍然太接近；
2. 局部 close-up 仍主要是轮廓级，不能写五官/手型/衣物细节；
3. 继续调采样预算会原地撞墙；
4. 必须切换到 canonical SMPL-X surfel residual / weak-region completion 表示。

下一轮核心：

RGB/mask/camera/VGGT full-forward outputs
        +
VGGT baseline high-confidence human points
        +
canonical SMPL-X surfel/graph bank
        ->
only-missing-or-weak-region residual completion
        ->
full-scene RGB point cloud with real environment

硬门：

- 主证据仍是 3D full-scene RGB point cloud；
- projection/metrics 只作辅助；
- no agent/subagent；
- no promotion/registry/V50 change；
- 不得把 V124/V125 旧 surfel 单序列结果当最终，只能作为结构表示参考；
- 如果四序列不能稳定显示 true > baseline/controls，继续 TRUE_EXTERNAL_HARD_BLOCK。
