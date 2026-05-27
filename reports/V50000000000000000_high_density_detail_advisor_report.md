# 基于 Full VGGT Forward 与 SMPL-X 结构先验的高密度人体场景点云补全

# 先给结论

当前状态：`V60000000000000000_HIGH_DENSITY_DETAIL_FIDELITY_MENTOR_READY_NOT_PROMOTED`。不 promotion，active candidate unchanged。导师主图：`boards/V30000000000000000_advisor_high_density_main.png`。

# 一、为什么 V200 仍需降级

V200 的 full-forward smoke 是重要进步，但不是 per-case full proof；V200 人体点云约 2k 点，controls still close，local detail only 2/4 improvement。

# 二、本轮路线定位

本轮完成 per-case full-forward effect、60k high-density human points、SMPL-X feature binding、detail-fidelity branch 和 visible environment branch。

# 三、架构图

```text
RGB/mask/camera
    -> full VGGT forward tokens and outputs
    + SMPL-X 3D feature bank
    -> high-density detail adapter
    -> human-main full-scene RGB point cloud
```

# 四、导师主图

- high-density main: `boards/V30000000000000000_advisor_high_density_main.png`
- same-scene controls: `boards/V30000000000000000_same_scene_high_density_controls.png`
- environment: `boards/V32000000000000000_environment_realism_v3.png`
- viewer: `viewer/V60000000000000000_high_density_viewer.html`

# 五、局部细节

只声明 head/hair/face contour、hand/arm、clothing boundary，不夸成五官。

# 六、Controls

包括 posthoc、same topology、tiny、source-label only、baseline highconf detail only、scaffold-only 等。

# 七、边界

not promotion；not paper-grade generalized；multi-sequence limits and local detail limits remain.

# 八、给导师看的文件

- `reports/V40000000000000000_final_high_density_mentor_gate.json`
- `reports/V55000000000000000_bundle_integrity.json`
- `reports/V60000000000000000_final_status.json`
