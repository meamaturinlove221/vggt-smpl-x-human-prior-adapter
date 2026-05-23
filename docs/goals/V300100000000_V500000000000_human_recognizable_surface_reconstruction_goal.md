# V300100000000–V500000000000 Human-Recognizable Surface Reconstruction Goal

你现在接手 D:\vggt\vggt-feature-adapter 的下一阶段超长期任务。

repo 固定为：

D:\vggt\vggt-feature-adapter

evidence root 固定为：

D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild

二者不能混写。

本轮允许使用 /goal，但本文件是主执行规范。
本轮严禁发起 agent / subagent。
只能由当前主线程长期执行、自动路由、自动修复、自动续跑。

重要原则：

V300000000000_POINT_CLOUD_MORPHOLOGY_READY_NOT_PROMOTED 不能作为最终导师视觉通过。
它只说明 morphology metric gate 通过。
导师要求看点云形状，当前图仍像稀疏 blob / sheet / sparse clusters，不像完整人体 3D 点云。
所以必须继续执行，直到 Human-Recognizable Shape Gate 通过。

当前事实必须接受：

1. V300000000000 已完成。
2. 最终状态是：
   V300000000000_POINT_CLOUD_MORPHOLOGY_READY_NOT_PROMOTED
3. 当前 true group:
   true_surface_guided_morphology_repair
4. 当前 final true rank = 1。
5. 当前真实 SMC projection 使用：
   0021_03_annots.smc
6. 当前 final inside-mask ratio:
   0.232004
7. original true inside-mask ratio:
   0.198833
8. local smoothing inside-mask ratio:
   0.178999
9. full_body / head_face / hairline / left_hand / right_hand 在指标上 passed。
10. selected/control NPZ readable。
11. sampled PLY headers verified。
12. no promotion。
13. no registry。
14. no V50/V50R2 changes。
15. active candidate unchanged:
    V11700_gap_reduction_branch_520
16. worktree dirty must be honestly reported。

但当前仍有关键问题：

1. V250 final_fullbody 图仍像稀疏椭圆/片状点云，不像完整人体。
2. true_surface_guided_morphology_repair 有横向漂浮/条带状点。
3. V250 final_parts 中 head/hair/hand 仍是稀疏点簇，看不出局部人体结构。
4. V250 final_projection_overlay 是 ratio heatmap，不是真正点云投影到 RGB/mask 上。
5. inside-mask ratio 只有约 0.23，不足以证明点云贴合人体 silhouette。
6. 当前 morphology_score 是统计指标，不等于导师肉眼可识别的人体点云。
7. 不能把 V300 作为最终可交导师视觉版本。

本轮总目标：

从 morphology metric ready 升级为：

Human-Recognizable Surface Reconstruction Ready

也就是：

导师看图时必须能明显看出：
- baseline 点云哪里不完整；
- true route 得到更像人体的 3D 结构；
- random / shuffled / smoothing / support / observation 没有同等人体形态；
- full body 能看出头、躯干、手臂、腿、朝向；
- head/hair/hand 局部不再只是稀疏点簇；
- 点云投影回真实 mask/RGB 后贴合人体 silhouette；
- 如果生成 surface/mesh，必须不只是 SMPL 直接替换，而是 VGGT observation + SMPL structure 的补全结果。

允许最终返回状态只有两个：

A. V500000000000_HUMAN_RECOGNIZABLE_SURFACE_READY_NOT_PROMOTED

必须满足：
- full-body human-recognizable shape gate passed
- head/hair/hand recognizable local structure passed
- true route visually better than V11700 and controls
- projection overlay on real SMC mask/RGB passed
- silhouette inside ratio significantly improved and not trivial
- background leakage bounded
- SMPL/skeleton overlay confirms surface consistency
- controls do not produce same human shape
- selected/control NPZ readable
- PLY/mesh samples readable
- Yuque-style report complete
- upload-safe bundles complete
- cleanup complete
- no promotion / no registry / no V50 changes
- active candidate unchanged

B. V500000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

只允许：
- required predictions missing/corrupted
- SMPL-X mesh/camera/mask assets missing
- Modal/GPU unavailable when training required
- disk/permission/Git hard block
- user must manually move/authorize files

任何其他状态不能返回。
如果 human-recognizable shape gate failed，必须自动进入 repair route。

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止把 morphology_score rank=1 当 human-recognizable success。
- 禁止把 inside-mask ratio heatmap 当 projection overlay。
- 禁止只出 2D delta map。
- 禁止只出 sparse scatter blob。
- 禁止只出 bar chart。
- 禁止复用旧图。
- 禁止没有 equal-axis。
- 禁止没有 body-part coloring。
- 禁止没有 SMPL/skeleton overlay。
- 禁止没有 true-vs-controls same-scale comparison。
- 禁止 dirty worktree 时声称 clean。
- 禁止中途 limitation / route exhausted 返回。
- 禁止只写计划不执行。
- 禁止 sleep 伪造长时间运行。

最低运行原则：

- 非 true external hard block，不允许短于 12 小时返回。
- 若数据和 Modal 允许，目标运行 24–96 小时。
- runtime 必须来自真实点云重建、surface completion、可视化、训练/修复、打包、审计。
- 所有阶段必须有 progress JSON、decision JSON、source manifest。
- 如果某一路线失败，必须自动生成下一代 human-shape repair route 并继续执行。

============================================================
V300100000000：V300 morphology visual audit
============================================================

读取并审计：

- reports/V300000000000_final_status.json
- reports/V300000000000_completion_audit.json
- reports/V260000000000_morphology_advisor_report.md
- reports/V250000000000_final_morphology_decision.json
- reports/V250000000000_final_morphology_metrics.csv
- reports/V204000000000_projection_morphology_metrics.csv
- boards/V250000000000_final_fullbody.png
- boards/V250000000000_final_parts.png
- boards/V250000000000_final_projection_overlay.png
- pointcloud samples PLY
- selected/control predictions

必须自动判定：

1. final_fullbody 是否像完整人体。
2. 是否可见头、躯干、手臂、腿。
3. 是否有异常横向漂浮点/条带。
4. part close-up 是否足够显示 head/hair/hand。
5. projection overlay 是否是真 overlay，还是 heatmap。
6. inside-mask ratio 是否足够高。
7. true 与 controls 是否肉眼可分。
8. 是否存在 SMPL直接补模板的嫌疑。
9. 是否需要 surface completion。

输出：

- reports/V300100000000_visual_shape_audit.json
- reports/V300100000000_human_recognizability_failure.md
- reports/V300100000000_next_shape_requirements.json

硬门：
如果当前图仍不像完整人体，必须进入 V310。
不得返回。

============================================================
V310000000000：dense point cloud reconstruction v2
============================================================

目标：
从 current sampled/fused point cloud 升级到 dense, human-readable point cloud。

输入：

- V11700 baseline
- true_camera_bound_surface_backend
- true_surface_guided_morphology_repair
- random_surface_semantic
- shuffled_surface_semantic
- local_knn_smoothing
- support_only
- observation_only
- no_sparseconv_mlp
- confidence
- normal
- body part map
- SMPL-X mesh/skeleton
- 0021_03 camera/masks

必须实现：

1. Fuse all valid points, not only sampled 18k.
2. Multi-threshold fusion:
   - high confidence
   - medium confidence
   - all human-mask points
3. Outlier rejection:
   - statistical outlier removal
   - radius outlier removal
   - mask projection gating
4. Duplicate removal:
   - voxel downsample
   - view-aware merge
5. Preserve local details:
   - head/hair/hand should not be over-smoothed away
6. Export:
   - dense .ply
   - sampled .ply
   - npz with metadata
   - per-part point clouds

输出：

- tools/v310000000000_dense_pointcloud_reconstruction.py
- output/V310000000000_dense_pointclouds/*.ply
- output/V310000000000_dense_pointclouds/*.npz
- reports/V310000000000_dense_pointcloud_inventory.csv
- reports/V310000000000_reconstruction_policy.json

硬门：
If dense point clouds still look like blob, enter V330.

============================================================
V320000000000：true projection overlay, not heatmap
============================================================

目标：
生成真正的投影叠加图，不再只是 inside-mask ratio heatmap。

For each camera 00/01/15/30/45/59:

1. Load real SMC mask.
2. If RGB available, load RGB.
3. Project 3D point clouds onto image.
4. Overlay:
   - mask boundary
   - projected points
   - point color by group / region / confidence
5. Compare:
   - V11700
   - true repair
   - random
   - smoothing
   - support
   - observation
6. Generate per-camera panels.

输出：

- boards/V320000000000_projection_overlay_camera00.png
- boards/V320000000000_projection_overlay_camera01.png
- boards/V320000000000_projection_overlay_camera15.png
- boards/V320000000000_projection_overlay_camera30.png
- boards/V320000000000_projection_overlay_camera45.png
- boards/V320000000000_projection_overlay_camera59.png
- boards/V320000000000_projection_overlay_grid.png
- reports/V320000000000_projection_overlay_metrics.csv

硬门：
If true projects mostly outside mask or still weak, enter V330/V350 repair.

============================================================
V330000000000：SMPL-X surface-guided human shape completion v2
============================================================

目标：
生成可识别人体形状，但不能简单 SMPL 替换。

方法：

1. Use SMPL-X as structural support.
2. Sample SMPL-X visible surface points.
3. For each SMPL point, estimate whether it is supported by:
   - VGGT observation
   - confidence
   - silhouette
   - nearest prediction point
   - normal agreement
4. Add completion points only where:
   - silhouette supports them
   - region density is low
   - background leakage is bounded
5. Preserve non-template residual:
   - hairline offset
   - clothing residual
   - hand local residual
6. Generate:
   - completed fullbody point cloud
   - completed part point clouds
   - confidence map
   - source label per point:
     observed / completed / rejected / control

Controls:
- SMPL-only
- random semantic completion
- smoothing completion
- no observation completion

输出：

- tools/v330000000000_surface_guided_human_shape_completion_v2.py
- output/V330000000000_completed_pointclouds/*.ply
- reports/V330000000000_completion_source_stats.csv
- reports/V330000000000_completion_eval.json
- boards/V330000000000_completion_visual.png

硬门：
If result is just SMPL-only template replacement, fail.
If controls also improve same way, fail.
If true improves human shape uniquely, continue.

============================================================
V340000000000：mesh / surface reconstruction route
============================================================

如果点云仍太稀疏，尝试 surface reconstruction。

Methods:
1. Poisson reconstruction from true point cloud + normals.
2. Ball pivoting if feasible.
3. Alpha shape / marching cubes if SDF exists.
4. SMPL-guided surface patch completion.
5. Export mesh and sampled point cloud.

Controls:
- V11700
- random
- smoothing
- SMPL-only

输出：

- output/V340000000000_meshes/*.ply
- output/V340000000000_mesh_sampled_pointclouds/*.ply
- reports/V340000000000_mesh_reconstruction_eval.json
- boards/V340000000000_mesh_visual.png

硬门：
If mesh creates artifacts or template-only, fail and continue V350.

============================================================
V350000000000：part-local high-density completion
============================================================

目标：
解决 head/hair/hand 仍是稀疏点簇的问题。

Regions:
- head_face
- hairline
- left_hand
- right_hand

Methods:
1. local surface patch sampling
2. local residual offset
3. confidence-gated densification
4. mask projection check
5. connected component cleanup
6. local normal smoothing
7. hand/hair boundary preservation

Controls:
- random semantic
- smoothing
- support only
- SMPL-only

输出：

- output/V350000000000_part_completed_pointclouds/*.ply
- boards/V350000000000_head_face_dense.png
- boards/V350000000000_hairline_dense.png
- boards/V350000000000_left_hand_dense.png
- boards/V350000000000_right_hand_dense.png
- reports/V350000000000_part_completion_eval.csv

硬门：
Each part must become visually interpretable or limitation triggers next repair.

============================================================
V360000000000：human-recognizable visual boards
============================================================

生成导师可看的最终大图。

必须包含：

1. full body turntable board:
   - front
   - back
   - side
   - top
   - oblique
   - equal-axis
   - body-part colors
   - SMPL skeleton overlay

2. true vs baseline:
   - baseline missing
   - true completed
   - delta/highlight completed points

3. true vs controls:
   - random
   - shuffled
   - smoothing
   - support
   - observation
   - SMPL-only

4. part closeups:
   - head_face
   - hairline
   - left_hand
   - right_hand

5. projection overlay:
   - real SMC mask/RGB if available
   - projected points
   - inside/outside colors

6. source label:
   - observed points
   - completed points
   - rejected/outlier points

输出：

- boards/V360000000000_fullbody_turntable.png
- boards/V360000000000_true_vs_baseline_completion.png
- boards/V360000000000_true_vs_controls_human_shape.png
- boards/V360000000000_part_closeups_dense.png
- boards/V360000000000_projection_overlay_real.png
- boards/V360000000000_source_label_visual.png

硬门：
If a human viewer cannot recognize rough body/head/hands from fullbody and closeups, fail.

============================================================
V370000000000：human-recognizable metrics
============================================================

Compute human-shape metrics:

Full body:
- skeleton coverage
- SMPL surface coverage
- torso/head separation
- limb separation
- hand/torso separation
- silhouette projection coverage
- inside-mask ratio
- background leakage
- completed vs observed ratio
- outlier ratio

Head/hair/hands:
- local point count
- local component count
- local surface coverage
- nearest surface distance
- projection inside ratio
- boundary coverage
- hand compactness
- hairline boundary continuity

Human recognizability:
- no blob score
- no floating streak score
- humanoid part visibility score
- true vs controls separability

输出：

- reports/V370000000000_human_shape_metrics.csv
- reports/V370000000000_human_shape_control_ranking.csv
- reports/V370000000000_human_shape_summary.json

============================================================
V380000000000：human-recognizable decision
============================================================

HUMAN_RECOGNIZABLE_SURFACE_READY requires:

1. fullbody recognizable.
2. head/hair/hand local structure recognizable.
3. true beats V11700.
4. true beats random/shuffled/smoothing/support/observation/SMPL-only.
5. projection overlay real and passed.
6. completed points not mostly background.
7. source labels show not pure SMPL replacement.
8. part dense completion not template-only.
9. selected/control PLY readable.
10. report and visuals ready.

If passed:
enter V450 report/package.

If failed:
classify:
- blob_shape
- floating_streaks
- sparse_parts
- projection_fail
- template_overfit
- controls_too_close
- no_rgb_overlay
- data_insufficient

Then enter V390 auto repair.

输出：

- reports/V380000000000_human_recognizable_decision.json
- reports/V380000000000_failure_attribution.md

============================================================
V390000000000：automatic human-shape repair loop
============================================================

If V380 fails, do not return.

Route A: blob_shape
- stronger skeleton constraint
- PCA/alignment correction
- surface sampling cleanup
- outlier rejection

Route B: floating_streaks
- reject unsupported completion points
- projection gating
- distance-to-surface cap
- component cleanup

Route C: sparse_parts
- high-density part completion
- local mesh patch
- hand/hair specialist v3

Route D: projection_fail
- recalibrate binding
- per-view projection refinement
- mask erosion/dilation robustness

Route E: template_overfit
- require observed support
- residual preservation
- compare SMPL-only control

Route F: controls_too_close
- stronger true semantic dependence
- contrastive control loss
- random/shuffled hard negatives

Each route must:
- implement repair
- rerun V330–V380
- if fails, try next route

If all fail:
- auto-generate next route file:
  docs/goals/V390000000000_auto_next_human_shape_route.md
- execute it automatically
- do not return.

输出：

- reports/V390000000000_repair_history.csv
- reports/V390000000000_auto_next_route.json

============================================================
V450000000000：Yuque-style advisor report
============================================================

Write Chinese report in Yuque-style.

Structure:

# 先给结论

明确：
- camera-bound metric 已通过；
- point-cloud morphology 已通过；
- human-recognizable surface 是否通过；
- 是否可以给导师看；
- 不 promotion。

# 为什么又补这一轮

说明：
V300 morphology metric passed，但图仍不像完整人体，因此新增 human-recognizable gate。

# 方法

说明：
- dense point cloud fusion
- surface-guided completion
- part-local densification
- mesh/surface route if used
- projection overlay
- controls

# 点云形态证据

分：
- full body
- head_face
- hairline
- left_hand
- right_hand

# 真实投影证据

说明：
- 0021_03 SMC
- cameras
- inside mask
- leakage
- overlays

# Controls

说明：
random/shuffled/smoothing/support/observation/SMPL-only。

# 仍然限制

必须诚实写：
- single true-match sequence
- SMPL proxy/support
- not promotion
- possible template bias

# 给导师看的图

列所有关键图片路径和用途。

输出：

- reports/V450000000000_human_shape_advisor_report.md
- reports/V450000000000_one_page.md
- reports/V450000000000_limitations.md

============================================================
V470000000000：upload-safe packaging
============================================================

打包：

- core evidence <=50MB
- reports <=50MB
- visuals <=150MB
- selected predictions <=250MB
- controls <=250MB
- dense pointcloud samples <=250MB
- mesh samples <=250MB if available
- omitted manifest
- authoritative sidecar manifest

必须：
- zip clean
- inner npz readable
- PLY readable
- mesh readable if included
- hashes match
- final status in core
- cleanup in core
- no self-referential hash issue

输出：

- archive/V470000000000_core_evidence_bundle.zip
- archive/V470000000000_reports_bundle.zip
- archive/V470000000000_visuals_bundle.zip
- archive/V470000000000_selected_predictions_bundle.zip
- archive/V470000000000_controls_bundle.zip
- archive/V470000000000_pointcloud_samples_bundle.zip
- archive/V470000000000_mesh_samples_bundle.zip
- reports/V470000000000_upload_manifest_sidecar.json
- reports/V470000000000_omitted_large_file_manifest.json

============================================================
V490000000000：post-push cleanup
============================================================

commit/push 后检查：

- git status clean or honestly dirty
- branch
- commit
- remote contains commit or patch bundle exists
- Modal apps
- Python workers
- registry diff
- V50/V50R2 diff
- active candidate
- staged/untracked files if dirty

输出：

- reports/V490000000000_post_push_cleanup.json

============================================================
V500000000000：final return gate
============================================================

Only two final return states are allowed:

1. V500000000000_HUMAN_RECOGNIZABLE_SURFACE_READY_NOT_PROMOTED

Requires:
- human recognizable shape decision passed
- fullbody visual passed
- part visual passed
- projection overlay passed
- true beats controls
- advisor report complete
- bundles complete
- cleanup complete
- no promotion / no registry / no V50 changes
- active candidate unchanged

2. V500000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

Requires:
- all automatic human-shape repair routes attempted or impossible
- user action checklist precise
- evidence bundles complete
- cleanup complete

Any other state cannot return.
If failed, return to V390 auto repair loop.
