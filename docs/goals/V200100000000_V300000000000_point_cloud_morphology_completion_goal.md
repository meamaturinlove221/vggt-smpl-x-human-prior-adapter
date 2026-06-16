# V200100000000–V300000000000 Point Cloud Morphology Completion Goal

你现在接手 D:\vggt\vggt-feature-adapter 的下一阶段超长期任务。

repo 固定为：

D:\vggt\vggt-feature-adapter

evidence root 固定为：

D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild

二者不能混写。

本轮允许使用 /goal，但本文件是主执行规范。
本轮严禁发起 agent / subagent。
只能由当前主线程长期执行、自动路由、自动修复、自动续跑。

当前事实必须接受：

1. V200000000000 已经完成。
2. 当前状态是：
   V200000000000_PAPER_GRADE_CAMERA_BOUND_SURFACE_BACKEND_READY_NOT_PROMOTED
3. 但这只说明 camera-bound score / report-level evidence 通过。
4. 当前 3D point-cloud morphology 证据不足。
5. V127 fullbody 图看起来像稀疏椭圆点团，不像完整人体点云。
6. V127 head/hair/hand close-up 点云过稀，无法证明人体局部形状完整。
7. V124 part specialist 的 separability 指标提升，但肉眼图仍不够强。
8. 导师要求看的是点云形状，不只是 camera-bound score。
9. 因此当前不能把“人体 3D 点云完整性”写成已证明。
10. 当前最多可写：
    camera-bound metric paper-grade passed,
    point-cloud morphology paper-grade not yet proven.
11. No promotion。
12. No strict registry。
13. No V50/V50R2 modification。
14. Active candidate remains V11700_gap_reduction_branch_520。
15. Worktree dirty must be honestly disclosed.

本轮总目标：

把 camera-bound metric evidence 升级成 point-cloud morphology evidence。

核心目标：

导师看图时必须能看出：
1. V11700 baseline 人体点云哪里不完整。
2. true route 如何让人体点云更完整。
3. random/shuffled/smoothing/support/observation 没有同等形态改善。
4. full body 具有可辨认人体结构。
5. head_face / hairline / left_hand / right_hand 具有更清晰局部点云结构。
6. 点云贴合 SMPL-X surface / skeleton / camera silhouette，不是背景噪点。
7. normal / curvature / density 不只是指标，而能辅助解释点云形态。

允许最终返回状态只有两个：

A. V300000000000_POINT_CLOUD_MORPHOLOGY_READY_NOT_PROMOTED

必须满足：
- full-body 3D morphology gate passed
- head-face morphology gate passed
- hairline morphology gate passed
- left-hand morphology gate passed
- right-hand morphology gate passed
- true route beats controls in morphology metrics
- true route beats controls in visual boards
- projection overlay confirms silhouette consistency
- SMPL overlay confirms surface consistency
- selected/control NPZ readable
- report complete
- bundles complete
- cleanup complete
- no promotion / no registry / no V50 changes
- active candidate unchanged

B. V300000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

只允许：
- required predictions missing or corrupted
- SMPL-X mesh/camera/mask assets missing
- Modal/GPU unavailable when training required
- disk/permission/Git hard block
- user must manually move/authorize files

任何其他状态不能返回。
如果 morphology gate failed，必须自动进入 repair route。

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止把 camera-bound rank=1 当作 morphology success。
- 禁止把 V127 当前图直接当作导师点云形状证据。
- 禁止只靠 mean score / bar chart。
- 禁止只出 2D delta map。
- 禁止只出 projection overlay。
- 禁止复用旧图当新图。
- 禁止没有 equal-axis 的 3D scatter。
- 禁止没有 body-part coloring。
- 禁止没有 SMPL/skeleton overlay。
- 禁止 dirty worktree 时声称 clean。
- 禁止中途 limitation / route exhausted 返回。
- 禁止只写计划不执行。

最低运行原则：

- 非 true external hard block，不允许短于 12 小时返回。
- 若数据和 Modal 允许，目标运行 24–72 小时。
- runtime 必须来自真实点云融合、形态评估、补全训练/修复、可视化、打包。
- 所有阶段必须有 progress JSON、decision JSON、source manifest。
- 如果某一路线失败，必须自动生成下一代 morphology repair route 并继续执行。

============================================================
V200100000000：current package and morphology audit
============================================================

读取并审计：

- reports/V200000000000_final_status.json
- reports/V128000000000_paper_grade_decision.json
- reports/V126000000000_control_ranking.csv
- reports/V126000000000_paper_grade_matrix.csv
- reports/V124000000000_part_specialist_eval.csv
- reports/V123000000000_normal_residual_eval.json
- reports/V121000000000_cross_smc_classification.json
- reports/V180000000000_paper_grade_advisor_report.md
- archive/V190000000000_* bundles
- boards/V127000000000_fullbody.png
- boards/V127000000000_head_face.png
- boards/V127000000000_hairline.png
- boards/V127000000000_left_hand.png
- boards/V127000000000_right_hand.png
- boards/V124000000000_part_specialist_head_hair_hand.png
- selected/control predictions.npz

必须检查：

1. zip clean。
2. inner NPZ readable。
3. selected/control world_points shapes。
4. baseline V11700 world_points shape。
5. current visual boards 是否能看出完整人体。
6. 是否有 body-part labels。
7. 是否有 SMPL mesh/skeleton overlay。
8. 是否有 equal-axis multi-view 3D scatter。
9. 是否有 true vs controls same-scale comparison。
10. 是否有 projection overlay。
11. 是否有 morphology metrics。
12. current report 是否把 camera-bound score 等同于 point-cloud morphology success。

输出：

- reports/V200100000000_morphology_audit.json
- reports/V200100000000_visual_weakness_report.md
- reports/V200100000000_pointcloud_data_inventory.json

硬门：
如果当前图不能显示完整人体 3D 点云，不得使用 V200 final 作为 morphology success。
必须进入 V201.

============================================================
V201000000000：full-view point cloud reconstruction from predictions
============================================================

目标：
从 full-view predictions 重新构建真正可看的 3D 点云，而不是当前稀疏散点图。

输入：

- V11700 baseline predictions
- true_camera_bound_surface_backend predictions
- random_surface_semantic predictions
- shuffled_surface_semantic predictions
- local_knn_smoothing predictions
- support_only predictions
- observation_only predictions
- no_sparseconv_mlp predictions
- confidence maps
- camera binding:
  0021_03_annots.smc
  inverse_rt_camera_to_world
  flip_z
  scale 1.5
- SMPL-X mesh if available
- masks / silhouettes if available
- body part maps if available

必须实现：

1. Convert each view's world_points into common camera-bound / SMC frame.
2. Apply confidence threshold sweep:
   - 0
   - p25
   - p50
   - p75
   - top-k per view
3. Remove obvious background leakage via masks if available.
4. Fuse six views:
   - concatenate
   - voxel downsample
   - duplicate removal
   - outlier removal
5. Keep per-point metadata:
   - source view
   - group
   - confidence
   - body part
   - nearest SMPL distance
   - projection mask status
   - normal
6. Export fused point cloud:
   - .npz
   - .ply
   - lightweight sampled .ply for upload
7. Generate point count stats.

输出：

- tools/v201000000000_reconstruct_fused_pointclouds.py
- output/V201000000000_fused_pointclouds/*.npz
- output/V201000000000_fused_pointclouds/*.ply
- reports/V201000000000_fused_pointcloud_inventory.csv
- reports/V201000000000_fusion_policy.json

硬门：
如果 fused true point cloud still cannot form human-like structure，enter V220 repair.

============================================================
V202000000000：equal-axis multi-view 3D visualization
============================================================

目标：
生成导师真正能看的 3D 点云图。

必须生成：

1. full body multi-view:
   - front
   - side
   - top
   - oblique
   - equal axis
   - same coordinate bounds
   - same point size
   - same sampling policy

2. groups:
   - V11700 baseline
   - true_camera_bound_surface_backend
   - random_surface_semantic
   - shuffled_surface_semantic
   - local_knn_smoothing
   - support_only
   - observation_only

3. color modes:
   - single color by group
   - body-part coloring
   - confidence coloring
   - nearest-SMPL-distance coloring
   - source-view coloring

4. overlays:
   - SMPL-X skeleton
   - SMPL-X mesh transparent if available
   - joint locations
   - camera frustums if useful

输出：

- boards/V202000000000_fullbody_multiview_equal_axis.png
- boards/V202000000000_fullbody_bodypart_colored.png
- boards/V202000000000_fullbody_smpl_overlay.png
- boards/V202000000000_fullbody_confidence_colored.png
- boards/V202000000000_fullbody_distance_colored.png
- reports/V202000000000_visualization_policy.json

硬门：
如果 equal-axis fullbody still looks like blob and not human, enter V220 repair.

============================================================
V203000000000：part-local 3D morphology close-ups
============================================================

目标：
生成 head/hair/hand 的真实 3D close-up，而不是 sparse dot clusters。

Regions:

1. head_face
2. hairline
3. left_hand
4. right_hand
5. optional torso/full upper body

For each region:

- extract points by body-part map or SMPL nearest part
- align local PCA frame
- equal-axis 3D scatter
- same crop bounds across groups
- same point count policy
- SMPL local surface overlay
- skeleton/joint overlay
- confidence coloring
- nearest-SMPL distance coloring
- normal arrows for sampled points

Groups:
- V11700
- true
- random
- shuffled
- smoothing
- support
- observation

输出：

- boards/V203000000000_head_face_morphology.png
- boards/V203000000000_hairline_morphology.png
- boards/V203000000000_left_hand_morphology.png
- boards/V203000000000_right_hand_morphology.png
- boards/V203000000000_part_local_smpl_overlay.png
- reports/V203000000000_part_closeup_policy.json

硬门：
如果 close-up 仍无法看出局部结构，enter V220 part-local densification repair.

============================================================
V204000000000：projection and silhouette morphology overlay
============================================================

目标：
证明 3D 点云不是背景噪点，而是投影回人体 silhouette。

For each group and camera:

1. Project fused point cloud to camera.
2. Overlay on RGB/mask/silhouette.
3. Compute:
   - inside mask ratio
   - silhouette coverage
   - background leakage
   - projected density
   - region coverage
   - head/hair/hand projected coverage
4. Generate boards:
   - true vs controls
   - baseline vs true
   - failure cameras

输出：

- boards/V204000000000_projection_overlay_true_vs_controls.png
- boards/V204000000000_projection_overlay_baseline_vs_true.png
- boards/V204000000000_projection_failure_cases.png
- reports/V204000000000_projection_morphology_metrics.csv

硬门：
If true points project outside silhouette more than controls, fail morphology.

============================================================
V205000000000：morphology metrics
============================================================

Compute true 3D morphology metrics.

Metrics:

Full body:
- humanoid_extent_ratio
- torso_head_leg_separation_score
- point_count
- coverage_ratio
- hole_ratio
- connected_components
- outlier_ratio
- nearest_smpl_distance_mean
- nearest_smpl_distance_p95
- background_leakage
- silhouette_inside_ratio
- surface_coverage

Head-face:
- head_point_count
- head_surface_coverage
- head_component_count
- head_outlier_ratio
- head_nearest_smpl_distance
- head_local_density

Hairline:
- hairline_boundary_coverage
- hairline_density
- hairline_sharpness
- hairline_outlier_ratio

Hands:
- left_hand_point_count
- right_hand_point_count
- hand_component_count
- hand_surface_coverage
- hand_separation_score
- hand_outlier_ratio
- hand_local_density

Comparisons:
- true vs V11700
- true vs random
- true vs shuffled
- true vs smoothing
- true vs support
- true vs observation

输出：

- reports/V205000000000_morphology_metrics.csv
- reports/V205000000000_morphology_control_ranking.csv
- reports/V205000000000_morphology_summary.json

硬门：
true must beat controls in morphology, not only in camera-bound score.

============================================================
V206000000000：morphology decision
============================================================

Decision rules:

POINT_CLOUD_MORPHOLOGY_READY requires:

1. full_body morphology passed.
2. head_face morphology passed.
3. hairline morphology passed.
4. left_hand morphology passed.
5. right_hand morphology passed.
6. true beats random/shuffled/smoothing/support/observation in at least 4/5 regions.
7. true projection overlay inside-mask ratio not worse.
8. true background leakage not worse.
9. true nearest-SMPL distance not worse.
10. visual boards show interpretable human structure.
11. no old image reuse.
12. source manifests clean.

If passed:
- enter V260 report/package.

If failed:
- classify failure:
  A. blob_fullbody
  B. sparse_head
  C. sparse_hairline
  D. sparse_hand
  E. background_leakage
  F. no_humanoid_shape
  G. control_too_close
  H. missing_part_labels
  I. camera_projection_fail

Then enter V220 automatic repair route.

输出：

- reports/V206000000000_morphology_decision.json
- reports/V206000000000_morphology_failure_attribution.md

============================================================
V220000000000：automatic morphology repair route
============================================================

If morphology fails, do not return.

Choose repair based on failure:

A. blob_fullbody / no_humanoid_shape:
- SMPL surface-guided point completion
- TSDF/SDF surface backend
- skeleton-constrained filtering

B. sparse_head / sparse_hairline / sparse_hand:
- part-local densification
- local surface patch decoder
- high-confidence region completion
- part specialist v2

C. background_leakage:
- silhouette-gated filtering
- camera-bound leakage loss
- outlier rejection

D. control_too_close:
- stronger semantic/topology contrast
- penalize smoothing mimic
- noGraph/randomGraph hard negative

E. missing_part_labels:
- derive nearest SMPL part labels
- skeleton/joint local region fallback

F. camera_projection_fail:
- re-run binding calibration
- learned binding correction
- per-view camera-local fallback

Must implement at least one repair and rerun V201–V206.

输出：

- reports/V220000000000_morphology_repair_plan.md
- reports/V220000000000_morphology_repair_result.json
- boards/V220000000000_repair_visual.png

If repair fails:
- auto-generate next route file:
  docs/goals/V220000000000_auto_next_morphology_route.md
- execute it automatically.
- do not return.

============================================================
V230000000000：SMPL surface-guided point completion route
============================================================

If needed, implement surface-guided completion.

Idea:
Use SMPL-X surface as structural support, but do not simply replace VGGT points.

Steps:

1. For each predicted point, find nearest SMPL surface point.
2. Estimate residual distribution.
3. Complete missing low-density regions by:
   - SMPL surface sampling
   - learned residual offset
   - confidence gating
   - silhouette gating
4. Preserve non-SMPL clothing/hair residual if supported by observations.
5. Prevent template overfit via controls:
   - SMPL-only
   - random semantic
   - no observation
   - smoothing

Outputs:
- completed point cloud
- residual field
- confidence
- controls

Run morphology metrics again.

输出：
- reports/V230000000000_surface_completion_eval.json
- boards/V230000000000_surface_completion_visual.png

============================================================
V240000000000：TSDF/SDF morphology backend route
============================================================

If point completion insufficient, implement TSDF/SDF backend.

Steps:
1. Fuse multi-view points into TSDF/SDF volume.
2. Use SMPL surface as prior.
3. Decode surface.
4. Sample point cloud.
5. Project back to views.
6. Compare against controls.

Outputs:
- SDF/TSDF volume summary
- sampled surface points
- normals
- morphology metrics

输出：
- reports/V240000000000_tsdf_sdf_morphology_eval.json
- boards/V240000000000_tsdf_sdf_visual.png

============================================================
V250000000000：final morphology rerun
============================================================

Run full morphology evaluation after repairs:

- V11700
- true original
- true repaired
- random
- shuffled
- smoothing
- support
- observation
- SMPL-only
- TSDF/SDF if available

Generate final boards and metrics.

输出：
- reports/V250000000000_final_morphology_metrics.csv
- reports/V250000000000_final_morphology_decision.json
- boards/V250000000000_final_fullbody.png
- boards/V250000000000_final_parts.png
- boards/V250000000000_final_projection_overlay.png

============================================================
V260000000000：Yuque-style advisor morphology report
============================================================

Write Chinese report in Yuque-style.

Structure:

# 先给结论

必须明确说：
- camera-bound metric 是否通过；
- point-cloud morphology 是否通过；
- 是否可以给导师看；
- 是否仍不 promotion。

# 为什么要补这一轮

说明：
V200 虽然 paper-grade metric passed，但点云图还不像完整人体，需要 morphology gate。

# 数据与方法

说明：
- 读取哪些 predictions
- 如何 fuse multi-view point cloud
- 如何做 equal-axis visualization
- 如何做 SMPL overlay
- 如何做 projection overlay
- 如何做 part metrics

# 点云形态证据

分：
- full body
- head_face
- hairline
- left_hand
- right_hand

# Controls 对比

说明 true vs random/shuffled/smoothing/support/observation。

# 如果做了修复

说明：
- surface-guided completion
- TSDF/SDF
- part-local densification

# 仍然限制

诚实写：
- 是否仍是 single-sequence
- 是否依赖 SMPL proxy
- 是否不是 promotion

# 给导师看的图

列路径和说明。

输出：
- reports/V260000000000_morphology_advisor_report.md
- reports/V260000000000_one_page.md
- reports/V260000000000_limitations.md

============================================================
V280000000000：upload-safe packaging
============================================================

打包：

- core evidence <=50MB
- reports <=50MB
- visuals <=150MB
- selected predictions <=250MB
- controls <=250MB
- morphology pointcloud samples <=250MB
- omitted manifest
- authoritative sidecar manifest

必须：
- zip clean
- inner npz readable
- PLY readable if included
- hashes match
- final status in core
- cleanup in core
- no self-referential hash issue

输出：
- archive/V280000000000_core_evidence_bundle.zip
- archive/V280000000000_reports_bundle.zip
- archive/V280000000000_visuals_bundle.zip
- archive/V280000000000_selected_predictions_bundle.zip
- archive/V280000000000_controls_bundle.zip
- archive/V280000000000_pointcloud_samples_bundle.zip
- reports/V280000000000_upload_manifest_sidecar.json
- reports/V280000000000_omitted_large_file_manifest.json

============================================================
V290000000000：post-push cleanup
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
- reports/V290000000000_post_push_cleanup.json

============================================================
V300000000000：final return gate
============================================================

Only two final return states are allowed:

1. V300000000000_POINT_CLOUD_MORPHOLOGY_READY_NOT_PROMOTED

Requires:
- point-cloud morphology decision passed
- fullbody / head / hair / hands visual boards passed
- true beats controls in morphology metrics
- advisor morphology report complete
- bundles complete
- cleanup complete
- no promotion / no registry / no V50 changes
- active candidate unchanged

2. V300000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

Requires:
- all automatic morphology repair routes attempted or impossible
- user action checklist precise
- evidence bundles complete
- cleanup complete

Any other state cannot return.
If failed, return to V220 auto repair loop.
