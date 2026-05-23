# V30010000000-V90000000000 Camera-Bound Semantic Transport Auto-Evolution Goal

你现在接手 D:\vggt\vggt-feature-adapter 的下一阶段超长期任务。

repo 固定为：

D:\vggt\vggt-feature-adapter

evidence root 固定为：

D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild

二者不能混写。

本轮允许使用 /goal，但本文件是主执行规范。
本轮严禁发起 agent / subagent。
只能由当前主线程长期执行、自动路由、自动修复、自动续跑。

你不能在中间失败状态返回用户。
如果一个阶段失败，必须自动写出下一阶段路线并继续执行。
只有以下情况允许最终返回：

A. 导师要求真正满足：
- semantic/topology causality 可信；
- camera-bound reprojection 可信；
- full body/head/hair/hand 点云视觉改善可信；
- 报告和 bundles 完成。

B. 真正外部硬阻断：
- Modal 权限/额度/GPU 完全不可用；
- 必需数据文件损坏或确实缺失；
- 所有自动坐标系绑定策略穷尽仍失败，并写出精确用户行动清单。

除此之外，任何 limitation / route exhausted / smoothing dominant / random dominant / camera binding failed 都不能返回用户，只能触发下一代路线。

当前事实必须接受：

1. V15000000000 已经不是可返回终点。
2. V145/V150 证明 true_surface_transport 在数值上强于 random/shuffled/smoothing 等 controls，但导师视觉证据仍不够强。
3. V30000000000 返回 V∞_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION。
4. 这个返回原因是 camera-bound reprojection gate 失败：
   - 4K4D SMC K/RT 可以读取；
   - camera ids 00,01,15,30,45,59 可绑定；
   - 当前 V11700/V920 world_points 与该 K/RT 坐标不一致；
   - 非零 residual 的最优 reprojection-aware scale 退化为 0.0。
5. 不能把当前结果写成导师目标达成。
6. 不能 promotion。
7. 不能写 strict registry。
8. 不能修改 V50/V50R2。
9. 不能替换 active candidate。
10. Active candidate remains V11700_gap_reduction_branch_520。
11. Push 已成功：
    commit 0217d0f36f6d1e7905ee0f5a0f3196d478533294。
12. Worktree 仍可能 dirty，只能 honestly dirty。
13. G:\数据集\datasets\data_used_in_4K4D\annotations 下有 8 个 SMC：
    - 0012_11_annots.smc
    - 0013_01_annots.smc
    - 0013_03_annots.smc
    - 0013_09_annots.smc
    - 0013_11_annots.smc
    - 0019_08_annots.smc
    - 0021_03_annots.smc
    - 0023_06_annots.smc
14. 0012_11_annots.smc 包含：
    - Camera_Parameter
    - Keypoints_2D
    - Keypoints_3D
    - Mask
    - SMPLx
    - actor_id=12
    - performance_id=11
    - camera ids 00..59
15. 0012_11_annots.smc 是合理候选，但不能证明就是 V11700/V920 的匹配源。
16. V270 reprojection metrics 中部分 camera 出现 1e8-1e9 级 shift，说明当前坐标假设严重错误。
17. 不允许继续让用户手动猜坐标系；必须自动求解坐标绑定。

本轮总目标：

构建 Camera-Bound Semantic Transport 体系。

核心问题：

不是“SMPL semantic 能不能让 metric 变大”，而是：

SMPL semantic/topology 修改后的点云，能不能在真实 4K4D camera/mask/silhouette/reprojection gate 下可信成立？

本轮必须完成：

1. 自动扫描 8 个 SMC，并判断哪个最可能匹配 V11700/V920。
2. 自动求解 VGGT world_points -> 4K4D world/camera 的 Sim(3)/SE(3)/scale/axis/RT convention。
3. 自动处理 RT 方向、单位、轴翻转、view order、mask resize、distortion、crop/resize convention。
4. 生成可信 T_vggt_to_4k4d。
5. 如果绑定成功，重新跑 camera-bound semantic transport matrix。
6. 如果绑定失败，自动进入替代路线：
   - camera-free visual mentor route；
   - SMPL-X projection-derived pseudo camera route；
   - mask-only silhouette route；
   - temporal/canonical self-consistency route；
   - learned camera alignment route。
7. 不能因为坐标失败就立即返回。
8. 只有所有自动 coordinate routes 穷尽，才允许 external hard block。

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止把当前 0012_11 直接当作绝对匹配源。
- 禁止只测试一个 RT convention。
- 禁止只测试一个 scale。
- 禁止 scale=0 后直接返回用户。
- 禁止把 1e8 级 reprojection shift 当普通误差。
- 禁止用 proxy visual 替代 camera-bound evidence。
- 禁止只靠 mean_delta 判断成功。
- 禁止只出 heatmap。
- 禁止复用旧图当新图。
- 禁止 dirty worktree 时声称 clean。
- 禁止中途 failure checkpoint 返回用户。
- 禁止只写计划不执行。

最低运行原则：

- 非 true external hard block，不允许短于 12 小时返回。
- 若数据和 Modal 允许，目标运行 24-96 小时。
- 不允许 sleep 伪造运行时间。
- runtime 必须来自真实坐标搜索、训练、评估、可视化、打包、审计。
- 如果提前终止，必须写 hard-blocker 证据和 user-action checklist。
- 所有自动修复路线必须有 progress JSON、decision JSON、source manifest。

============================================================
V30010000000：goal manifest and current evidence audit
============================================================

保存本文件并生成：

reports/V30010000000_goal_file_manifest.json

读取并审计：

- reports/V30000000000_final_status.json
- reports/V30000000000_advisor_report.md
- reports/V30000000000_user_action_checklist.json
- reports/V30000000000_smc_match_recheck.json
- reports/V29000000000_decision.json
- reports/V27000000000_camera_binding_report.json
- reports/V27000000000_reprojection_metrics.csv
- reports/V26000000000_tsdf_metrics.csv
- reports/V25000000000_region_metrics.csv
- reports/V9900000000_causality_summary.json
- reports/V10100000000_decision.json
- reports/V14000000000_advisor_report.md
- reports/V14800000000_post_push_cleanup.json
- reports/V30000000000_post_push_cleanup.json
- archive/V30000000000 bundles
- archive/V14500000000 bundles

必须检查：

1. 当前 bundles 是否 zip clean。
2. selected/control npz 是否内部可读。
3. V145/V300 visual boards 是否新图。
4. V145/V300 metrics 互相是否一致。
5. V270 reprojection 失败细节。
6. 当前 SMC recheck 结果。
7. 当前 git commit/push 状态。
8. worktree dirty 原因。
9. Modal apps 是否空。
10. active candidate 是否未替换。
11. registry/V50/V50R2 是否未改。

输出：

- reports/V30010000000_evidence_audit.json
- reports/V30010000000_camera_failure_attribution.md
- reports/V30010000000_next_route_requirements.json

硬门：
如果 scale=0 / reprojection fail 是当前唯一 blocker，必须进入 V30100000000 coordinate resolver，不得返回用户。

============================================================
V30100000000：SMC sequence auto matcher
============================================================

目标：
自动判断 V11700/V920 最可能对应哪个 SMC sequence。

扫描：

G:\数据集\datasets\data_used_in_4K4D\annotations\*.smc

对于每个 SMC：

1. 读取 attrs：
   - actor_id
   - performance_id
   - gender
   - height
   - weight
2. 读取 top-level keys：
   - Camera_Parameter
   - Keypoints_2D
   - Keypoints_3D
   - Mask
   - SMPLx
3. 读取 camera ids。
4. 检查是否包含 required cameras：
   00,01,15,30,45,59
5. 读取每个 camera 的 masks。
6. 下采样或 resize 到 518×518。
7. 与 V11700/V920 valid human mask / semantic mask / foreground mask 做 IoU。
8. 与 V16300/V340 semantic masks 做 IoU。
9. 若 RGB inputs 可找到，比较 RGB frame/mask outline。
10. 若 SMPLx 可读，投影 SMPLx silhouette 到每个 camera，与 V11700 mask 比较。

输出：

- reports/V30100000000_smc_sequence_scan.json
- reports/V30100000000_smc_mask_iou.csv
- reports/V30100000000_smc_match_ranking.json
- boards/V30100000000_smc_match_visual.png

硬门：
不能只测试 0012_11。
必须测试 8 个 SMC。
如果没有任何 SMC mask 与 V11700 mask 匹配，进入 V302 alternative mask source route。

============================================================
V30200000000：camera parameter convention inventory
============================================================

对排名靠前的 SMC，解析 camera parameter convention。

必须读取：

- K / intrinsic
- RT / extrinsic
- R
- T
- D / distortion
- image size if present
- camera order
- world-to-camera or camera-to-world possible interpretations
- row-major / col-major shape
- units

生成 convention candidates：

1. RT as world_to_camera
2. inverse RT as camera_to_world
3. R transposed
4. T as column / row
5. T unit m
6. T unit mm
7. T unit cm
8. x/y/z axis flips:
   - identity
   - flip x
   - flip y
   - flip z
   - flip xy
   - flip xz
   - flip yz
   - flip xyz
9. OpenCV convention:
   x right, y down, z forward
10. OpenGL convention:
   x right, y up, z backward

输出：

- reports/V30200000000_camera_convention_candidates.json
- reports/V30200000000_intrinsic_resize_policy.json

硬门：
不能只用一种 RT convention。

============================================================
V30300000000：VGGT world coordinate diagnosis
============================================================

目标：
判断 V11700/V920 world_points 是哪种坐标系。

对 world_points 统计：

1. xyz min/max/mean/std。
2. depth z range。
3. per-view xyz distribution。
4. cross-view centroid consistency。
5. 是否 camera-local。
6. 是否 normalized。
7. 是否 first-view anchored。
8. 是否 global world。
9. 是否 scale arbitrary。
10. 是否 z forward / z backward。

检查：
- V11700 camera predictions if available。
- VGGT extrinsics if stored。
- any transforms in V11700 metadata。
- any normalization scale in source manifest。
- any crop/resize metadata。

输出：

- reports/V30300000000_vggt_world_coordinate_diagnosis.json
- reports/V30300000000_vggt_metadata_scan.json
- boards/V30300000000_world_points_distribution.png

硬门：
如果 world_points 明显是 per-view camera-local，不得直接用 4K4D world RT。
必须进入 per-view camera alignment route。

============================================================
V30400000000：Sim(3) / SE(3) coordinate binding search
============================================================

目标：
自动求解 VGGT world_points 到 4K4D camera/world 的变换。

候选输入：

- SMC candidates from V301
- camera convention candidates from V302
- world coordinate diagnosis from V303
- V11700/V920 points
- masks/silhouettes
- SMPLx mesh if available
- keypoints 2D/3D if available

搜索参数：

1. scale:
   - logspace 1e-6 to 1e6
   - also units m/mm/cm
   - robust scale from bounding boxes
   - scale from SMPL height
   - scale from mask size

2. rotation:
   - identity
   - axis flips
   - principal-axis alignment
   - Umeyama rotation
   - ICP rotation
   - camera-frustum-based rotation

3. translation:
   - centroid alignment
   - camera depth median alignment
   - mask center alignment
   - SMPL pelvis alignment

4. RT convention:
   all candidates from V302

5. view order:
   - given [00,01,15,30,45,59]
   - permutations only if mask similarity suggests mismatch
   - cyclic/order tests if source order unknown

Objectives:

- in-frame ratio
- silhouette IoU
- mask coverage
- projected point inside human mask
- reprojection distance to SMPLx silhouette
- keypoint consistency if 2D/3D available
- depth sign validity
- cross-view consistency
- nonzero residual feasibility

必须避免：
- scale=0 trivial solution
- all points projected outside
- huge reprojection shifts accepted
- single-view overfit

输出：

- tools/v30400000000_coordinate_binding_search.py
- reports/V30400000000_coordinate_binding_candidates.csv
- reports/V30400000000_best_binding.json
- boards/V30400000000_binding_visual_grid.png

硬门：
如果 best nonzero transform score passes threshold，进入 V305.
如果 all candidates fail，进入 V306 alternative coordinate routes.
不得直接返回用户。

============================================================
V30500000000：camera-bound verification of semantic predictions
============================================================

用 V304 best transform 验证已有 V145/V300 predictions。

评估：

1. V11700 baseline reprojection.
2. true_surface_transport reprojection.
3. random semantic reprojection.
4. shuffled semantic reprojection.
5. local smoothing reprojection.
6. support/observation controls.
7. dense teacher/TSDF controls if available.

Metrics:

- silhouette IoU
- in-frame ratio
- human-mask coverage
- background leakage
- reprojection shift
- region IoU:
  full_body / head_face / hairline / left_hand / right_hand
- camera-bound score
- nonzero residual allowed ratio

输出：

- reports/V30500000000_camera_bound_eval.csv
- reports/V30500000000_camera_bound_decision.json
- boards/V30500000000_camera_bound_visual.png

硬门：
如果 true improves under camera-bound metrics，进入 V320 camera-bound training.
如果 binding works but true fails, enter auto repair route.
If binding fails, enter V306.

============================================================
V30600000000：alternative coordinate routes if direct binding fails
============================================================

如果 V304 未找到可信直接绑定，不得返回用户。
必须自动尝试替代路线：

Route 1: per-view camera-local binding
- 假设 VGGT world_points 是 per-view camera coordinates。
- 不使用 global world RT。
- 每个 view 单独用 K/resize/mask 做 alignment。

Route 2: silhouette-only binding
- 不依赖 depth/world metric。
- 用 2D mask coverage 约束 residual candidate。

Route 3: SMPL-X projection-derived pseudo camera
- 从 SMPLx mesh 和 masks 估 pseudo projection。
- 用 pseudo camera 作为 trust gate。

Route 4: learned camera alignment network
- 学一个 small Sim3/scale/shift calibrator。
- 控制组必须保留。

Route 5: camera-free mentor visual route
- 如果真实 camera gate不可用，退到 camera-free，但必须明确 limitation，不得 mentor-ready。

每条 route 必须输出：
- route report
- metrics
- visual board
- whether can continue

输出：

- reports/V30600000000_alternative_coordinate_routes.json

硬门：
只有 V306 全部失败，且原因不可由代码继续搜索，才允许 external hard block。
否则继续 V320/V330.

============================================================
V31000000000：coordinate-bound training dataset
============================================================

如果 V304/V306 找到可信 binding，构建 camera-bound training dataset。

Dataset 包含：

- full 81-channel semantic payload
- surface-indexed topology payload
- V11700/V920 points
- camera-bound projected points
- masks/silhouettes
- K/RT/convention metadata
- T_vggt_to_4k4d
- resize/distortion policy
- region labels

Groups:

1. true_surface_transport
2. random_surface_semantic
3. shuffled_surface_semantic
4. local_knn_smoothing_surface
5. no_surface_graph
6. random_surface_graph
7. observation_only
8. support_only
9. no_sparseconv_mlp
10. no_teacher

输出：

- tools/v31000000000_build_camera_bound_dataset.py
- reports/V31000000000_dataset_manifest.json
- reports/V31000000000_pairing_audit.json

============================================================
V32000000000：camera-bound semantic transport model
============================================================

实现或修复：

models/v32000000000_camera_bound_semantic_transport.py

新增：

1. CameraBindingModule
- applies T_vggt_to_4k4d
- applies K/RT convention
- projects to masks
- returns reprojection loss

2. ReprojectionTrustHead
- estimates whether residual is camera-trustworthy

3. CameraBoundTransportDecoder
- semantic/topology value
- observation context
- support mask
- camera projection consistency

4. LearnedNormalResidualHead
- learned residual, not only geometric recompute

Losses:

- point residual
- surface topology
- camera reprojection
- mask IoU
- background leakage
- normal residual
- hair/hand region
- anti-smoothing
- random/shuffled contrastive

输出：

- models/v32000000000_camera_bound_semantic_transport.py
- reports/V32000000000_model_contract.json
- reports/V32000000000_loss_design.md

============================================================
V33000000000：camera-bound formal matrix
============================================================

运行 camera-bound formal matrix。

Groups:
1. true_surface_transport
2. random_surface_semantic
3. shuffled_surface_semantic
4. local_knn_smoothing_surface
5. no_surface_graph
6. random_surface_graph
7. observation_only
8. support_only
9. no_sparseconv_mlp
10. no_teacher

每组：
- minimum 5 seeds
- target 10 seeds
- GPU formal run
- training_steps >= 200
- meaningful runtime
- no CPU pilot

每 run 输出：
- predictions.npz
- camera_bound_eval.json
- source_manifest.json
- board.png
- training log
- GPU type
- runtime

输出：

- reports/V33000000000_camera_bound_matrix.csv
- reports/V33000000000_seed_metrics.csv
- reports/V33000000000_modal_job_manifest.json

硬门：
如果 camera-bound training impossible，enter V340 alternative route, not return.

============================================================
V34000000000：camera-bound decision
============================================================

MENTOR_READY_CAMERA_BOUND requires:

1. true > random semantic.
2. true > shuffled semantic.
3. true > local smoothing.
4. true > no graph/random graph.
5. true > observation/support.
6. true improves silhouette IoU.
7. true reduces background leakage.
8. true has nonzero residual feasible under camera gate.
9. full_body/head/hair/hand visually improved.
10. learned normal residual valid.
11. source manifests clean.

如果 pass：
进入 V80000000000 mentor package.

如果 fail：
自动路由。

Failures:
- coordinate still weak -> V350 learned binding route
- smoothing still dominant -> V360 topology/SDF route
- visual weak -> V370 visual-first route
- normal weak -> V380 normal route
- part weak -> V390 part specialist route

输出：
- reports/V34000000000_camera_bound_decision.json

============================================================
V35000000000：learned binding route
============================================================

如果 coordinate binding 不稳定：

训练 small calibrator：

Input:
- V11700 world_points stats
- SMPLx projected masks
- camera K/RT candidates
- semantic masks

Output:
- Sim3 transform
- per-view scale/shift if needed
- confidence

Controls:
- random semantic
- random camera
- shuffled view order
- no SMPL
- mask only

Goal:
Find trustworthy camera binding without user.

输出：
- reports/V35000000000_learned_binding_eval.json

If improved, return to V310/V330.
If failed, enter V360/V370.

============================================================
V36000000000：TSDF/SDF camera-bound backend route
============================================================

如果 point residual route fails：

Build SMPL-guided TSDF/SDF backend.

Inputs:
- SMPL-X surface
- VGGT observations
- camera binding if available
- masks

Outputs:
- SDF residual
- surface points
- normals
- full-view projection

Controls:
- random semantic
- no topology
- local smoothing
- no camera

Run matrix and evaluate camera-bound metrics.

输出：
- reports/V36000000000_tsdf_sdf_backend_eval.json

If improved, return to V340.
If failed, enter V370.

============================================================
V37000000000：visual-first mentor route
============================================================

If camera metrics remain impossible but visual route promising:

目标：
不声明 paper-grade camera causality，但生成导师可判断的 visual-first package。

必须：
- true vs random/shuffled/smoothing full-view closeups
- full body/head/hair/hand
- same view same scale
- no heatmap only
- no proxy-only
- clear limitations

输出：
- reports/V37000000000_visual_first_eval.json
- boards/V37000000000_visual_first_mentor_board.png

If visual proof strong, continue; if weak, enter V380/V390.

============================================================
V38000000000：learned normal and geometry residual repair
============================================================

If normal or geometry residual weak:

- train learned normal residual
- add curvature loss
- add part boundary normal loss
- add hand/hair normal consistency
- compare geometric vs learned residual

输出：
- reports/V38000000000_normal_repair_eval.json

If improved, return to V330.

============================================================
V39000000000：part-local high-res repair
============================================================

If head/hair/hand weak:

- HeadFace specialist
- Hairline specialist
- LeftHand specialist
- RightHand specialist
- local surface graph
- part-specific residuals
- object/phone leakage guard

输出：
- reports/V39000000000_part_local_eval.json
- boards/V39000000000_part_local_visual.png

If improved, return to V330/V340.

============================================================
V40000000000：auto evolution gate
============================================================

If none of V350-V390 reaches mentor-ready:

Do not return.

Generate next route file:

docs/goals/V40000000000_auto_evolved_route.md

The route must include:
- failure attribution
- next architecture hypothesis
- why previous route failed
- new hard gates
- new data/training/eval matrix
- new visual proof requirements

Candidate route families:

1. differentiable SMPL renderer route
2. dense 4D canonical fusion route
3. neuralbody latent surface code route
4. point transformer surface graph route v2
5. SDF/NeuS local human field route
6. VGGT token injection after camera-bound gate
7. learned teacher distillation route with stronger leakage audit

After writing the file, immediately execute it.
Do not return user.

============================================================
V80000000000：mentor-ready package
============================================================

Only if mentor requirement truly satisfied.

Write Yuque-style Chinese advisor report:

# 先给结论

# 本轮解决的问题

# 核心证据

# Camera-bound 坐标解决方案

# Semantic/topology 因果证据

# Full body / head / hair / hand 图

# Normal 证据

# Controls 证据

# 仍然的限制

# 为什么不 promotion

# 下一步论文路线

输出：
- reports/V80000000000_advisor_report.md
- reports/V80000000000_one_page.md
- reports/V80000000000_limitations.md

打包：
- archive/V80000000000_core_evidence_bundle.zip
- archive/V80000000000_reports_bundle.zip
- archive/V80000000000_visuals_bundle.zip
- archive/V80000000000_selected_predictions_bundle.zip
- archive/V80000000000_controls_bundle.zip
- reports/V80000000000_upload_manifest_sidecar.json
- reports/V80000000000_omitted_large_file_manifest.json

All under 500MB.
All zip clean.
Internal npz readable.

============================================================
V85000000000：post-push cleanup
============================================================

commit/push 后检查：

- git status clean or honestly dirty
- branch
- commit
- Modal apps
- Python workers
- registry diff
- V50/V50R2 diff
- active candidate
- staged/untracked files if dirty

输出：
- reports/V85000000000_post_push_cleanup.json

============================================================
V90000000000：final return gate
============================================================

Only two final return states are allowed:

1. V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED

Requires:
- mentor requirement satisfied
- camera binding solved or acceptable limitation explained
- semantic/topology causality evidence
- visual improvement
- advisor report
- bundles
- cleanup

2. V90000000000_TRUE_EXTERNAL_HARD_BLOCK_AFTER_COORDINATE_EXHAUSTION

Requires:
- all SMCs scanned
- all coordinate conventions tried
- Sim3/SE3/scale/axis/view-order search exhausted
- alternative coordinate routes tried
- user action checklist precise
- advisor report
- bundles
- cleanup

Any other state cannot return.
Failure states must route to V40000000000 auto evolution.
