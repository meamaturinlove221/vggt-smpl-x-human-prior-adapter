# V90010000000–V120000000000 Advisor-Defense Hardening Goal

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

1. V90000000000 已经达到：
   V90000000000_MENTOR_READY_CAMERA_BOUND_SEMANTIC_TRANSPORT_NOT_PROMOTED

2. 但这不是可以直接停止的最终研究版本。
   现在要升级为 advisor-defense-ready。

3. 当前结论：
   - mentor requirement: satisfied
   - semantic/topology causality: camera-bound gate 下确认
   - camera binding: 自动求解成功
   - best binding: 0021_03_annots.smc
   - rt convention: inverse_rt_camera_to_world
   - axis flip: flip_z
   - scale: 1.0
   - true projection rank: 1
   - true margin: 0.002552
   - Modal GPU: NVIDIA A10
   - normal nonzero ratio: 1.0
   - regions ok: full_body / head_face / hairline / left_hand / right_hand

4. 仍然存在防御风险：
   - true margin 很小；
   - projection ranking 图中各组非常接近；
   - fullbody 点云图肉眼差异不明显；
   - head/hair/hand close-up 仍稀疏；
   - residual-vs-input ranking 仍偏向 low-motion controls；
   - coordinate binding 是自动搜索得出，还需要 heldout/ablation 稳健性；
   - 当前 push failed，local branch ahead 1；
   - worktree dirty，只能 honestly dirty；
   - no promotion / no registry / no V50/V50R2 modification；
   - active candidate remains V11700_gap_reduction_branch_520。

本轮总目标：

不要再以 mentor-ready 作为返回条件。
把 V900 结果升级为 advisor-defense-ready。

只有以下情况允许返回：

A. V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED

必须满足：
1. push 成功或生成可恢复 patch bundle。
2. artifact / manifest / npz / visuals 全部自洽。
3. camera binding 通过 heldout robustness。
4. true route 的 margin 在 bootstrap / view ablation 下仍为正。
5. true route 在 camera-bound score 上稳定优于 random/shuffled/noGraph/smoothing/support/observation。
6. fullbody/head/hair/left hand/right hand 有更强的可视图。
7. normal 有 learned/residual 或明确几何来源，并且不能被写成 learned success。
8. region metrics 不只是 proxy。
9. 导师报告是语雀项目文档风格，能解释所有风险。
10. upload-safe bundles 完成。
11. cleanup 完成。
12. no promotion / no registry / no V50/V50R2 modification。
13. active candidate 不替换，只作为 candidate proposal。

B. V120000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

只允许：
1. GitHub 长时间不可达，且已生成 patch/backup bundle。
2. Modal 权限/额度/GPU 完全不可用。
3. 必需数据文件损坏或确实缺失。
4. 磁盘空间不足且无法清理。
5. 文件系统权限问题。
6. 需要用户手动授权或移动资产。

任何实验失败、margin 不够、visual 不够、binding 不稳、normal 不够、dirty worktree，都不能直接返回，必须自动修复或自动生成下一代路线继续执行。

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止在 push failed 状态下写“remote clean”。
- 禁止把 mentor-ready 当 final advisor-defense-ready。
- 禁止只靠 projection rank=1。
- 禁止忽略 true margin 过小的问题。
- 禁止复用旧图当新证据。
- 禁止只出 bar chart。
- 禁止 dirty worktree 时声称 clean。
- 禁止 limitation / route exhausted 直接返回。
- 禁止只写计划不执行。
- 禁止 sleep 伪造长时间运行。

最低运行原则：

- 非 true external hard block，不允许短于 12 小时返回。
- 若数据和 Modal 允许，目标运行 24–72 小时。
- runtime 必须来自真实 push/retry、审计、稳健性验证、训练/评估、可视化、打包。
- 所有阶段必须有 progress JSON、decision JSON、source manifest。

============================================================
V90010000000：post-V900 audit and push recovery
============================================================

第一优先级：修复 push failed。

必须读取：

- reports/V90000000000_final_status.json
- reports/V80000000000_advisor_report.md
- reports/V85000000000_post_push_cleanup.json
- reports/V80000000000_upload_manifest_sidecar.json
- reports/V41600000000_camera_bound_mentor_gate_decision.json
- reports/V41500000000_camera_bound_projection_decision.json
- reports/V41500000000_region_metrics.csv
- reports/V41500000000_camera_bound_projection_metrics.csv
- archive/V80000000000_* bundles

必须执行：

1. 检查 git branch。
2. 检查 local commit 00b85ad5408b7a28e7121caf2ff350d627272126。
3. 检查 remote 是否已有该 commit。
4. 如果 remote 没有，retry push。
5. 如果 push 失败：
   - 生成 patch bundle；
   - 生成 git status report；
   - 生成 remote failure report；
   - 不删除证据。
6. 检查 dirty worktree，分类：
   - 当前研究必须提交的文件；
   - 历史遗留文件；
   - 可清理临时文件；
   - 不应删除的证据文件。
7. 不得把 dirty 写成 clean。

输出：

- reports/V90010000000_push_recovery.json
- reports/V90010000000_dirty_worktree_classification.md
- reports/V90010000000_artifact_audit.json

硬门：
push 或 patch bundle 必须完成其一，才能进入 V901。

============================================================
V90100000000：artifact and visual evidence audit
============================================================

重新审计 V800/V900 包。

必须检查：

1. zip clean。
2. sidecar manifest hash 与实际 zip hash 一致。
3. selected predictions npz 可读。
4. controls predictions npz 可读。
5. world_points / normal / confidence shape。
6. normal nonzero。
7. visuals 是否为新图。
8. visual 是否能肉眼区分 true 和 controls。
9. projection ranking 是否 margin 过小。
10. region metrics 是否充分。
11. residual-vs-input ranking 是否仍支持 low-motion controls。

输出：

- reports/V90100000000_artifact_visual_audit.json
- reports/V90100000000_visual_weakness_table.md
- reports/V90100000000_margin_risk_report.json

硬门：
如果 visual 证据不足，必须进入 V930 visual hardening。
如果 margin 风险高，必须进入 V910 robustness。

============================================================
V91000000000：camera binding robustness
============================================================

目标：
证明 0021_03 + inverse_rt_camera_to_world + flip_z + scale 1.0 不是偶然搜索结果。

必须做：

1. 重新扫描 8 个 SMC。
2. 保留 top-5 binding candidates。
3. 对 0021_03 做 view ablation：
   - remove camera 00
   - remove camera 01
   - remove camera 15
   - remove camera 30
   - remove camera 45
   - remove camera 59
4. 做 bootstrap over views。
5. 做 RT convention ablation。
6. 做 axis flip ablation。
7. 做 scale perturbation：
   - 0.5
   - 0.75
   - 1.0
   - 1.25
   - 1.5
8. 做 translation perturbation。
9. 检查 true rank 是否稳定。
10. 检查 margin 是否仍为正。

输出：

- reports/V91000000000_binding_robustness.json
- reports/V91000000000_view_ablation.csv
- reports/V91000000000_binding_bootstrap.csv
- boards/V91000000000_binding_robustness.png

硬门：
如果 0021_03 binding 不稳，进入 V920 learned binding repair。
如果 binding 稳，进入 V930 visual hardening。

============================================================
V92000000000：learned binding repair if needed
============================================================

如果 V910 失败，训练 small binding calibrator。

输入：
- V11700/V920 world points
- SMC K/RT/masks
- SMPL-X projections
- semantic masks

输出：
- calibrated Sim3
- view-specific residual correction
- confidence

Controls：
- random SMC
- random view order
- random axis
- no SMPL
- mask-only

输出：

- reports/V92000000000_learned_binding_eval.json
- boards/V92000000000_learned_binding_visual.png

如果修复成功，回到 V910。
如果失败，进入 V960 alternative route，不得返回。

============================================================
V93000000000：visual evidence hardening
============================================================

目标：
导师看图要能看出差别，不只是 rank=1。

必须生成新图，不准复用旧图。

图 1：fullbody same-scale overlay
- V11700
- true_camera_bound_transport
- random_surface_semantic
- shuffled_surface_semantic
- local_knn_smoothing
- observation_only
- support_only

图 2：head_face close-up
同列。

图 3：hairline close-up
同列。

图 4：left_hand close-up
同列。

图 5：right_hand close-up
同列。

图 6：delta map / displacement magnitude
显示 true 和 controls 的差异区域。

图 7：camera projection overlay
投影点云到 SMC masks 上，显示 true/control 与 mask 的关系。

图 8：normal board
normal / normal_conf / learned or geometric source。

图 9：failure cases
展示哪些视角仍弱。

必须要求：
- same scale
- same camera angle
- same point size
- same region crop
- 标注 group / score / region metric
- 不准只出柱状图

输出：

- boards/V93000000000_fullbody_overlay.png
- boards/V93000000000_head_face_closeup.png
- boards/V93000000000_hairline_closeup.png
- boards/V93000000000_left_hand_closeup.png
- boards/V93000000000_right_hand_closeup.png
- boards/V93000000000_delta_maps.png
- boards/V93000000000_projection_overlay.png
- boards/V93000000000_normal_board.png
- boards/V93000000000_failure_cases.png
- reports/V93000000000_visual_hardening_report.json

硬门：
如果肉眼仍然看不出差异，进入 V940 visual-first repair。

============================================================
V94000000000：visual-first repair
============================================================

如果 V930 图像不够强，不能返回。

修复路线：

1. Region-local residual amplification：
   只在 camera-bound safe region 放大 true residual，必须不增加 background leakage。

2. Hand/hair specialist：
   对 head/hair/hand 分别训练或合成更清晰 local residual。

3. Confidence-gated visualization：
   剔除低置信噪点，突出可信点云结构。

4. Part-local densification：
   在 SMPL surface topology 上对 head/hair/hand 做局部 densification。

5. Compare against random/shuffled/smoothing：
   放大必须只对 true 有效，controls 不得同样改善。

运行 reduced matrix：
- true
- random
- shuffled
- smoothing
- observation
- support

输出：

- reports/V94000000000_visual_repair_eval.json
- boards/V94000000000_visual_repair_board.png

如果成功，回到 V930。
如果失败，进入 V960 alternative route。

============================================================
V95000000000：normal and residual defense
============================================================

目标：
把 normal 证据讲清楚。

必须区分：

1. geometric recomputed normal
2. model output normal
3. learned residual normal
4. normal confidence

如果当前 normal 只是 model export 但非 learned residual：
- 不能写 learned normal head。
- 必须写 normal valid but not learned residual。
- 如果能训练 residual normal head，则训练并评估。

输出：

- reports/V95000000000_normal_source_audit.json
- reports/V95000000000_normal_residual_eval.json
- boards/V95000000000_normal_defense.png

硬门：
normal 证据不清楚时，不得写 learned normal success。

============================================================
V96000000000：auto alternative architecture route if evidence still weak
============================================================

如果 V910/V930/V950 后仍然无法 advisor-defense-ready，不得返回。

自动生成下一代路线：

docs/goals/V96000000000_auto_next_route.md

根据失败原因选择：

A. Binding weak：
- learned binding route
- differentiable renderer route
- mask-only camera route

B. Visual weak：
- part-local densification route
- camera-safe residual amplification route
- high-res crop route

C. Random/control too close：
- stronger contrastive semantic route
- topology graph transformer route
- TSDF/SDF route

D. Normal weak：
- learned normal residual route
- normal teacher distillation route

E. Region weak：
- hand/hair specialist route
- object/phone exclusion route

写完后必须立即执行，不得返回用户。

输出：

- reports/V96000000000_auto_next_route_generation.json

============================================================
V100000000000：defense rerun matrix
============================================================

对修复后的 best route 跑 defense matrix：

核心组：
1. true
2. random semantic
3. shuffled semantic
4. local smoothing
5. no graph
6. random graph
7. observation only
8. support only

每组至少 5 seeds。

必须评估：
- camera-bound score
- region metrics
- visual score
- normal score
- background leakage
- true margin bootstrap CI

输出：

- reports/V100000000000_defense_matrix.csv
- reports/V100000000000_bootstrap_ci.json
- reports/V100000000000_defense_decision.json

============================================================
V110000000000：Yuque-style advisor defense report
============================================================

写中文导师文档，文风参考项目复盘 / 语雀项目文档。

结构：

# 先给结论

必须一句话说：
是否达到 advisor-defense-ready。

# 本轮做了什么

列：
- push recovery
- artifact audit
- camera binding robustness
- visual hardening
- normal defense
- repair matrix

# 为什么这次比 V900 更可信

必须解释：
- V900 的弱点；
- 本轮如何补；
- margin 是否稳；
- heldout / bootstrap 是否稳；
- controls 是否仍被压过；
- 图是否更直观。

# 坐标绑定解决方案

写：
- SMC
- RT convention
- axis
- scale
- translation
- robustness
- view ablation

# 点云证据

写：
- full body
- head_face
- hairline
- left_hand
- right_hand

# Controls 证据

写：
- random
- shuffled
- smoothing
- no graph
- observation
- support

# Normal 证据

写来源，不夸大。

# 仍然的限制

必须诚实写。

# 为什么不 promotion

写明 active candidate 未替换。

# 下一步论文路线

写可执行计划。

输出：

- reports/V110000000000_advisor_defense_report.md
- reports/V110000000000_one_page.md
- reports/V110000000000_limitations.md

============================================================
V115000000000：upload-safe package
============================================================

打包：

- core evidence <=50MB
- reports <=50MB
- visuals <=150MB
- selected predictions <=250MB
- controls <=250MB
- omitted manifest
- authoritative sidecar manifest

必须：
- zip clean
- inner npz readable
- hashes match
- final status in core
- cleanup in core
- no self-referential hash issue
- controls include predictions

输出：
- archive/V115000000000_core_evidence_bundle.zip
- archive/V115000000000_reports_bundle.zip
- archive/V115000000000_visuals_bundle.zip
- archive/V115000000000_selected_predictions_bundle.zip
- archive/V115000000000_controls_bundle.zip
- reports/V115000000000_upload_manifest_sidecar.json
- reports/V115000000000_omitted_large_file_manifest.json

============================================================
V118000000000：post-push cleanup
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

- reports/V118000000000_post_push_cleanup.json

============================================================
V120000000000：final return gate
============================================================

只有以下两种状态允许返回：

1. V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED

Requires:
- push or patch bundle complete
- camera binding robust
- margin robust
- visual evidence stronger
- normal source clear
- region metrics complete
- controls beaten or limitation precisely bounded
- advisor report complete
- bundles complete
- cleanup complete
- no promotion / no registry / no V50 change
- active candidate not replaced

2. V120000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

Requires:
- all automatic repair routes attempted or impossible
- user action checklist precise
- evidence bundles complete
- cleanup complete

任何其他状态不得返回。
如果失败，自动进入 V960 写下一代路线并执行。
