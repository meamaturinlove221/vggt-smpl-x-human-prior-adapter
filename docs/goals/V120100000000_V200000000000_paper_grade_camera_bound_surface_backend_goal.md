# V120100000000–V200000000000 Paper-Grade Camera-Bound Surface Backend Goal

你现在接手 D:\vggt\vggt-feature-adapter 的下一阶段超长期任务。

repo 固定为：

D:\vggt\vggt-feature-adapter

evidence root 固定为：

D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild

二者不能混写。

本轮允许使用 /goal，但本文件是主执行规范。
本轮严禁发起 agent / subagent。
只能由当前主线程长期执行、自动路由、自动修复、自动续跑。

你不能在中间 failure / limitation / route exhausted 状态返回用户。
你也不能因为已经 advisor-defense-ready 就返回用户。
本轮目标是从 advisor-defense-ready 升级到 paper-grade / multi-sequence / learned-normal / reproducibility-ready。

当前事实必须接受：

1. V120000000000 已完成。
2. 当前状态是：
   V120000000000_ADVISOR_DEFENSE_READY_NOT_PROMOTED
3. 当前已经满足 advisor-defense-ready，但不能作为最终 paper-grade 结果停止。
4. 当前 calibrated binding：
   - SMC: 0021_03_annots.smc
   - RT convention: inverse_rt_camera_to_world
   - axis: flip_z
   - scale: 1.5
5. 当前 base margin: 0.017372。
6. 当前 bootstrap p05 margin: 0.009333。
7. 当前 view ablation min margin: 0.012319。
8. 当前 semantic/topology causality 在 camera-bound gate 下成立。
9. 当前 normal source clear，但不是 learned residual normal success。
10. 当前 active candidate remains V11700_gap_reduction_branch_520。
11. no promotion。
12. no strict registry。
13. no V50/V50R2 modification。
14. worktree still dirty from historical/unrelated files。
15. Push succeeded at commit 526b56667e3e0127ca59c4d31172c6bf13052c6b。
16. V115 selected/control bundles include readable NPZ。
17. V11700 baseline normal remains zero, true/control normals are nonzero。
18. Current visual boards are improved but still not visually overwhelming.
19. Current evidence is strongest on 0021_03, not yet cross-SMC.

Important audit warning:

The latest user-text hashes for selected predictions and controls match the uploaded bundles, but uploaded core/reports/visuals bundle hashes may not match the newest text-reported hashes. You must perform artifact reconciliation before using them as authoritative.

本轮总目标：

把 current advisor-defense package 升级为 paper-grade camera-bound human surface backend evidence.

必须完成：

1. Artifact reconciliation：
   - confirm latest local V115 bundles
   - confirm hashes
   - confirm uploaded-vs-local mismatch
   - regenerate if needed

2. Cross-SMC generalization:
   - test 0021_03 plus all other available SMCs
   - distinguish true match from non-match
   - run binding search and controls
   - prove current binding is not overfitted

3. Learned residual normal:
   - implement learned normal residual head
   - train/evaluate against geometric normal teacher
   - keep geometric normal source separate
   - cannot claim learned normal if only recompute

4. Part-local surface specialists:
   - head-face
   - hairline
   - left hand
   - right hand
   - use surface graph / local curvature / region masks
   - improve close-up visual separability

5. Differentiable camera-bound loss:
   - silhouette/mask projection
   - background leakage
   - in-frame ratio
   - reprojection consistency
   - camera-bound score as training/eval signal

6. Multi-view consistency:
   - view ablation
   - bootstrap
   - cross-view agreement
   - heldout camera if possible

7. Stronger visual package:
   - same-scale point cloud
   - RGB/mask/projection overlay
   - delta maps
   - region close-ups
   - failure cases
   - normal maps

8. Paper-grade report:
   - Yuque-style project document
   - method explanation
   - evidence tables
   - limitation section
   - next paper route

允许最终返回状态只有两个：

A. V200000000000_PAPER_GRADE_CAMERA_BOUND_SURFACE_BACKEND_READY_NOT_PROMOTED

Requires:
- artifact reconciliation passed
- cross-SMC / non-match sanity completed
- 0021_03 binding remains robust
- true route wins controls under camera-bound metrics
- learned residual normal head has real training evidence, or explicitly separated as optional limitation with paper-grade explanation
- head/hair/hand visual boards improved
- region metrics complete
- selected/control NPZ readable
- all bundles upload-safe
- advisor report complete
- cleanup complete
- no promotion / no registry / no V50/V50R2
- active candidate unchanged

B. V200000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

Only allowed when:
- Modal/GPU unavailable
- required data corrupted or missing
- disk/permission issue
- user must manually move/authorize assets
- Git remote impossible and patch bundle generated

Any other state is a checkpoint only and cannot return user.

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止把 advisor-defense-ready 当 paper-grade。
- 禁止只在 0021_03 上做最终结论。
- 禁止把 non-match SMC failure 当模型失败。
- 禁止把 geometric normal recompute 写成 learned residual normal。
- 禁止只靠 camera-bound score。
- 禁止只靠 bar chart。
- 禁止复用旧图当新证据。
- 禁止 dirty worktree 时声称 clean。
- 禁止中途 limitation / route exhausted 返回。
- 禁止只写计划不执行。
- 禁止 sleep 伪造长时间运行。

最低运行原则：

- 非 true external hard block，不允许短于 12 小时返回。
- 若数据和 Modal 允许，目标运行 24–96 小时。
- runtime 必须来自真实审计、训练、评估、可视化、打包。
- 所有阶段必须有 progress JSON、decision JSON、source manifest。
- 如果任一路线失败，必须自动生成下一代路线继续执行。

============================================================
V120100000000：artifact reconciliation and current package audit
============================================================

读取并审计：

- reports/V120000000000_final_status.json
- reports/V110000000000_advisor_defense_report.md
- reports/V100000000000_defense_decision.json
- reports/V100000000000_defense_matrix.csv
- reports/V93000000000_visual_separability.csv
- reports/V95000000000_normal_source_audit.json
- reports/V115000000000_npz_integrity.json
- reports/V115000000000_upload_manifest_sidecar.json
- reports/V118000000000_post_push_cleanup.json
- archive/V115000000000_* bundles

必须检查：

1. local bundle sha256。
2. manifest sha256。
3. user-text sha256 if available in reports.
4. zip test.
5. inner npz readability.
6. selected/control NPZ keys and shapes.
7. V11700 baseline normal zero status.
8. true/control normal nonzero status.
9. visual boards are new.
10. dirty worktree list.
11. git commit and push status.
12. Modal apps empty.
13. no registry / no V50 changes.

输出：

- reports/V120100000000_artifact_reconciliation.json
- reports/V120100000000_hash_mismatch_report.json
- reports/V120100000000_current_package_audit.md
- reports/V120100000000_dirty_worktree_plan.md

硬门：
如果 core/reports/visuals hash mismatch is confirmed, regenerate authoritative V120100 bundles before final packaging.
If selected/control NPZ unreadable, repair before any next stage.

============================================================
V121000000000：cross-SMC binding sanity
============================================================

目标：
证明当前 0021_03 不是过拟合偶然，同时明确哪些 SMC 是 true-match / non-match。

扫描：

G:\数据集\datasets\data_used_in_4K4D\annotations\*.smc

SMCs:
- 0012_11
- 0013_01
- 0013_03
- 0013_09
- 0013_11
- 0019_08
- 0021_03
- 0023_06

对每个 SMC 执行：

1. Load K/RT/mask/SMPLx/keypoints if available.
2. Run binding search:
   - RT convention
   - axis flip
   - scale
   - translation
   - unit
3. Compute:
   - mean bbox IoU
   - silhouette IoU
   - center error
   - mask coverage
   - in-frame ratio
   - positive depth ratio
   - true vs controls camera-bound rank
4. Classify:
   - true-match
   - plausible-match
   - non-match
5. For non-match, verify expected failure:
   - low mask IoU
   - unstable binding
   - true route not necessarily rank1

输出：

- reports/V121000000000_cross_smc_binding_scan.csv
- reports/V121000000000_cross_smc_classification.json
- boards/V121000000000_cross_smc_binding_grid.png

硬门：
0021_03 must remain top true-match.
Non-match SMCs should not be used to disprove current route.
If multiple SMCs match, run multi-sequence eval.

============================================================
V122000000000：binding stress test v2
============================================================

对 0021_03 做更强 stress test：

1. View ablation.
2. Bootstrap over views.
3. Camera subset:
   - front-ish only
   - side-ish only
   - sparse 3-view
   - full 6-view
4. Translation perturbation.
5. Scale perturbation around 1.5.
6. Axis perturbation.
7. RT perturbation.
8. Mask erosion/dilation.
9. Resolution perturbation:
   - 518
   - 512
   - 520
10. Confidence threshold perturbation.

输出：

- reports/V122000000000_binding_stress_test.csv
- reports/V122000000000_binding_stress_decision.json
- boards/V122000000000_binding_stress_visual.png

硬门：
margin p05 must stay positive or limitation must trigger repair.

============================================================
V123000000000：learned residual normal head
============================================================

目标：
从 valid exported/geometric-compatible normal 升级到 learned residual normal evidence。

实现：

models/v123_learned_normal_residual_head.py
training/losses/v123_normal_residual_losses.py
tools/v123_train_normal_residual_head.py

Inputs:
- true_camera_bound_transport_world_points
- geometric normals computed from points
- V920 surface normals
- semantic/topology features
- region masks

Outputs:
- learned_normal_residual
- learned_normal
- normal_conf
- residual_magnitude
- source_manifest

Losses:
- cosine loss vs geometric teacher
- smoothness regularization
- curvature consistency
- part boundary normal loss
- hand/hair normal loss
- random/shuffled semantic control margin

Controls:
- no semantic
- random semantic
- shuffled semantic
- geometric-only
- MLP-only

输出：

- reports/V123000000000_normal_residual_training.csv
- reports/V123000000000_normal_residual_eval.json
- boards/V123000000000_normal_residual_visual.png

硬门：
If learned residual head fails, do not claim learned normal success.
Proceed with limitation only after trying repair.

============================================================
V124000000000：part-local surface specialists
============================================================

目标：
让 head/hair/hand close-up 更强。

实现四个 specialists：

1. HeadFaceSurfaceSpecialist
2. HairlineBoundarySpecialist
3. LeftHandSurfaceSpecialist
4. RightHandSurfaceSpecialist

Each uses:
- surface graph local crop
- local curvature
- part id
- barycentric/face id
- camera mask
- region confidence
- anti-background leakage loss

Outputs:
- part-local delta_point
- part-local delta_normal
- part-local confidence
- boundary mask

Controls:
- same specialist with random semantic
- same specialist with shuffled semantic
- local smoothing
- support only
- observation only

输出：

- reports/V124000000000_part_specialist_eval.csv
- reports/V124000000000_part_specialist_decision.json
- boards/V124000000000_part_specialist_head_hair_hand.png

硬门：
Part specialist must improve at least two of:
- visual separability
- region metric
- boundary sharpness
- background leakage
- normal consistency

If fails, enter V125 repair.

============================================================
V125000000000：differentiable camera-bound loss prototype
============================================================

目标：
把 camera-bound score 从离线评估升级为可训练 loss prototype。

实现：

training/losses/v125_differentiable_camera_bound_losses.py

Loss components:
- soft silhouette IoU
- projected point inside mask
- background leakage
- bbox center alignment
- depth positivity
- in-frame ratio
- part-region projection consistency
- camera-bound contrastive margin true vs random/shuffled/smoothing

要求：
- differentiable where possible
- fallback approximation documented
- no hard argmax only
- controls included

输出：

- reports/V125000000000_camera_bound_loss_design.md
- reports/V125000000000_camera_bound_loss_smoke.json

硬门：
If differentiability incomplete, mark prototype, not success.

============================================================
V126000000000：paper-grade rerun matrix
============================================================

Run a paper-grade defense matrix with improved modules.

Groups:
1. true_camera_bound_surface_backend
2. random_surface_semantic
3. shuffled_surface_semantic
4. local_knn_smoothing
5. no_surface_graph
6. random_surface_graph
7. observation_only
8. support_only
9. no_sparseconv_mlp
10. no_teacher
11. geometric_normal_only
12. learned_normal_residual

Minimum:
- 5 seeds per group
Target:
- 10 seeds per group if resources allow

Must record:
- runtime
- GPU type
- training steps
- seed variance
- source manifest
- camera-bound metrics
- region metrics
- visual separability
- normal metrics

输出：

- reports/V126000000000_paper_grade_matrix.csv
- reports/V126000000000_seed_metrics.csv
- reports/V126000000000_control_ranking.csv
- output/V126000000000_predictions/*

硬门：
If true does not beat controls, enter auto repair.
If learned normal does not help, keep as limitation.

============================================================
V127000000000：stronger visual boards
============================================================

Generate new paper-grade boards.

Boards:

1. fullbody:
   V11700 / V120 true / V126 true / random / smoothing / support / observation

2. head-face close-up:
   same columns

3. hairline close-up:
   same columns

4. left-hand close-up:
   same columns

5. right-hand close-up:
   same columns

6. projection overlay:
   true/control projected into 0021_03 masks

7. cross-SMC sanity:
   0021_03 vs non-match SMC

8. normal:
   geometric / learned residual / controls

9. part specialist:
   before/after local repair

10. failure cases:
   weak cameras / weak regions

要求：
- same scale
- same viewpoint
- same point size
- same crop
- region labels
- metric labels
- no old image reuse

输出：
- boards/V127000000000_fullbody.png
- boards/V127000000000_head_face.png
- boards/V127000000000_hairline.png
- boards/V127000000000_left_hand.png
- boards/V127000000000_right_hand.png
- boards/V127000000000_projection_overlay.png
- boards/V127000000000_cross_smc.png
- boards/V127000000000_normals.png
- boards/V127000000000_part_specialist.png
- boards/V127000000000_failure_cases.png

============================================================
V128000000000：paper-grade decision
============================================================

PAPER_GRADE_READY requires:

1. V120 advisor-defense still valid.
2. Cross-SMC sanity complete.
3. 0021_03 binding robust.
4. true route beats random/shuffled/smoothing/support/observation/noSparse.
5. paper-grade matrix complete.
6. region metrics complete.
7. normal source clear, learned residual either valid or limitation explicitly bounded.
8. head/hair/hand visuals improved or limitation bounded.
9. selected/control NPZ readable.
10. bundles planned.
11. no promotion/no registry/no V50 modification.

输出：

- reports/V128000000000_paper_grade_decision.json

If fails:
- auto route based on failure:
  - random/smoothing close -> stronger contrastive/topology
  - normal weak -> normal repair
  - part weak -> high-res part specialist
  - cross-SMC weak -> multi-sequence binding calibrator
  - visual weak -> visual-first densification

Do not return; execute repair route.

============================================================
V130000000000：auto repair loop
============================================================

If V128 fails, run repair loops until success or true external hard block.

Routes:

A. Cross-SMC weak:
- multi-sequence binding calibrator
- sequence classifier
- false-match rejection
- heldout camera validation

B. Visual weak:
- part-local densification
- confidence-gated rendering
- surface neighborhood interpolation
- anti-noise filter

C. Normal weak:
- residual normal head v2
- normal teacher distillation
- part-specific normal loss

D. Random/smoothing close:
- topology contrastive loss
- graph transformer
- TSDF/SDF residual backend
- differentiable renderer loss

E. Hand/hair weak:
- high-res crop
- object leakage exclusion
- local SMPL surface patch decoder

For each repair:
- write route file
- execute route
- evaluate
- if improved, rerun V126/V127/V128
- if failed, try next repair

输出：
- reports/V130000000000_auto_repair_history.csv
- docs/goals/V130000000000_auto_generated_routes/*.md

============================================================
V180000000000：Yuque-style paper-grade advisor report
============================================================

Write Chinese project document in Yuque-style.

Structure:

# 先给结论

# 当前版本定位

# 相比 V120 解决了什么

# 方法架构

# 数据与坐标绑定

# 多 SMC 复核

# 模型与训练

# Controls 与消融

# 点云可视证据

# Normal 证据

# Hand / Hair / Head 局部证据

# 仍然的限制

# 不 promotion 的原因

# 下一步论文计划

输出：
- reports/V180000000000_paper_grade_advisor_report.md
- reports/V180000000000_one_page.md
- reports/V180000000000_limitations.md

============================================================
V190000000000：upload-safe packaging
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
- archive/V190000000000_core_evidence_bundle.zip
- archive/V190000000000_reports_bundle.zip
- archive/V190000000000_visuals_bundle.zip
- archive/V190000000000_selected_predictions_bundle.zip
- archive/V190000000000_controls_bundle.zip
- reports/V190000000000_upload_manifest_sidecar.json
- reports/V190000000000_omitted_large_file_manifest.json

============================================================
V195000000000：post-push cleanup
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
- reports/V195000000000_post_push_cleanup.json

============================================================
V200000000000：final return gate
============================================================

Only two final return states are allowed:

1. V200000000000_PAPER_GRADE_CAMERA_BOUND_SURFACE_BACKEND_READY_NOT_PROMOTED

Requires:
- paper-grade decision passed
- report complete
- bundles complete
- cleanup complete
- no promotion / no registry / no V50/V50R2
- active candidate unchanged

2. V200000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

Requires:
- all automatic repair routes attempted or impossible
- user action checklist precise
- evidence bundles complete
- cleanup complete

Any other state cannot return.
If failed, return to V130 auto repair loop.
