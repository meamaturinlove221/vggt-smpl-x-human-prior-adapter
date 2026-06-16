# V6010000000–V9000000000 Formal GPU Full-View Matrix Execution Goal

你现在接手 D:\vggt\vggt-feature-adapter 的下一阶段任务。

证据根目录固定为：

D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild

注意：
repo 是 D:\vggt\vggt-feature-adapter
evidence root 是 D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild
二者不能混写。

本轮允许使用 /goal，但本文件是主执行规范。不要发起 agent。不要使用 subagent。必须由当前主线程长期执行、自动路由、自动修复、自动续跑，直到达到导师目标或真实不可绕过硬阻断。

当前事实必须接受：

1. G:\数据集\datasets\smplx\SMPLX_NEUTRAL.npz 已找到。
2. 不能再返回 missing SMPL-X model。
3. Schema v2 已达到 6 x 81 x 518 x 518。
4. Schema v2 仍不是完全 paper-grade：
   - body_part_id = kinematic_derived
   - surface_distance = model_derived
   - curvature = mesh_derived_proxy
5. V380 是 CPU remote pilot，不是 formal GPU full-view training。
6. V430 GPU smoke 已跑通，Modal A10G cuda_available=true。
7. GPU smoke 不等于正式 V460 40-run matrix。
8. V460 要求的 8 groups x 5 seeds = 40 runs 没有完成。
9. V380 pilot 里 random semantic 和 local KNN smoothing 都超过 true_full。
10. V380 input normals 全 0，本轮重算 geometric normal，nonzero ratio 约 0.9923，但这不是 learned normal head 成功。
11. V390 region metrics 是 not_evaluated，不能写 paper-grade region proof。
12. V415 selected bundle 曾有二层 CRC 错误，V422 已修 selected/control predictions，但下一轮必须重新审计内部 npz 可读性。
13. V415 controls bundle 原本没有 controls predictions，不能写 controls complete。
14. 旧灰度图/旧 close-up 图不能复用成本轮视觉证据。
15. Worktree 仍可能 dirty；cleanup 只能诚实写 dirty，不能伪称 clean。
16. No promotion。
17. No strict registry。
18. No V50/V50R2 modification。
19. Do not replace active candidate。
20. Active candidate remains V11700_gap_reduction_branch_520。

上一轮最终状态：

V6000000000_INVALID_CONTROLLER_NOT_IMPLEMENTED

原因：
V430 GPU smoke exists, but V460 formal GPU full-view 40-run matrix was not completed. 因此没有足够 formal matrix evidence 判断 semantic causality / smoothing dominance / random dominance。

本轮总目标：

实现并执行真正 formal GPU full-view matrix，不再停在 smoke/pilot/controller audit。

必须完成：

1. 重新审计 V560 包、内层 npz、manifest hash。
2. 修复 worktree dirty 策略，提交或诚实记录。
3. 修复/实现 V460 formal GPU full-view controller。
4. 运行核心正式矩阵：
   8 groups x 5 seeds = 40 formal GPU runs。
5. 每个 run 必须是 GPU formal run，不是 CPU remote pilot。
6. 每个 run 必须产出 readable full-view predictions.npz。
7. 每个 prediction 必须包含：
   - world_points: 6 x 518 x 518 x 3
   - depth: 6 x 518 x 518
   - confidence or world_points_conf
   - normal: 6 x 518 x 518 x 3
   - normal_conf
8. 每个 run 必须产出：
   - eval.json
   - source_manifest.json
   - quality.json
   - board.png
   - training log
   - runtime
   - Modal app id
   - GPU type
9. 必须评估 region metrics：
   - full_body
   - head_face
   - hairline
   - left_hand
   - right_hand
10. 必须生成新的 paper-grade point cloud boards，不准复用旧图。
11. 只有 true_full 在正式矩阵中赢过 random/shuffled/support/observation/noSparse/local smoothing，且 region/normal/structure 指标不差，才能写 semantic causal confirmed。
12. 如果失败，必须进入自动修复循环，而不是立即返回。
13. 自动修复最多 5 轮，每轮都要跑 reduced matrix 证明修复是否有效。
14. 如果仍失败，才进入允许终态。
15. 最终必须生成新版导师文档、upload-safe bundles、omitted manifest、post-push cleanup。

允许最终状态：

1. V9000000000_FULLVIEW_GPU_SEMANTIC_CAUSAL_CONFIRMED_NOT_PROMOTED
2. V9000000000_FULLVIEW_GPU_SEMANTIC_IMPROVED_LIMITATIONS_DISCLOSED
3. V9000000000_RANDOM_SEMANTIC_DOMINANT_LIMITATIONS_DISCLOSED
4. V9000000000_SUPPORT_DOMINANT_LIMITATIONS_DISCLOSED
5. V9000000000_OBSERVATION_DOMINANT_LIMITATIONS_DISCLOSED
6. V9000000000_SMOOTHING_DOMINANT_LIMITATIONS_DISCLOSED
7. V9000000000_NORMAL_HEAD_FAILED_WITH_ANALYSIS
8. V9000000000_TRUE_HARD_BLOCKED_MODAL_OR_DATA
9. V9000000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS
10. V9000000000_INVALID_CONTROLLER_NOT_IMPLEMENTED
11. V9000000000_INVALID_FAST_RETURN

禁止事项：

- 禁止 promotion。
- 禁止 strict registry。
- 禁止修改 V50/V50R2。
- 禁止替换 active candidate。
- 禁止发起 agent / subagent。
- 禁止把 GPU smoke 当 formal matrix。
- 禁止把 CPU remote pilot 当 formal GPU training。
- 禁止把 3 seeds pilot 当 40-run formal matrix。
- 禁止把 random semantic 赢过 true_full 包装成 semantic success。
- 禁止把 local KNN smoothing 赢过 true_full 包装成 semantic success。
- 禁止把 normal 几何重算包装成 learned normal head success。
- 禁止只靠 mean_delta 判断成功。
- 禁止只出 heatmap / grayscale depth map。
- 禁止复用旧 V460/V560/V1050 图当新图。
- 禁止 selected_predictions 中存在坏 npz 仍声明 bundle fully verified。
- 禁止 controls bundle 没有 controls predictions 时写 controls complete。
- 禁止 repo dirty 时声称 clean。
- 禁止中途返回用户。
- 禁止达到中间 failure state 就返回。必须先自动修复、重跑、再判断。
- 只有达到 V9000000000 允许终态，且完成 advisor report + bundles + cleanup，才能返回。

最低运行原则：

- 非 true hard block，不允许短于 12 小时返回。
- 若 Modal 和数据允许，目标运行 24–72 小时。
- 不允许 sleep 伪造运行时间。
- runtime 必须来自真实 GPU training、eval、visual、packaging、audit。
- 如果提前终止，必须有 hard blocker 文件证据。
- 如果只是 smoke/pilot，状态必须写 INVALID_FAST_RETURN 或 INVALID_CONTROLLER_NOT_IMPLEMENTED，不得写 limitation final。

============================================================
V6010000000：goal and artifact preflight
============================================================

保存本目标文件，并生成：

reports/V6010000000_goal_file_manifest.json

读取并审计：

- reports/V6000000000_final_status.json
- reports/V5000000000_causality_decision.json
- reports/V4300000000_gpu_smoke.json
- reports/V5400000000_advisor_report.md
- reports/V5600000000_upload_manifest_sidecar.json
- reports/V5800000000_post_push_cleanup.json
- archive/V5600000000_core_evidence_bundle.zip
- archive/V5600000000_reports_bundle.zip
- archive/V5600000000_visuals_bundle.zip
- archive/V5600000000_selected_predictions_bundle.zip
- archive/V5600000000_controls_bundle.zip

必须检查：

1. 上传包实际 sha256。
2. sidecar manifest 记录 sha256。
3. 是否一致。
4. 外层 zip 是否 clean。
5. 内层 npz 是否 clean。
6. 每个 npz 是否可 np.load。
7. world_points/depth/normal/conf 是否可读。
8. normal 是否非零。
9. controls bundle 是否含 predictions。
10. visuals 是否是新图还是旧图。
11. worktree dirty 原因。
12. commit 和 branch。
13. Modal apps 是否清空。
14. registry/V50/V50R2/active candidate 是否未改。

输出：

- reports/V6010000000_preflight_audit.json
- reports/V6010000000_npz_integrity_report.json
- reports/V6010000000_hash_reconciliation.json
- reports/V6010000000_dirty_worktree_plan.md

硬门：
如果发现 selected/control predictions 内部 npz 坏，先修复再进入 V610。
如果 manifest hash 和实际上传 hash 不一致，后续必须重打包。
如果 worktree dirty，必须决定 commit / stash / intentional dirty，不得声称 clean。

============================================================
V6100000000：formal matrix controller implementation audit
============================================================

目标：
确认 V460 正式 GPU full-view matrix controller 是否真实实现。

必须查找：

- tools/v4300000000_modal_gpu_fullview_semantic_trainer.py
- tools/v4600000000_formal_gpu_fullview_matrix_controller.py
- modal scripts
- dataset loader
- model loader
- loss loader
- eval script
- normal recompute script
- region metric script

必须判定：

1. 是否存在正式 matrix controller。
2. 是否能提交 8 x 5 = 40 runs。
3. 是否支持 resume。
4. 是否支持 retry。
5. 是否支持 Modal volume pull。
6. 是否支持 CLI hang recovery。
7. 是否区分 GPU formal run 和 CPU pilot。
8. 是否写 source_manifest。
9. 是否检查 npz internal readability。
10. 是否输出 full-view predictions。
11. 是否写 region metrics。
12. 是否写 normal metrics。

输出：

- reports/V6100000000_controller_audit.json
- reports/V6100000000_missing_controller_items.md

硬门：
如果 controller 不存在或只是 smoke wrapper，必须实现 V620。
不得再返回 INVALID_CONTROLLER_NOT_IMPLEMENTED，除非实现失败且给出具体原因。

============================================================
V6200000000：implement formal GPU matrix controller
============================================================

实现：

tools/v6200000000_formal_gpu_fullview_matrix_controller.py

要求：

Core groups:
1. true_full
2. same_support_random_semantic
3. same_support_shuffled_semantic
4. support_only
5. observation_only
6. no_sparseconv_mlp
7. local_knn_smoothing
8. no_teacher

每组 5 seeds：
seed = 0,1,2,3,4

总计 40 runs。

每个 run 必须使用：

- Modal GPU
- cuda_available = true
- gpu_type != modal_cpu_remote_pilot
- training_steps > 1
- runtime_seconds > meaningful threshold
- full-view prediction output
- normal output nonzero or recomputed with source marked
- source_manifest

Controller 必须支持：

- progress JSON
- per-run log
- retry up to 3
- resume from completed run
- remote complete but local pull incomplete recovery
- npz CRC repair / repull
- Modal app cleanup
- failure manifest
- partial matrix summary
- no duplicate rerun unless invalid

输出：

- tools/v6200000000_formal_gpu_fullview_matrix_controller.py
- reports/V6200000000_controller_implementation.md
- reports/V6200000000_dry_run_plan.json

硬门：
dry run 必须列出 40 run specs。
每个 run spec 必须含 group/seed/gpu/formal_mode/output path。

============================================================
V6300000000：formal GPU matrix dry run
============================================================

不训练，只做 dry run validation。

检查：

1. dataset exists
2. semantic schema exists
3. full-view tensors shape valid
4. support/semantic/observation split valid
5. true/random/shuffled have matched support and matched observation
6. Modal image can import torch
7. Modal image can access GPU
8. output root exists
9. controller can resume
10. no old residual composer/postcompose target

输出：

- reports/V6300000000_dry_run_validation.json
- reports/V6300000000_run_spec_table.csv

硬门：
dry run fail 必须自动修复，不得返回。

============================================================
V6400000000：formal GPU matrix pilot sanity, not final
============================================================

跑 1 group x 1 seed sanity：

- true_full seed0

要求：
- Modal GPU
- cuda_available true
- runtime meaningful
- training_steps > 1
- predictions full-view readable
- normal nonzero
- eval exists
- source_manifest clean

输出：
- reports/V6400000000_gpu_sanity.json
- output/V6400000000_gpu_sanity/true_full_seed0/predictions.npz

硬门：
V640 只是 sanity，不是 final。
不得在 V640 后返回。

============================================================
V6500000000：formal GPU core matrix wave 1
============================================================

运行前 20 runs：

Groups:
1. true_full seeds 0-4
2. same_support_random_semantic seeds 0-4
3. same_support_shuffled_semantic seeds 0-4
4. local_knn_smoothing seeds 0-4

为什么先跑这四组：
- 直接验证 true semantic 是否赢 random/shuffled
- 直接验证 smoothing 是否仍超过 true

输出：

- reports/V6500000000_wave1_progress.json
- reports/V6500000000_wave1_matrix.csv
- reports/V6500000000_wave1_failures.json
- output/V6500000000_wave1_predictions/*/predictions.npz

硬门：
如果 random semantic 或 smoothing 明显高于 true_full，不能提前返回。
必须进入 repair cycle V700 后重跑 reduced wave。
如果 runs fail，retry/repull/recover，不得短返。

============================================================
V6600000000：formal GPU core matrix wave 2
============================================================

运行后 20 runs：

Groups:
1. support_only seeds 0-4
2. observation_only seeds 0-4
3. no_sparseconv_mlp seeds 0-4
4. no_teacher seeds 0-4

输出：

- reports/V6600000000_wave2_progress.json
- reports/V6600000000_wave2_matrix.csv
- reports/V6600000000_wave2_failures.json
- output/V6600000000_wave2_predictions/*/predictions.npz

硬门：
40 runs 全部 valid 才能写 formal_gpu_fullview_matrix_completed=true。

============================================================
V6700000000：formal matrix validation
============================================================

验证 40 runs：

1. all runs complete
2. all predictions readable
3. all eval readable
4. all source manifests readable
5. all normals nonzero or justified
6. all region masks nonempty
7. all run GPU formal, not CPU pilot
8. all group/seed unique
9. no old postcompose leakage
10. no active candidate replacement

输出：

- reports/V6700000000_formal_matrix_validation.json
- reports/V6700000000_seed_level_metrics.csv
- reports/V6700000000_control_ranking.csv

硬门：
如果 any required run invalid，return to V650/V660 retry, not final.

============================================================
V6800000000：region and structure metrics
============================================================

Compute for all 40 runs.

Regions:
- full_body
- head_face
- hairline
- left_hand
- right_hand

Metrics:
- mean_delta
- local_delta
- normal_consistency
- normal_nonzero_ratio
- outlier_ratio
- background_leakage
- point_density
- component_count
- hand_isolated_ratio
- right_hand_planar_score
- hairline_boundary_sharpness
- horizontal_band_artifact
- surface_continuity
- reprojection_consistency

Comparisons:
- true_full vs random semantic
- true_full vs shuffled semantic
- true_full vs support_only
- true_full vs observation_only
- true_full vs noSparse MLP
- true_full vs local KNN smoothing
- true_full vs no_teacher
- true_full vs V11700
- true_full vs V770/V999 if available

输出：

- reports/V6800000000_region_metrics.csv
- reports/V6800000000_structure_metrics.csv
- reports/V6800000000_region_decision.json

硬门：
region not_evaluated 不得进入 success。
hand/hair 必须单独评估。

============================================================
V6900000000：formal visual boards
============================================================

生成新图，不得复用旧图。

Boards:

1. boards/V6900000000_fullbody_pointcloud.png
   columns:
   V770 / V999 / V11700 / true_full / random_semantic / smoothing / best_final

2. boards/V6900000000_head_hair_hand_closeups.png
   rows:
   head_face / hairline / left_hand / right_hand
   columns:
   V11700 / true_full / random_semantic / smoothing / best_final

3. boards/V6900000000_counterfactual_controls.png
   true vs random vs shuffled

4. boards/V6900000000_smoothing_controls.png
   true vs noSparse vs local KNN

5. boards/V6900000000_normal_comparison.png
   learned/input/geometric normals, normal confidence

6. boards/V6900000000_failure_cases.png

硬门：
不允许 heatmap 冒充点云。
不允许只用 grayscale depth。
必须标明 group/seed/candidate/source。

============================================================
V7000000000：formal causality decision
============================================================

判断：

SEMANTIC_CAUSAL_CONFIRMED requires:
- formal 40-run GPU matrix complete
- true_full mean > random_semantic mean
- true_full mean > shuffled_semantic mean
- true_full mean > support_only mean
- true_full mean > observation_only mean
- true_full mean > no_sparseconv_mlp mean
- true_full mean > local_knn_smoothing mean
- true_full mean > no_teacher mean or justified
- region metrics true_full not worse
- head_face positive
- hairline positive
- left/right hand positive
- normal nonzero and consistent
- source manifests clean
- no teacher/postcompose leakage

If true_full beats random/shuffled but not smoothing:
SMOOTHING_DOMINANT or SEMANTIC_IMPROVED_LIMITATIONS, depending margins.

If random semantic beats true_full:
RANDOM_SEMANTIC_DOMINANT_LIMITATIONS.

If local KNN smoothing beats true_full:
SMOOTHING_DOMINANT_LIMITATIONS.

If support/observation beats true:
SUPPORT/OBSERVATION_DOMINANT.

输出：

- reports/V7000000000_causality_decision.json
- reports/V7000000000_control_ranking.csv

硬门：
不准因为 V700 失败就返回。
必须进入 V710 repair cycles，除非 Modal/data hard block.

============================================================
V7100000000：automatic repair cycle 1
============================================================

根据 V700 结果自动选择修复路线。

If smoothing dominant:
- increase anti-smoothing margin
- add curvature loss
- add boundary loss
- add local normal variance loss
- penalize local KNN mimic
- add hand/hair structure loss

If random semantic dominant:
- strengthen semantic contrastive loss
- verify random semantic generation
- reduce observation shortcut
- increase semantic token capacity
- add semantic consistency loss

If support dominant:
- reduce support capacity
- support binary mask only
- increase support dropout
- stricter support invariance

If observation dominant:
- observation dropout
- detach residual
- view count normalization
- observation-only penalty

If normal failed:
- geometric normal as teacher
- train normal residual head
- re-evaluate normal

Run reduced repair matrix:
- true_full seeds 0-2
- random_semantic seeds 0-2
- shuffled_semantic seeds 0-2
- local_knn_smoothing seeds 0-2
- no_sparseconv_mlp seeds 0-2

输出：
- reports/V7100000000_repair_cycle1.csv
- reports/V7100000000_repair_cycle1_decision.json

============================================================
V7200000000：automatic repair cycle 2
============================================================

If V710 improves:
- run expanded repair matrix 5 groups x 5 seeds.
If no improvement:
- switch repair strategy based on next dominant failure.

输出：
- reports/V7200000000_repair_cycle2.csv
- reports/V7200000000_repair_cycle2_decision.json

============================================================
V7300000000：automatic repair cycle 3
============================================================

Final repair attempt.

If still failing:
- route exhausted with failure analysis.
If improved:
- rerun full 40-run formal matrix.

输出：
- reports/V7300000000_repair_cycle3.csv
- reports/V7300000000_repair_cycle3_decision.json

============================================================
V7400000000：post-repair formal rerun if warranted
============================================================

If any repair cycle shows clear improvement:
rerun formal 40-run matrix with best repair config.

输出：
- reports/V7400000000_post_repair_matrix.csv
- reports/V7400000000_post_repair_seed_metrics.csv
- boards/V7400000000_post_repair_visuals.png

============================================================
V7600000000：candidate synthesis
============================================================

Generate candidates:

- best true_full
- best repaired true_full
- conservative no-regression candidate
- best semantic-improved candidate
- best normal-fixed candidate
- best hand/hair candidate
- limitation-disclosed candidate

Raw candidates >= 100
Unique candidates >= 20

输出：
- reports/V7600000000_candidate_synthesis.csv
- reports/V7600000000_uniqueness_audit.csv

============================================================
V7800000000：strict final evaluation
============================================================

Compare:
- V770
- V999
- V11700
- V420
- V600
- V650/V660/V740 candidates
- random/shuffled/support/observation/noSparse/smoothing controls

Hard gates:
- upload safe
- no promotion
- no registry
- no V50/V50R2
- source manifests clean
- no teacher/postcompose leakage
- formal GPU matrix complete or hard-blocked
- semantic causal margin positive or limitation disclosed
- random semantic not beating true or limitation disclosed
- local smoothing not beating true or limitation disclosed
- normal valid or limitation disclosed
- region metrics complete or limitation disclosed
- hand/hair not worse or limitation disclosed
- full body no regression or limitation disclosed
- selected/control npz readable
- candidate uniqueness acceptable

输出：
- reports/V7800000000_strict_eval.json
- reports/V7800000000_ranked_candidates.csv

============================================================
V8000000000：advisor report
============================================================

写中文导师报告：

必须包含：
1. V600 为什么 invalid controller。
2. 本轮是否完成 formal GPU 40-run matrix。
3. true/random/shuffled semantic 结果。
4. support/observation/noSparse/local smoothing 结果。
5. normal 状态。
6. full body/head/hair/hand 点云图。
7. 是否确认 SMPL semantic causality。
8. 如果没确认，主导因素是什么。
9. repair cycles 做了什么。
10. 为什么不 promotion。
11. 下一步论文路线。

输出：
- reports/V8000000000_advisor_report.md
- reports/V8000000000_one_page.md
- reports/V8000000000_limitations.md

============================================================
V8400000000：upload-safe packaging
============================================================

打包：

- core evidence <=50MB
- reports <=50MB
- visuals <=150MB
- selected predictions <=250MB
- controls predictions <=250MB
- omitted large file manifest
- authoritative sidecar manifest

必须：
- zip clean
- all under 500MB
- final status in core
- cleanup in core
- sidecar manifest records actual hashes after zip close
- all selected predictions npz internally readable
- all controls predictions npz internally readable
- no CRC errors
- controls bundle includes control predictions or explicitly limitation-disclosed

输出：
- archive/V8400000000_core_evidence_bundle.zip
- archive/V8400000000_reports_bundle.zip
- archive/V8400000000_visuals_bundle.zip
- archive/V8400000000_selected_predictions_bundle.zip
- archive/V8400000000_controls_bundle.zip
- reports/V8400000000_upload_manifest_sidecar.json
- reports/V8400000000_omitted_large_file_manifest.json

============================================================
V8600000000：post-push cleanup
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
- reports/V8600000000_post_push_cleanup.json

硬门：
dirty 不得声称 clean。
not committed 必须解释。

============================================================
V9000000000：final return
============================================================

最终返回必须包含：

- final status
- formal GPU full-view matrix completed or not
- semantic causality confirmed or not
- dominant factor if failed
- repair cycles summary
- normal head status
- best candidate
- advisor report path
- upload bundles and sha256
- omitted manifest
- cleanup report
- branch/commit

只有达到 V9000000000 允许终态才能返回用户。
