# 当前项目 vs 备份项目完整对比报告（2026-04-01）

## 对比范围

- 当前目录：`F:\vggt\vggt-main`
- 备份目录：`G:\项目备份\vggt\vggt-main`
- 源码级文件统计采用 `rg --files` 对两边做同口径扫描。
- 额外说明：当前工作区还包含 `.venv5080`、`output`、`.vscode`、`__pycache__` 等运行态目录，它们不属于备份源树的一部分，我单独列在“运行态差异”里。

## 结论先看

这不是“小修小补”，而是一次把原始 VGGT 项目扩展成 ZJU 几何训练与研究工作区的演进。

源码级结果：

- 当前文件数：`461`
- 备份文件数：`169`
- 新增文件：`292`
- 删除文件：`0`
- 直接修改原有文件：`11`

工作主线可以概括成 5 件事：

1. 把项目目标从“原始推理展示”扩展到“ZJU 人体场景下的 depth+camera 几何验证与微调”。
2. 新增一套 ZJU 几何数据集接入层，支持 source policy、raw/cached 视图池、anchor、hardcase manifest。
3. 在训练链路里加入 `unproject_geometry` 及一整套面向 anchor / conf-depth / residual tail 的损失与调度机制。
4. 新增大量本地 / Modal / 夜间批处理 / 审核门控脚本，把实验从手动试跑变成可记录、可审核、可回退的流程。
5. 留下了完整研究文档链，显示方向从“验证 depth+camera 是否值得继续”推进到“source policy 提升后，如何做 residual hardcase coverage rebalancing”。

## 运行态差异（当前目录额外存在）

- .venv5080 (28148 files)
- .vscode (1 files)
- output (141602 files)
- __pycache__ (5 files)
- tmp_driver_live_20260329_013543.log (62182 bytes)
- tmp_driver_live_20260329_014446.log (246687 bytes)

解释：

- `.venv5080` 说明当前目录已经被当作本地训练/验证工作区使用，而不是纯备份源码。
- `output` 体量很大，说明大量实验结果、摘要、门控记录和夜间运行产物都保存在当前工作区外链输出目录里。
- `tmp_*` 日志对应 2026-03-29 的研究/守护脚本运行痕迹。

## 11 个直接修改的原文件

- `demo_gradio.py` / `visual_util.py`：把默认可视化分支从 pointmap 切到 `Depthmap and Camera Branch`，说明主推路径已从原始 point map 转向 depth+camera 几何链。
- `vggt/models/vggt.py`：把 `torch.cuda.amp.autocast` 改成设备无关的 `torch.amp.autocast(device_type=...)`，兼容新的 AMP 调用方式。
- `training/data/dataset_util.py`：`read_image_cv2` 增加 `cv2.imdecode(np.fromfile(...))` 回退逻辑，解决中文/非 ASCII 路径读图失败。
- `training/data/composed_dataset.py`：把 `conf_depth_point_masks`、`depth_conf_maps`、`foreground_masks`、anchor 信息、manifest 标记送进 batch，给后续 loss 和采样逻辑使用。
- `training/data/dynamic_dataloader.py`：允许单机非 DDP 运行，补充 `pin_memory_device`、`prefetch_factor`，并按 `torch.distributed` 实际状态创建 sampler。
- `training/data/worker_fn.py` / `training/train_utils/distributed.py`：为 `WORLD_SIZE=1` 和缺失 `RANK`/`LOCAL_RANK` 的 Windows 本地训练兜底。
- `training/launch.py`：自动补 `sys.path`，并支持 Hydra overrides，便于脚本化切配置跑实验。
- `training/trainer.py`：新增单进程训练路径、`torch.compile`、`train_progress` 注入、batch 翻转时同步 anchor 索引、loss 阶段标记与 `loss_objective` 兼容记录。
- `training/loss.py`：是本轮最大改动。核心新增包括 `unproject_geometry` 辅助损失、loss warmup / two-stage 权重调度、anchor/sample-manifest 条件缩放、conf/reg disagreement 与 unproject consistency 路由、`active_view_mean` 聚合、更加稳健的 quantile mask 与 gradient regularization。

修改文件清单：

- demo_gradio.py
- training\data\composed_dataset.py
- training\data\dataset_util.py
- training\data\dynamic_dataloader.py
- training\data\worker_fn.py
- training\launch.py
- training\loss.py
- training\train_utils\distributed.py
- training\trainer.py
- vggt\models\vggt.py
- visual_util.py

## 292 个新增文件的分组含义

### 1. 文档 `docs/`（`64` 个）

- `geometry_*` 文档（含 6src、12src、post-v9、source policy、hardcase bucket）：记录每轮实验、门控结果和下一步决策。
- `modal_*` / `zju_*` 文档：记录 Modal 运行、ZJU 几何 smoke、深度目标可靠性等专项结果。
- `mentor_*` 文档：保留方向讨论与整理稿，用来约束后续研究方向。

完整清单：

- docs\current_vs_backup_compare_20260401.md
- docs\geometry_6src_b13_b23_v9_upgrade_gate_20260324.md
- docs\geometry_6src_b2_policy_probe_20260324.md
- docs\geometry_6src_b2_v7_upgrade_gate_20260324.md
- docs\geometry_6src_b2_v8_completion_gate_20260324.md
- docs\geometry_6src_frameaware_v6_multiframe_transfer_gate_20260324.md
- docs\geometry_6src_hardcase_region_status_20260323.md
- docs\geometry_6src_hardcontrol_local_completion_20260324.md
- docs\geometry_6src_hybrid_policy_standardized_20260323.md
- docs\geometry_6src_hybrid_v5_legacy_backfill_20260324.md
- docs\geometry_6src_hybrid_v5_multiframe_local_gate_20260324.md
- docs\geometry_6src_hybrid_v5_transfer_gate_20260324.md
- docs\geometry_6src_source_loo_status_20260323.md
- docs\geometry_6src_source_swap_probe_20260323.md
- docs\geometry_6src_stage2_local_probes_20260323.md
- docs\geometry_6src_uniform_policy_followup_20260323.md
- docs\geometry_baseline_workflow.md
- docs\geometry_bottom_only_unproject_local_gate_20260323.md
- docs\geometry_confdepth_only_local_gate_20260325.md
- docs\geometry_direction_status_20260322_cloud_compare_completed.md
- docs\geometry_direction_status_20260322_night.md
- docs\geometry_direction_status_20260323_confgate_completed.md
- docs\geometry_direction_status_20260323_pair4000_completed.md
- docs\geometry_direction_status_20260323_threshold_and_pow2_completed.md
- docs\geometry_direction_status_20260323_warmup_pair_completed.md
- docs\geometry_direction_status_20260323_weight_sweep_completed.md
- docs\geometry_first_status_20260321.md
- docs\geometry_hardcase_bucket_definition_20260328.md
- docs\geometry_hardcase_bucket_mix_contract_20260328.md
- docs\geometry_minimal_finetune.md
- docs\geometry_next_manual_problem_after_promotion_20260328.md
- docs\geometry_next_step_after_20260323.md
- docs\geometry_post_promotion_gap_mining_20260328.md
- docs\geometry_post_v9_b11_stop_gate_20260324.md
- docs\geometry_post_v9_next_training_question_20260325.md
- docs\geometry_post_v9_nightly_automation_status_20260325.md
- docs\geometry_region_policy_contrast_20260323.md
- docs\geometry_reliable_region_unproject_local_gate_20260323.md
- docs\geometry_source_policy_acceleration_long_process_plan_20260328.md
- docs\geometry_source_policy_failure_boundary_and_next_problem_20260327.md
- docs\geometry_source_policy_next_family_manual_draft_20260328.md
- docs\geometry_source_policy_rawpool_coverage_matrix_20260325.md
- docs\geometry_source_policy_rawpool_local_gate_20260325.md
- docs\geometry_source_policy_rawpool_nightly_plan_20260325.md
- docs\geometry_source_policy_training_hook_local_gate_20260325.md
- docs\geometry_sparse_policy_followup_20260323.md
- docs\geometry_sparse_region_contrast_20260323.md
- docs\geometry_zju_12src_policy_probe_20260322.md
- docs\geometry_zju_baseline.md
- docs\geometry_zju_checkpoint_eval_20260322.md
- docs\geometry_zju_diagnostics_20260322.md
- docs\geometry_zju_sparse_policy_compare_20260322.md
- docs\geometry_zju_view_sweep_round1_20260322.md
- docs\geometry_zju_view_sweep_round2_targetaware_20260322.md
- docs\mentor_direction_cleaned_transcript_cn_20260321.md
- docs\mentor_direction_final_cn_20260321.md
- docs\modal_geometry_minimal_finetune.md
- docs\modal_local_stability_20260322.md
- docs\modal_zju_geometry_ablation_pair_20260322.md
- docs\modal_zju_geometry_minimal_finetune.md
- docs\modal_zju_geometry_progress_20260322_evening.md
- docs\zju_depth_target_reliability_status_20260323_completed.md
- docs\zju_depth_target_reliability_status_20260323_launched.md
- docs\zju_vggt_geom_smoke_20260321.md
- docs\zju_vggt_unproject_geometry_probe_20260321.md

### 2. 脚本 `scripts/`（`150` 个）

脚本前缀分布：

- analyze: 7
- audit: 7
- bootstrap: 1
- check: 1
- compare: 7
- find: 2
- generate: 8
- invoke: 1
- materialize: 3
- monitor: 1
- other: 23
- package: 18
- plan: 1
- probe: 1
- pull: 1
- run: 32
- search: 1
- start: 2
- summarize: 23
- sync: 2
- synthesize: 1
- validate: 1
- write: 6

这些脚本说明当前项目已经形成完整实验流水线：

- `compare_*` / `run_*`：跑本地基线、ZJU 视角 sweep、legacy backfill、source policy 候选、nightly/guard/long gate。
- `audit_*` / `analyze_*`：做深度置信度、区域质量、objective balance、cached geometry 归因分析。
- `materialize_*` / `package_*` / `summarize_*` / `write_*`：把“研究问题”包装成可执行 ticket、manifest、blueprint、readiness 报告和手工审批材料。
- `run_zju_source_policy_research_loop.py` 这类脚本说明研究流程已经被状态机化，核心约束是 `IDLE_GUARD -> 1x1 -> 10x5 -> 100x20 -> VERDICT -> RETURN_TO_GUARD`，并强调单问题、单候选、云端默认关闭。

完整清单：

- scripts\analyze_zju_conf_depth_quality_signal.py
- scripts\analyze_zju_depth_conf.py
- scripts\analyze_zju_quality_candidate_postmortem.py
- scripts\analyze_zju_quality_continuous_soft_rule_sweep.py
- scripts\analyze_zju_quality_region_boundary.py
- scripts\analyze_zju_quality_rule_shape_sweep.py
- scripts\analyze_zju_unproject_gate_thresholds.py
- scripts\arm_zju_source_policy_approved_problem.py
- scripts\audit_zju_cached_depth_targets.py
- scripts\audit_zju_conf_depth_attribution.py
- scripts\audit_zju_early_fl_tax_localization_daytime_20260330.py
- scripts\audit_zju_geom_cache_inventory.py
- scripts\audit_zju_objective_balance_daytime_20260330.py
- scripts\audit_zju_source_policy_supervision.py
- scripts\audit_zju_supervised_geom_view_quality.py
- scripts\bootstrap_local_5080_env.ps1
- scripts\check_zju_post_v9_consistency.py
- scripts\compare_geometry_branches.py
- scripts\compare_geometry_branches_zju_report.py
- scripts\compare_zju_finetune_runs.py
- scripts\compare_zju_geom_cache_variants.py
- scripts\compare_zju_geometry_sweeps.py
- scripts\compare_zju_region_batch_summaries.py
- scripts\compare_zju_vggt_geom_probes.py
- scripts\find_co3d_candidates.ps1
- scripts\find_co3d_candidates.py
- scripts\generate_zju_camera_focal_objective_isolation_authored_patch_review_note.py
- scripts\generate_zju_camera_focal_objective_isolation_design_contract.py
- scripts\generate_zju_camera_focal_objective_isolation_execution_prep_design.py
- scripts\generate_zju_camera_focal_objective_isolation_execution_prep_implementation_sketch.py
- scripts\generate_zju_camera_focal_objective_isolation_hygiene_review_and_execution_prep_promotion_note.py
- scripts\generate_zju_camera_focal_objective_isolation_patch_authoring_approval_note.py
- scripts\generate_zju_camera_focal_objective_isolation_pseudodiff_map.py
- scripts\generate_zju_camera_focal_objective_isolation_review_packet.py
- scripts\invoke_modal_zju_preflight.ps1
- scripts\manifests\zju_next_cloud_pair_v1.json
- scripts\manifests\zju_next_training_question_v1.json
- scripts\manifests\zju_post_v9_residual_cluster_v1.json
- scripts\manifests\zju_source_policy_rawpool_local_nightly_v1.json
- scripts\materialize_zju_anchor_balance_reserve_manifest.py
- scripts\materialize_zju_contract_segment_stratified_hardtail_bucket.py
- scripts\materialize_zju_promoted_hardcase_manifest.py
- scripts\monitor_modal_depth_target_pair.py
- scripts\package_zju_camera_focal_objective_isolation_manual_problem.py
- scripts\package_zju_default_stream_intrinsics_counterbalance_manual_problem.py
- scripts\package_zju_hardtail_bucket_granularity_manual_problem.py
- scripts\package_zju_hybrid_tail_exposure_manual_problem.py
- scripts\package_zju_residual_case_coverage_manual_approval.py
- scripts\package_zju_soft_tail_exposure_manual_problem.py
- scripts\package_zju_tail_anchor_reserve_hybridization_manual_problem.py
- scripts\package_zju_tail_anchor_stabilization_manual_problem.py
- scripts\package_zju_tail_conf_branch_decoupling_manual_problem.py
- scripts\package_zju_tail_contract_anchor_replay_manual_problem.py
- scripts\package_zju_tail_contract_viewset_replay_manual_problem.py
- scripts\package_zju_tail_counterbalance_cohort_manual_problem.py
- scripts\package_zju_tail_dual_supervision_rebalancing_manual_problem.py
- scripts\package_zju_tail_intrinsics_branch_decoupling_manual_problem.py
- scripts\package_zju_tail_manifest_focal_reinforcement_manual_problem.py
- scripts\package_zju_tail_pose_branch_decoupling_manual_problem.py
- scripts\package_zju_tail_source_pool_tempering_manual_problem.py
- scripts\package_zju_tail_stream_selective_focal_reinforcement_manual_problem.py
- scripts\plan_zju_multiview_addon_cache.py
- scripts\probe_zju_vggt_geom_dataset.py
- scripts\pull_modal_zju_geometry_pair_summary.ps1
- scripts\run_geometry_baseline_batch.py
- scripts\run_geometry_minimal_finetune.ps1
- scripts\run_legacy_backfill_from_current_manifest.ps1
- scripts\run_legacy_gap_batch.py
- scripts\run_legacy_uniform_backfill_from_current_summaries.ps1
- scripts\run_local_geometry_baseline.ps1
- scripts\run_modal_geometry_minimal_finetune.ps1
- scripts\run_modal_zju_depth_target_reliability_pair.ps1
- scripts\run_modal_zju_geometry_branch_compare.ps1
- scripts\run_modal_zju_geometry_minimal_finetune.ps1
- scripts\run_modal_zju_unproject_geometry_ablation_pair.ps1
- scripts\run_zju_custom_source_set_region_probe.py
- scripts\run_zju_depth_target_reliability_pair.ps1
- scripts\run_zju_execution_ready_preparation_design_guard_only.py
- scripts\run_zju_geometry_baseline_batch.py
- scripts\run_zju_geometry_baseline_from_report.ps1
- scripts\run_zju_geometry_diagnostics.py
- scripts\run_zju_geometry_primary_from_report.ps1
- scripts\run_zju_geometry_region_diagnostics.py
- scripts\run_zju_geometry_view_sweep.py
- scripts\run_zju_post_v9_residual_cluster_nightly.py
- scripts\run_zju_source_leave_one_out_region_diagnostics.py
- scripts\run_zju_source_policy_rawpool_guard_daemon.py
- scripts\run_zju_source_policy_rawpool_local_nightly.py
- scripts\run_zju_source_policy_rawpool_long_gate.py
- scripts\run_zju_source_policy_rawpool_overnight_watch.py
- scripts\run_zju_source_policy_research_candidate.py
- scripts\run_zju_source_policy_research_loop.py
- scripts\run_zju_source_policy_research_watch.py
- scripts\run_zju_two_stage_objective_decoupling_task_mode.py
- scripts\run_zju_unproject_geometry_ablation_pair.ps1
- scripts\run_zju_vggt_geom_minimal_finetune.ps1
- scripts\search_zju_hybrid_source_sets.py
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v1.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v2_b12_only.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v3_b12_b4.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v4_b1_b4_b12.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v5_b1_b4_b12_b15.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v5_b1_b4_b12_b15_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v6_b1_b4_b12_frameaware_b15_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v7_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v8_b1_b2_b4_b12_frameaware_b15_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_6src_hardcontrol_hybrid_v9_b1_b2_b4_b12_frameaware_b13_b15_b23_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b11_s1_011_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b11_s1_017_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b13_s1_005_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b13_s1_021_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b2_s1_013_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b23_s1_008_probe_frames0_600_1080_1170.json
- scripts\source_policy_overrides\zju_b23_s1_016_probe_frames0_600_1080_1170.json
- scripts\start_modal_background_run.ps1
- scripts\start_powershell_background_job.ps1
- scripts\summarize_geometry_pair_summaries.py
- scripts\summarize_legacy_backfill_manifest.py
- scripts\summarize_legacy_uniform_backfill.py
- scripts\summarize_zju_default_stream_intrinsics_counterbalance_readiness.py
- scripts\summarize_zju_geometry_baselines.py
- scripts\summarize_zju_geometry_policy_followup.py
- scripts\summarize_zju_hardcase_bucket_readiness.py
- scripts\summarize_zju_hardtail_bucket_granularity_readiness.py
- scripts\summarize_zju_hybrid_tail_exposure_readiness.py
- scripts\summarize_zju_region_case_summaries.py
- scripts\summarize_zju_soft_tail_exposure_readiness.py
- scripts\summarize_zju_tail_anchor_reserve_hybridization_readiness.py
- scripts\summarize_zju_tail_anchor_stabilization_readiness.py
- scripts\summarize_zju_tail_conf_branch_decoupling_readiness.py
- scripts\summarize_zju_tail_contract_anchor_replay_readiness.py
- scripts\summarize_zju_tail_contract_viewset_replay_readiness.py
- scripts\summarize_zju_tail_counterbalance_cohort_mixing_readiness.py
- scripts\summarize_zju_tail_dual_supervision_rebalancing_readiness.py
- scripts\summarize_zju_tail_intrinsics_branch_decoupling_readiness.py
- scripts\summarize_zju_tail_manifest_focal_reinforcement_readiness.py
- scripts\summarize_zju_tail_pose_branch_decoupling_readiness.py
- scripts\summarize_zju_tail_source_pool_tempering_readiness.py
- scripts\summarize_zju_tail_stream_selective_focal_reinforcement_readiness.py
- scripts\sync_zju_daytime_manual_diagnosis_after_two_stage_failure.py
- scripts\sync_zju_residual_case_coverage_manual_decision.py
- scripts\synthesize_selection_contract_mechanism.py
- scripts\validate_zju_camera_focal_objective_isolation_execution_prep_baseline.py
- scripts\write_zju_execution_prep_baseline_decision.py
- scripts\write_zju_execution_ready_boundary.py
- scripts\write_zju_execution_ready_discussion_approval_note.py
- scripts\write_zju_execution_ready_discussion_decision.py
- scripts\write_zju_execution_ready_prep_plan.py
- scripts\write_zju_execution_ready_preparation_design.py
- scripts\zju_geometry_region_utils.py

### 3. 训练层 `training/`（`64` 个）

- `training/data/datasets/zju_vggt_geom.py`：新增 ZJU 几何数据集，支持 source policy、raw/cached 视图池、manifest 采样、anchor 选择、监督视图质量过滤。
- `training/config/*.yaml`：从最小 ZJU 几何基线，逐步扩展到 `unproject_geometry`、`confdepth_dropworst`、`gradconfmask`、`nearest_plus_uniform_tail`、`hardcase bucket mix`、`anchorb1` 等试验族。

完整清单：

- training\config\zju_vggt_geom_minimal.yaml
- training\config\zju_vggt_geom_unproject_bottom_only_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_confgate_pow2_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_confgate_t50_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_confgate_t75_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_confgate_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_minimal.yaml
- training\config\zju_vggt_geom_unproject_reliable_region_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_confgate_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1bottom20confscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1confscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1disagreementjointconf0875reg1125_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qge275wholefgconfmaskdrop_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qge275wholefgjointdepthscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qlinearfginteriorerode5confscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgconfscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qlinearwholefgjointdepthscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5decoupleddepthreg09375conf0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfp60jointdepthscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5depthconfsmoothstepp60jointdepthscale0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5jointdepthscale0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticfgedge5partialjointconfsmoothstepp60jointdepthscale0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgconfscale05_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgdecoupleddepthreg09375conf0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgjointdepthscale075_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1qquadraticwholefgjointdepthscale0875_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_anchorb1unprojectconsistencyjointconf0875reg1125_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_unprojectauxconfgatew005_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_viewmean_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_trainmix50_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_dropworst_supervised_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_anchorb13softtailtaperreg095conf095_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_anchorb13iqrconfmaskdrop_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_anchorb13softguardreg095conf095_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_hardcasecamerascale0_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_hardcasefocalscale0_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_hardcasemaxdepthanchor_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_hardcasetrainmix50_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailbucketmix4to1_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_depthgainthencamerareconciliation_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_defaultfocal105_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_focalscale1125_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_hardtailfocalscale1125_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_hardcasemaxdepthanchor_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_manifestanchorreplay_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_manifestanchorreplay_minsup2_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_manifestviewsetreplay_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_hardcasebucketmix4to1_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_uniform_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_uniform_rawpool_minimal.yaml
- training\config\zju_vggt_geom_unproject_source_policy_uniform_union_debug_minimal.yaml
- training\config\zju_vggt_geom_unproject_w005_minimal.yaml
- training\config\zju_vggt_geom_unproject_w01_minimal.yaml
- training\config\zju_vggt_geom_unproject_warmup_minimal.yaml
- training\data\datasets\zju_vggt_geom.py

### 4. 顶层新增文件（`14` 个）

- `modal_geometry_minimal_finetune.py`：最小云端几何微调封装。
- `modal_zju_geometry_minimal_finetune.py`：面向 ZJU 几何训练的 Modal 任务封装，包含 promoted / previous source policy 选择。
- `modal_zju_geometry_branch_compare.py`：云端批量比较 `point_map` 与 `depth_unproject`。
- `requirements_training.txt`：补齐训练依赖。
- `tmp_*`：2026-03-29 的研究运行日志与摘要快照。

完整清单：

- modal_geometry_minimal_finetune.py
- modal_zju_geometry_branch_compare.py
- modal_zju_geometry_minimal_finetune.py
- requirements_training.txt
- tmp_driver_stdout_20260329_014446.err
- tmp_driver_stdout_20260329_014446.txt
- tmp_modal_log_20260329_014446.err
- tmp_modal_log_20260329_014446.txt
- tmp_modal_log_20260329_020541.err
- tmp_modal_log_20260329_020541.txt
- tmp_modal_log_20260329_020541_latest.err
- tmp_modal_log_20260329_020541_latest.txt
- tmp_probe_summary_20260329_020541.err
- tmp_probe_summary_20260329_020541.md

## 这批工作背后的原理

### A. 为什么默认分支从 pointmap 改成 depth+camera

从新增文档链可以看出，研究先在本地和 ZJU 上反复比较 `point map` 与 `depth + camera -> unprojection`。方向结论不是“无条件完胜”，而是：

- dense/full-rig 条件下，`depth + camera` 明显值得继续；
- sparse source set 下，胜负高度依赖 source policy；
- 因此不回退旧 ghost stack，而是继续沿几何链和 source policy 优化。

这也是为什么 UI 默认值被切到 `Depthmap and Camera Branch`。

### B. 为什么新增 ZJU 数据集和 manifest 机制

原备份版没有 ZJU 几何训练闭环。当前版本新增 `ZjuVggtGeomDataset`，核心目的是把“实验假设”直接编码进数据采样：

- 用 `source_policy` 控制多视角来源怎么选；
- 用 `source_view_pool` 控制只用缓存视图还是允许 raw pool；
- 用 `source_anchor_policy` 固定 anchor；
- 用 `sample_manifest_path` 和相关字段把 hardcase bucket、contract replay、anchor replay 等训练样本子集显式化；
- 同时把 `depth_conf_maps`、`foreground_masks`、anchor 索引等监督辅助信息一并送到 loss。

原理上，这是把“哪类视角组合更利于 depth_unproject”从线下分析推进到线上训练控制。

### C. 为什么 `training/loss.py` 改这么大

这部分说明当前工作已经从“看结果”进入“定向调监督”的阶段。

核心原理是：

- 不再只依赖统一的 depth/camera loss，而是引入 `unproject_geometry`，让预测深度和相机组成的几何链直接接受约束。
- 不是全图一刀切调权，而是根据 `train_progress`、manifest、anchor、conf-depth、branch disagreement、unproject consistency 等信号做局部或阶段性加权。
- 通过 `conf_depth_point_masks`、foreground 区域、anchor view index，把监督更集中地打到“真正想修”的区域和视图上。
- 通过 quantile / confidence regularization 调整，避免极端噪声和无效像素把梯度拉偏。

简言之：从统一监督变成了“条件化、分阶段、面向残差模式”的监督设计。

### D. 为什么会出现这么多 `package_/write_/seed/blueprint` 风格脚本

这说明当前目录不只是代码实验，还内置了研究治理机制。它在约束：

- 一晚只允许一个 approved problem；
- 一个 family 只发一个 first candidate；
- 先 smoke，再短 gate，再长 gate；
- 失败就 return-to-guard，不自动继续 cousin sweep；
- 云端默认保持关闭，很多阶段先做本地 gate。

原理上，这是为了防止研究在 source policy / routing / hardcase family 上无限发散，保证每一步都有审查、证据链和止损条件。

### E. 目前工作推进到哪一步

从文档时间线看，演进是：

1. `2026-03-21`：确认 geometry-first 方向成立，本地/Modal smoke 与基线比较完成。
2. `2026-03-22 ~ 2026-03-24`：大量做 sparse/dense source policy 比较、6src/12src 局部修复、legacy backfill、region diagnostics。
3. `2026-03-25 ~ 2026-03-27`：把 source policy、conf-depth、gradconfmask、anchor 路由等训练族系统化，失败 family 被文档化冻结。
4. `2026-03-28`：source-policy promoted lead 基本定型，问题转成 post-promotion residual gap mining。
5. `2026-03-28 ~ 2026-03-30`：开始把 residual hardcase coverage rebalancing、hardtail bucket granularity、two-stage objective decoupling 等问题封装成可审批 family。

## 最重要的文件参考

- 方向切换与总体状态：`docs/geometry_first_status_20260321.md`
- 6src/12src 修复闭环与后续结论：`docs/geometry_next_step_after_20260323.md`
- 失败边界与下一个研究问题：`docs/geometry_source_policy_failure_boundary_and_next_problem_20260327.md`
- promoted lead 之后的剩余问题：`docs/geometry_post_promotion_gap_mining_20260328.md`
- ZJU 数据集实现：`training/data/datasets/zju_vggt_geom.py`
- 最大核心改动：`training/loss.py`
- 研究状态机：`scripts/run_zju_source_policy_research_loop.py`

## 一句话结论

和 `G:\项目备份\vggt\vggt-main` 相比，当前项目已经从“原始 VGGT 推理/训练代码”演进成“围绕 ZJU 几何链、source policy、hardcase coverage 和人工审批式研究流程的完整实验工作区”；差异是系统级的，不是零星补丁。


