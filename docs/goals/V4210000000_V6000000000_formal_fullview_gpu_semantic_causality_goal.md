# V4210000000-V6000000000 Formal GPU Full-View Semantic Causality and Anti-Smoothing Route

This file preserves the active execution goal for the current route. The working
repository is `D:\vggt\vggt-feature-adapter`; the correct evidence and artifact
root from the preceding route history is
`D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild`.

## Accepted Facts

1. `V3000000000-V4200000000` completed with final state
   `V4200000000_SMOOTHING_DOMINANT_LIMITATIONS_DISCLOSED`.
2. The old missing SMPL-X model conclusion is obsolete. The SMPL-X model was
   found at `G:\数据集\datasets\smplx\SMPLX_NEUTRAL.npz`.
3. The SMPL-X model contains `f`, `v_template`, `shapedirs`, `posedirs`,
   `J_regressor`, `kintree_table`, and `weights`.
4. Schema v2 exists with shape `6 x 81 x 518 x 518`.
5. Schema v2 is not complete:
   - `body_part_id = kinematic_derived`
   - `nearest_surface_distance = model_derived`
   - `curvature = mesh_derived_proxy`
6. `V380` remote Modal pilot ran `8 groups x 3 seeds` and produced full-view
   `6 x 518 x 518` predictions.
7. `V380` is not formal training:
   - `fullview_modal_matrix_complete = false`
   - `gpu_type = modal_cpu_remote_pilot`
   - runtime was about 120 seconds
   - region metrics were not evaluated
   - `normal_score = 0`
   - at least one selected prediction appears to have zero normals
8. `V380` pilot metrics:
   - `true_full = 0.000167812599`
   - `same_support_random_semantic = 0.000276559964`
   - `same_support_shuffled_semantic = 0.000041952837`
   - `no_sparseconv_mlp = 0.000151030035`
   - `local_knn_smoothing = 0.000176209331`
   - `no_teacher = 0.000117470583`
   - `smoothing_gap = -8.3967e-06`
   - random semantic beats true semantic
   - local KNN smoothing beats true semantic
9. Semantic causality is not confirmed. Do not promote, do not write strict
   registry, do not modify V50/V50R2, and do not replace active candidate
   `V11700_gap_reduction_branch_520`.
10. `V415` upload audit found that an outer zip can be clean while an inner
    selected prediction npz is unreadable. The reported failure was
    `same_support_random_semantic_seed0/predictions.npz` with a CRC error in
    `world_points.npy`.
11. Controls bundle is too small and may not contain complete control
    predictions.
12. Visuals are not paper-grade 3D close-up boards.
13. Full-body, head-face, hairline, left-hand, and right-hand region metrics
    were not truly evaluated.
14. Cleanup must not claim a clean worktree when the worktree is dirty.

## Required Outcome

Do not continue CPU pilot work. Do not claim that `V380` is formal GPU training.
Do not rely only on mean delta or bar/heatmap visualizations. This route must
either complete a formal GPU full-view matrix or stop in an allowed hard-blocked
or limitation-disclosed state with concrete evidence.

The route must:

1. Repair artifact and selected/control prediction integrity.
2. Implement a formal GPU Modal full-view trainer.
3. Run the formal core matrix if Modal GPU and data are available:
   `8 groups x 5 seeds = 40 runs`.
4. Ensure normals are nonzero or produce a normal-head failure analysis.
5. Evaluate true region metrics for full body, head-face, hairline, left hand,
   and right hand.
6. Require true semantic to beat random, shuffled, support-only,
   observation-only, no-SparseConv, and local smoothing before any semantic
   causality claim.
7. Generate real 3D point-cloud close-up boards.
8. Build upload-safe bundles where selected and control npz files are internally
   readable.
9. Produce honest post-push cleanup.

## Allowed Final States

1. `V6000000000_FULLVIEW_GPU_SEMANTIC_CAUSAL_CONFIRMED_NOT_PROMOTED`
2. `V6000000000_FULLVIEW_GPU_SEMANTIC_IMPROVED_LIMITATIONS_DISCLOSED`
3. `V6000000000_RANDOM_SEMANTIC_DOMINANT_LIMITATIONS_DISCLOSED`
4. `V6000000000_SUPPORT_DOMINANT_LIMITATIONS_DISCLOSED`
5. `V6000000000_OBSERVATION_DOMINANT_LIMITATIONS_DISCLOSED`
6. `V6000000000_SMOOTHING_DOMINANT_LIMITATIONS_DISCLOSED`
7. `V6000000000_NORMAL_HEAD_FAILED_WITH_ANALYSIS`
8. `V6000000000_TRUE_HARD_BLOCKED_MODAL_OR_DATA`
9. `V6000000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS`
10. `V6000000000_INVALID_CONTROLLER_NOT_IMPLEMENTED`
11. `V6000000000_INVALID_FAST_RETURN`

## Bans

- No promotion.
- No strict registry.
- No V50/V50R2 edits.
- Do not replace active candidate.
- Do not report a CPU remote pilot as formal GPU training.
- Do not report a 3-seed pilot as a formal full matrix.
- Do not package local KNN smoothing dominance as semantic success.
- Do not package random semantic dominance as semantic success.
- Do not package all-zero normals as normal-aware success.
- Do not use mean delta alone for final decisions.
- Do not use heatmaps, grayscale depth maps, or old V460/V560/V1050 images as
  new proof.
- Do not claim selected predictions are verified when inner npz files fail.
- Do not claim controls complete when the controls bundle lacks control
  predictions.
- Do not claim clean when the worktree is dirty.

## Agents

Agents are allowed, but only with `gpt-5.5` and `xhigh` reasoning. The master
controller owns the final decision. Agents must not promote, write registry,
edit V50/V50R2, or replace the active candidate.

Suggested roles:

- Agent A: artifact-integrity
- Agent B: gpu-modal-trainer
- Agent C: normal-and-structure-head
- Agent D: formal-matrix
- Agent E: region-eval-visuals
- Agent F: causality-decision
- Agent G: advisor-package

## Stage V4210000000: Artifact Integrity and Pilot Audit

Audit:

- `V4200000000_final_status.json`
- `V4100000000_route_decision.json`
- `V3800000000_fullview_matrix.csv`
- `V3800000000_seed_metrics.csv`
- `V3900000000_structure_metrics.csv`
- `V3900000000_region_metrics.csv`
- `V4150000000_upload_manifest_sidecar.json`
- `V4180000000_post_push_cleanup.json`
- local V415 selected predictions bundle

Check outer zip hashes, outer zip tests, inner npz tests, readable keys,
readable `world_points`/`depth`/`normal`/`confidence`, CRC errors, zero normals,
missing control predictions, CPU pilot status, dirty worktree, Modal app state,
and registry/V50/V50R2/active candidate safety.

Outputs:

- `reports/V4210000000_artifact_integrity_audit.json`
- `reports/V4210000000_npz_integrity_report.json`
- `reports/V4210000000_pilot_vs_formal_audit.md`
- `reports/V4210000000_dirty_worktree_plan.md`

## Stage V4220000000: Repair Selected Predictions and Controls Bundle

Repair or repull representative selected predictions:

- true_full seed0
- same_support_random_semantic seed0
- local_knn_smoothing seed0
- no_sparseconv_mlp seed0
- support_only seed0
- observation_only seed0
- same_support_shuffled_semantic seed0

Each repaired prediction must pass `np.load`, key read, finite check, shape
check, and internal npz zip test. Controls bundle must contain prediction,
evaluation, and source-manifest evidence, or clearly disclose missing controls.

Outputs:

- `reports/V4220000000_prediction_repull_report.json`
- `reports/V4220000000_controls_prediction_manifest.json`
- `output/V4220000000_repaired_predictions/*/predictions.npz`

## Stage V4300000000: Formal GPU Full-View Trainer Implementation

Implement `tools/v4300000000_modal_gpu_fullview_semantic_trainer.py` with Modal
A100/L40/A10 support, CUDA checks, optional spconv, noSparseConv MLP, local KNN
smoothing, random adjacency SparseConv, normal head training, anti-smoothing
loss, structure-sensitive losses, checkpointing, retry, CLI hang recovery,
Modal volume pull, source manifests, quality reports, boards, and resume.

Outputs:

- `tools/v4300000000_modal_gpu_fullview_semantic_trainer.py`
- `reports/V4300000000_trainer_implementation_report.md`
- `reports/V4300000000_gpu_smoke.json`

Formal training cannot be counted when `cuda_available=false` or
`gpu_type=modal_cpu_remote_pilot`.

## Stage V4400000000: Normal Head Repair

Normals must not be all zero. If a learned normal head is unavailable, recompute
geometric normals from full-view world points and record the source separately.

Outputs:

- `tools/v4400000000_recompute_fullview_normals.py`
- `training/losses/v4400000000_normal_structure_losses.py`
- `reports/V4400000000_normal_head_repair.json`
- `reports/V4400000000_normal_nonzero_audit.csv`
- `boards/V4400000000_normal_visual.png`

If valid human-region `normal_nonzero_ratio <= 0.1`, the route can at most end
as `V6000000000_NORMAL_HEAD_FAILED_WITH_ANALYSIS`.

## Stage V4500000000: Formal Matrix Config

Core groups:

1. true_full
2. same_support_random_semantic
3. same_support_shuffled_semantic
4. support_only
5. observation_only
6. no_sparseconv_mlp
7. local_knn_smoothing
8. no_teacher

Each core group requires five GPU seeds with the same steps, same dataset, and
same evaluation.

Outputs:

- `reports/V4500000000_formal_matrix_config.json`
- `reports/V4500000000_expected_runtime_budget.md`

## Stage V4600000000: Formal GPU Full-View Matrix Core Run

Run the 40-run core matrix if Modal GPU and data are available.

Per run outputs:

- full-view `predictions.npz`
- `eval.json`
- `source_manifest.json`
- `quality.json`
- `board.png`
- `training.log`
- checkpoint or omitted reason
- runtime, Modal app id, and GPU type

Outputs:

- `reports/V4600000000_formal_matrix_progress.json`
- `reports/V4600000000_formal_matrix.csv`
- `reports/V4600000000_seed_metrics.csv`
- `reports/V4600000000_modal_job_manifest.json`
- `output/V4600000000_formal_fullview_predictions/*/predictions.npz`

Core matrix complete requires all 40 runs valid, all predictions readable, all
source manifests clean, and nonzero normals or justified normal handling.

## Stage V4700000000: Formal Matrix Expansion

Only run if the core matrix does not already exhaust the route. Expansion groups:
semantic_only, no_support, no_observation, random_adjacency_sparseconv,
semantic_gate_disabled, auxiliary_loss_disabled, and
support_firewall_disabled_ablation.

Outputs:

- `reports/V4700000000_expansion_matrix.csv`
- `reports/V4700000000_expansion_seed_metrics.csv`

## Stage V4800000000: Structure-Sensitive Region Evaluation

Evaluate full_body, head_face, hairline, left_hand, and right_hand. Required
metrics include mean delta, local delta, normal consistency,
normal_nonzero_ratio, outlier ratio, background leakage, point density,
component count, hand isolated ratio, right-hand planar score, hairline boundary
sharpness, horizontal band artifact, surface continuity, and reprojection
consistency.

Outputs:

- `reports/V4800000000_region_metrics.csv`
- `reports/V4800000000_structure_metrics.csv`
- `reports/V4800000000_region_decision.json`

Any `not_evaluated` region blocks paper-grade claims.

## Stage V4900000000: Paper-Grade Visual Boards

Generate real full-view point-cloud boards:

- `boards/V4900000000_fullbody_pointcloud.png`
- `boards/V4900000000_head_hair_hand_closeups.png`
- `boards/V4900000000_counterfactual_controls.png`
- `boards/V4900000000_smoothing_controls.png`
- `boards/V4900000000_normal_comparison.png`
- `boards/V4900000000_failure_cases.png`

Do not use heatmaps, grayscale-only depth maps, or old figures as proof.

## Stage V5000000000: Causality Decision

`SEMANTIC_CAUSAL_CONFIRMED` requires formal GPU matrix completion, true_full
mean beating same-support random semantic, same-support shuffled semantic,
support_only, observation_only, no_sparseconv_mlp, and local_knn_smoothing;
region metrics not worse; full_body/head_face/hairline/left/right hand positive;
normal nonzero and consistent; source manifests clean; no teacher/postcompose
leakage; and all selected/control predictions readable.

If random semantic beats true, use
`V6000000000_RANDOM_SEMANTIC_DOMINANT_LIMITATIONS_DISCLOSED`.

If smoothing beats true, use
`V6000000000_SMOOTHING_DOMINANT_LIMITATIONS_DISCLOSED`.

If formal GPU matrix cannot be run, use
`V6000000000_TRUE_HARD_BLOCKED_MODAL_OR_DATA`.

Outputs:

- `reports/V5000000000_causality_decision.json`
- `reports/V5000000000_control_ranking.csv`

## Stage V5100000000: Automatic Repair Cycles

Run up to three repair cycles when not hard blocked. Repair smoothing,
random-semantic dominance, normal failure, support dominance, or observation
dominance with reduced matrices and explicit results.

Outputs:

- `reports/V5100000000_repair_cycles.csv`
- `reports/V5100000000_best_repair_summary.md`

## Stage V5200000000: Candidate Synthesis

Generate at least 100 raw candidates and 20 unique candidates if the route
reaches a trainable state. Include best true_full, repaired true_full,
conservative no-regression, semantic-improved, limitation-disclosed,
normal-fixed, and hand/hair candidates.

Outputs:

- `reports/V5200000000_candidate_synthesis.csv`
- `reports/V5200000000_uniqueness_audit.csv`

## Stage V5400000000: Advisor Report

Write Chinese advisor reports covering why V420 was not successful, whether
formal GPU full-view matrix completed, true/random/shuffled results,
support/observation/noSparse/local smoothing controls, normal repair, visual
boards, semantic causality status, dominant factor if not causal, no-promotion
rationale, and next paper route.

Outputs:

- `reports/V5400000000_advisor_report.md`
- `reports/V5400000000_one_page.md`
- `reports/V5400000000_limitations.md`

## Stage V5600000000: Upload-Safe Packaging

Create upload-safe bundles:

- `archive/V5600000000_core_evidence_bundle.zip`
- `archive/V5600000000_reports_bundle.zip`
- `archive/V5600000000_visuals_bundle.zip`
- `archive/V5600000000_selected_predictions_bundle.zip`
- `archive/V5600000000_controls_bundle.zip`
- `reports/V5600000000_upload_manifest_sidecar.json`
- `reports/V5600000000_omitted_large_file_manifest.json`

All bundles must be under 500 MB, zip clean, and recorded in a sidecar manifest
after zip close. All selected and control npz files included in bundles must be
internally readable without CRC errors.

## Stage V5800000000: Post-Push Cleanup

After commit/push if performed, report git status, branch, commit, Modal apps,
Python workers, registry diff, V50/V50R2 diff, active candidate, and dirty file
list when applicable.

Output:

- `reports/V5800000000_post_push_cleanup.json`

## Stage V6000000000: Final Return

Final response must include final status, whether formal GPU full-view matrix
completed, whether semantic causality is confirmed, whether random semantic,
smoothing, support, or observation dominates, normal-head status, best
candidate, advisor report path, upload bundle paths and sha256 values, omitted
manifest, cleanup report, branch, and commit.
