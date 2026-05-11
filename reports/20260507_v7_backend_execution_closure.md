# V7 Backend Execution Closure

Date: 2026-05-07
Repo: D:\vggt\vggt-main

## Verdict

V7 local execution is complete as a research-only backend run. It produced new 3D artifacts for the requested main lines, refreshed the region referee, and rechecked the formal cloud guard.

Formal cloud train/infer/export was not launched because the local D-line guard still blocks it:

- strict_candidate_passes = 0
- strict_teacher_passes = 0
- candidate cloud guard: blocked, reason strict_candidate_passes is 0
- teacher-supervised cloud guard: blocked, reasons strict_candidate_passes is 0 and strict_teacher_passes is 0

No strict pass, teacher export, candidate export, predictions export, or strict registry promotion was written.

## Thread Coordination

Existing completed agent threads were recovered and closed:

- Zeno: B-hair1 backend smoke, 2240 rooted strand-chain points, failed controls.
- Copernicus: v6 D-line guard/referee refresh, strict gate red.
- Faraday: B-Fus3D input audit, confirmed token/query/grid/template inputs.

Additional v7 agents were also closed before final reporting:

- D-line v7 region referee: completed and closed.
- B-GS2: completed and closed.
- B-hair2: hung, closed; main thread completed implementation/run.
- B-hand9: hung, closed; main thread completed implementation/run.

## P0 Region Autopsy

Implemented and ran:

- tools/v7_region_failure_autopsy.py

Outputs:

- output/surface_research_preflight_local/V7_region_failure_autopsy/v7_region_failure_autopsy_summary.json
- output/surface_research_preflight_local/V7_region_failure_autopsy/v7_region_failure_autopsy_report.md
- reports/20260507_v7_region_failure_autopsy.md

This produced region contact sheets for B-Fus3D0-v2, B-GS1, B-hair1, and B-hand8. Caveat: the first autopsy score is a coarse spread/coverage heuristic and can saturate; use the contact sheets and later referee metrics as the stronger decision evidence.

## P1 B-Fus3D1

Implemented and ran:

- tools/b_fus3d1_trainable_latent_sdf_overfit.py

Outputs:

- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_trainable_latent_sdf_overfit_summary.json
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_trainable_latent_sdf_fields.npz
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_real_trainable_latent_sdf_occupied_points.ply
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_shuffle_trainable_latent_sdf_occupied_points.ply
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_zero_trainable_latent_sdf_occupied_points.ply
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_random_view_trainable_latent_sdf_occupied_points.ply
- output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit/b_fus3d1_render_contact_sheet.png

Key metrics:

- real mean_iou = 0.766999236729143
- real mean_overfill_ratio = 0.1630971334366442
- real mean_target_recall = 0.903681261464083
- real_minus_shuffle_iou = 0.09650189385319619
- real_minus_zero_iou = 0.0007789858853624043
- real_minus_random_view_iou = 0.007898243950160366
- real_minus_zero_recall = -0.004590155985574218

Decision: fail closed. B-Fus3D1 trained and wrote artifacts, but real did not robustly beat zero/random and did not satisfy hard-region improvement criteria.

## P2 B-GS2

Implemented and ran:

- tools/b_gs2_fus3d_guided_residual_gaussian_layer.py

Outputs:

- output/surface_research_preflight_local/B_GS2_fus3d_guided_residual_gaussian_layer/b_gs2_fus3d_guided_combined.ply
- output/surface_research_preflight_local/B_GS2_fus3d_guided_residual_gaussian_layer/b_gs2_fus3d_guided_residual_only.ply
- output/surface_research_preflight_local/B_GS2_fus3d_guided_residual_gaussian_layer/b_gs2_fus3d_guided_residual_summary.json
- output/surface_research_preflight_local/B_GS2_fus3d_guided_residual_gaussian_layer/b_gs2_fus3d_guided_residual_diagnostics.npz

Key metrics:

- guided_minus_constrained_iou = -0.00008494176398321507
- guided_minus_constrained_overfill = 0.0013858772551179621
- guided_minus_constrained_rgb_residual = -0.014547229806582185
- guided_minus_random_iou = 0.00873755061737902
- guided_minus_random_overfill = -0.00891302967586094

Decision: fail closed. B-GS2 beats random residual behavior, but it still does not beat constrained baseline on IoU/overfill.

## P3 B-Hand9

Implemented and ran:

- tools/b_hand9_finger_aware_local_surface_decoder.py

Outputs:

- output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder/b_hand9_combined_finger_aware_local_surface.ply
- output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder/b_hand9_left_finger_aware_local_surface.ply
- output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder/b_hand9_right_finger_aware_local_surface.ply
- output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder/b_hand9_finger_contact_sheet.png
- output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder/b_hand9_finger_aware_summary.json

Local side metrics:

- left finger_structure_visible = true
- left wrist_connected = true
- left palm_continuity = true
- left not_smplx_scaffold_only = true
- right finger_structure_visible = true
- right wrist_connected = true
- right palm_continuity = true
- right not_smplx_scaffold_only = true

Decision: research-only progress. B-hand9 produced a procedural finger-aware local surface connected to B-hand8 wrist anchors, but it is not a learned VGGT-token-margin pass and does not unlock strict/candidate/cloud.

## P4 B-Hair2

Implemented and ran:

- tools/b_hair2_image_first_hair_surface_backend.py

Outputs:

- output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend/b_hair2_real_image_real_token_image_first_strand_chain.ply
- output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend/b_hair2_real_image_zero_token_image_first_strand_chain.ply
- output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend/b_hair2_mask_only_image_first_strand_chain.ply
- output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend/b_hair2_image_first_contact_sheet.png
- output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend/b_hair2_image_first_summary.json

Key metrics:

- real_minus_mask_only_iou = -0.0003582739478504432
- real_minus_mask_only_root_score = 0.06912397802289993
- real_minus_real_image_zero_token_iou = 0.00006290533705088225
- real_minus_real_image_zero_token_root_score = -0.012860796269443253
- real_minus_shuffle_token_iou = 0.00010437235521566779
- real_minus_zero_token_iou = 0.00017470716031036532
- real_overfill_minus_hair1 = 0.03018185463119749

Decision: fail closed. B-hair2 produced image-first hairline/strand artifacts, but real+token did not beat mask-only/zero strongly enough and overfill worsened versus B-hair1.

## P6 D-Line V7 Referee

Implemented/reran:

- tools/research_artifact_referee_v7_region.py
- tools/research_artifact_referee.py
- tools/check_cloud_gate_status.py --json
- tools/check_cloud_gate_status.py --teacher-supervised --json
- tools/test_cloud_gate_status_guard.py

Outputs:

- reports/20260507_v7_region_referee_agent.json
- reports/20260507_v7_region_referee_agent.md
- output/surface_research_preflight_local/research_artifact_referee/research_artifact_referee_summary.json
- output/surface_research_preflight_local/research_artifact_referee/research_artifact_referee_report.md

V7 region referee:

- status = research_only_v7_region_referee_strict_gate_red
- artifact_count = 60
- report_count = 54
- dirs_with_3d_output = 36
- dirs_with_contact_sheet = 10
- strict_gate_red = true
- strict_candidate_passes = 0
- strict_teacher_passes = 0
- formal_cloud_train_infer_export = blocked

Generic referee:

- status = research_only_referee_strict_gate_red
- artifact_count = 72
- verdict = strict gate red; research artifacts remain blocked from formal cloud/export

Cloud guard tests:

- tools/test_cloud_gate_status_guard.py passed.

## Final State

Completed locally:

- v7 region autopsy
- B-Fus3D1 trainable latent SDF overfit smoke
- B-GS2 Fus3D-guided residual Gaussian layer smoke
- B-hand9 finger-aware local surface decoder smoke
- B-hair2 image-first hair surface backend smoke
- v7 D-line region referee
- cloud guard refresh and guard tests
- existing agent thread recovery/closure

Blocked by design:

- formal cloud train/infer/export
- teacher export
- candidate export
- predictions export
- strict pass writing

Reason: strict_candidate_passes = 0 and strict_teacher_passes = 0. The correct next step is not formal cloud; it is another research iteration that makes at least one hard-region artifact robustly beat controls without increasing overfill or scaffold dependency.
