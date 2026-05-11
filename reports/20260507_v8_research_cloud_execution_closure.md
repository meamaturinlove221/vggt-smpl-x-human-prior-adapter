# V8 Research Cloud Execution Closure

Date: 2026-05-07
Repo: D:\vggt\vggt-main

## Verdict

V8 research-cloud execution is complete. A separate research cloud guard was implemented, formal guard stayed red, and the bounded Cloud-A/B/C/D jobs were launched on Modal, completed, downloaded, and refereed.

Formal cloud train/infer/export remains blocked:

- strict_candidate_passes = 0
- strict_teacher_passes = 0
- candidate formal guard = blocked, reason strict_candidate_passes is 0
- teacher-supervised formal guard = blocked, reasons strict_candidate_passes is 0 and strict_teacher_passes is 0

Research cloud preflight is allowed and was used:

- tools/check_research_cloud_gate_status.py
- research_cloud_job_manifest.schema.json
- tools/research_cloud_artifact_referee.py
- modal_v8_research_cloud.py

No predictions.npz, teacher package, candidate package, strict pass, or strict registry write was produced.

## Agents

Closed:

- Agent A / Pascal completed B-Fus3D2 local dataset smoke.
- Agent B / Sagan disconnected before completion; main thread implemented A5-X intake.
- Agent C / Franklin disconnected before completion; main thread implemented hand/hair/cloth specialists.
- Agent D / Huygens completed launcher dry-run package.

Main thread completed cloud launch, download, referee, and final reporting.

## Cloud Runs

Modal runs completed:

- Cloud-B A5-X external dense intake: https://modal.com/apps/shimakaze22333/main/ap-RxBzpiR9k3vHJMvkMVDaw1
- Cloud-C B-hand10: https://modal.com/apps/shimakaze22333/main/ap-lZYidK4VVYXx5tQXvwDOKh
- Cloud-D B-hair3: https://modal.com/apps/shimakaze22333/main/ap-kcQpy9FhmV5XXHD6OcezTU
- Cloud-A B-Fus3D2: https://modal.com/apps/shimakaze22333/main/ap-GcE81cg0TvoQBLO2L7fVR6
- Cloud-C B-cloth0: https://modal.com/apps/shimakaze22333/main/ap-zjR5q7GKROZ8bKhvJhFQQI

Cloud-A had two environment/data misses before success:

- missing torch in the first lightweight image
- missing PyYAML in the first torch image
- missing local query/template cache in the remote container

Fixes applied:

- added a torch/PyYAML Modal image for Cloud-A
- added procedural human SDF fallback in training/data/datasets/human_surface_sdf_dataset.py

## Cloud-A: B-Fus3D2 Dataset-Level Train

Script:

- tools/b_fus3d2_human_dataset_train.py
- training/data/datasets/human_surface_sdf_dataset.py
- training/config/b_fus3d2_human_dataset.yaml

Downloaded output:

- output/surface_research_cloud_preflight/Cloud_A/b_fus3d2_human_dataset_train/summary.json
- output/surface_research_cloud_preflight/Cloud_A/b_fus3d2_human_dataset_train/report.md
- output/surface_research_cloud_preflight/Cloud_A/b_fus3d2_human_dataset_train/b_fus3d2_control_contact_sheet.png
- output/surface_research_cloud_preflight/Cloud_A/b_fus3d2_human_dataset_train/*surface_sdf_points.ply
- output/surface_research_cloud_preflight/Cloud_A/b_fus3d2_human_dataset_train/b_fus3d2_human_dataset_train_diagnostics.npz

Remote result:

- status = research_only_dataset_train_smoke_no_export
- success = false
- real_minus_zero_eval_iou = 0.010723740039150331
- real_minus_random_eval_iou = 0.013770778998959732
- real_minus_shuffle_eval_iou = 0.029914898680786894

Decision: fail closed. Cloud-A ran successfully, but it does not meet the v8 target of real_minus_zero/random > 0.03.

## Cloud-B: A5-X External Dense Intake

Script:

- tools/a5x_external_dense_teacher_intake_smoke.py

Downloaded output:

- output/surface_research_cloud_preflight/Cloud_B/a5x_external_dense_teacher_intake/summary.json
- output/surface_research_cloud_preflight/Cloud_B/a5x_external_dense_teacher_intake/report.md
- output/surface_research_cloud_preflight/Cloud_B/a5x_external_dense_teacher_intake/a5x_external_dense_teacher_intake_contact_sheet.png
- output/surface_research_cloud_preflight/Cloud_B/a5x_external_dense_teacher_intake/*dense_candidate_points.ply

Remote result:

- status = research_only_a5x_external_dense_intake_no_export
- success = true
- best_method = must3r_family_pointmap
- weak_teacher_pool_only = true
- best full score = 0.9888384865664477
- best face score = 0.9835164835164835
- best hairline score = 0.9923664122137404
- best hands score = 0.9885931558935361
- best head score = 0.9888268156424582

Decision: research-only weak teacher pool. It is not a strict teacher and writes no teacher export.

## Cloud-C: B-Hand10 Specialist

Script:

- tools/b_hand10_hggt_style_hand_decoder_smoke.py

Downloaded output:

- output/surface_research_cloud_preflight/Cloud_C/b_hand10_hggt_style_hand_decoder/summary.json
- output/surface_research_cloud_preflight/Cloud_C/b_hand10_hggt_style_hand_decoder/report.md
- output/surface_research_cloud_preflight/Cloud_C/b_hand10_hggt_style_hand_decoder/b_hand10_hggt_style_hand_contact_sheet.png
- output/surface_research_cloud_preflight/Cloud_C/b_hand10_hggt_style_hand_decoder/*hggt_style_points.ply

Remote result:

- status = research_only_b_hand10_hggt_style_no_export
- success = true
- left_real_minus_zero_iou = 0.7053571627365369
- left_real_minus_shuffle_iou = 0.5352399302647773
- left_real_minus_mask_only_iou = 0.5437908232391659
- right_real_minus_zero_iou = 0.6783394615321459
- right_real_minus_shuffle_iou = 0.4442793462109955
- right_real_minus_mask_only_iou = 0.4026447401881834

Decision: research-only progress for hand hard gate. It is not a strict candidate.

## Cloud-C: B-Cloth0 Specialist

Script:

- tools/b_cloth0_silhouette_surface_residual_smoke.py

Downloaded output:

- output/surface_research_cloud_preflight/Cloud_C/b_cloth0_silhouette_surface_residual/summary.json
- output/surface_research_cloud_preflight/Cloud_C/b_cloth0_silhouette_surface_residual/report.md
- output/surface_research_cloud_preflight/Cloud_C/b_cloth0_silhouette_surface_residual/b_cloth0_silhouette_surface_contact_sheet.png
- output/surface_research_cloud_preflight/Cloud_C/b_cloth0_silhouette_surface_residual/*surface_points.ply

Remote result:

- status = research_only_b_cloth0_silhouette_residual_no_export
- success = false
- residual_minus_constrained_iou = -0.014885414676362507
- residual_minus_random_iou = 0.4254439608586098
- residual_minus_constrained_overfill = 0.0

Decision: fail closed against constrained baseline. It generated artifacts but does not prove the clothing residual route yet.

## Cloud-D: B-Hair3 Specialist

Script:

- tools/b_hair3_hairgs_topology_smoke.py

Downloaded output:

- output/surface_research_cloud_preflight/Cloud_D/b_hair3_hairgs_topology/summary.json
- output/surface_research_cloud_preflight/Cloud_D/b_hair3_hairgs_topology/report.md
- output/surface_research_cloud_preflight/Cloud_D/b_hair3_hairgs_topology/b_hair3_hairgs_topology_contact_sheet.png
- output/surface_research_cloud_preflight/Cloud_D/b_hair3_hairgs_topology/*hairgs_strand_points.ply

Remote result:

- status = research_only_b_hair3_hairgs_topology_no_export
- success = true
- real_minus_image_only_iou = 0.19466191251324194
- real_minus_mask_only_iou = 0.24204545454545445
- real_minus_shuffle_iou = 0.3579545454545454
- real_minus_zero_token_iou = 0.73597510078442

Decision: research-only progress for hair hard gate. It is not a strict candidate.

## Referee

Research cloud artifact referee:

- status = research_cloud_artifact_referee_research_only
- artifact_count = 11
- dirs_with_3d_output = 10
- dirs_with_contact_sheet = 10
- dirs_with_controls = 9
- dirs_with_forbidden_path_hits = 0
- dirs_with_suspicious_fields = 0
- research_cloud_allowed = true
- formal_cloud_train_infer_export = blocked
- predictions_export = blocked
- teacher_export = blocked
- candidate_export = blocked
- strict_pass_write = blocked

Formal v7/strict referee:

- status = research_only_v7_region_referee_strict_gate_red
- strict_candidate_passes = 0
- strict_teacher_passes = 0

## Final State

Completed:

- V8-D research cloud guard and schema
- V8-D research cloud artifact referee
- V8 Modal research runner
- V8-A B-Fus3D2 dataset-level cloud train smoke
- V8-B A5-X external dense intake cloud smoke
- V8-C B-hand10 cloud smoke
- V8-C B-hair3 cloud smoke
- V8-C B-cloth0 cloud smoke
- post-job referee and formal guard checks

Blocked by design:

- formal candidate cloud
- formal teacher-supervised cloud
- predictions export
- teacher export
- candidate export
- strict registry write
- strict pass write

Next technical conclusion:

- B-Fus3D2 needs a real multi-case scanned/synthetic asset set or stronger hard-negative design; the remote procedural fallback did not clear real-vs-zero/random margins.
- A5-X, B-hand10, and B-hair3 are positive research-only signals.
- B-cloth0 still fails against the constrained baseline.
