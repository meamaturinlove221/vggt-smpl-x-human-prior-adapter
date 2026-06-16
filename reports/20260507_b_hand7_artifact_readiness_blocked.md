# 2026-05-07 B-Hand7 Artifact Readiness

Status: `blocked_missing_continuous_connected_hand_surface_artifact`

This is a B-hand readiness report only. It does not train, run inference,
patch predictions, export a teacher/candidate, call cloud, or write a strict
registry pass.

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher export = blocked
candidate export = blocked
predictions export = blocked
strict registry write = blocked
```

## Decision

`B_hand7_continuous_connected_hand_surface_review` cannot be produced from the
current local artifacts.

The current B-hand tree reaches B-hand6 plus a joint surface/hand contract
probe. Those artifacts prove ROI/token evidence, a decoder skeleton, weak
label/depth diagnostics, bbox-level COLMAP depth presence, and a scaffold
connected proxy. They do not prove a continuous left/right hand surface
connected to wrists/arms.

## Evidence Read

Existing B-hand artifacts:

```text
output/surface_research_preflight_local/B_hand0_evidence_cache_60v_humancrop_hybrid6/b_hand_evidence_cache.json
output/surface_research_preflight_local/B_hand1_token_backend_smoke_hybrid6/b_hand_token_backend_smoke_summary.json
output/surface_research_preflight_local/B_hand2_decoder_skeleton_smoke_hybrid6/b_hand_decoder_skeleton_smoke_summary.json
output/surface_research_preflight_local/B_hand3_smplx_wrist_arm_connected_precheck_hybrid6/raw_smplx_mesh_hand_anchor_summary.json
output/surface_research_preflight_local/B_hand4_connected_mesh_precheck_hybrid6/b_hand_connected_mesh_precheck_summary.json
output/surface_research_preflight_local/B_hand5_label_evidence_learnability_probe_hybrid6/b_hand_label_evidence_learnability_summary.json
output/surface_research_preflight_local/B_hand6_colmap_depth_evidence_probe_hybrid12/b_hand_colmap_depth_evidence_summary.json
output/surface_research_preflight_local/B_joint_surface_hand_contract_probe_hybrid6/b_joint_surface_hand_contract_summary.json
```

No existing local `B_hand7`, `continuous_connected_hand_surface_review`, or
`hand_surface_review` artifact/report was found.

The blocking B-hand3 readout is:

```text
body_gate.pass = true
body_gate.views_passing_body_anchor = 60
hand_gate.pass = false
hand_gate.eligible_views_with_hand_candidates = 60
hand_gate.views_passing_raw_hand_anchor = 0
hand_gate.views_with_compact_3d_hand_boxes = 0
top_level_pass = false
```

The blocking B-hand4 connected-mesh readout is:

```text
connected_proxy_built = true
pass = false
truthful_status = fail_precheck_only_not_candidate_not_teacher
gate_decision = fail_hand_gate_not_passed
upstream_hand_gate_pass = false
combined mesh / pointcloud / Open3D proxy renders exist
left and right sides are single-component scaffold proxies
```

B-hand4 explicitly records these blockers:

```text
B-hand3 upstream raw SMPL-X hand gate did not pass, so this is fail.
Connected template/SMPL-X hand topology is a scaffold only, not hand success.
No teacher, candidate, predictions, registry, training, inference, or cloud export was created.
Open3D-readable connected proxy still requires human visual review before any future gate claim.
```

B-hand6 is useful evidence but not sufficient:

```text
roi_count = 14
mapped_depth_roi_count = 12
missing_depth_roi_count = 2
colmap_depth_present_for_both_hands = true
interpretation = bbox_level_depth_signal_exists_but_not_connected_hand_surface
```

The joint surface/hand contract is also not sufficient:

```text
status = research_only_joint_contract_probe_not_decoder_not_teacher_not_candidate
combined_beats_hand_only = false
combined_no_absolute_x_survives = false
```

## Exact Missing Artifact

Missing:

```text
B_hand7_continuous_connected_hand_surface_review
```

Required contents:

```text
one local summary JSON
one local markdown report
left Open3D screenshots: front, side, top, iso
right Open3D screenshots: front, side, top, iso
combined hands+wrist+forearm Open3D screenshots: front, side, top, iso
largest connected component and fragmentation stats per side
explicit left/right hand presence
explicit wrist-to-forearm connection check
explicit palm continuity check
explicit finger-structure readout, or a failure stating fingers are missing
explicit note that SMPL-X/MediaPipe/template topology is weak support only
```

Current blockers mean this cannot be replaced by:

```text
B_hand4 connected-template or SMPL-X scaffold proxy
ROI boxes only
bbox-level COLMAP depth presence only
MediaPipe hand landmarks or patches only
sparse query proposals without wrist/palm/finger surface continuity
numeric-only pass without Open3D visual review
```

## Data Needed

To unblock B-hand7, provide or produce one same-frame dense hand/arm surface
artifact:

```text
<same_frame_dense_connected_hand_arm_surface.ply|obj|npz>
```

It must contain left and right hands, wrists, and enough forearm/arm context to
verify that both hands are connected to arms rather than detached sheets or
floating blobs. It must show palm continuity and finger surface structure, or
truthfully fail those checks. SMPL-X, MediaPipe, and connected-template topology
may be provenance or weak support, but not the success source.

## Commands Needed After Data Exists

First verify the dense surface source exists:

```powershell
Test-Path -LiteralPath '<same_frame_dense_connected_hand_arm_surface.ply>'
```

Then render the dense surface for review:

```powershell
D:\anaconda\envs\g3splat\python.exe -B tools\render_teacher_surface_review.py `
  --teacher-mesh <same_frame_dense_connected_hand_arm_surface.ply> `
  --predictions-npz output\local_inference_results\r34_raw518_r27_on6v_fullbody\predictions.npz `
  --output-dir output\surface_research_preflight_local\B_hand7_continuous_connected_hand_surface_review `
  --roi hands `
  --camera-view-indices 0,24,36,45,57 `
  --sample-points 160000
```

Only after that output has front/side/top/iso views for left, right, and
combined hands plus per-side connected-component statistics should a B-hand7
summary JSON/markdown be written.

## Final Readiness State

B-hand remains frozen for strict progress. Current local artifacts are useful
research diagnostics, but they are not enough to produce
`B_hand7_continuous_connected_hand_surface_review`.
