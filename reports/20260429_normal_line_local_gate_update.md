# 2026-04-29 normal-line local gate update

## Current truthful status

The normal line is still **not mentor-final**.

Current local probes can raise face ROI point counts, but they do not produce a modeled face/head surface. The repeated failure mode is:

- face/head side views remain shell-like;
- face lacks stable nose / eye / mouth geometry;
- several variants create multi-layer ghost surfaces;
- high `world_points` counts often come from confidence / teacher patch artifacts rather than real geometry;
- hands are present at a coarse level in full-body views, but finger/hand detail is not reliable.

No cloud run is allowed from this state.

## Newly checked candidate group

Visual audit sheet:

- `output\normal_line_multiview_20260428\depth_roi_v2_o035_world_sculpt_visual_audit_sheet.png`

Result:

- `depth_roi_v2_o035_world_sculpt_profile_mid`: negative.
- `depth_roi_v2_o035_world_sculpt_profile_strong`: negative.
- `depth_roi_v2_o035_world_sculpt_all_mid`: negative.

Reason:

- face numbers improve, but Open3D still shows shell / ghost;
- side views do not show reliable facial volume;
- view-consistent sculpting did not translate into a modeled head/face.

## Existing-candidate scan

Same-protocol scan output:

- `output\normal_line_multiview_20260428\on6v_headshoulder_candidate_scan.csv`

Notable high-count candidates were re-rendered:

- `20260425_original6v_sapiens_normal_guided_depth_r2_inference_on6v_headshoulder`
- `20260425_original6v_mesh_raycast_facecore_r4_inference_on6v_headshoulder`
- `20260424_sparseproto_humancrop_pointnormal_r1_on6v_headshoulder`
- `smplx_face_scaffold_a085_from_signfix`

Visual audit sheets:

- `output\normal_line_multiview_20260428\review_top_old_candidates_face_sheet.png`
- `output\normal_line_multiview_20260428\review_smplx_face_scaffold_sheet.png`

Result:

- all remain negative.

Reason:

- high `world_points` face counts are often scatter / ray artifacts;
- depth-unprojection is cleaner but remains a smooth texture shell;
- SMPL-X scaffold creates repeated / displaced template-like fragments and cannot be used as a face pass.

## Full-body and hands gate

The full-body gate must run on the real full-body scene, not only the headshoulder crop:

- `output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_human_crop`

Candidate scan:

- `output\normal_line_multiview_20260428\fullbody_humancrop_candidate_scan.csv`

Implemented/updated tools:

- `tools\audit_fullbody_hand_integrity.py`
- `tools\render_open3d_pointcloud.py --roi hands --roi-source 2d`

Hand ROI was tightened to avoid treating bare legs as hands in the full-body crop.

Visual evidence:

- `output\normal_line_multiview_20260428\fullbody_hands_visual_audit_sheet.png`
- `output\normal_line_multiview_20260428\fullbody_hands_visual_audit_sheet_v2.png`
- `output\normal_line_multiview_20260428\fullbody_hands_visual_audit_sheet_v3.png`

Full-body result:

- body point cloud is largely connected;
- no obvious full-body collapse or large body hole in the checked full-body candidates;
- lower body / shoes are represented at coarse scale.

Hand result:

- hand/forearm regions are not entirely missing in visible views;
- however, hand ROI is still coarse and mixes hand / forearm / phone / clothing edges;
- fingers are not modeled reliably;
- therefore hands are only a bottom-line sanity check at this stage, not a quality pass.

Updated rule:

- face/head/hairline remains the primary pass gate;
- full-body must have no large holes, broken limbs, or severe ghosting;
- hands must not be obviously amputated or fully missing;
- hand detail remains secondary but cannot be ignored in final Open3D review.

## 60-view teacher check

The local non-RPC 60-view headshoulder teacher `predictions.npz` failed `np.load` with `BadZipFile`.

The RPC copy rendered successfully:

- `output\normal_line_multiview_20260428\review_60v_teacher_face_sheet.png`

Result:

- 60-view teacher is more complete than 6-view, but still smooth/shell-like in face side views;
- it is not a sufficient high-quality face teacher for mentor-final sparse-view delivery.

## Next local action

Do not continue `r16 + epoch` or depth-offset/sculpt-only probes.

The remaining blocking problem is: normal/depth/point consistency and confidence manipulation can make shells cleaner or denser, but they do not create true head/face surface detail.

Next local iteration must target the actual geometry branch:

- keep signed normal convention fixed;
- keep point/depth branch targeted coupling;
- add full-body/hands evaluation as mandatory candidate packaging;
- only consider a new candidate if both `world_points` and `depth_unprojection` improve face/head without introducing shell/ghost artifacts and without full-body/hands regressions.
