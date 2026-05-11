# 2026-05-01 HART-style PnP and Full-Body Gate Local Status

## Current Truth

No local candidate is mentor-final. Do not upload to cloud.

The HART-style camera-from-pointmap ablation has been completed in the isolated worktree:

```text
D:\vggt\vggt-main-hart-pnp
branch: codex/hart-pnp-camera-ablation
```

Aggregate output:

```text
D:\vggt\vggt-main-hart-pnp\output\hart_style_pnp_camera_ablation\aggregate_signfix_r16_r20_r52_r57_r58\aggregate_report.md
```

Conclusion: PnP solves cameras and reduces pointmap reprojection in some cases, but it does not beat the VGGT camera head on GT-aligned camera metrics and does not improve Open3D head/face quality. Therefore the camera head should not be removed. HART should only be cited as camera-from-pointmap inspiration, not as a reason to make the pipeline camera-free.

## Newly Fixed Local Gate Issue

The previous full-body/hands audit was too easy to misread. It could show `Bottom-line pass: True` even when Open3D hands were visually fragmented or not hand-like.

Updated files:

```text
tools\audit_fullbody_hand_integrity.py
tools\package_normal_candidate_gate.py
```

Local gate changes:

- hand audit now checks compact 3D support inside each MediaPipe hand box;
- elongated hand/forearm/shell fragments are blocked by `max_hand_box_3d_extent` and `max_hand_box_depth_range`;
- package report now labels this as a coverage/metric screen, not a visual pass;
- package report now adds a full-body provenance gate, so a head/face candidate cannot silently inherit an older shared full-body NPZ and claim candidate-specific full-body success;
- `render_artifacts_ok` is explicitly only screenshot/PLY generation, not visual quality;
- full-body/hands visual pass remains controlled by explicit Open3D review keys.

Regression outputs:

```text
output\normal_line_multiview_20260430\gate_regression_r57_strict_fullbody_hand_package
output\normal_line_multiview_20260430\gate_regression_r58_strict_fullbody_hand_package
output\normal_line_multiview_20260430\gate_regression_r59_strict_fullbody_hand_package
```

All are correctly blocked from pass/cloud.

## Negative Candidates Confirmed

### r57

r57 can pass numeric/normal/shape under the old automatic view, but it is still not mentor-final:

- Open3D face/head is still shell-like rather than clearly modeled;
- full-body input is inherited from r24, not candidate-specific;
- stricter hand gate fails: only 1 view has compact 3D hand support and 2 hand boxes are implausible.

### r58

r58 is negative:

- same-protocol p40 face ROI regresses below signfix;
- shape gate fails on face depth p40 thinness boundary;
- full-body/hands strict gate fails;
- full-body input is inherited.

### r59

r59 is negative:

- true-highres teacher pointcloud projection into r57 lowers same-protocol p40 full/head/face counts;
- fixed threshold also regresses;
- shape gate fails multiple head/face coverage/protrusion checks;
- visual quick renders remain shell/blank-like;
- full-body/hands strict gate fails.

## Mentor Gate Reminder

A candidate may only be considered local pass when all are true:

- same-protocol 6v headshoulder numeric gate passes for `world_points` and `depth_unprojection`;
- head/face/hairline Open3D closeups show modeled geometry, not texture shell;
- normal-depth-point consistency does not regress;
- shape metrics do not indicate shell collapse;
- full-body Open3D looks like a normal human from front/side/back/iso;
- no large body holes, broken limbs, severe ghosting, or implausible body proportions;
- hands are not missing, amputated, or only scattered noise;
- full-body result is candidate-specific or explicitly justified;
- explicit visual review JSON passes all required checks.

## Next Non-Wall Direction

Do not continue by:

- adding epochs to r16/r18/r19;
- increasing confidence boosts;
- repeating MVS/Kinect/external pointcloud projection patches;
- claiming success from normal metrics alone;
- claiming success from true-highres crop alone as a replacement for same-protocol headshoulder.

Next useful direction is teacher-gate first:

1. Find or construct a continuous, aligned head/face surface teacher that can be projected back into the original 6-view headshoulder protocol without creating shells or lowering ROI counts.
2. Validate that teacher locally with the strict gate before any new training.
3. Only if the teacher passes visual/numeric sanity, run a small point-branch-targeted or one-frame ROI overfit smoke.
4. Package every candidate with same-protocol head/face, full-body/hands, p40/fixed threshold, normal consistency, shape metrics, Open3D sheet, and explicit visual review.

Current status remains blocked for cloud upload.
