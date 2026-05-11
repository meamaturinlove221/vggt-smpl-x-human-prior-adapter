# 2026-05-07 B-Hand Next Unblocker Decision

Status: `blocked_until_continuous_connected_hand_surface_artifact`

This note is a local decision report for the B-hand side line. It does not run
training, inference, reconstruction, teacher export, candidate export, strict
registry writes, or cloud jobs.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher export = blocked
candidate export = blocked
strict pass write = blocked
```

## Local Facts Read

Relevant local reports and scripts already cover the cheap B-hand questions:

```text
reports/20260507_b_hand_evidence_cache_status.md
reports/20260507_b_hand_label_evidence_learnability_status.md
reports/20260507_b_hand_colmap_depth_evidence_status.md
reports/20260507_b_joint_surface_hand_contract_status.md
tools/b_hand_evidence_cache.py
tools/b_hand_token_backend_smoke.py
tools/b_hand_decoder_skeleton_smoke.py
tools/b_hand_connected_mesh_precheck.py
tools/b_hand_label_evidence_learnability_probe.py
tools/b_hand_colmap_depth_evidence_probe.py
tools/b_joint_surface_hand_contract_probe.py
```

The local output tree currently reaches B-hand6 and the joint surface/hand
contract probe only:

```text
B_hand0 evidence cache
B_hand1 token backend smoke
B_hand2 decoder skeleton smoke
B_hand3 SMPL-X wrist/arm connected precheck
B_hand4 connected mesh precheck
B_hand5 label/evidence learnability probe
B_hand6 COLMAP depth evidence probe
B_joint surface/hand contract probe
```

No B-hand7 or equivalent continuous hand-surface artifact exists locally.

## Evidence Summary

B-hand3/B-hand4:

```text
SMPL-X/body scaffold can be connected topologically.
raw SMPL-X hand gate is still false.
B-hand4 connected proxy pass = false.
connected template/SMPL-X topology is weak scaffold evidence only.
```

B-hand5:

```text
roi_examples = 14
side weak label signal = not reliable after controls
connection risk signal = not reliable
depth/span risk signal = weak diagnostic only
upstream connected hand gate remains failed
```

B-hand6:

```text
roi_count = 14
mapped_depth_roi_count = 12
depth_valid_ratio_total = 0.3086
left depth_valid_ratio_total = 0.5425
right depth_valid_ratio_total = 0.2450
```

This proves bbox-level COLMAP depth presence for both sides, not a continuous
hand mesh, not wrist/palm/finger recovery, and not an Open3D hand gate pass.

B-joint:

```text
combined_beats_hand_only = false
combined_no_absolute_x_survives = false
surface_context_only_accuracy = 0.4286
hand_only_depth_risk_accuracy = 0.7857
combined_accuracy = 0.7857
```

The joint interface adds no stable signal beyond the existing hand controls.

## Frozen Lines

The following B-hand lines are frozen and must not be repeated as the next
unblocker:

```text
B-hand0/B-hand1/B-hand2 evidence-cache, token-smoke, or decoder-skeleton reruns
B-hand3/B-hand4 SMPL-X or connected-template topology as hand success
B-hand5 weak label learnability reruns, ridge/prototype tweaks, or feature ablations
B-hand6 COLMAP bbox-depth threshold, view-set, or valid-range tuning
B-joint surface+hand feature concatenation or no-absolute-x control reruns
MediaPipe hand detection, landmarks, or patch relief as a success claim
SMPL-X hand residual, MANO-like scaffold, or template residual as a success claim
hidden-size, step-count, smoothing, threshold, component-filter, or confidence loops
teacher export, candidate export, predictions export, strict registry writes
formal cloud train/infer/export
```

These are frozen because they answer only evidence existence, weak label
predictability, bbox depth presence, or topology-scaffold connectivity. None of
them produces the missing continuous hand surface connected to arms.

## Allowed Minimal Artifact

The only non-repeated B-hand unblocker is a small local artifact package that
proves a continuous, arm-connected hand surface exists before any decoder,
teacher, candidate, or cloud step is considered.

Allowed artifact name:

```text
B_hand7_continuous_connected_hand_surface_review
```

Allowed contents:

```text
one local summary JSON
one local markdown report
left/right Open3D screenshots: front, side, top, iso
one combined hands+wrist+forearm Open3D screenshot set
optional diagnostic PLY/OBJ only if it is explicitly marked research-only
```

Required checks:

```text
left and right hands are both present
wrists are connected to forearms/arms, not detached sheets or floating blobs
palm surface is continuous enough to read as one hand surface
fingers are visible as surface structure or explicitly reported missing
largest connected component and fragmentation stats are reported per side
front/side/top/iso Open3D views are included for both sides and combined hands
SMPL-X/MediaPipe/template contribution is labeled weak support, not success
strict_candidate_passes = 0 and strict_teacher_passes = 0 remain written in the report
```

Allowed inputs:

```text
existing B-hand evidence cache, token smoke, B-hand4 proxy, B-hand6 depth presence
existing B-Fus3D raw-image/support diagnostics as context only
one genuinely new dense same-frame surface artifact, if it appears on disk
```

Not allowed as the minimal artifact:

```text
ROI boxes only
bbox-level COLMAP depth presence only
SMPL-X hand topology only
MediaPipe hand landmarks or patches only
point clusters without a continuous surface
sparse query proposals without wrist/palm/finger surface continuity
numeric-only pass without Open3D visual review
```

## Blocked Loops

Until the allowed minimal artifact exists, B-hand has no useful local execution
loop left. The following loops are explicitly blocked:

```text
rerun B-hand5 with new thresholds/features until weak labels look better
rerun B-hand6 with different COLMAP depth min/max, view ids, or bbox scales
promote SMPL-X wrist/hand residuals into predictions or teacher targets
patch hand pixels with MediaPipe, landmarks, silhouettes, or template overlays
convert B-hand4 connected topology proxy into a pass by component filtering
train a hand decoder from B-hand5/B-hand6 weak labels
combine B-Fus3D context with B-hand features again without new surface evidence
write strict pass, teacher, candidate, or predictions artifacts from these probes
upload/run formal cloud train, infer, or export
```

## Freeze Condition

If no continuous connected hand-surface review artifact can be produced from a
new same-frame dense surface source or a genuinely new rendered-surface backend
contract, B-hand is frozen.

Frozen means:

```text
do not run another B-hand evidence probe
do not tune weak-label or bbox-depth readouts
do not claim hand progress from SMPL-X or MediaPipe patches
do not unblock training, inference, export, registry writes, or cloud
move effort to the shared surface backend / dense artifact lane that can supply
the missing continuous surface evidence
```

## Decision

The next B-hand step is not another local probe. The only non-repeated
unblocker is:

```text
Produce or import one local B_hand7_continuous_connected_hand_surface_review
artifact showing continuous left/right hand surfaces connected to wrists/arms
under Open3D front/side/top/iso review.
```

If that artifact is not available, B-hand should remain frozen. B-hand5/B-hand6
may be cited only as weak support diagnostics for a future backend, not as a
teacher, candidate, strict pass, or cloud unblocker.
