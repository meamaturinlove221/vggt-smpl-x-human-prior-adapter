# 2026-05-07 Plan v5 Four-Mainline Landing

Status: `landing_contract_no_gate_change`

This note maps the proposed four-mainline plan onto the current local checkout
state. It does not run reconstruction, train, infer, export a teacher/candidate,
write predictions, edit the strict registry, launch cloud, or change guards.

## Non-Negotiable Runtime Contract

```text
agent model requirement = GPT-5.5
agent reasoning requirement = xhigh
local Open3D runtime = D:\anaconda\envs\g3splat\python.exe
local Open3D helpers = tools/open3d_view_pointcloud.py, scripts/open_pointcloud_open3d.ps1
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher-supervised cloud route = blocked
research-preflight = diagnostic only
```

The mentor gate remains visual and geometric, not name-driven. Numeric point
counts, bbox depth presence, weak landmarks, SMPL-X topology, diagnostic
coordinate passes, or research wrapper completion do not count unless the
current strict full/head/face/hairline/hands Open3D protocol passes.

## Current Local Truth

The local repo already contains the decisive A5/B-Fus3D/B-hand/D-line evidence
needed to stop several loops:

```text
reports/20260507_plan_v4_parallel_unblocker_status.md
reports/20260507_a5_next_unblocker_decision.md
reports/20260507_b_fus3d_b19_bounded_surface_sdf_render_status.md
reports/20260507_b_hand_next_unblocker_decision.md
reports/20260507_dline_post_b16_guard_audit.md
reports/20260507_strict_gate_registry_refresh.md
```

Guard check on this machine still returns:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
teacher-supervised route additionally blocked by strict_teacher_passes = 0
```

## A5 Same-Frame Dense Teacher

Priority: `near_term_main_but_colmap_cuda_frozen_after_decision_set`

What is already answered locally:

```text
v8 triangulated adj6 fused_points = 2206
v8 face_core depth-compatible = 0
v8 head_face depth-compatible = 0
v8 hairline depth-compatible = 0

known_direct adj6 fused_points = 93692
face_core depth-compatible = 2
head_face depth-compatible = 62
hairline depth-compatible = 3

known_direct adj12 fused_points = 153062
face_core depth-compatible = 4
head_face depth-compatible = 193
hairline depth-compatible = 25

known_direct hybrid12 fused_points = 101420
face_core depth-compatible = 47
head_face depth-compatible = 1203
hairline depth-compatible = 20
```

Landing decision:

```text
A5 COLMAP CUDA = functional backend smoke
A5 COLMAP CUDA = not a teacher
A5 more view-count/source-pair/fusion-threshold loops = frozen
A5 Poisson/BPA/surface extraction from current sparse/misaligned clouds = blocked
```

Allowed next A5 action:

```text
Import exactly one new same-frame dense surface artifact or mutually consistent
calibrated multi-view depth set, then run tools/a5_external_dense_backend_preflight.py
and the strict Open3D full/head/face/hairline/hands teacher review.
```

If no new external dense artifact exists on disk, A5 is parked. Do not spend
cycles on adjacent-20/30, hybrid variants, COLMAP thresholds, visual hull,
SMPL-X patching, landmark patching, or per-view shell fusion.

## B-Fus3D Learned SDF Backend

Priority: `long_term_main_but_current_b19_instance_frozen`

What is already answered locally:

```text
VGGT aggregator token extraction works.
Layer-23 token cache exists for hybrid6 with face/hair/hand ROI coverage.
B17 surface/SDF contract preflight exists.
B19 bounded query-to-carrier rendered smoke ran real/shuffle/zero controls.
```

B19 result:

```text
real_minus_shuffle_iou = 0.0
real_minus_zero_iou = -0.0000904728
real_rgb_better_than_shuffle = false
real_rgb_better_than_zero = true
decision = freeze this bounded query-to-carrier implementation
```

Landing decision:

```text
Do not tune B19 hidden size, steps, thresholds, smoothing, or component filters.
Do not revive B2/B16/B18/B19 as small residual/carrier-offset loops.
```

Allowed next B-Fus3D action:

```text
B-Fus3D0-v2 must be a genuinely different representation:
VGGT latent plus raw RGB/masks/known cameras -> human canonical 3D latent grid
or local surface-token grid -> 2D-to-3D cross attention -> 3D refinement ->
SDF/occupancy/normal residual/visibility/confidence.
```

Acceptance for the next implementation is not a strict pass. It is a local
single-frame overfit decision smoke where:

```text
real > shuffle and real > zero on rendered geometry metrics
face/hair/hands visibly depart from the SMPL-X/template shell
depth/normal/render consistency improve together
Open3D is not just the connected carrier or SMPL-X shell
```

If real again does not beat shuffle/zero, freeze that B-Fus3D variant.

## B-GS SMPL-X Anchored Plus Free Gaussians

Priority: `new_human_representation_mainline`

Current local gap:

```text
No B-GS0 report or script is present.
No Gaussian-based single-frame overfit artifact is present.
```

This is the most important newly added implementation lane because it attacks
the template-shell failure with representation capacity instead of another mesh
offset. It should be started as a new research-only local line, not as a
renamed B2/B19 loop.

B-GS0 minimal contract:

```text
input = raw 60v RGB, masks, cameras, SMPL-X weak anchor
representation = each SMPL-X vertex gets one constrained Gaussian plus K free
Gaussians for clothing, hair, silhouette, sleeves, and template-outside regions
supervision = mask, RGB, depth order, silhouette, weak landmarks
outputs = Gaussian point review, Open3D review, raster protocol diagnostics
no predictions.npz, no checkpoint export, no teacher/candidate export
```

Hard acceptance:

```text
hairline/clothing are not template shell
hands remain attached
full body does not break
Open3D shows non-template outer structure
geometry is not transparent spikes, floating dots, or RGB-only improvement
```

The first landing task is to add only the B-GS0 contract/preflight skeleton and
fail-closed output schema. After that, run a tiny local single-frame smoke.

## B-Hand And B-Hair Hard-Gate Lines

Priority: `local_blocker_mainline`

B-hand current local truth:

```text
B-hand0..B-hand6 and B-joint already answer evidence/cache/weak-label/depth
presence/topology questions.
No B_hand7_continuous_connected_hand_surface_review exists.
Weak labels, bbox COLMAP depth, MediaPipe, MANO-like scaffolds, and SMPL-X hand
topology are not success.
```

B-hand landing decision:

```text
B-hand is blocked until one continuous arm-connected left/right hand surface
artifact exists and passes Open3D front/side/top/iso review.
Do not run more weak-label or bbox-depth threshold loops.
```

B-hair current local truth:

```text
No dedicated B-hair0 report or script is present.
A5 and teacher gates repeatedly show hairline coverage collapsing.
```

B-hair0 minimal contract:

```text
scalp root support
hairline boundary support
strand, curve, or Gaussian-chain primitives
photometric refinement
topological connectivity metric
Open3D head/hairline/head-top review
```

Hard acceptance:

```text
hairline no longer 0/6 or near-empty
head top is not a coarse template cap
result is not floating dots
result is not only the template head shell
```

The B-hand/B-hair line should initially produce contracts plus one smoke each.
Any successful local primitive must later merge back into the full-body protocol.

## D-Line Strict Gate And Cloud Guard

Priority: `resident_referee`

D-line remains red and should stay resident:

```text
refresh registry
run tools/check_cloud_gate_status.py --json
run tools/check_cloud_gate_status.py --teacher-supervised --json
scan new research scripts for pass/export/cloud-unblock risk wording
keep research-preflight metadata research_only/no_export/no_strict_pass_write
build or refresh Open3D contact sheets for full/head/face/hairline/hands
```

D-line must block:

```text
formal cloud train/infer/export
teacher/candidate export
strict registry pass writes
research-preflight reports being described as mentor passes
numeric-only or visual-only green claims
```

## Execution Split

Use agents only when they are explicitly launched as GPT-5.5 xhigh.

```text
Agent A: A5 Dense Teacher
  Current action: parked unless a new external dense artifact appears.
  If artifact appears: run one artifact-intake preflight and strict Open3D review.

Agent B: B-Fus3D
  Current action: do not tune B19. Design/implement B-Fus3D0-v2 only if it is a
  true latent-grid or surface-token-grid SDF representation with controls.

Agent C: B-GS
  Current action: start. Add B-GS0 fail-closed contract/preflight skeleton, then
  a tiny single-frame Gaussian smoke.

Agent D: B-hand/B-hair
  Current action: B-hand waits for continuous hand surface artifact; B-hair0
  needs first contract/smoke.

Agent E: D-line
  Current action: keep strict guard red, registry fresh, and contact sheets honest.
```

## Immediate Local Landing Order

```text
1. Treat A5 COLMAP CUDA as frozen until a new dense artifact is imported.
2. Treat B19 as frozen because real did not beat shuffle/zero.
3. Start B-GS0 contract/preflight skeleton as the first new code lane.
4. Start B-hair0 contract/preflight skeleton if a second lane is available.
5. Keep B-hand parked until B_hand7 continuous hand surface evidence exists.
6. Keep D-line guard checks running before any cloud or export claim.
```

## Commands To Reuse

```powershell
python tools/check_cloud_gate_status.py --json
python tools/check_cloud_gate_status.py --teacher-supervised --json
powershell -ExecutionPolicy Bypass -File scripts/open_pointcloud_open3d.ps1 -InputPath <ply-or-folder>
```

Open3D must be invoked through:

```text
D:\anaconda\envs\g3splat\python.exe
```

unless a current machine audit proves another environment has working Open3D.

## Bottom Line

The quoted plan is directionally right, but local evidence changes the first
move. A5 has already answered the bounded COLMAP decision set and should not
consume more parameter-search time. B-Fus3D has also answered the current B19
bounded representation negatively. The next real implementation gap is B-GS0,
with B-hair0 as the other missing hard-gate contract, while D-line keeps the
strict mentor gate red until a true full/head/face/hairline/hands surface exists.
