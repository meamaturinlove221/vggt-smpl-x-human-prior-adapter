# Research Preflight Guard Audit

Date: 2026-05-06

Status: `strict_gate_blocked_research_preflight_only`

This D-line audit checks the truthfulness of the current research-preflight
state. It is report-only. It does not edit the strict registry, does not change
Modal entrypoints, does not change A/B tool logic, and does not touch training
code.

## Current Strict Gate Truth

Read-only guard command:

```text
python tools/check_cloud_gate_status.py --json
```

Current result:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reason = strict_candidate_passes is 0
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-06T03:57:15.701453+00:00
```

Teacher-supervised guard command:

```text
python tools/check_cloud_gate_status.py --teacher-supervised --json
```

Current result:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reasons = strict_candidate_passes is 0;
          teacher-supervised route requested but strict_teacher_passes is 0
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-06T03:57:15.701453+00:00
```

Therefore the only truthful gate state is:

```text
formal cloud train = blocked
formal cloud infer = blocked
formal cloud export = blocked
teacher-supervised route = blocked
candidate export = blocked
teacher export = blocked
```

## Formal Cloud Guard Boundary

The formal cloud guard in `tools/check_cloud_gate_status.py` authorizes cloud
only when all required strict conditions hold:

```text
strict_candidate_passes > 0
registry schema == 20260504_visual_fullbody_hands_v2
registry age <= max_age_hours, default 24
```

For teacher-supervised routes it additionally requires:

```text
strict_teacher_passes > 0
```

The current registry has zero strict candidate passes and zero strict teacher
passes, so no formal train/infer/export route may be treated as unblocked.

This audit did not run `--refresh` and did not write:

```text
reports/*strict_gate_registry*.json
```

## Research-Preflight Allowed Boundary

Research-preflight Modal work is allowed only as isolated artifact generation.
Its allowed scope is:

```text
run raw asset readiness checks
run A-line/B-line surface research smokes
write research summaries, meshes, contact sheets, and diagnostics
record failure or interim evidence
```

Its forbidden scope is:

```text
write strict pass state
write or mutate strict registry
export a teacher
export a candidate
call or unblock formal VGGT cloud train/infer/export
claim mentor pass from a completed research wrapper
claim pass from IoU/precision/recall alone without strict visual gate success
```

Completed research wrappers only prove that a research lane ran. They do not
override `strict_candidate_passes=0` or `strict_teacher_passes=0`.

## Lane Truthfulness

The current A/B research-preflight lanes must be interpreted as follows:

| Lane | Current truthful state | Teacher/candidate/pass status |
| --- | --- | --- |
| A1 visual-hull mesh refinement | Silhouette shrink preflight; useful gradient/plumbing signal. | Must not be teacher, candidate, or pass. |
| B1 surface-token backend smoke | Cloud nvdiffrast/token diagnostics run; geometry signal too weak. | Must not be teacher, candidate, or pass. |
| A2 raw neural occupancy field | Raw RGB/mask/camera signal smoke; thick overfilled occupancy volume. | Must not be teacher, candidate, or pass. |
| A4 NeuS/SDF surface field | Promising representation change; thinner mostly connected SDF surface but visual fail remains. | Must not be teacher, candidate, or pass. |

No A1/B1/A2/A4 result currently satisfies strict teacher or strict candidate
requirements.

## A2 Eval Bug And Fixed State

Earlier A2 thin run:

```text
output/surface_research_preflight/A2_neural_field_t64_step160_train4_eval2_thin_gpu
intended_train_views = 0,20,40,50
intended_eval_views = 10,30
actual_eval_views = 0,20,40,50
```

Truthfulness decision:

```text
This run must not be cited as held-out evidence.
It is train-view evidence only because the wrapper did not pass held-out eval
views in that revision.
```

Fixed A2 held-out run:

```text
output/surface_research_preflight/A2_neural_field_t64_step40_train4_eval2_thin_fixed_gpu
cmd includes --eval-view-indices 10,30
train_views = 0,20,40,50
eval_views = 10,30
eval_avg_render_iou = 0.6454713535
eval_avg_render_precision = 0.6589585395
eval_avg_render_recall = 0.9693406593
occupied_fraction = 0.3034362793
mesh_component_count = 75
mesh_largest_component_ratio = 0.9731001096
```

Truthfulness decision:

```text
freeze_current_tiny_occupancy_field_as_raw_signal_smoke
```

Reason:

```text
A2 proves the raw calibrated image/mask path has a learnable signal, but the
mesh is a thick occupancy volume. High recall comes from over-filling, held-out
precision remains low, and the contact-sheet review lacks modeled face,
hairline, connected hands, and mentor-level full-body surface detail.
```

Disallowed next moves for A2:

```text
more occupancy threshold tuning
more render threshold tuning
more step-count tuning
using train-view-only metrics as held-out evidence
teacher export from A2
candidate export from A2
strict pass write from A2
```

## A4 Interim State

A4 changed representation from A2's free occupancy volume to an SDF-derived
zero-level surface with eikonal and ray-concentration diagnostics.

First 4-train/2-eval run:

```text
output/surface_research_preflight/A4_neus_sdf_t64_step96_train4_eval2_first_gpu
train_views = 0,20,40,50
eval_views = 10,30
train_avg_render_iou = 0.8120585709
train_avg_render_precision = 0.8387501475
train_avg_render_recall = 0.9610909465
eval_avg_render_iou = 0.7293989059
eval_avg_render_precision = 0.7478524754
eval_avg_render_recall = 0.9664835165
near_surface_fraction = 0.0349273682
mesh_component_count = 3
mesh_largest_component_ratio = 0.9958049292
```

Truthfulness decision:

```text
promising_representation_change_but_not_mentor_pass
```

Reason:

```text
A4 materially improves the held-out IoU/precision state over the fixed A2 split
and extracts a thinner, mostly connected SDF surface. Visual review still fails:
the mesh remains a slim template-like body without modeled face relief,
hairline, hands/fingers, clothing, or full-body detail. It cannot be a strict
teacher or candidate.
```

## Stop Conditions And Next Allowed Moves

Stop A1/B1/A2 as pass-seeking loops now:

```text
A1 plus more silhouette steps is stopped
B1 plus hidden-size/step/weight tuning is stopped
A2 plus threshold/step tuning is stopped
```

Continue A4 only if the next change adds a genuinely new surface signal or local
surface mechanism, such as:

```text
view-consistent rendered depth/normal supervision
part-local face/hair/hand residual surface carriers
true NeuS CDF formulation
external same-frame dense reconstruction backend adapter
```

Stop A4 and freeze the internal tiny-field line if the next run preserves the
same visual failure:

```text
thin template-like body
no modeled face relief
no hairline surface
no connected hands/fingers
no clothing or person-specific full-body detail
```

Formal cloud remains stopped until a local strict gate package truthfully shows:

```text
strict_candidate_passes > 0
and, for teacher-supervised routes, strict_teacher_passes > 0
and current schema visual full-body/head/face/hairline/attached-hand review pass
```

## Files Read For This Audit

```text
tools/check_cloud_gate_status.py
reports/20260506_surface_research_preflight_status.md
reports/20260506_a1_b1_a2_freeze_and_next_unblockers.md
reports/20260506_dline_surface_research_preflight_gate_audit.md
reports/20260504_cloud_guard_and_no_wall_status.md
```

## Files Modified By This Audit

```text
reports/20260506_research_preflight_guard_audit.md
```
