# 2026-05-07 D-Line Guard Status

Status: `strict_gate_blocked_research_preflight_only`

This is a D-line guard/report audit only. No experiment script, strict registry,
teacher artifact, candidate artifact, or pass state was edited.

## Current Gate Truth

Strict registry read:

```text
registry = reports/20260504_strict_gate_registry.json
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-06T03:57:15.701453+00:00
check_time = 2026-05-07T11:42:35+08:00 / 2026-05-07T03:42:35Z
registry_age_hours_by_guard = 23.72
```

Registry counts:

```text
candidates = 26
teachers = 81
strict_candidate_passes = 0
strict_teacher_passes = 0
kinect_coord_passes = 2
raw_sensor_fullbody_hand_passes = 0
smplx_weak_anchor_passes = 1
full_gate_numeric_pass_visual_fail = 1
teacher_numeric_pass_visual_fail = 0
other_teacher_fail = 81
visible_surface_teacher_passes = 0
```

The two Kinect coordinate positives are head/head-face ROI diagnostics, not full
strict teacher passes. The SMPL-X weak-anchor positive is also not a strict
teacher or strict candidate pass. Numeric-only or ROI-local positives cannot
authorize cloud, teacher export, candidate export, or mentor pass claims.

`python tools/check_cloud_gate_status.py --json` reports:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reason = strict_candidate_passes is 0
```

`python tools/check_cloud_gate_status.py --teacher-supervised --json` reports:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reasons = strict_candidate_passes is 0;
          teacher-supervised route requested but strict_teacher_passes is 0
```

Therefore the current formal state is:

```text
formal cloud train = blocked
formal cloud infer = blocked
formal cloud export = blocked
teacher-supervised route = blocked
teacher export = blocked
candidate export = blocked
```

## Schema And Freshness

The registry schema matches the guard requirement:

```text
required_schema = 20260504_visual_fullbody_hands_v2
actual_schema = 20260504_visual_fullbody_hands_v2
schema_status = current
```

Freshness is currently within the default guard window but close to the edge:

```text
default_max_age_hours = 24
observed_age_hours = 23.72
freshness_status_default = current, but refresh soon
```

With a stricter one-hour check, the same registry is stale:

```text
python tools/check_cloud_gate_status.py --max-age-hours 1 --json
reasons = strict gate registry is stale (23.7h > 1.0h);
          strict_candidate_passes is 0
```

No formal cloud decision should rely on this registry once it crosses the
24-hour guard window; refresh the registry first and keep the same strict pass
requirements.

## Formal Guard Relationship

`tools/check_cloud_gate_status.py` is read-only and never contacts Modal. It
allows cloud only when:

```text
registry exists and is readable
schema_version == 20260504_visual_fullbody_hands_v2
registry_age_hours <= max_age_hours
strict_candidate_passes > 0
```

Teacher-supervised routes additionally require:

```text
strict_teacher_passes > 0
```

The formal Modal train/infer guards use the same registry, schema, 24-hour
freshness, and strict candidate count. Training also conservatively treats route
names containing markers such as teacher, teacher_target, surface_target,
supervised, from60v, kinect, colmap, mvs, or depthpro as teacher-supervised and
requires a strict teacher pass. Inference-side remote exports are guarded as
cloud inference/export actions.

No clear guard bug was found in this audit, so no code change is recommended.

## Research-Preflight Boundary

`modal_surface_research_preflight.py` is intentionally separate from formal
train/infer/export. Its local registry status helper can print whether formal
cloud remains blocked, but the entrypoint remains research-only even if the
formal gate becomes green later.

Allowed research-preflight scope:

```text
upload research scene/assets through upload_research_assets
run ping_scene
run A_readiness
run A1_refine_visual_hull_mesh
run A2_neural_field
run A3_visual_hull_init
run A4_neus_sdf_surface
run A4_1_part_local_sdf
run A5_known_camera_colmap_workspace as a research/preflight workspace only
run B0_surface_tokens
run B1_surface_tokens
run B2_surface_tokens
download existing research outputs
write JSON/MD/PLY/PNG diagnostic artifacts and wrapper summaries
```

Forbidden research-preflight scope:

```text
write or mutate reports/*strict_gate_registry*.json as a pass signal
write strict pass state
export teacher targets
export candidate packages
name remote data/output paths as strict_pass, teacher_export, or candidate_export
call formal VGGT cloud train/infer/export
claim mentor pass from wrapper status=completed
claim pass from IoU/precision/recall/numeric improvements alone
promote head/head-face ROI, weak prior, or numeric-only positives to strict pass
continue frozen routes as unblockers: r16/r18/r19/r57-r66 epoch loop,
HART PnP replacement, TSDF/signfix shell teacher, Kinect patch without strict
coord+visual pass, support/threshold/confidence loops, or numeric-only pass
```

The wrapper rejects research asset/output paths containing:

```text
strict_pass
teacher_export
candidate_export
```

The remote run metadata writes:

```text
research_only = true
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
formal_cloud_train_infer_export = blocked unless local strict gate passes
```

## Local Research Artifact Cross-Check

Local research wrapper summaries under `output/surface_research_preflight` were
checked for launch guard consistency:

```text
TOTAL research_preflight_summary.json = 40
bad_flags = 0
all summaries have research_only = true
all summaries have no_teacher_export = true
all summaries have no_candidate_export = true
all summaries have no_strict_pass_write = true
all launch-time strict_candidate_passes = 0
all launch-time strict_teacher_passes = 0
all launch-time formal_cloud_allowed = false
```

Lane counts:

```text
A_readiness = 2
A1_refine_visual_hull_mesh = 3
A2_neural_field = 6
A3_visual_hull_init = 3
A4_1_part_local_sdf = 2
A4_neus_sdf_surface = 2
A5_known_camera_colmap_workspace = 11
B0_surface_tokens = 1
B1_surface_tokens = 2
B2_surface_tokens = 8
```

No research-preflight file scan found a true strict candidate pass, true strict
teacher pass, true teacher/candidate visual pass, `formal_cloud_allowed=true`,
`cloud_allowed=true`, or a false no-export/no-pass wrapper flag. No local
research-preflight file named like `teacher_targets`, `candidate`,
`strict_pass`, `teacher_export`, or `candidate_export` was found in the checked
research-preflight output roots.

## Report Wording Check

The current 2026-05-06 guard/preflight reports are directionally consistent:
they state strict candidate/teacher passes are zero, formal cloud is blocked,
and research artifacts are not teachers, candidates, or pass signals.

Potentially ambiguous words that must stay scoped:

```text
completed = wrapper/process completion only, not pass
asset_ready_for_research_preflight = input readiness only, not teacher/candidate
unblocked_surface_not_solved = backend/plumbing status only, not cloud unblock
positive IoU/precision/recall = diagnostic metrics only, not strict visual pass
```

No report wording was found that should currently be treated as a mentor pass or
cloud authorization. The safest phrasing going forward is to pair any
`completed`, `ready`, `green`, `positive`, or `unblocked` wording with the
explicit guard state:

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher/candidate export = blocked
research-only diagnostic
```

## Commands Run

```text
Get-Content -Raw reports/20260504_strict_gate_registry.json
Get-Content -Raw tools/check_cloud_gate_status.py
Get-Content -Raw modal_surface_research_preflight.py
python tools/check_cloud_gate_status.py --json
python tools/check_cloud_gate_status.py --teacher-supervised --json
python tools/check_cloud_gate_status.py
python tools/check_cloud_gate_status.py --max-age-hours 1 --json
Select-String guard/report scans over Modal guard files and 2026-05-04/06 reports
Get-ChildItem output/surface_research_preflight -Filter research_preflight_summary.json
Select-String research-preflight outputs for pass/export/cloud-allowed true flags
```

Note: `rg` was attempted first for text search but was denied by the Windows
environment, so PowerShell `Select-String`/`Get-ChildItem` scans were used.
