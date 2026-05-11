# D-Line Surface Research Preflight Gate Audit

Date: 2026-05-06

Status: `strict_gate_blocked_research_artifacts_only`

This D-line audit cross-checks:

- `reports/20260506_surface_research_preflight_status.md`
- `output/surface_research_preflight/**/research_preflight_summary.json`
- `tools/check_cloud_gate_status.py`

No `pass` field was edited. No formal cloud train/infer/export route is
unblocked. No teacher or candidate is accepted by this audit.

## Gate Truth

`tools/check_cloud_gate_status.py --json` reports:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reason = strict_candidate_passes is 0
exit_code = 2
```

`tools/check_cloud_gate_status.py --teacher-supervised --json` reports:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
reasons = strict_candidate_passes is 0;
          teacher-supervised route requested but strict_teacher_passes is 0
exit_code = 2
```

The registry schema is current for the local guard:

```text
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-06T03:57:15.701453+00:00
```

## Modal Artifact Summary Cross-Check

All downloaded wrapper summaries under `output/surface_research_preflight`
agree with the strict gate:

```text
research_only = true
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
formal_cloud_allowed_at_launch = false
strict_candidate_passes_at_launch = 0
strict_teacher_passes_at_launch = 0
```

Observed wrapper summaries:

| Output subdir | Lane | Wrapper status | Return code | Artifact note |
| --- | --- | ---: | ---: | --- |
| `A_readiness_60v_humancrop_t96_cpu_v2` | `A_readiness` | `completed` | 0 | Older readiness smoke; v3 supersedes camera intrinsics interpretation. |
| `A_readiness_60v_humancrop_t96_cpu_v3_intrinsics` | `A_readiness` | `completed` | 0 | Asset readiness only; `asset_ready_for_research_preflight=true`, not a teacher/candidate. |
| `B0_surface_tokens_t64_step2_gpu` | `B0_surface_tokens` | `completed` | 0 | GPU/nvdiffrast path ran; inner summary says `formal_cloud=blocked` and strict passes are 0. |
| `A3_visual_hull_init_t96_g56_s4` | `A3_visual_hull_init` | `completed` | 0 | Coarse visual-hull support diagnostic only. |
| `A3_visual_hull_mesh_t96_g56_s4` | `A3_visual_hull_init` | `completed` | 0 | Mesh extracted: 14208 vertices, 28586 faces; still research-only. |
| `A3_visual_hull_mesh_project_t96_g56_s4` | `A3_visual_hull_init` | `completed` | 0 | Mesh projection summary was downloaded after the status report text; mean IoU 0.6611346666, recall 0.9996233214, precision 0.6613312293. Not a pass signal. |

## Report Consistency Finding

`reports/20260506_surface_research_preflight_status.md` is directionally
consistent with the gate: it says strict candidate/teacher passes are 0,
formal cloud train/infer/export is blocked, and the A/B/A3 artifacts are not
teachers or candidates.

One additive mismatch was found:

```text
output/surface_research_preflight/A3_visual_hull_mesh_project_t96_g56_s4
```

exists locally but is not listed in the status report. This audit records it
as an additional research-only A3 visual-hull projection artifact. Its summary
does not change the gate: it carries the same no-export/no-pass flags and the
same launch-time strict pass counts of zero.

## Local Visual Review Cross-Check

The local B0 Open3D review decision remains fail:

```text
visual_pass = false
teacher_visual_pass = false
candidate_visual_pass = false
decision = fail
strict_candidate_passes = 0
strict_teacher_passes = 0
```

The reason is that the short B0 run does not show mentor-level non-template
face, hairline, hands, or full-body surface quality; numeric IoU deltas cannot
be promoted to strict pass accounting.

## Current True State

The only truthful state after this audit is:

```text
research artifacts exist
some research wrappers completed with returncode 0
A3 visual-hull meshes/projection diagnostics exist
B0 smoke artifacts exist
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher-supervised route = blocked
candidate export = blocked
teacher export = blocked
```

## Commands Run

```text
git status --short --branch
Get-Content -Raw reports/20260506_surface_research_preflight_status.md
Get-ChildItem -Recurse output/surface_research_preflight | Select-Object FullName,Length,LastWriteTime
Get-Content -Raw tools/check_cloud_gate_status.py
Get-ChildItem -Recurse output/surface_research_preflight -Filter research_preflight_summary.json
python tools/check_cloud_gate_status.py
python tools/check_cloud_gate_status.py --json
python tools/check_cloud_gate_status.py --teacher-supervised --json
Get-Content -Raw output/surface_research_preflight/*/research_preflight_summary.json
Select-String -Path reports/20260506_surface_research_preflight_status.md -Pattern 'success|pass|formal|blocked|completed|teacher|candidate|strict' -CaseSensitive:$false
Select-String -Path output/surface_research_preflight/**/*.json -Pattern '"pass"|strict|formal_cloud|no_teacher|no_candidate|no_strict|completed|extracted' -CaseSensitive:$false
Get-ChildItem output/surface_research_preflight -Directory | Select-Object -ExpandProperty Name
Get-Content -Raw output/surface_research_preflight/A3_visual_hull_mesh_project_t96_g56_s4/research_preflight_summary.json
Get-Content -Raw output/surface_research_preflight/B0_surface_tokens_t64_step2_gpu/B0_surface_tokens/surface_token_b0_summary.json
Get-Content -Raw output/surface_research_preflight_local/B0_surface_tokens_t96_step20/surface_token_b0_summary.json
Get-Content -Raw output/surface_research_preflight_local/B0_surface_tokens_t96_step20/visual_review_codex_fail_or_pass.json
Get-Content -Raw output/surface_research_preflight/A3_visual_hull_mesh_project_t96_g56_s4/A3_visual_hull_init/visual_hull_init_summary.json
```

Note: `rg` was attempted for file/text scanning but was denied by the Windows
environment, so PowerShell enumeration and `Select-String` were used instead.
