# 2026-05-07 D-Line Post-B16 Guard Audit

Status: `strict_guard_red_formal_cloud_blocked`

Scope: D-line guard/read/report only. No experiment was run, no Modal job was
launched, no registry/result artifact was edited, and no pass state was written.

## Strict Registry

Registry checked:

```text
reports/20260504_strict_gate_registry.json
```

Current registry truth:

```text
schema_version = 20260504_visual_fullbody_hands_v2
generated_at = 2026-05-07T03:58:35.809940+00:00
candidates = 26
teachers = 81
strict_candidate_passes = 0
strict_teacher_passes = 0
strict_candidate_passes list count = 0
strict_teacher_passes list count = 0
```

The registry is still red. The positive diagnostic counters in the registry are
not strict candidate or strict teacher passes:

```text
kinect_coord_passes = 2
smplx_weak_anchor_passes = 1
full_gate_numeric_pass_visual_fail = 1
other_teacher_fail = 81
visible_surface_teacher_passes = 0
```

These do not authorize formal cloud train, infer, export, candidate export, or
teacher export.

## Cloud Guard

`python tools/check_cloud_gate_status.py --json` returned:

```text
cloud_allowed = false
reasons = strict_candidate_passes is 0
strict_candidate_passes = 0
strict_teacher_passes = 0
registry_age_hours = 2.578
```

`python tools/check_cloud_gate_status.py --teacher-supervised --json` returned:

```text
cloud_allowed = false
reasons = strict_candidate_passes is 0;
          teacher-supervised route requested but strict_teacher_passes is 0
strict_candidate_passes = 0
strict_teacher_passes = 0
```

The formal cloud guard remains blocked. The train/infer Modal entrypoints still
raise on missing, stale, wrong-schema, or red strict registry state, and the
teacher-supervised train route additionally requires `strict_teacher_passes > 0`.

Current formal state:

```text
formal cloud train = blocked
formal cloud infer = blocked
formal cloud export = blocked
teacher-supervised route = blocked
teacher export = blocked
candidate export = blocked
```

## Modal Research-Preflight Discipline

`modal_surface_research_preflight.py` still reads the strict registry and records
research-only launch metadata. It rejects research asset/output paths containing:

```text
strict_pass
teacher_export
candidate_export
```

The remote launch metadata still includes:

```text
research_only = true
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
formal_cloud_train_infer_export = blocked unless local strict gate passes
```

This wrapper remains a research/preflight path only. Wrapper completion,
diagnostic metrics, COLMAP fused output, token/surface smoke output, or local
audit output must not be interpreted as a mentor pass, a strict teacher pass, a
strict candidate pass, or a formal cloud unblock signal.

## New Research Script Risk-Word Scan

`rg` was attempted first, but Windows returned access denied for `rg.exe`, so
PowerShell `git ls-files`, `Get-Content`, and `Select-String` were used instead.

Untracked `tools/*.py` scripts currently found:

```text
untracked_tools_py = 103
B-line/research-like subset counted by filename pattern = 30
```

Broad risk-word scan:

```text
pattern includes pass/export/cloud/unblock/strict_pass/teacher_export/candidate_export
matches = 266
files_with_matches = 42
```

Interpretation of broad hits:

```text
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
teacher_export = blocked
candidate_export = blocked
formal cloud unblock signal = blocked
cannot unblock formal cloud
not a mentor pass
pass --overwrite
```

These are protective, negative, reporting, or ordinary CLI wording hits. They
are not pass/export/cloud-unblock actions.

Narrow dangerous-state scan looked for truthy or allowing forms such as:

```text
no_teacher_export = false
no_candidate_export = false
no_strict_pass_write = false
teacher_export = allowed
candidate_export = allowed
strict_pass = true
writes_strict_pass = true
formal_cloud_allowed = true
cloud_allowed = true
CLOUD ALLOWED
unblock formal cloud
formal cloud unblock
```

Narrow scan findings:

```text
strict_danger_scan_matches = 3
```

All three are non-actionable in this audit:

```text
tools/audit_b2_surface_token_support.py:621
  "formal cloud unblock signal = blocked"

tools/b_fus3d_raw_image_normal_linesearch_probe.py:230
  "- This probe cannot unblock formal cloud or strict gates."

tools/check_cloud_gate_status.py:146
  print("CLOUD ALLOWED: local strict candidate gate is green.")
```

The first two are explicitly blocked/cannot-unblock statements. The third is
inside the formal guard checker and only prints on a green strict candidate gate;
the current registry is red, and the actual command output is `cloud_allowed =
false`.

No new research script scan result was found that should be treated as writing a
strict pass, exporting teacher/candidate artifacts, or unblocking formal cloud.

## Commands Run

```text
git status --short
rg -n "strict_candidate_passes|strict_teacher_passes|formal cloud|cloud blocked|research-preflight|preflight|cloud guard|strict registry|registry" -S .
rg --files
Get-Content -Raw -LiteralPath reports/20260504_strict_gate_registry.json
Get-Content -Raw -LiteralPath tools/check_cloud_gate_status.py
Get-Content -Raw -LiteralPath modal_surface_research_preflight.py
Get-Content -Raw -LiteralPath reports/20260507_dline_guard_status.md
Get-Content -Raw -LiteralPath reports/20260507_modal_research_preflight_runtime_audit.md
python tools/check_cloud_gate_status.py --json
python tools/check_cloud_gate_status.py --teacher-supervised --json
Select-String -Path modal_4k4d_vggt_train.py,modal_4k4d_vggt_infer.py,modal_surface_research_preflight.py,tools/check_cloud_gate_status.py -Pattern "strict_candidate_passes|strict_teacher_passes|cloud_allowed|formal_cloud_allowed|CLOUD BLOCKED|formal cloud|teacher-supervised|strict gate|teacher_export|candidate_export|strict_pass" -CaseSensitive:$false
Get-Content -Raw reports/20260504_strict_gate_registry.json | ConvertFrom-Json
git ls-files --others --exclude-standard tools
git ls-files --modified tools modal_surface_research_preflight.py modal_4k4d_vggt_train.py modal_4k4d_vggt_infer.py reports/20260504_strict_gate_registry.json
Select-String over untracked tools/*.py for broad pass/export/cloud/unblock risk words
Select-String over untracked tools/*.py for narrow dangerous truthy/allowed risk forms
Select-String -Path modal_4k4d_vggt_train.py,modal_4k4d_vggt_infer.py -Pattern "def _assert_cloud|candidate_passes|teacher_passes|raise RuntimeError|Cloud .*blocked|strict gate registry is stale|strict gate registry schema" -Context 2,3
git diff -- reports/20260504_strict_gate_registry.json | Select-Object -First 80
Get-Date -Format o
```

Note: the two `rg` commands failed with Windows `Access is denied`; they did not
produce scan results and were replaced by PowerShell scans.

## Bottom Line

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher/candidate export = blocked
research-preflight remains research-only
new research-script risk-word scan found no actionable pass/export/cloud-unblock risk
```
