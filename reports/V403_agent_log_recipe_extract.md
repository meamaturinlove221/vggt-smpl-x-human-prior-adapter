# V403 Agent Log Recipe Extract

Generated: 2026-05-09 05:33 +08:00  
Workspace: `D:\vggt\vggt-main`  
Source root: `C:\Users\WINDOWS\.codex\sessions\2026\05`

This is a read-only log report. I only wrote:

- `reports\V403_agent_log_recipe_extract.json`
- `reports\V403_agent_log_recipe_extract.md`

## Executive Verdict

The logs contain a usable V50 generation recipe and provenance trail, but they do not contain the original binary NPZ/ZIP payloads.

- Bitwise V50 recovery from logs alone: **not possible**
- Recipe rerun from logs plus workspace/cloud inputs: **possible**, but it would produce a new hash lineage
- Original V50 package/hash still locally provable after the later cleanup incident: **no**
- Logs do show the original V50 pass happened before that incident: **yes**

## Primary Logs

- Main orchestration log: `C:\Users\WINDOWS\.codex\sessions\2026\05\07\rollout-2026-05-07T16-16-22-019e0182-634f-75a3-92a1-b9d19d2c8cad.jsonl`
- Initial V42/V43 subagent: `C:\Users\WINDOWS\.codex\sessions\2026\05\08\rollout-2026-05-08T21-27-33-019e07c5-a3fa-7543-9b63-5cc3dd994c1f.jsonl`
- Initial V44-V50 subagent: `C:\Users\WINDOWS\.codex\sessions\2026\05\08\rollout-2026-05-08T21-27-38-019e07c5-b88a-7113-be29-e7d0d4c40dac.jsonl`
- Later artifact searches: `C:\Users\WINDOWS\.codex\sessions\2026\05\09\rollout-2026-05-09T05-24-07-019e0979-f49f-7882-87ee-2685540fb196.jsonl`, `...\019e0979-f4f4-7a22-9e6a-ab331e1a6d70.jsonl`

## Original Chain

V42 first failed closed, then was rerouted.

Initial commands:

```powershell
python -m py_compile tools\v42_prior_enabled_predictions_rerun.py tools\v43_replay_with_prior_enabled_predictions.py
python tools\v42_prior_enabled_predictions_rerun.py
python tools\v43_replay_with_prior_enabled_predictions.py
```

That first result was hard-impossible because the chain only had scaffold/surrogate checkpoint evidence. The main log later added `modal_v42_prior_enabled_predictions.py` and ran:

```powershell
python -m py_compile modal_v42_prior_enabled_predictions.py
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; modal run modal_v42_prior_enabled_predictions.py
```

Final response says V42 passed via the `remote-hf` route: remote `facebook/VGGT-1B` was used to construct a 30-channel SMPL-X prior-enabled VGGT and emit research-only depth/points/normals/confidence/control payload. It did not write formal `predictions.npz`.

V44 then passed:

```powershell
python tools\v44_strict_visual_pre_promotion_gate.py
```

Logged V44 outputs include:

- `visual_review_codex_pass.json`
- candidate points: `798357` bytes
- candidate geometric normals: `796589` bytes

V49 passed as dry-run:

```powershell
python tools\v49_package_dry_run.py
```

It produced dry-run manifests and a dry-run registry entry, but did not write the final strict registry.

V50 final promotion then ran:

```powershell
python tools\v50_final_promotion_transaction.py
python -m py_compile tools\v50_final_promotion_transaction.py; python tools\v50_final_promotion_transaction.py; python tools\v44_v50_completion_audit.py; python tools\v37_v50_completion_audit.py
```

Final V50 logged state:

- `status = DONE_PASS`
- `strict_candidate_passes = 1`
- `strict_teacher_passes = 0`
- `formal_cloud_unblocked = true`
- `forbidden_hit_count = 0`
- `V37-V50 completion audit = COMPLETE_AUDIT_PASS`

Final V50 package paths logged:

- `output\surface_research_preflight_local\V50_final_promotion_transaction\candidate_package_v50`
- `output\surface_research_preflight_local\V50_final_promotion_transaction\candidate_package_v50\manifest.json`
- `output\surface_research_preflight_local\V50_final_promotion_transaction\strict_registry_entry_v50.json`

## Package Files Logged

The frozen package/hash stage later listed these files and sizes:

| File | Size |
|---|---:|
| `candidate_files__candidate_normals.npz` | 796589 |
| `candidate_files__candidate_points.npz` | 798357 |
| `candidate_files__hand_patch.npz` | 338682 |
| `candidate_files__head_face_patch.npz` | 256326 |
| `candidate_files__temporal_teacher.npz` | 2169505 |
| `candidate_files__visual_review.json` | 262 |
| `v42_prior_enabled_payload__control_audit.json` | 19263 |
| `v42_prior_enabled_payload__research_confidence.npz` | 85602711 |
| `v42_prior_enabled_payload__research_depths.npz` | 30775206 |
| `v42_prior_enabled_payload__research_normals_geometric.npz` | 100071633 |
| `v42_prior_enabled_payload__research_points_world.npz` | 105107787 |
| `v42_prior_enabled_payload__research_prior_effect.json` | 35089 |

One complete logged hash found in visible extraction:

- `candidate_files__candidate_points.npz`: `9f032e87125b1c204cc7cc83b9dbaf73f448ab7a99f1a4852bb0b84644bff12b`

The logs mention more hashes and hash manifests, but not every hash was visible as a full value in the extracted console snippets.

## V64 Followup

V64/V62-V120 treated V50 as frozen evidence and created/archive-checked the release chain.

Main command:

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; python tools\v62_v120_multibranch_controller.py
```

Logged V64/V62-V120 state:

- `ALL_BRANCHES_TERMINAL_V2`
- `strict_candidate_passes = 1`
- `strict_teacher_passes = 0`
- `formal_cloud_unblocked = true`
- `candidate_package_still_immutable = true`
- `forbidden_hit_count = 0`

Logged archive evidence:

- `archive\V64_candidate_pass_bundle`
- `archive\V64_candidate_pass_bundle.zip`
- zip size: `325743314` bytes
- local write time: `2026-05-09 02:32:25 +08:00`

## Later Incident

A later V223 cleanup command over-deleted. The log explicitly says source, `.git`, reports, output, archive, and V50 frozen candidate evidence were affected.

Recovery final state in the log:

- `strict_candidate_passes_current = NOT_REASSERTED_AFTER_LOCAL_ARTIFACT_LOSS`
- `strict_teacher_passes_current = 0`
- `V50 original hash locked = false`
- `candidate_pass_written_after_incident = false`
- `teacher_pass_written_after_incident = false`

Current snapshot during this V403 report also found:

- `output\frozen_candidates\V50_smplx_native_candidate_pass` exists, but its package file snapshot is empty
- `output\surface_research_preflight_local\V50_final_promotion_transaction` exists, but its candidate package file snapshot is empty
- `archive\V64_candidate_pass_bundle.zip` is missing
- `archive\package_files.zip` is missing

So the pre-incident V50 pass is documented in logs, but the original local binary package is not currently recoverable from the checked local paths.

## Binary Payload Finding

The logs contain:

- commands
- script patches
- JSON manifests/reports
- final responses
- file paths
- file sizes
- selected hashes
- base64 screenshots/images
- encrypted reasoning blobs

The logs do **not** contain complete NPZ or ZIP binary payload bytes. Therefore:

- You cannot reconstruct original `*.npz` or `*.zip` bit-for-bit from logs alone.
- You can reconstruct the recipe and rerun the chain if the workspace/cloud/model/data inputs are available.
- Any rerun must be labeled as a new rebuild/rerun with new hashes, not as the original V50 bytes.

