# V42/V44-V50 Dependency Audit

Status: `DONE_PATCHED_AND_AUDITED`

This audit fixed the dependency wiring after V39 without writing formal outputs.

## Findings

- V42 was looking for `V39_adapter_only_microfit`; the actual directory is `output/surface_research_preflight_local/V39_adapter_microfit`.
- V44/V49/V50 were still checking the old `20260508_v30_prior_enabled_vggt_predictions.json` readiness path through the shared common helper.
- The V39 compact adapter payload is now accepted as a valid rerun dependency when paired with the V38 prior-enabled scaffold.
- V42 still cannot pass because the actual research prediction payload files have not been generated or downloaded.

## Patched

- `tools/v42_prior_enabled_predictions_rerun.py`: searches `V39_adapter_microfit`, supports V41b-style directories, and validates compact V39/V41b payloads by report status, route, checkpoint files, V38 scaffold, prior channels, control wins, and region wins.
- `tools/v44_v50_common.py`: added `v42_prior_prediction_ready()` and points readiness at `reports/20260509_v42_prior_enabled_predictions_rerun.json` plus the V42 research files.
- `tools/v44_strict_visual_pre_promotion_gate.py`, `tools/v49_package_dry_run.py`, `tools/v50_final_promotion_transaction.py`: refreshed labels/imports so downstream blockers name V42, not stale V30.

## Current State

- V42: `DONE_FAIL_ROUTED`
- V42 dependency gate: passed from `V39_adapter_microfit`
- V42 missing files: `research_depths.npz`, `research_points_world.npz`, `research_confidence.npz`, `research_normals_geometric.npz`, `research_prior_effect.json`, `control_real_zero_shuffle_random_dropout.json`
- V44: `DONE_FAIL_ROUTED` on missing V42 payload
- V49: `DONE_FAIL_ROUTED` on V44 plus missing V42 payload
- V50: `DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE`, no package/registry/pass written
- V44-V50 completion audit: `COMPLETE_AUDIT_PASS`

## Verification

- `python -m py_compile tools\v42_prior_enabled_predictions_rerun.py tools\v44_v50_common.py tools\v44_strict_visual_pre_promotion_gate.py tools\v49_package_dry_run.py tools\v50_final_promotion_transaction.py`
- `python tools\v42_prior_enabled_predictions_rerun.py`
- `python tools\v44_strict_visual_pre_promotion_gate.py`
- `python tools\v49_package_dry_run.py`
- `python tools\v50_final_promotion_transaction.py`
- `python tools\v44_v50_completion_audit.py`

No `predictions.npz`, candidate package, teacher package, strict registry, or strict pass was written by this audit.
