# V17-V20 Execution Summary

Status: `v20_promotion_fail_closed_no_strict_write`

V17 through V20 were executed after V16 routed the failed trainable microfit to weak-residual / residual-surface follow-up. All outputs are research-only.

## V17

- Tool: `tools/v17_smplx_residual_surface_optimizer.py`
- Status: `v17_smplx_residual_surface_research_stub_ready`
- Output PLY: `D:\vggt\vggt-main\output\surface_research_preflight_local\V17_smplx_residual_surface_optimizer\v17_smplx_residual_surface_points.ply`
- Result: residual research artifact exists, but it is not a strict teacher/candidate.

## V18

- Tool: `tools/v18_residual_teacher_distillation_case.py`
- Status: `v18_residual_teacher_case_ready_research_only`
- Output target NPZ: `D:\vggt\vggt-main\output\surface_research_preflight_local\V18_residual_teacher_distillation\v18_residual_teacher_targets.npz`
- Result: bounded residual teacher distillation targets were generated for research training, not formal promotion.

## V19

- Tool: `tools/v19_temporal_canonical_residual_teacher.py`
- Status: `v19_temporal_assets_ready_predictions_missing_research_only`
- Result: frame0000/0001/0002 TMF scenes exist, but adjacent-frame VGGT predictions are missing, so a real temporal canonical teacher was not built.

## V20

- Tool: `tools/v20_final_promotion_transaction.py`
- Status: `v20_promotion_fail_closed_no_strict_write`
- Result: D-line promotion transaction ran and correctly refused to write strict registry/package/pass.

## Final Guard

- `strict_candidate_passes = 0`
- `strict_teacher_passes = 0`
- `formal_cloud_unblocked = false`
- no candidate package
- no teacher package
- no strict registry write

## Current Blockers

- V16 trainable microfit did not beat zero/shuffle controls.
- V19 adjacent-frame VGGT predictions are missing.
- V17/V18 are research artifacts and lack strict visual/region/6-view audits.
