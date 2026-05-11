# V16 Final Execution Summary

Status: `v16_smplx_native_routed_research_complete_no_strict_pass`

This is a research-only closure for V16-DLINE-ROUTED-SMPLX. No formal prediction bundle, candidate package, teacher package, strict registry entry, or strict pass was written.

## Completed Local Gates

- V15 autopsy: `v15_negative_is_raw_softsurfel_only_not_true_vggt_training`
- Prior metric autopsy: sparse 6-view support audited; full 60-view reraster still required for final selection.
- HumanPriorAdapter probe: `v16_human_prior_adapter_nonzero_trainable`
- SMPL-X loss probe: `v16_smplx_prior_loss_nonzero_with_gradients`
- View support selector: `v16_view_support_selector_ready_sparse6_needs_60view_reraster`
- SMPL-X native ROI builder: `v16_smplx_native_region_roi_builder_ready_with_documented_fallbacks`
- Local microfit: `v16_vggt_smplx_m0_m1_cpu_smoke_negative_m2_m3_modal_route`
- V17 residual stub: `v17_smplx_residual_surface_research_stub_ready`

## Cloud Run

- Modal app run completed: `completed_research_only`
- Local download: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V16_vggt_smplx_microfit_runner`
- Remote case: `surface_research_cloud_preflight/V16_smplx_native_prior_case`
- Remote output: `surface_research_cloud_preflight/V16_vggt_smplx_microfit_runner`
- Cloud result: M0 real beat controls, but M1 adapter-only did not beat zero/shuffle; M2/M3 remain routed to research cloud/full VGGT lane.

## D-Line

- Current D-line router status: `v16_dline_failure_routed_no_strict_write`
- Passing layers: D0 SMPL-X asset, D1 prior raster, D2 V15 autopsy, D3 adapter/loss, D4 view/ROI, D6 research route, D7 no-promotion-write.
- Failing layers: D5 microfit negative/inconclusive and D8 final strict gate.
- Route: `route_weak_residual_or_v17`

## Key Paths

- `D:\vggt\vggt-main\reports\20260508_v16_dline_failure_router.md`
- `D:\vggt\vggt-main\reports\20260508_v16_vggt_smplx_microfit_runner.md`
- `D:\vggt\vggt-main\reports\20260508_v16_execution_rollup.md`
- `D:\vggt\vggt-main\reports\20260508_v17_smplx_residual_surface_optimizer.md`
- `D:\vggt\vggt-main\output\surface_research_preflight_local\V17_smplx_residual_surface_optimizer\v17_smplx_residual_surface_points.ply`
- `D:\vggt\vggt-main\output\surface_research_cloud_preflight\V16_vggt_smplx_microfit_runner\v16_modal_research_summary.json`

## Final State

`strict_candidate_passes = 0` and `strict_teacher_passes = 0`.

V16 did not satisfy the mentor strict surface gate. It did finish the requested routed SMPL-X-native proof layers and cloud research execution. The next valid branch is V17: replace hard SMPL-X supervision with weak residual / SMPL-X anchored residual surface optimization, then rerun D-line only if V17 becomes research-positive.
