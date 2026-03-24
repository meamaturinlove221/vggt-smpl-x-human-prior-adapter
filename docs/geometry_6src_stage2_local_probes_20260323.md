# 6src Stage-2 Local Probes 2026-03-23

## Goal

This note records the next local-only step after the standardized `B12`-only
hybrid conclusion.

The purpose was twofold:

- finish repairing the local runner chain so it always uses the repo-local GPU
  environment by default
- continue probing unresolved `6src` cases without opening any cloud run

## Local Runner Fix

The custom local runners previously inherited the current shell's
`sys.executable`.

That was unsafe on this machine because the active shell Python could be
`D:\anaconda\python.exe`, whose PyTorch build cannot run on the local
`RTX 5080 (sm_120)`.

The following scripts now default to the repo-local
`.venv5080\Scripts\python.exe` when `--python_exe` is not explicitly provided:

- [run_zju_custom_source_set_region_probe.py](/f:/vggt/vggt-main/scripts/run_zju_custom_source_set_region_probe.py)
- [run_zju_source_leave_one_out_region_diagnostics.py](/f:/vggt/vggt-main/scripts/run_zju_source_leave_one_out_region_diagnostics.py)
- [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py)

The leave-one-out runner was also standardized to the same default render cap as
the current compare/sweep path:

- `render_max_points = 750000`

## Local Runner Verification

The fix was verified from the current shell, without manually forcing
`.venv5080` on the command line.

Successful local GPU runs:

- [B15 stage-2 custom probe](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B15_stage2_std750k/summary.md)
- [B1 stage-2 custom probe](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B1_stage2_std750k/summary.md)
- [leave-one-out smoke](/f:/vggt/vggt-main/output/geometry_source_loo_diagnostics_smoke_20260323_b1_fixverify/summary.md)
- [view-sweep smoke](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/smoke_runner_fix_b1_v1/summary.md)

So the local execution path itself is now stable again.

## `Camera_B15` Stage-2 Probe

Output:

- [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B15_stage2_std750k/summary.md)

Reference `uniform`:

- `Camera_B11,Camera_B9,Camera_B7,Camera_B21,Camera_B4,Camera_B14`

Best bottom-improving / full-improving variant:

- `hybrid_b22_b3_keep_b4`
- source set:
  - `Camera_B11,Camera_B9,Camera_B4,Camera_B21,Camera_B22,Camera_B3`

Delta vs `uniform`:

- full-frame depth-minus-point MAE: `-0.000715`
- `fg_human` depth-minus-point MAE: `+0.008632`
- `bg_far` depth-minus-point MAE: `-0.001470`
- `bg_bottom_band` depth-minus-point MAE: `-0.001584`
- decision: still `point_map`

Foreground-preserving alternative:

- `hybrid_b22_b3_keep_b7`
- full-frame depth-minus-point MAE: `-0.001409`
- `fg_human` depth-minus-point MAE: `-0.007428`
- `bg_bottom_band` depth-minus-point MAE: `+0.001498`

Readout:

- `B15` still has source-policy structure
- but the simple local hybrids split into two bad families:
  - variants that help full-frame and bottom-band but hurt `fg_human`
  - variants that help full-frame and `fg_human` but fail on bottom-band
- so there is still no safe `B15` override to promote

## `Camera_B1` Stage-2 Probe

Output:

- [summary.md](/f:/vggt/vggt-main/output/geometry_source_swap_probe_20260323/B1_stage2_std750k/summary.md)

Reference `uniform`:

- `Camera_B13,Camera_B16,Camera_B18,Camera_B7,Camera_B5,Camera_B3`

Best bottom-improving / full-improving variant:

- `swap_b18_to_b10`
- source set:
  - `Camera_B13,Camera_B16,Camera_B10,Camera_B7,Camera_B5,Camera_B3`

Delta vs `uniform`:

- full-frame depth-minus-point MAE: `-0.000231`
- `fg_human` depth-minus-point MAE: `+0.002665`
- `bg_far` depth-minus-point MAE: `-0.000320`
- `bg_bottom_band` depth-minus-point MAE: `-0.001646`
- decision: still `point_map`

Foreground-preserving alternative:

- `hybrid_b6_b10_keep_b5`
- full-frame depth-minus-point MAE: `-0.000474`
- `fg_human` depth-minus-point MAE: `-0.005335`
- `bg_bottom_band` depth-minus-point MAE: `+0.000391`

Readout:

- `B1` also still has source-policy structure
- but the same tradeoff appears:
  - the move that helps bottom-band also hurts `fg_human`
  - the move that preserves / improves `fg_human` no longer helps bottom-band
- so there is still no safe `B1` override to promote

## Decision

After this stage-2 local pass:

- the local runner chain itself is repaired and verified
- but the geometry/source-policy problem is still **not** locally solved
- `B12` remains the only currently validated hard/control override
- `B1`, `B4`, and `B15` still block any cloud promotion

So the operational rule remains unchanged:

- keep Modal off
- keep cloud clean
- do not promote a new cloud run until the local failure set is fixed further or
  clearly bounded

## Next Local Step

The next local-only work should stay narrow:

- treat `B12` as the only validated override
- treat `B1` and `B15` as unresolved tradeoff cases, not as new leads
- continue searching for source-set moves that improve `bg_bottom_band` without
  regressing `fg_human`
- keep the search local and standardized before any cloud reconsideration
