# Geometry Direction Status 2026-03-23 Threshold And Pow2 Completed

## Executive status

- The follow-up gate-family check is complete.
- Two additional variants were tested on Modal after `confgate_w0.05`:
  - `confgate_t50_w0.05`
  - `confgate_pow2_w0.05`
- Neither variant beat baseline.
- Neither variant beat the earlier `confgate_w0.05` candidate.

## What was tested

- Hard threshold gate:
  - [zju_vggt_geom_unproject_confgate_t50_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_confgate_t50_w005_minimal.yaml)
  - keeps only higher-confidence pixels using a `p50` threshold
- Power-weighted gate:
  - [zju_vggt_geom_unproject_confgate_pow2_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_confgate_pow2_w005_minimal.yaml)
  - keeps all pixels but squares detached depth confidence weights
- Threshold diagnostics used to choose the hard-gate attempt:
  - [analyze_zju_unproject_gate_thresholds.py](/f:/vggt/vggt-main/scripts/analyze_zju_unproject_gate_thresholds.py)
  - [unproject_gate_thresholds.md](/f:/vggt/vggt-main/output/depth_conf_analysis/20260323_gate_thresholds_v1/unproject_gate_thresholds.md)

## Main results

- Best existing candidate remains:
  - [confgate_w0.05 summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_065320_zju_geom_modal_pair_4000step_a10080fast_confgate_w005_v1/summary.md)
  - val objective delta: `+0.0014`
- Hard threshold result:
  - [confgate_t50_w0.05 summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_084159_zju_geom_modal_pair_4000step_a10080fast_confgate_t50_w005_v1/summary.md)
  - val objective delta: `+0.0032`
  - val `loss_unproject_geometry`: `0.0000`
- Power-weighted result:
  - [confgate_pow2_w0.05 summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_095523_zju_geom_modal_pair_4000step_a10080fast_confgate_pow2_w005_v1/summary.md)
  - val objective delta: `+0.0040`
  - val `loss_unproject_geometry`: `0.0269`

## Interpretation

- `confgate_t50_w0.05` is not viable for this setup.
  - The hard threshold makes validation geometry loss collapse to zero.
  - That means the auxiliary signal is effectively missing on val.
- `confgate_pow2_w0.05` is healthier than thresholding.
  - Geometry loss stays active.
  - But overall objective still moves further away from baseline.
- The gate-family conclusion is now stable:
  1. `confgate_w0.05` is the only surviving auxiliary candidate in this line
  2. hard thresholding should be dropped
  3. stronger confidence weighting via `pow2` should also be deprioritized

## Operational status

- Current Modal app state has been re-checked:
  - existing apps are `stopped`
  - no active detached training app is left running
- Repo-scoped local launcher residue has also been re-checked:
  - no lingering training or log-stream process remains

## Recommendation

- Keep the mentor-aligned main line unchanged:
  - original VGGT
  - geometry-chain first
  - no return to the old `ghost` stack
- If another experiment is needed, it should probably move outside this gate-tuning family.
- If we need a single auxiliary geometry-loss candidate to keep on the shelf, keep only `confgate_w0.05`.
