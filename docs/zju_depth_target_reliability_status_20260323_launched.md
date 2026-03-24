# ZJU Depth Target Reliability Status 2026-03-23

## Summary

- Goal: validate whether filtering unreliable cached pseudo depth targets improves the current original-VGGT ZJU baseline without changing the loss formula or network structure.
- Mentor concern being tested: cached depth targets in ZJU may be unreliable, especially around ground/background regions.
- Current handling strategy: keep the original baseline recipe fixed and change only `zju_min_depth_conf`.

## Verified Inputs

- Region diagnostics already showed:
  - `depth_unproject` helps `fg_human`
  - degradation is concentrated in `bg_bottom_band`, and for `12src_nested` also appears in `bg_far`
- Cached target audit selected:
  - `fg_human p60 = 5.913640410988592`
  - this is the highest sampled threshold that still kept `>=100` valid pixels in every sampled train/val batch

## Launched Run

- launch date: `2026-03-23`
- modal app id: `ap-oao2n9iBQmARGeAHirTFLr`
- app state at launch check: `ephemeral (detached)`
- run type: `baseline pair`
- paired comparison:
  - baseline A: `zju_vggt_geom_minimal` with `zju_min_depth_conf=0.0`
  - candidate B: `zju_vggt_geom_minimal` with `zju_min_depth_conf=5.913640410988592`
- fixed conditions:
  - `NumImages=4`
  - `LimitTrainBatches=4000`
  - `LimitValBatches=20`
  - throughput profile: `a10080_fast`
  - GPU target: `A100-80GB`
- requested output subdir base:
  - `depth_target_pairs/20260323_153500_zju_depth_target_reliability_pair_4000step_p60_a10080fast_v1`

## Intended Next Check

- once the pair finishes:
  - compare baseline vs filtered-target training logs
  - regenerate the `6src_hist` and `12src_nested` control cases
  - rerun region diagnostics on those control outputs
  - decide whether the same target filtering should be carried into the `unproject_geometry` branch

## Process Rule

- cloud training must not be launched until the relevant local bug is confirmed fixed
- this specific Modal run was stopped after launch because the local-bug-fix gate had not been fully closed yet
- next action is local-only debugging/validation first, then a fresh cloud launch
