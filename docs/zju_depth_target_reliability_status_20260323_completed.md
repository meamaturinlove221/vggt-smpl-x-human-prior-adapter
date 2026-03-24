# ZJU Depth Target Reliability Status 2026-03-23 Completed

## Summary

- Goal: validate whether filtering unreliable cached pseudo depth targets improves the current original-VGGT ZJU baseline without changing the loss formula or network structure.
- Diagnosis stayed valid:
  - earlier region diagnostics still support the mentor concern that degradation is concentrated in `bg_bottom_band`, and for `12src_nested` also in `bg_far`
  - cached target audit still selected `fg_human p60 = 5.913640410988592` as the highest threshold that preserved `>=100` valid pixels in sampled train/val batches
- Treatment did **not** validate:
  - the paired cloud run showed that `baseline + zju_min_depth_conf=p60` did not beat the raw-target baseline
  - by the project rule, this means the thresholded target filtering should **not** be carried into `unproject_geometry` yet

## Completed Pair Run

- Modal app id: `ap-6noqdXamIzu50ctQqF5Lyz`
- final app state: `stopped`
- stop time: `2026-03-23 17:53:55+08:00`
- remote output root:
  - `depth_target_pairs/20260323_170500_zju_depth_target_reliability_pair_4000step_p60_v2`
- fixed conditions:
  - baseline A: `zju_vggt_geom_minimal`, `zju_min_depth_conf=0.0`
  - candidate B: `zju_vggt_geom_minimal`, `zju_min_depth_conf=5.913640410988592`
  - `NumImages=4`
  - `LimitTrainBatches=4000`
  - `LimitValBatches=20`
  - throughput profile: `a10080_fast`
  - GPU target: `A100-80GB`

## Result

- pulled comparison summary:
  - [summary.md](/f:/vggt/vggt-main/output/zju_depth_target_reliability_pair_cloud/20260323_170500_zju_depth_target_reliability_pair_4000step_p60_v2/summary.md)
  - [summary.json](/f:/vggt/vggt-main/output/zju_depth_target_reliability_pair_cloud/20260323_170500_zju_depth_target_reliability_pair_4000step_p60_v2/summary.json)
- validation-level decision:
  - baseline `loss_objective`: `0.0559`
  - candidate `loss_objective`: `0.0577`
  - delta: `+0.0018` for the candidate, so the filtered-target run is slightly worse on the main validation objective
- side metrics:
  - candidate reduced `loss_grad_depth`
  - candidate drove `loss_conf_depth` strongly downward
  - but candidate clearly worsened `loss_camera`, `loss_T`, and `loss_reg_depth`
- practical decision:
  - stop at diagnosis
  - do **not** propagate this scalar `zju_min_depth_conf=p60` filtering into the next `unproject_geometry` run

## Monitoring Fix Closure

- the earlier observability bug is now closed locally and validated:
  - [modal_zju_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_zju_geometry_minimal_finetune.py) now mirrors subprocess output into `driver_live.log` and commits the output volume periodically during training
  - [monitor_modal_depth_target_pair.py](/f:/vggt/vggt-main/scripts/monitor_modal_depth_target_pair.py) now prefers `driver_live.log` instead of relying only on `logs/log.txt`
- this was validated locally before relaunch, then confirmed remotely during the completed pair:
  - baseline progressed visibly from `1059 -> 3999`
  - candidate progressed visibly from `1063 -> 3999`

## Throughput Follow-Up

- current completed pair used the conservative fast profile and still showed:
  - GPU memory around `31 GB`
  - effective throughput consistent with under-filled `A100-80GB`
- local throughput improvements have now been implemented and validated for future cloud runs:
  - [modal_zju_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_zju_geometry_minimal_finetune.py)
    - automatically enables `data.train.pin_memory=True`
    - automatically enables `data.val.pin_memory=True`
    - automatically enables `data.train.persistent_workers=True` and `data.val.persistent_workers=True` when `num_workers > 0`
    - automatically enables `cuda.cudnn_benchmark=True`
  - [run_modal_zju_depth_target_reliability_pair.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_depth_target_reliability_pair.ps1)
    - now includes `a10080_high_util`
    - this profile sets:
      - `ModalCpu = 16`
      - `ModalMemoryMb = 147456`
      - `MaxImgPerGpu = 24`
      - `NumWorkers = 16`
- these throughput changes were locally validated by:
  - Python syntax check
  - dry-run command assembly
  - direct `_build_overrides(...)` verification
- they were **not** applied retroactively to the completed pair; they are reserved for the next cloud run after the next local gate passes

## Next Step

- keep the region-diagnostic conclusion, but drop this specific scalar threshold treatment
- next investigation should move back to one of:
  - source-view policy sensitivity
  - camera prediction / render gap versus legacy
  - a narrower reliable-region treatment that is more targeted than a single global `min_depth_conf`
