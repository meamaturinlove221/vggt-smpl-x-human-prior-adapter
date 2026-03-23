# Geometry Direction Status 2026-03-23 Pair4000 Completed

## Executive status

- The mentor-aligned first-round question is already answered:
  - confirm the current render path
  - compare `point_map / world_points` against `depth + camera -> unprojected points`
  - do it first without restoring the old `ghost` stack
- The first-round branch comparison already supported staying on the geometry direction overall:
  - `depth_unproject` wins `2 / 3`
  - `point_map` wins `1 / 3`
- The second-round overnight cloud pair is now also complete:
  - baseline: `zju_vggt_geom_minimal`
  - candidate: `zju_vggt_geom_unproject_minimal`
  - same `NumImages=4`
  - same paired schedule
  - `4000` train batches on Modal `A100-80GB`
- Result:
  - the geometry direction is still the right main line
  - but the current minimal `unproject_geometry` training term still does **not** beat the plain baseline on `val_loss_objective`
  - therefore the next step is **not** “merge the new geometry loss as-is”

## What is now firmly established

The mentor repeatedly emphasized two concrete questions:

1. confirm which point-source path the current render is actually using
2. once that is confirmed, try the alternate path:
   - `camera + depth -> unprojected points -> render`

That part is no longer ambiguous.

- The repo-side rendering path under discussion is the `point / world_points` path.
- The alternate `depth + camera -> unprojected points` branch has now been tested in a clean first-round A/B comparison.
- That first-round branch compare is documented here:
  - [first-round cloud compare](/f:/vggt/vggt-main/docs/geometry_direction_status_20260322_cloud_compare_completed.md)
  - [first-round batch summary](/f:/vggt/vggt-main/output/geometry_compare_cloud/20260322_234949_zju_geometry_branch_compare_batch_v2_payloadfix/batch_summary.md)

First-round readout:

- `depth_unproject` wins `2 / 3`
- advantage becomes clearer as view count increases
- sparse-view `6src_hist` remains the hardest borderline case

This is enough to keep the main direction on the geometry chain and to avoid reviving the old image-side `ghost` pipeline.

## Second-round overnight cloud pair

- Modal app:
  - `ap-LBcsLcwEvarLScdVgiqNK0`
- Description:
  - `vggt-zju-geometry-minimal-finetune`
- Final state:
  - `stopped` after successful completion
- Remote pair root:
  - `/geometry_pairs/20260323_005034_zju_geom_modal_pair_4000step_a10080fast_v1`
- Local outputs:
  - [4000-step summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_005034_zju_geom_modal_pair_4000step_a10080fast_v1/summary.md)
  - [4000-step summary json](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_005034_zju_geom_modal_pair_4000step_a10080fast_v1/summary.json)
  - [500-step summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260322_201558_zju_geom_modal_pair_500step_remote_v1/summary.md)

### 500-step vs 4000-step comparison

Validation objective:

- 500-step
  - baseline: `0.0995`
  - unproject_geometry: `0.1117`
  - delta: `+0.0122`
- 4000-step
  - baseline: `0.0546`
  - unproject_geometry: `0.0589`
  - delta: `+0.0043`

Train objective:

- 500-step
  - baseline: `0.2365`
  - unproject_geometry: `0.2533`
  - delta: `+0.0168`
- 4000-step
  - baseline: `0.1260`
  - unproject_geometry: `0.1348`
  - delta: `+0.0088`

Geometry-related side terms at 4000-step:

- `unproject_geometry` slightly improves some depth-side averages:
  - `val loss_conf_depth`: `0.0068` vs baseline `0.0071`
  - `val loss_reg_depth`: `0.0068` vs baseline `0.0071`
  - `val loss_T`: `0.0067` vs baseline `0.0068`
- but it is still worse on overall validation objective:
  - `0.0589` vs baseline `0.0546`

### Interpretation

This means:

- the longer run did **not** reverse the 500-step conclusion
- the gap shrank a lot, so the idea is not obviously broken
- but the current minimal `unproject_geometry` loss is still not strong enough to justify replacing the baseline training recipe

So the honest conclusion is:

- `depth + camera -> unprojected points` is still the better **geometry branch direction**
- but the current second-round implementation of extra geometry supervision is only a partial improvement, not yet a winning training config

## Recommended next step

The next step should stay disciplined:

1. Keep the repo on the original VGGT line, not the old `ghost` branch.
2. Treat the first-round point-source swap result as the main confirmed evidence.
3. Do **not** promote `zju_vggt_geom_unproject_minimal` to the new default fine-tune config yet.
4. Keep the current baseline training recipe as the control.
5. If continuing, only test one lighter geometry-side refinement at a time, for example:
   - delayed / warmup-enabled `unproject_geometry`
   - smaller geometry-loss weight
   - masked or confidence-gated reconstruction on the unprojected render path
6. Do **not** restore `ghost score`, bbox scoring, or mixed image-side weight stacks.
7. Keep `6src_hist` as the sparse-view regression case.

In short:

> The mentor's geometry-first direction is supported.  
> The point-source swap baseline is worth keeping.  
> The current extra unprojection loss is not yet good enough to become the new training mainline.

## Stability and cleanup status

- No active redundant Modal app remains.
- The overnight app completed and stopped normally.
- The local crash-prone `modal app logs` monitor chains were manually cleaned up before the overnight wait.
- Final local memory check after cleanup stayed around:
  - free memory: about `15.7 GB`
  - used memory: about `15.5 GB`
- No repo-scoped lingering `powershell/python/modal/cmd` monitor processes remained at the end of this run.
