# Geometry Direction Status 2026-03-23 Weight Sweep Completed

## Executive status

- The overnight low-weight sweep is now complete.
- Tested candidates on top of the same geometry-side idea:
  - fixed `unproject_geometry` weight `0.2`
  - warmup `0.2`
  - fixed `0.1`
  - fixed `0.05`
- All runs used the same `4000-step`, `NumImages=4`, `A100-80GB fast` comparison regime.
- Result:
  - none of the current `unproject_geometry` training variants beat the plain baseline
  - among the tested variants, fixed `0.05` is the closest to baseline on validation objective
  - the geometry-chain direction is still supported by the first-round branch swap, but the current auxiliary geometry-loss formulation remains a weak add-on rather than a proven improvement

## New artifacts from this sweep

- New configs:
  - [zju_vggt_geom_unproject_w01_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_w01_minimal.yaml)
  - [zju_vggt_geom_unproject_w005_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_w005_minimal.yaml)
- New sweep summary script:
  - [summarize_geometry_pair_summaries.py](/f:/vggt/vggt-main/scripts/summarize_geometry_pair_summaries.py)
- Generated sweep table:
  - [geometry_candidate_sweep_4000step_summary.md](/f:/vggt/vggt-main/output/geometry_pairs_cloud/geometry_candidate_sweep_4000step_summary.md)

## Cloud runs completed

- Fixed `0.1`:
  - app: `ap-VI57JUgDdbQYgfhlVITyHd`
  - summary: [w0.1 summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_040540_zju_geom_modal_pair_4000step_a10080fast_w01_v1/summary.md)
- Fixed `0.05`:
  - app: `ap-qdW5yHsx2wO7DaIICWMwb9`
  - summary: [w0.05 summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_053650_zju_geom_modal_pair_4000step_a10080fast_w005_v1/summary.md)
- Earlier references:
  - fixed `0.2`: [summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_005034_zju_geom_modal_pair_4000step_a10080fast_v1/summary.md)
  - warmup `0.2`: [summary](/f:/vggt/vggt-main/output/geometry_pairs_cloud/20260323_022730_zju_geom_modal_pair_4000step_a10080fast_warmup_v1/summary.md)

## Main readout

Validation objective deltas against baseline:

- fixed `0.05`
  - baseline: `0.0558`
  - candidate: `0.0598`
  - delta: `+0.0040`
- fixed `0.2`
  - baseline: `0.0546`
  - candidate: `0.0589`
  - delta: `+0.0043`
- fixed `0.1`
  - baseline: `0.0548`
  - candidate: `0.0596`
  - delta: `+0.0048`
- warmup `0.2`
  - baseline: `0.0545`
  - candidate: `0.0629`
  - delta: `+0.0084`

What this says:

- lowering the weight helps more than warmup
- `0.05` is the current best candidate in this family
- but even `0.05` still does not cross baseline

## Interpretation against the mentor's direction

The mentor's main decision still stands:

1. stay on original VGGT
2. keep the geometry-first reading of the problem
3. do not return to the old `ghost` stack

What changed after the overnight sweep is simply the more precise technical conclusion:

- the branch-level geometry direction is supported
- the current auxiliary `unproject_geometry` loss is not yet a reliable training improvement
- if this line continues, the only reasonable survivor today is the smallest fixed-weight version, not warmup

## Recommended next step

The cleanest next move is:

1. stop broadening this auxiliary-loss sweep
2. keep `fixed_w0.05` as the only remaining candidate from this family
3. if one more geometry-loss experiment is needed, make it even more targeted, for example:
   - confidence-gated unprojection regression
   - mask-restricted valid region for geometry loss
   - loss only on better-covered views
4. otherwise, pause new training-loss additions and shift attention back to:
   - branch-level geometry rendering evidence
   - sparse-view failure analysis on `6src_hist`
   - camera/depth quality diagnosis

In short:

> Geometry branch swap: keep.  
> Current auxiliary geometry loss: not yet good enough.  
> From the tested loss variants, only fixed `0.05` is still worth keeping on the table.

## Cleanup status

- No active redundant Modal app remains.
- All recent Modal apps completed and are now `stopped`.
- No repo-scoped local `powershell/python/modal/cmd` process remained after final cleanup.
