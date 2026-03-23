# Geometry-First Status 2026-03-21

This note captures the current state after switching back to the original VGGT codebase and validating the `depth + camera` geometry direction first.

## What Is Already Done

- local `RTX 5080 16GB` environment fixed with a CUDA build that supports the card
- original VGGT inference confirmed to run in `.venv5080`
- first-round geometry baseline implemented and executed
- geometry-first primary ZJU output path implemented and verified
- minimal local fine-tune wrapper implemented and smoke-tested on local Windows 5080
- first minimal extra-supervision variant implemented and locally probed
- 20-step and 100-step paired local comparisons completed for baseline vs `unproject_geometry`
- first large ZJU target-view sweep completed overnight
- minimal Modal fine-tune scaffold prepared
- ZJU-specific Modal fine-tune scaffold prepared and dry-run validated locally

## First-Round Baseline Result

Primary artifacts:

- [kitchen summary](/f:/vggt/vggt-main/output/geometry_baseline/kitchen8/summary.md)
- [batch summary](/f:/vggt/vggt-main/output/geometry_baseline_batch/examples8/batch_summary.md)

High-level read:

- `kitchen`: `depth + camera` is clearly better than `point map`
- `llff_fern`: lower MAE on `depth + camera`, but lower coverage
- `llff_flower`: lower MAE on `depth + camera`, but lower coverage
- `room`: `point map` has lower MAE, while `depth + camera` has higher coverage

So the current conclusion is:

- the geometry chain is supported strongly enough to continue
- it is not yet an unconditional winner on every scene
- we should keep the next step minimal and geometry-first instead of restoring the old ghost stack

## Human-Domain Follow-up

I also added a ZJU/CoreView-specific baseline that is closer to the real project setup:

- it uses only source views as VGGT input
- it re-renders into the real target camera from the ZJU calibration
- it compares `point map` vs `depth + camera` on the human-domain case directly

Artifacts:

- [coreview390_batch_summary.md](/f:/vggt/vggt-main/output/geometry_baseline_zju/coreview390_batch_summary.md)
- [geometry_zju_baseline.md](/f:/vggt/vggt-main/docs/geometry_zju_baseline.md)
- [primary_summary.md](/f:/vggt/vggt-main/output/geometry_primary_zju/coreview390_6src_hist_primary/primary_summary.md)

Current readout on `CoreView_390 / frame 1080 / Camera_B5`:

- `6src_hist`: `depth + camera` wins
- `12src_nested`: tie, but `depth + camera` still has lower MAE
- `23cam_fullset`: `depth + camera` wins

This is the strongest local evidence so far for keeping the next step on the geometry chain.

## Large View-Level Sweep

I then expanded the ZJU check from one target view into the first larger target-view sweep:

- [geometry_zju_view_sweep_round1_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_view_sweep_round1_20260322.md)
- [round1 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.md)

Scope:

- `459` cases
- `9` frames
- multiple target cameras
- three source-view profiles
- `0` failures

High-level read:

- overall: `199` depth wins / `163` point wins / `97` ties
- `23cam_fullset`: strong support for `depth + camera`
- `6src_hist`: mixed, with `point_map` often stronger
- `12src_nested`: mixed, with `point_map` often stronger

This adds an important nuance to the earlier single-case baseline:

- the geometry chain is strongly supported in dense-view settings
- but sparse fixed source subsets do not yet generalize cleanly across target views

So the correct status is now:

- keep the geometry-chain direction
- do not restore the old ghost stack
- but do not claim a universal sparse-view win yet

## Round-2 Target-Aware Sparse Sweep

I then pushed the sparse-view question one step further by changing only the source-selection policy:

- keep the same original VGGT checkpoint
- keep the same render-path comparison
- keep the same evaluation rule
- replace the old fixed sparse source subset with a target-aware rotated version of the same template pattern

Artifacts:

- [geometry_zju_view_sweep_round2_targetaware_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_view_sweep_round2_targetaware_20260322.md)
- [round2 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.md)
- [round1 vs round2 comparison](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_vs_round2_targetaware_v1/comparison.md)
- [geometry_zju_sparse_policy_compare_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_sparse_policy_compare_20260322.md)

Scope:

- `414` sparse target-aware cases
- `9` frames
- `6src_hist`
- `12src_nested`

High-level read:

- `6src_hist` improved materially under target-aware source selection
- common-case depth wins increased from `28 -> 46`
- common-case average geometry gain delta is `+0.000896`
- common-case average coverage gain delta is `+0.024030`
- `12src_nested` improved only slightly in depth-win count, but remained negative overall

So the updated sparse-view status is now:

- fixed sparse subsets were indeed part of the problem
- this explanation is strong for `6src_hist`
- this explanation is not sufficient for `12src_nested`
- dense/full-rig evidence remains the strongest support for `depth + camera`

## Round-3 `12src` Policy Probe

I then isolated the remaining weak profile and tested two new `12src` source-selection strategies:

- `nearest_ring`
- `uniform_ring`

Artifacts:

- [geometry_zju_12src_policy_probe_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_12src_policy_probe_20260322.md)
- [round3 nearest summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round3_12src_nearest_v1/summary.md)
- [round3 uniform summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round3_12src_uniform_v1/summary.md)
- [rotate vs uniform](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_rotate_vs_round3_uniform_12src_v1/comparison.md)

High-level read:

- `nearest_ring` improved `12src`, but only modestly
- `uniform_ring` improved it much more strongly
- round-2 rotated template depth wins: `14`
- round-3 nearest depth wins: `23`
- round-3 uniform depth wins: `48`

So the updated `12src` status is now:

- `12src` is more coverage-sensitive than locality-sensitive
- a uniform ring policy is the best current `12src` baseline
- but `12src` still remains less geometry-friendly than dense/full-rig and less clean than the best `6src` target-aware result

## Cross-Round Diagnostics And Training Gate

After the three inference rounds were in place, I aggregated them into one diagnostics pass:

- [geometry_zju_diagnostics_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_diagnostics_20260322.md)
- [diagnostics summary](/f:/vggt/vggt-main/output/geometry_diagnostics_zju/round1_round2_round3_v1/summary.md)

This pass answers the next three questions directly:

- whether bad sparse cases are mainly explained by worse post-alignment
- whether `point_map` wins mostly when the two point-source branches diverge more
- whether the ring-coverage statistics separate the source policies clearly enough to choose the next baseline

The current read is:

- `23cam_fullset` is the cleanest geometry-chain confirmation
  - depth-win rate among decisive cases: `0.874`
  - avg geometry gain: `+0.001915`
  - avg coverage gain: `+0.021472`
- `6src_hist + rotate_template_offsets` is the best current sparse baseline if a sparse regime is required now
  - depth-win rate among decisive cases: `0.428`
  - avg geometry gain: `-0.000764`
  - avg coverage gain: `+0.021843`
- `12src_uniform` is the best current `12src` policy, but still not yet a primary long-training sparse baseline
  - depth-win rate among decisive cases: `0.310`
  - avg geometry gain: `-0.000248`
  - avg coverage gain: `-0.021101`

The main diagnostic conclusion is:

- dense/full-rig behavior is strongly consistent with the geometry-chain hypothesis
- sparse failures are not explained only by post-alignment error
- source-policy geometry matters a lot, especially ring coverage and gap uniformity
- the right next move is still geometry/source-policy work, not reviving the old ghost stack

## Local Smoke Fine-Tune Result

To avoid being blocked by the missing CO3D path, I also validated a local pseudo-geometry fine-tune chain on ZJU:

- dataset root: `G:\数据集\datasets\ZJU_MoCap\data\zju_mocap`
- sequence: `CoreView_390`
- pseudo-geometry cache: `vggt_geom`
- checkpoint: `G:\项目备份\vggt_小感度不起作用\vggt\model.pt`
- model heads enabled: `camera=True`, `depth=True`, `point=False`, `track=False`

What is now confirmed:

- the wrapper can resolve the default local paths correctly
- the minimal config can instantiate train and val datasets
- local single-GPU smoke can run on Windows without DDP
- one train batch and one val batch complete successfully
- checkpoints and TensorBoard events are written correctly

Artifacts:

- [zju_vggt_geom_smoke_20260321.md](/f:/vggt/vggt-main/docs/zju_vggt_geom_smoke_20260321.md)
- [log.txt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/log.txt)
- [checkpoint.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/ckpts/checkpoint.pt)
- [checkpoint_0.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_smoke_local_v3/ckpts/checkpoint_0.pt)

Representative smoke metrics from `zju_vggt_geom_smoke_local_v3`:

- train objective: `4.3503`
- train camera loss: `0.0515`
- train depth confidence loss: `3.5250`
- val objective: `3.4253`
- val camera loss: `0.0561`
- val depth confidence loss: `2.5854`

This means the local 5080 route is now good enough for:

- wrapper verification
- dataset and checkpoint sanity checks
- minimal smoke fine-tune runs
- quick geometry-only ablations before moving heavier jobs to Modal

## First Extra-Supervision Candidate

After the geometry-first baseline was confirmed, I added exactly one new auxiliary term:

- `loss_unproject_geometry`
- built from predicted `depth + camera`
- differentiably unprojected into world points
- regressed against GT `world_points`

This keeps the training target aligned with the mentor's direction:

- no point head re-enabled
- no image-side ghost stack restored
- no network-structure change

Artifacts:

- [zju_vggt_unproject_geometry_probe_20260321.md](/f:/vggt/vggt-main/docs/zju_vggt_unproject_geometry_probe_20260321.md)
- [unproject smoke log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_smoke_local_v1/log.txt)
- [baseline probe log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_baseline_probe_local_v1/log.txt)
- [unproject probe log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_probe_local_v1/log.txt)

Short read:

- the new term is numerically stable on local Windows 5080
- 1-train + 1-val smoke succeeded
- 5-train + 2-val probe also succeeded
- camera and depth sub-losses stayed essentially unchanged versus the same-length baseline probe
- the added term behaves like a clean auxiliary geometry constraint, not a destabilizing rewrite

Current best minimal extension beyond baseline is therefore:

- original VGGT
- `camera + depth` heads only
- optional `unproject_geometry` auxiliary loss

## Longer Paired Local Result

I also pushed the first extra-supervision candidate beyond the 5-step probe and ran longer paired local schedules:

- [20-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_20step_v1/summary.md)
- [20-step pair json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_20step_v1/summary.json)
- [100-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_100step_v1/summary.md)
- [100-step pair json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_100step_v1/summary.json)
- [500-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.md)
- [500-step pair json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.json)

20-step read:

- train camera stayed effectively unchanged: `0.0528 -> 0.0530`
- train depth confidence and regression stayed slightly better numerically on the unproject run: `0.4408 -> 0.4400`, `0.1873 -> 0.1865`
- val camera stayed effectively unchanged: `0.0592 -> 0.0594`
- val depth losses remained very close: `0.0511 -> 0.0526`
- the extra term stayed finite and decreased during training: train average `0.2828`, val average `0.2031`

100-step read:

- train camera stayed effectively unchanged: `0.0406 -> 0.0404`
- train `T` and `R` stayed effectively unchanged: `0.0324 -> 0.0323`, `0.0028 -> 0.0027`
- train depth confidence and regression matched exactly at the reported average: `0.1597 -> 0.1597`, `0.0867 -> 0.0867`
- train depth gradient improved slightly: `0.0125 -> 0.0107`
- val camera stayed effectively unchanged: `0.0304 -> 0.0308`
- val depth confidence and regression improved slightly on the unproject run: `0.0294 -> 0.0277`
- the extra term stayed finite and lower than the 20-step run: train average `0.1690`, val average `0.1010`

So the longer paired results support the same conclusion as the 5-step probe:

- `unproject_geometry` is still behaving like a clean auxiliary geometry term
- it is not obviously destabilizing the original `camera + depth` training path
- the 100-step pair is now the strongest local evidence for keeping it as the first extra term

500-step read:

- train camera is still effectively unchanged and slightly better on the unproject run: `0.0253 -> 0.0247`
- train `T` is also slightly better on the unproject run: `0.0200 -> 0.0194`
- train depth confidence and regression remain matched within `0.0002`: `0.0783 -> 0.0785`, `0.0484 -> 0.0486`
- val camera remains effectively unchanged: `0.0123 -> 0.0122`
- val depth confidence and regression are slightly worse but still extremely close: `0.0168 -> 0.0174`
- val depth gradient is slightly better on the unproject run: `0.0046 -> 0.0044`
- the extra geometry term keeps decreasing further: train average `0.0968`, val average `0.0421`

So the current strongest local training read is now:

- `unproject_geometry` remains numerically stable even on the longer 500-step paired run
- it still does not derail the original `camera + depth` path
- the 500-step pair is now the strongest local evidence for keeping it as the first extra term

## Checkpoint-After-Training Render Evaluation

I then checked the paired 500-step checkpoints on the render side instead of stopping at training losses:

- [checkpoint eval note](/f:/vggt/vggt-main/docs/geometry_zju_checkpoint_eval_20260322.md)
- [small compare](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_compare_v1/comparison.md)
- [full-target compare](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_fulltargets_compare_v1/comparison.md)

Small follow-up probe, `48` common cases:

- both checkpoints already prefer `depth + camera` on all `48/48` cases
- the unproject checkpoint still improves the margin
- average geometry gain delta: `+0.000715`
- average coverage gain delta: `+0.038546`
- `41/48` cases improved in both geometry and coverage
- `7/48` cases regressed in both, concentrated around `Camera_B8`

Full-target follow-up, `184` common cases:

- baseline decisions: depth `168`, point `0`, tie `16`
- unproject decisions: depth `169`, point `0`, tie `15`
- one case improved from `tie -> depth`
- zero cases regressed in decision category
- average geometry gain delta: `+0.000632`
- average coverage gain delta: `+0.035864`

This is the current strongest combined read:

- the 500-step unproject run is not only stable in training
- it also produces small but positive average render-side deltas afterward
- the `Camera_B8` hotspot is real and should be monitored, but it does not overturn the overall geometry-chain result

## Scripts Added For This Direction

- [compare_geometry_branches.py](/f:/vggt/vggt-main/scripts/compare_geometry_branches.py)
- [run_local_geometry_baseline.ps1](/f:/vggt/vggt-main/scripts/run_local_geometry_baseline.ps1)
- [run_geometry_baseline_batch.py](/f:/vggt/vggt-main/scripts/run_geometry_baseline_batch.py)
- [run_zju_geometry_primary_from_report.ps1](/f:/vggt/vggt-main/scripts/run_zju_geometry_primary_from_report.ps1)
- [run_zju_geometry_view_sweep.py](/f:/vggt/vggt-main/scripts/run_zju_geometry_view_sweep.py)
- [compare_zju_geometry_sweeps.py](/f:/vggt/vggt-main/scripts/compare_zju_geometry_sweeps.py)
- [run_geometry_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_geometry_minimal_finetune.ps1)
- [run_zju_vggt_geom_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_zju_vggt_geom_minimal_finetune.ps1)
- [run_zju_unproject_geometry_ablation_pair.ps1](/f:/vggt/vggt-main/scripts/run_zju_unproject_geometry_ablation_pair.ps1)
- [compare_zju_finetune_runs.py](/f:/vggt/vggt-main/scripts/compare_zju_finetune_runs.py)
- [modal_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_geometry_minimal_finetune.py)
- [run_modal_geometry_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_modal_geometry_minimal_finetune.ps1)
- [modal_zju_geometry_minimal_finetune.py](/f:/vggt/vggt-main/modal_zju_geometry_minimal_finetune.py)
- [run_modal_zju_geometry_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_modal_zju_geometry_minimal_finetune.ps1)
- [modal_zju_geometry_minimal_finetune.md](/f:/vggt/vggt-main/docs/modal_zju_geometry_minimal_finetune.md)
- [zju_vggt_geom_unproject_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_minimal.yaml)

## Important Supporting Fixes

- [training/launch.py](/f:/vggt/vggt-main/training/launch.py) now accepts Hydra-style runtime overrides and can run as a standalone script without losing the repo root on `sys.path`
- [training/trainer.py](/f:/vggt/vggt-main/training/trainer.py) now supports local single-process smoke runs when `WORLD_SIZE=1`, fixes checkpoint saving without DDP, fixes exact batch-limit semantics, and aligns objective logging
- [training/data/dynamic_dataloader.py](/f:/vggt/vggt-main/training/data/dynamic_dataloader.py) now falls back to a single-replica sampler when torch distributed is not initialized
- [training/loss.py](/f:/vggt/vggt-main/training/loss.py) now supports an optional `unproject_geometry` loss built from predicted `depth + camera`
- [bootstrap_local_5080_env.ps1](/f:/vggt/vggt-main/scripts/bootstrap_local_5080_env.ps1) now supports:
  - training dependencies
  - optional Modal dependency install
- [requirements_training.txt](/f:/vggt/vggt-main/requirements_training.txt) centralizes the training dependency set

## Current Recommendation

1. Keep the codebase on original VGGT.
2. Treat `depth + camera` as the current main branch to test.
3. If you start fine-tuning, keep `point` and `track` disabled.
4. Do not restore ghost/mask/bbox/confidence stacks as training objectives.
5. If you move beyond the pure baseline, `unproject_geometry` is the first extra term to keep.
6. Only add one extra geometry or reconstruction term at a time.
7. Treat the 500-step paired local run plus the checkpoint-after-training render eval as the current reference before any longer Modal job.
8. When discussing rendering-path replacement, distinguish:
   - dense/full-rig cases, where `depth + camera` is strongly supported
   - sparse `6src` cases, where target-aware source selection helps materially
   - sparse `12src` cases, where uniform ring coverage is better than both rotated-template and nearest-ring selection
9. Use the new diagnostics note as the gate for longer runs:
   - `23cam_fullset` passes strongly
   - `6src_hist + rotate_template_offsets` is the best immediate sparse baseline
   - `12src_uniform` remains a secondary probe, not the primary sparse long-run setting

## What Is Still Blocked By Data

The main remaining blocker for the generic CO3D training path is still the real CO3D location:

- local run needs the actual `CO3D_DIR`
- local run needs the actual `CO3D_ANNOTATION_DIR`
- Modal run needs the matching volume-relative subdirectories

That said, the ZJU pseudo-geometry route is no longer blocked locally, and I also prepared a ZJU-specific Modal launcher for the same geometry-first setup. That cloud path still requires you to place the ZJU root and checkpoint into a Modal data volume, but it no longer depends on finding CO3D first.

I also added a helper scan:

- [find_co3d_candidates.ps1](/f:/vggt/vggt-main/scripts/find_co3d_candidates.ps1)
- [find_co3d_candidates.py](/f:/vggt/vggt-main/scripts/find_co3d_candidates.py)
- latest targeted scan result: [co3d_candidates_targeted.md](/f:/vggt/vggt-main/output/co3d_candidates_targeted.md)
- broader follow-up scan result: [co3d_candidates_broader.md](/f:/vggt/vggt-main/output/co3d_candidates_broader.md)

The latest targeted and broader scans across `G:\数据集`, `G:\项目备份`, `F:\datasets`, `F:\dataset_practice`, `G:\NAS`, `D:\model`, `D:\BaiduNetdisk`, and `G:\BaiduNetdiskDownload` still did not find an obvious CO3D dataset or annotation directory.

Once those paths are known, the wrappers are ready for the next step.

## Current Practical Next Step

1. Keep using the local 5080 path for quick geometry-only checks.
2. Use the paired runner to compare baseline vs `unproject_geometry` on the same short local schedule before any longer run.
3. If you add one new supervision term, start from this `camera + depth only` smoke baseline or the paired `unproject_geometry` variant.
4. For the mentor's render-path question, use round 1 plus round 2 together:
   - round 1 for dense/full-rig evidence
   - round 2 and round 3 for sparse source-policy sensitivity
5. If you want a sparse geometry baseline right now:
   - use target-aware `6src_hist` as the cleaner sparse geometry-friendly setting
   - use `12src_uniform` only if you specifically need the 12-view profile
6. The next high-value offline experiment is a stronger uniform-coverage sparse policy or a longer minimal geometry run, not ghost-loss restoration.
7. Use the ZJU-specific Modal launcher only if you want to move the same minimal geometry experiment to cloud before CO3D is resolved.
8. Move only the minimal geometry version to Modal for longer runs.
9. Keep the old ghost stack out of both local and Modal training until the geometry branch clearly saturates.
10. If a longer run exposes new failures first, inspect target-camera hotspots such as `Camera_B8` before adding any new image-side loss.
