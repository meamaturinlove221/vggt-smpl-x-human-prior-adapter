# ZJU Unproject Geometry Probe 2026-03-21

This note records the first minimal extra-supervision experiment after the geometry-first baseline was confirmed.

## Goal

Add exactly one extra geometry term on top of the original `camera + depth` setup:

- do not re-enable `point_head`
- do not restore ghost/mask/bbox/confidence-stack logic
- do not change the main network structure

The new term is:

- `loss_unproject_geometry`
- built from `predicted depth + predicted camera`
- differentiably unprojected back to world points
- compared directly against batch `world_points`

In short:

`depth + camera -> unproject -> world point regression`

## Implementation

Main code changes:

- [loss.py](/f:/vggt/vggt-main/training/loss.py)
  - added `compute_unproject_geometry_loss`
  - added `unproject_depth_and_pose_to_world_points`
  - integrated optional `unproject_geometry` block into `MultitaskLoss`
- [zju_vggt_geom_unproject_minimal.yaml](/f:/vggt/vggt-main/training/config/zju_vggt_geom_unproject_minimal.yaml)
  - new minimal config with `loss.unproject_geometry`
  - logs `loss_unproject_geometry` for train and val
- [run_zju_vggt_geom_minimal_finetune.ps1](/f:/vggt/vggt-main/scripts/run_zju_vggt_geom_minimal_finetune.ps1)
  - now supports optional unproject-geometry overrides for future sweeps

Experiment setting:

- `weight = 0.2`
- `loss_type = l2`
- `valid_range = 0.98`
- `min_valid_points = 100`

## Smoke Result

1-train-batch + 1-val-batch smoke succeeded with:

- config: `zju_vggt_geom_unproject_minimal`
- run: [zju_vggt_geom_unproject_smoke_local_v1](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_smoke_local_v1/log.txt)

Key metrics:

- train objective: `4.4766`
- train unproject geometry: `0.6315`
- val objective: `3.5459`
- val unproject geometry: `0.6031`

This confirmed that the new term is numerically stable and the pipeline runs end-to-end locally.

## 5-Step Probe

To check whether the new term immediately destabilizes the original objectives, I ran a short probe:

- baseline run: [zju_vggt_geom_baseline_probe_local_v1](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_baseline_probe_local_v1/log.txt)
- unproject run: [zju_vggt_geom_unproject_probe_local_v1](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_probe_local_v1/log.txt)

Both runs used:

- 5 train batches
- 2 val batches
- same ZJU sequence
- same checkpoint
- same optimizer and frozen-module policy

## Short Comparison

Final averaged validation numbers after the 5-step probe:

Baseline:

- val objective: `1.3188`
- val camera: `0.0536`
- val conf depth: `0.5220`
- val reg depth: `0.5117`
- val grad depth: `0.0171`

Baseline + unproject geometry:

- val objective: `1.4373`
- val camera: `0.0537`
- val conf depth: `0.5214`
- val reg depth: `0.5108`
- val grad depth: `0.0171`
- val unproject geometry: `0.5970`

## 20-Step Paired Local Run

To make sure the first readout was not just a 5-step coincidence, I ran the paired automation on a longer short-run schedule:

- baseline: [zju_vggt_geom_pair_20step_v1_baseline](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_20step_v1_baseline/log.txt)
- unproject: [zju_vggt_geom_pair_20step_v1_unproject](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_20step_v1_unproject/log.txt)
- paired report: [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_20step_v1/summary.md)
- paired report json: [summary.json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_20step_v1/summary.json)

Both runs used:

- 20 train batches
- 4 val batches
- same sequence, checkpoint, optimizer, and frozen-module policy

Final averaged metrics:

Baseline:

- train camera: `0.0528`
- train conf depth: `0.4408`
- train reg depth: `0.1873`
- train grad depth: `0.0224`
- val camera: `0.0592`
- val conf depth: `0.0511`
- val reg depth: `0.0511`
- val grad depth: `0.0141`

Baseline + unproject geometry:

- train camera: `0.0530`
- train conf depth: `0.4400`
- train reg depth: `0.1865`
- train grad depth: `0.0221`
- val camera: `0.0594`
- val conf depth: `0.0526`
- val reg depth: `0.0526`
- val grad depth: `0.0140`
- train unproject geometry: `0.2828`
- val unproject geometry: `0.2031`

## 100-Step Paired Local Run

I then pushed the same paired setup further to a more realistic short fine-tune window:

- baseline: [zju_vggt_geom_pair_100step_v1_baseline](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_100step_v1_baseline/log.txt)
- unproject: [zju_vggt_geom_pair_100step_v1_unproject](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_100step_v1_unproject/log.txt)
- paired report: [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_100step_v1/summary.md)
- paired report json: [summary.json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_100step_v1/summary.json)

Both runs used:

- 100 train batches
- 10 val batches
- same sequence, checkpoint, optimizer, and frozen-module policy

Final averaged metrics:

Baseline:

- train camera: `0.0406`
- train T: `0.0324`
- train R: `0.0028`
- train conf depth: `0.1597`
- train reg depth: `0.0867`
- train grad depth: `0.0125`
- val camera: `0.0304`
- val conf depth: `0.0294`
- val reg depth: `0.0294`
- val grad depth: `0.0074`

Baseline + unproject geometry:

- train camera: `0.0404`
- train T: `0.0323`
- train R: `0.0027`
- train conf depth: `0.1597`
- train reg depth: `0.0867`
- train grad depth: `0.0107`
- val camera: `0.0308`
- val conf depth: `0.0277`
- val reg depth: `0.0277`
- val grad depth: `0.0070`
- train unproject geometry: `0.1690`
- val unproject geometry: `0.1010`

## Interpretation

The important part is not the raw objective increase by itself, because the new run includes one extra weighted term.

What matters is:

- camera loss stayed essentially unchanged
- depth sub-losses stayed essentially unchanged
- no early divergence appeared
- no gradient explosion appeared
- the added geometry term stayed finite and stable at about `0.60`

The 20-step paired run keeps the same pattern:

- camera loss is still effectively unchanged
- depth confidence, regression, and gradient terms remain very close
- the added geometry term keeps decreasing during training instead of blowing up
- the higher objective is still mostly explained by the presence of the extra weighted term itself

The 100-step paired run strengthens that conclusion:

- train camera, `T`, and `R` are still effectively unchanged
- train depth confidence and regression are exactly matched at the reported average
- train depth gradient is slightly better on the unproject run
- val depth confidence and regression are slightly better on the unproject run
- the extra geometry term keeps decreasing to `0.1690` train average and `0.1010` val average

This is now the strongest local training evidence so far that `unproject_geometry` can be kept as a minimal auxiliary term without derailing the original `camera + depth` path.

This means the new term behaves like a clean auxiliary geometry constraint, not like a disruptive objective rewrite.

## 500-Step Paired Local Run

I then pushed the same paired setup into a longer local overnight-scale schedule:

- baseline: [zju_vggt_geom_pair_500step_v1_baseline](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_500step_v1_baseline/log.txt)
- unproject: [zju_vggt_geom_pair_500step_v1_unproject](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_500step_v1_unproject/log.txt)
- paired report: [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.md)
- paired report json: [summary.json](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.json)

Both runs used:

- 500 train batches
- 20 val batches
- same sequence, checkpoint, optimizer, and frozen-module policy

Final averaged metrics:

Baseline:

- train camera: `0.0253`
- train `T`: `0.0200`
- train `R`: `0.0020`
- train conf depth: `0.0783`
- train reg depth: `0.0484`
- train grad depth: `0.0060`
- val camera: `0.0123`
- val conf depth: `0.0168`
- val reg depth: `0.0168`
- val grad depth: `0.0046`

Baseline + unproject geometry:

- train camera: `0.0247`
- train `T`: `0.0194`
- train `R`: `0.0021`
- train conf depth: `0.0785`
- train reg depth: `0.0486`
- train grad depth: `0.0060`
- val camera: `0.0122`
- val conf depth: `0.0174`
- val reg depth: `0.0174`
- val grad depth: `0.0044`
- train unproject geometry: `0.0968`
- val unproject geometry: `0.0421`

Readout:

- train camera is still effectively unchanged and slightly better numerically on the unproject run
- train `T` is also slightly better numerically on the unproject run
- train depth confidence and regression remain matched to within `0.0002`
- val camera is effectively unchanged
- val depth confidence and regression are slightly worse by `0.0006`, but still remain extremely close
- val depth gradient is slightly better on the unproject run
- the added geometry term keeps decreasing further to `0.0968` train average and `0.0421` val average

The 500-step pair therefore preserves the same main conclusion as the 20-step and 100-step runs:

- `unproject_geometry` stays numerically stable
- it does not derail the original `camera + depth` path
- the added objective increase is still mostly explained by the presence of the extra weighted term itself
- the longer run still supports keeping it as the first minimal extra geometry term

## Render-Side Follow-Up After The 500-Step Pair

I then evaluated the two 500-step checkpoints on the same render-side `point_map` vs `depth + camera` comparison used in the earlier ZJU sweeps:

- checkpoint eval note: [geometry_zju_checkpoint_eval_20260322.md](/f:/vggt/vggt-main/docs/geometry_zju_checkpoint_eval_20260322.md)
- small compare: [comparison.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_compare_v1/comparison.md)
- full-target compare: [comparison.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_fulltargets_compare_v1/comparison.md)

Key read:

- the unproject checkpoint keeps `depth + camera` ahead on the same cases where the baseline already preferred it
- the small `48`-case probe shows positive average deltas in both geometry and coverage
- the broader `184`-case full-target run still shows positive average deltas and one `tie -> depth` improvement with no decision regressions
- the main negative hotspot remains concentrated around `Camera_B8`, which means the issue is local and diagnosable rather than a general failure of the geometry-first direction

So the 500-step run now has two kinds of evidence behind it:

- training-side stability
- render-side positive follow-up

## Current Recommendation

The geometry-first sequence now looks like this:

1. keep original VGGT
2. keep `camera + depth` as the core training path
3. treat `unproject_geometry` as the first acceptable extra supervision candidate
4. if a longer local or Modal run is attempted next, prefer this term over reviving the old ghost stack
5. continue comparing it against the pure baseline with paired runs instead of stacking more losses
6. treat the 500-step paired run as the current local reference result

## Outputs

- [baseline probe log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_baseline_probe_local_v1/log.txt)
- [unproject probe log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_probe_local_v1/log.txt)
- [unproject smoke log](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_unproject_smoke_local_v1/log.txt)

## Automation Added After The Probe

To make this comparison repeatable, I also added:

- [compare_zju_finetune_runs.py](/f:/vggt/vggt-main/scripts/compare_zju_finetune_runs.py)
  - parses two `training/logs/.../log.txt` files
  - exports `summary.md` and `summary.json`
- [run_zju_unproject_geometry_ablation_pair.ps1](/f:/vggt/vggt-main/scripts/run_zju_unproject_geometry_ablation_pair.ps1)
  - runs baseline and unproject variants sequentially
  - automatically writes a comparison report

Generated comparison artifacts:

- [probe compare summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_probe_compare_v1/summary.md)
- [pair e2e smoke summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_e2e_smoke_v2/summary.md)
- [20-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_20step_v1/summary.md)
- [100-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_100step_v1/summary.md)
- [500-step pair summary](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.md)

Reference command for the current strongest local paired run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_zju_unproject_geometry_ablation_pair.ps1 `
  -ExpPrefix zju_vggt_geom_pair_500step_v1 `
  -LimitTrainBatches 500 `
  -LimitValBatches 20 `
  -NumImages 4 `
  -MaxImgPerGpu 4 `
  -AccumSteps 1 `
  -MaxEpochs 1 `
  -NumWorkers 0
```
