# Geometry Sparse-Policy Follow-Up 2026-03-23

## Current State

- `zju_min_depth_conf` reliability line is closed.
  - result: the scalar cached-target threshold did not beat the baseline
  - decision: do not carry that treatment into `unproject_geometry`
- the matched `12src_rotate -> 12src_uniform` current-current subset follow-up has now been materialized:
  - output: `output/geometry_sparse_policy_followup_20260323/rotate_vs_uniform_frame1080_matched/summary.md`
  - result:
    - `3 / 4` cases improved in geometry gain
    - average geometry gain delta: `+0.001796`
    - average coverage gain delta: `+0.025556`
    - `Camera_B19` flipped from `point_map` to `depth_unproject`
- the strict same-source `legacy native` backfill for `12src_uniform` has now also been completed locally:
  - output: `output/legacy_uniform_backfill_from_current/batch4_20260323/summary/summary.md`
  - result:
    - `depth_unproject` beats current `point_map` on all `4 / 4` matched-source cases
    - average legacy gap:
      - current point: `+0.001614`
      - current depth: `+0.000243`
    - `Camera_B19` remains the clearest remaining hard gap, but `depth_unproject` is still closer to legacy than `point_map`
- Modal state has been re-checked after completion.
  - no useful active cloud apps remain
  - latest checked apps were both `stopped`
- Repo-scoped residual monitor/train processes were re-checked locally.
  - no active repo/modal watcher process needs cleanup
  - the visible remaining `powershell/python` processes are VS Code terminal/LSP processes, not this experiment line

## Why The Next Step Changes

The current bottleneck is no longer "is `depth + camera -> unproject` valid at all?"

That part is already supported.

The more precise open question is:

> where does sparse-view current geometry still lose, and is the remaining gap better explained by source policy than by another global target-depth treatment?

The strongest evidence behind that shift is:

- hard-case legacy-gap batch:
  - `depth_better_than_point = 5 / 8`
  - average legacy gap:
    - `depth_unproject = 0.008434`
    - `point_map = 0.009209`
- sparse diagnostics:
  - `6src_hist + rotate_template_offsets` is still the best current `6src` sparse baseline
  - `12src_nested + uniform_ring` is the best current `12src` source policy, but has not yet been pushed through the legacy-gap-style hard/control check

So the smallest useful next action is not another training run.

It is an inference-only sparse-policy follow-up.

## Immediate Next Experiment

### Phase 1

First run a local/offline `current-current` follow-up on the exact `12src` hard/control targets that already exist in the sweep outputs:

- sequence: `CoreView_390`
- frame: `1080`
- profile: `12src_nested`
- source policy: `uniform_ring`
- targets:
  - `Camera_B5` control
  - `Camera_B3` depth-favored control/hard crossover
  - `Camera_B8` hard case
  - `Camera_B19` hard case

This keeps all of the following fixed:

- original VGGT checkpoint
- same branch compare pipeline
- same `point_map` vs `depth_unproject` evaluation
- same original VGGT checkpoint family

The only changed variable is the sparse `12src` source policy.

## Phase 1 Outcome

The matched current-current comparison is now complete on:

- `CoreView_390 / frame 1080 / 12src_nested`
- targets:
  - `Camera_B3`
  - `Camera_B5`
  - `Camera_B8`
  - `Camera_B19`

Readout:

- `12src_uniform` improved geometry gain on `3 / 4` targets
- `12src_uniform` improved average geometry gain from `-0.000425` to `+0.001371`
- `12src_uniform` improved average coverage gain from `-0.028563` to `-0.003008`
- per target:
  - `Camera_B19`: `point_map -> depth_unproject`
  - `Camera_B8`: `point_map -> tie`
  - `Camera_B5`: tie stays tie, but geometry and coverage both improve
  - `Camera_B3`: regresses relative to rotate

That means the current-current gate is passed.

So the next strict step is no longer "rerun current `12src_uniform`."

It is:

> prepare a matched-source legacy-native backfill for the same four targets, using the exact `source_cameras` extracted from the current `12src_uniform` case summaries.

That step is now complete locally.

## Phase 2 Outcome

The matched-source legacy-native backfill was run locally for:

- `Camera_B3`
- `Camera_B5`
- `Camera_B8`
- `Camera_B19`

under:

- `view_profile_tag = 12src_uniform_from_current`
- exact `src_cameras` copied from the current `round3_12src_uniform_v1` case summaries

Readout:

- current `depth_unproject` beats current `point_map` on all `4 / 4` same-source cases
- average MAE:
  - legacy native: `0.034940`
  - current point: `0.036554`
  - current depth: `0.035183`
- average legacy gap:
  - point: `+0.001614`
  - depth: `+0.000243`

This is the first clean same-source result in this branch that strongly supports:

1. keeping `12src_uniform` over `12src_rotate`
2. keeping `depth_unproject` over current `point_map`
3. not reopening the old `ghost` direction

### Why This Is The Smallest Useful Step

- it tests the current main hypothesis directly:
  - whether sparse geometry is still being limited more by source-view policy than by another training-side scalar gate
- it reuses existing current sweep outputs, so it does not require new training
- it does not require cloud fine-tuning
- it does not reopen:
  - `ghost`
  - `confgate / threshold / pow`
  - new loss design

## Secondary Queue After Phase 1

If the `12src_uniform` follow-up still leaves a large gap, the next diagnosis target should be the most consistently hard sparse cameras rather than another threshold experiment.

Current priority queue from the diagnostics summary:

- `6src_hist / rotate_template_offsets`
  - `Camera_B15`
  - `Camera_B12`
  - `Camera_B1`
  - `Camera_B4`
- `12src_nested / uniform_ring`
  - `Camera_B12`
  - `Camera_B2`
  - `Camera_B13`
  - `Camera_B15`

These are diagnosis targets, not a training launch list yet.

## Important Control-Variable Note

`12src_uniform` changes the source-camera set.

That means the old `12src_rotate` legacy-native reports are **not** a strict same-input legacy baseline for `12src_uniform`.

So the correct order is:

1. first compare `12src_rotate` vs `12src_uniform` on the current side with matched `frame / target / profile`
2. only if `12src_uniform` is clearly better, decide whether to backfill a **new matched legacy-native** batch for the `uniform_ring` source sets

This keeps the control variable clean.

## Local Gate

Do not launch cloud from this step.

The local gate is:

1. confirm the old template report, checkpoint, and local ZJU root still resolve
2. compare the existing `12src_uniform` four-case sweep outputs against the existing `12src_rotate` current hard/control cases
3. only if the current-current comparison is clearly favorable, queue a matched-source legacy-native backfill

Only if that local gate is clean and useful should we consider a new cloud-backed action.

## Cloud Gate

Cloud is allowed only after the local follow-up answers one of these:

1. `12src_uniform` clearly improves the matched current-side hard/control set, which justifies a matched-source legacy-native backfill
2. `12src_uniform` clearly fails, which justifies shifting effort to:
   - `6src` hard-camera geometry/render-gap diagnosis
   - not another target-threshold treatment

The first condition is now satisfied.

The remaining gate is operational:

- the matched-source backfill is now already complete locally
- there is still no immediate need for cloud just to validate this sparse-policy question
- the next cloud action, if any, should be chosen from a new training/ablation question rather than from missing inference evidence

## Reproduction Command For The Next Local Follow-Up

```powershell
.\.venv5080\Scripts\python.exe .\scripts\run_zju_geometry_view_sweep.py `
  --template_reports `
    "G:\项目备份\vggt_小感度不起作用\vggt\infer_out\vggt_raw_viewcount\12src_nested\CoreView_390\frame_001080_Camera_B5\run_20260316_110256\report.json" `
  --local_zju_root "G:\数据集\datasets\ZJU_MoCap\data\zju_mocap" `
  --checkpoint "G:\项目备份\vggt_小感度不起作用\vggt\model.pt" `
  --output_root "output/geometry_view_sweep_zju/round4_12src_uniform_legacygap1080_v1" `
  --frame_ids "1080" `
  --target_cameras "Camera_B3,Camera_B5,Camera_B8,Camera_B19" `
  --source_policy uniform_ring `
  --skip_existing
```

This command is only needed if the existing `round3_12src_uniform_v1` outputs are missing or need to be regenerated.

## Acceptance Rule

Promote `12src_uniform` to the next sparse follow-up baseline only if it satisfies both:

1. it improves `depth_unproject` over the current `12src_rotate` matched cases on at least `3 / 4` targets
2. it improves the average geometry gain on that matched four-case subset

If either condition fails, freeze `12src` as "improved but still secondary" and move the next effort to sparse hard-camera diagnosis rather than another training-side threshold line.
