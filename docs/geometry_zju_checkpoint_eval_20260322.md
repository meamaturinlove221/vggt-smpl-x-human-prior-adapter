# ZJU Checkpoint Eval 2026-03-22

This note records the first render-side follow-up after the 500-step paired local training run.

The question here is narrower than the earlier inference sweeps:

- keep the same original VGGT codebase
- keep the same `depth + camera` render-path comparison
- compare two trained checkpoints only
- ask whether the minimal `unproject_geometry` fine-tune helps the render-side geometry readout, not just the training losses

## Checkpoints Compared

- baseline checkpoint: [checkpoint.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_500step_v1_baseline/ckpts/checkpoint.pt)
- unproject checkpoint: [checkpoint.pt](/f:/vggt/vggt-main/training/logs/zju_vggt_geom_pair_500step_v1_unproject/ckpts/checkpoint.pt)
- paired 500-step training summary: [summary.md](/f:/vggt/vggt-main/output/zju_training_ablation/zju_vggt_geom_pair_500step_v1/summary.md)

Both checkpoints come from the same local paired setup:

- original VGGT initialization
- `camera + depth` heads active
- `point=False`
- same optimizer, same data, same frozen-module policy
- only one extra term differs: `loss_unproject_geometry`

## Small Follow-Up Probe

Artifacts:

- baseline sweep summary: [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_baseline_v1/summary.md)
- unproject sweep summary: [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_unproject_v1/summary.md)
- comparison: [comparison.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_compare_v1/comparison.md)

Scope:

- `48` common cases
- profiles: `23cam_fullset`, `6src_hist`
- frames: `0, 600, 1080, 1170`
- target cameras: `B1, B5, B8, B12, B15, B20`

Readout:

- baseline decisions: `48` depth wins, `0` point wins, `0` ties
- unproject decisions: `48` depth wins, `0` point wins, `0` ties
- average geometry gain delta: `+0.000715`
- average coverage gain delta: `+0.038546`
- `41/48` cases improved in both geometry and coverage
- `7/48` cases regressed in both geometry and coverage

Important nuance:

- the regressions in this small probe were concentrated around `Camera_B8`
- so the first follow-up question was whether that pattern persists after expanding to all target cameras

## Full-Target Check

Artifacts:

- baseline full-target summary: [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_fulltargets_baseline_v1/summary.md)
- unproject full-target summary: [summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_fulltargets_unproject_v1/summary.md)
- full-target comparison: [comparison.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/checkpoint_eval_500step_fulltargets_compare_v1/comparison.md)

Scope:

- `184` common cases
- same two profiles: `23cam_fullset`, `6src_hist`
- same four frames: `0, 600, 1080, 1170`
- all `23` target cameras

Overall result:

- baseline decisions: depth `168`, point `0`, tie `16`
- unproject decisions: depth `169`, point `0`, tie `15`
- improved decision count: `1` case moved from `tie -> depth`
- regressed decision count: `0`
- average geometry gain delta: `+0.000632`
- average coverage gain delta: `+0.035864`

By profile:

- `23cam_fullset`: geometry delta `+0.000800`, coverage delta `+0.051668`, one `tie -> depth` improvement
- `6src_hist`: geometry delta `+0.000465`, coverage delta `+0.020060`, no decision regressions

So the full-target read is stronger than the small probe in one important way:

- the positive average deltas survive the expansion to all target cameras
- the decision counts do not regress
- the extra geometry term now has both training-side stability evidence and render-side support

## Negative-Case Distribution

I also checked where the negative deltas cluster in the full-target comparison.

Counts across the `184` common cases:

- geometry delta negative: `40`
- coverage delta negative: `27`
- both negative together: `21`
- both positive together: `138`

Target-camera concentration for the `21` both-negative cases:

- `Camera_B8`: `7`
- `Camera_B10`: `3`
- `Camera_B11`: `3`
- `Camera_B7`: `2`
- `Camera_B14`: `2`
- `Camera_B19`: `2`
- `Camera_B21`: `1`
- `Camera_B9`: `1`

Profile split for the `21` both-negative cases:

- `6src_hist`: `14`
- `23cam_fullset`: `7`

`Camera_B8` follow-up:

- total `Camera_B8` cases in the full-target run: `8`
- both-negative `Camera_B8` cases: `7`
- average `Camera_B8` geometry delta: `-0.000839`
- average `Camera_B8` coverage delta: `-0.027200`

Profile breakdown for `Camera_B8`:

- `23cam_fullset`: `4/4` both negative, average geometry delta `-0.001809`, average coverage delta `-0.070144`
- `6src_hist`: `3/4` both negative, but the profile-average delta stays slightly positive because one case improves strongly enough to offset the others

Worst full-target regressions by geometry delta include:

- `23cam_fullset / frame 600 / Camera_B8`: geometry `-0.002465`, coverage `-0.094326`
- `23cam_fullset / frame 600 / Camera_B10`: geometry `-0.002419`, coverage `+0.024754`
- `23cam_fullset / frame 1080 / Camera_B8`: geometry `-0.002335`, coverage `-0.072006`

This means the small-probe `Camera_B8` warning was not noise:

- it does persist in the broader evaluation
- but it remains a local hotspot, not a global reversal of the geometry-chain trend

## Conclusion

The current checkpoint-level conclusion is:

1. the mentor's geometry-first direction is still supported after training, not just at raw inference time
2. `depth + camera` remains the correct render-side mainline to keep testing
3. the first extra term should remain `loss_unproject_geometry`, because it stayed stable in training and produced positive average render-side deltas
4. the next thing to watch is target-camera hotspot behavior, especially `Camera_B8`, not any return to the old ghost-loss stack

So if the next longer run goes to Modal, the minimal version to carry forward is still:

- original VGGT
- `camera + depth`
- optional `loss_unproject_geometry`
- no restored ghost/mask/bbox/confidence-stack training logic
