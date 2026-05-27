# ZJU SMPL Prior Head/Hair Modal Status

## Scope

- Date: `2026-04-16`
- Goal: continue from the latest remote `resume2` checkpoint, keep the run alive beyond the local terminal timeout, leave active monitoring in place, and determine whether the resumed cloud training fully completed.

## Final Conclusion

- `resume5_guarded` completed the configured training plan naturally; no further relaunch is required.
- There is no active Modal app remaining for `vggt-zju-geometry-smplprior-headhair`.
- The latest completed output root is:
  - `/mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume5_guarded`
- The latest completed checkpoint is:
  - `/mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume5_guarded/ckpts/checkpoint_7.pt`

Evidence:

- Launcher parameters for this run included:
  - `max_epochs: 8`
  - `limit_train_batches: 800`
  - `limit_val_batches: 200`
- Trainer semantics in `training/trainer.py` are:
  - on resume, `prev_epoch -> self.epoch = prev_epoch + 1`
  - training loop condition is `while self.epoch < self.max_epochs`
  - a checkpoint is saved after each train epoch
  - in-loop validation is skipped for the last train epoch
  - after `run_train()`, `run()` performs one final `run_val()`
- The last verified remote progress matches that exact terminal pattern:
  - `Train Epoch: [7][799/1000000]`
  - `Saving checkpoint at epoch 7 to .../ckpts/checkpoint_7.pt`
  - `Val Epoch: [7][199/1000000]`
- `modal app list --json` now shows the latest app as stopped:
  - app id `ap-5cYQCMNnm7VO9VhpD9g6hb`
  - created `2026-04-16 15:05:04 +08:00`
  - stopped `2026-04-16 15:59:31 +08:00`

## What Landed

- Probe/export completion had already been fixed and locally verified in:
  - `scripts/probe_zju_vggt_geom_dataset.py`
- A new local guard daemon now keeps the Modal launcher alive and watches cloud progress:
  - `scripts/run_modal_zju_geometry_guard_daemon.py`
- The Modal preflight repo-process allowlist now ignores the guard daemon itself so it does not trip local preflight checks:
  - `scripts/invoke_modal_zju_preflight.ps1`

## Final Guarded Run

- Modal app name: `vggt-zju-geometry-smplprior-headhair`
- Final app id: `ap-5cYQCMNnm7VO9VhpD9g6hb`
- App created at: `2026-04-16 15:05:04 +08:00`
- App stopped at: `2026-04-16 15:59:31 +08:00`
- Final state at last check: `stopped`
- Resume checkpoint:
  - `/mnt/out/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/ckpts/checkpoint_6.pt`
- New output root:
  - `/mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume5_guarded`
- Final checkpoints present:
  - `checkpoint_7.pt`
  - `checkpoint.pt`

## Resume4 Failure And Fix

- First guarded relaunch output root:
  - `/mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume4_guarded`
- App id:
  - `ap-JNBeCmUzyjYfRNIaINlmYz`
- It made real progress through approximately:
  - `Train Epoch: [7][350/1000000]`
- It then stopped at:
  - `2026-04-16 14:27:31 +08:00`
- Launcher tail showed:
  - `Stopping app - user stopped from CLI.`

Interpretation:

- The prior PowerShell launcher used `modal ...::run_remote_zju_geometry_finetune` directly.
- That path still tied the remote run lifetime to the local `modal run` client, even under `--detach`.
- Fix landed:
  - `modal_zju_geometry_minimal_finetune.py`
    - added local entrypoint `spawn_remote_zju_geometry_finetune`
  - `scripts/run_modal_zju_geometry_minimal_finetune.ps1`
    - `-Detach` now routes to the new local spawn entrypoint instead of binding directly to the remote function
  - `scripts/run_modal_zju_geometry_guard_daemon.py`
    - updated to treat a fast launcher exit as normal for the new spawn-based detach flow

## Verified Progress And Completion

- The current remote output root already contains:
  - `dataset_probe/summary.json`
  - `dataset_probe/aggregate_summary.json`
  - `dataset_probe/sample_human_prior_completion_depths.png`
  - `dataset_probe/sample_human_prior_completion_point_masks.png`
  - `dataset_probe/sample_human_prior_completion_world_points.ply`
  - `dataset_probe/sample_completed_world_points.ply`
  - `dataset_probe/sample_human_prior_target_mask.png`
  - `logs/log.txt`
  - `driver_live.log`
- Resume entered the real training loop again, not just initialization.
- The fixed detached-spawn path has now been verified on `resume5_guarded`:
  - local launcher exited cleanly with return code `0`
  - the Modal app remained active after launcher exit
  - remote training logs continued to advance after launcher exit
- Latest verified train progress:
  - `Train Epoch: [7][799/1000000]` at `2026-04-16 07:46:59 UTC`
- Latest verified validation progress:
  - `Val Epoch: [7][199/1000000]` at `2026-04-16 07:58:01 UTC`
- Latest verified checkpoint save:
  - `Saving checkpoint at epoch 7 to /mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume5_guarded/ckpts/checkpoint_7.pt`
- Guard status concluded:
  - `status: stopped`
  - `reason: active Modal app has ended after launcher exit`
- No newer relaunch is needed because this stop aligns with the configured terminal epoch boundary rather than an early interruption.
- No `Traceback`, `RuntimeError`, `Exception`, or `KeyboardInterrupt` was seen in the checked guard/launcher tail.

## Guard Daemon

- Guard session dir:
  - `output/modal_zju_geometry_guard_daemon/20260416_150457_smplprior_headhair_resume_guard`
- Guard status snapshot:
  - `output/modal_zju_geometry_guard_daemon/latest_session.json`
- Active lock:
  - `output/modal_zju_geometry_guard_daemon/active_guard.json`
- Local launcher log:
  - `output/modal_zju_geometry_guard_daemon/20260416_134608_smplprior_headhair_resume_guard/launcher_stdout.txt`
- The guard is configured to:
  - poll every `120` seconds
  - stop redundant active apps with the same Modal description if more than one appears
  - stop the active app if training progress stays unchanged for `3.0` hours
- During this run, no redundant active app needed to be killed.

## Notes

- A direct `modal run --detach` from the interactive terminal was not sufficient because the local client remained attached long enough to be killed by tool timeout, which in turn stopped the remote app.
- The new guard daemon avoids that failure mode by owning the launcher process in the background and continuously checking progress.
- The final stop of `resume5_guarded` is consistent with normal completion, not with the earlier CLI-bound shutdown failure seen on `resume4_guarded`.
- Volume inode pressure remains high:
  - `/mnt/out` around `91.4%`
  - `/mnt/data` around `84.9%`
