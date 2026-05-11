# ZJU SMPL Prior Head/Hair Modal Status

## Scope

- Date: `2026-04-16`
- Goal: continue from the latest remote `resume2` checkpoint, keep the run alive beyond the local terminal timeout, and leave active monitoring in place.

## What Landed

- Probe/export completion had already been fixed and locally verified in:
  - `scripts/probe_zju_vggt_geom_dataset.py`
- A new local guard daemon now keeps the Modal launcher alive and watches cloud progress:
  - `scripts/run_modal_zju_geometry_guard_daemon.py`
- The Modal preflight repo-process allowlist now ignores the guard daemon itself so it does not trip local preflight checks:
  - `scripts/invoke_modal_zju_preflight.ps1`

## Current Guarded Run

- Modal app name: `vggt-zju-geometry-smplprior-headhair`
- Active app id: `ap-JNBeCmUzyjYfRNIaINlmYz`
- App created at: `2026-04-16 13:46:21 +08:00`
- Current state at last check: `ephemeral (detached)` with `Tasks=1`
- Resume checkpoint:
  - `/mnt/out/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/ckpts/checkpoint_6.pt`
- New output root:
  - `/mnt/out/20260416_smplprior_headhair_longrun_eager_shmsafe_resume4_guarded`

## Verified Progress

- The new remote output root already contains:
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
- Latest verified launcher log tail at last manual check:
  - `Train Epoch: [7][0]` at `2026-04-16 05:49:03 UTC`
  - `Train Epoch: [7][1]` at `2026-04-16 05:49:34 UTC`
  - `Train Epoch: [7][2]` at `2026-04-16 05:50:06 UTC`
  - `Train Epoch: [7][3]` at `2026-04-16 05:50:07 UTC`
- No `Traceback`, `RuntimeError`, `Exception`, or `KeyboardInterrupt` was seen in the checked guard/launcher tail.

## Guard Daemon

- Guard session dir:
  - `output/modal_zju_geometry_guard_daemon/20260416_134608_smplprior_headhair_resume_guard`
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

## Notes

- A direct `modal run --detach` from the interactive terminal was not sufficient because the local client remained attached long enough to be killed by tool timeout, which in turn stopped the remote app.
- The new guard daemon avoids that failure mode by owning the launcher process in the background and continuously checking progress.
- Volume inode pressure remains high:
  - `/mnt/out` around `91.4%`
  - `/mnt/data` around `84.9%`
