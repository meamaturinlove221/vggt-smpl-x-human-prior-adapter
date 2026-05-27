# ZJU SMPL Prior Head/Hair Modal Status

## Scope

- Date: `2026-04-15`
- Goal: on the current repo version, finish the "human prior for face/hair detail + point-cloud completeness + Modal A100 training" task and leave a runnable cloud job with evidence.
- Current status at last check: an active detached A100 run exists again under app `ap-HrGYz8OsqwhkQfVpowrW07` and is writing resumed training logs.

## Code Landed

- `training/data/datasets/zju_vggt_geom.py`
  - Loads ZJU SMPL-related vertices from `new_vertices` / `vertices`
  - Projects SMPL vertices into each selected view
  - Emits:
    - `smpl_prior_masks`
    - `smpl_prior_feature_maps`
    - `human_prior_completion_masks`
    - `head_hair_region_masks`
    - `head_hair_detail_masks`
- `training/data/composed_dataset.py`
  - Forwards the above tensors into the training sample dict
- `training/loss.py`
  - Adds human-prior-aware depth and unproject-geometry weighting / completion constraints
  - Supports:
    - `human_prior_mask_key`
    - `human_prior_feature_map_key`
    - `human_prior_mask_floor`
    - `human_prior_scale`
    - `human_prior_reg_scale`
    - `human_prior_conf_scale`
    - `human_prior_conf_floor_weight`
    - `human_prior_conf_floor`
    - `human_prior_depth_presence_weight`
    - `human_prior_train_only`
    - `human_prior_anchor_view_only`
- `training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_smplprior_headhair_longrun.yaml`
  - Enables head/hair detail emphasis and completeness support using SMPL prior tensors
- `scripts/run_modal_zju_geometry_smplprior_headhair_longrun.ps1`
  - A100 long-run launcher for this setup
  - Supports `-DisableCompile`
  - Pins Modal app name to `vggt-zju-geometry-smplprior-headhair`
- `scripts/run_modal_zju_geometry_minimal_finetune.ps1`
  - Respects `-ModalAppName` so active-app checks and launches target the correct Modal app description
- `training/trainer.py`
  - Fixes checkpoint resume compatibility for the current optimizer wrapper shape
  - Loads optimizer state correctly when `construct_optimizers()` returns a list
  - Resumes from `prev_epoch` when older checkpoints store the completed epoch under that key instead of `epoch`

## Validation Completed In This Session

- `py_compile` passed for:
  - `training/data/datasets/zju_vggt_geom.py`
  - `training/data/composed_dataset.py`
  - `training/loss.py`
  - `modal_zju_geometry_minimal_finetune.py`
  - `training/trainer.py`
- Dataset smoke confirmed the new prior tensors exist and are non-zero on sample data.
- Loss smoke passed for the real training-style depth loss path with `gradient_loss_fn='grad+conf'`.
- Resume smoke passed after the `training/trainer.py` fix:
  - model weights loaded from the saved `checkpoint.pt`
  - optimizer state loaded without the previous `AttributeError`
  - training resumed at `Train Epoch: [1]` and started writing new training steps again

## Modal Run History

### 1. First compile-based attempt

- App ID: `ap-VUOzxad2fLC9FwsQ7lPjsN`
- App name: `vggt-zju-geometry-smplprior-headhair`
- Created at: `2026-04-15 13:35:17 +08:00`
- Stopped at: `2026-04-15 13:55:42 +08:00`
- Output root: `/zju_smplprior_headhair/20260415_smplprior_headhair_longrun`

Observed behavior:

- Dataset probe and log directories were created successfully.
- `torch.compile` / `inductor` kept producing long fallback-related output.
- TensorBoard event file stayed effectively empty and no first training step was written.
- To avoid wasting A100 wall-clock on compile-path stalls, this run was intentionally stopped and replaced with an eager rerun.

### 2. First eager A100 rerun

- App ID: `ap-Xo3EJEgCqlCeu0jvBI4Fg8`
- App name: `vggt-zju-geometry-smplprior-headhair`
- Created at: `2026-04-15 13:55:32 +08:00`
- Stopped at: `2026-04-15 14:21:34 +08:00`
- Output root: `/20260415_smplprior_headhair_longrun_eager`

Evidence that was pulled locally before the interruption:

- `driver_live.log` existed and was growing
- `logs/log.txt` contained real training-step lines
- TensorBoard event file existed under `/20260415_smplprior_headhair_longrun_eager/logs/tensorboard`
- the local mirror captured progress through approximately `Train Epoch: [0][750/1000000]`
- at that point, GPU memory was around `83.0 GB`

### 3. Stable eager `shmsafe` rerun

- App ID: `ap-GiX8Z0mb5C9sJfkQf7wC3q`
- App name: `vggt-zju-geometry-smplprior-headhair`
- Created at: `2026-04-15 14:31:47 +08:00`
- Stopped at: `2026-04-15 15:20:04 +08:00`
- Output root: `/20260415_smplprior_headhair_longrun_eager_shmsafe`

Evidence recovered after reconnecting:

- The run wrote real training and validation logs through the output volume.
- Checkpoints were saved successfully:
  - `/20260415_smplprior_headhair_longrun_eager_shmsafe/ckpts/checkpoint.pt`
  - `/20260415_smplprior_headhair_longrun_eager_shmsafe/ckpts/checkpoint_0.pt`
- Those checkpoints were last modified at `2026-04-15 14:57 +08:00`.
- The final visible training lines pulled from `logs/log.txt` reached:
  - `2026-04-15 07:19:57` inside the remote log stream
  - approximately `Train Epoch: [1][518/1000000]`
  - GPU memory around `80.0 GB`

Interpretation:

- The recovered tail showed no training-side Python traceback before the run stopped.
- Modal marked the app as `stopped`, not `failed`.
- Inference: the run appears to have been stopped externally or operationally after proving stable resumed training, rather than dying from an in-process model error.

### 4. First resume attempt from `checkpoint.pt`

- App ID: `ap-MhZx7QRBpC5mmDVZboBJ3M`
- App name: `vggt-zju-geometry-smplprior-headhair`
- Created at: `2026-04-15 16:33:04 +08:00`
- Stopped at: `2026-04-15 16:34:29 +08:00`
- Output root: `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume1`

Failure reason:

- The resume path itself was valid and the checkpoint was read successfully.
- Model weights loaded cleanly from:
  - `/mnt/out/20260415_smplprior_headhair_longrun_eager_shmsafe/ckpts/checkpoint.pt`
- The run failed during optimizer restore with:
  - `AttributeError: 'list' object has no attribute 'optimizer'`
- Root cause: `training/trainer.py` assumed a single optimizer wrapper on resume, but the current trainer stores optimizers as a list and checkpoints already save them in list-compatible form.

Action taken:

- Patched `training/trainer.py` to restore optimizer state correctly for list-based optimizer wrappers.
- Patched the same resume path to honor `prev_epoch` from saved checkpoints.

### 5. Current active resumed run

- App ID: `ap-HrGYz8OsqwhkQfVpowrW07`
- App name: `vggt-zju-geometry-smplprior-headhair`
- State at last check: `ephemeral (detached)`
- Tasks at last check: `1`
- Created at: `2026-04-15 16:39:35 +08:00`
- Output root: `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2`
- Resume checkpoint: `/mnt/out/20260415_smplprior_headhair_longrun_eager_shmsafe/ckpts/checkpoint.pt`

Current evidence from live volume pulls:

- The run cleared the previous failure point:
  - `Model state loaded. Missing keys: None. Unexpected keys: None.`
  - `Loading optimizer state dict (rank 0)`
  - `Single-process mode enabled; leaving model unwrapped by DDP.`
- The resumed run is writing new training steps again under `Train Epoch: [1]`.
- At the latest pulled point, it had reached approximately:
  - remote log time `2026-04-15 08:42:37`
  - `Train Epoch: [1][6/1000000]`
  - GPU memory `82.0 GB`
- TensorBoard output exists under:
  - `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/logs/tensorboard`

## Evidence Paths

### Local mirrors already present

- Compile attempt:
  - `output/modal_logs/20260415_smplprior_headhair/driver_live_compile.log`
  - `output/modal_logs/20260415_smplprior_headhair/dataset_probe_summary.md`
- Earlier eager pulls:
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/log_eager.txt`
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/driver_live_eager.log`
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/log_eager_shmsafe.txt`
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/driver_live_eager_shmsafe.log`
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/log_eager_shmsafe_latest.txt`
  - `output/modal_logs/20260415_smplprior_headhair/remote_pull/driver_live_eager_shmsafe_latest.log`

### Active remote output root

- `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/driver_live.log`
- `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/logs/log.txt`
- `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/logs/tensorboard/events.out.tfevents...`
- `/20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/dataset_probe/summary.md`

## Notes

- In the sampled windows so far, `loss_human_prior_conf_floor` and `loss_human_prior_depth_presence` remain logged as `0.0000`.
- This does not mean the human prior path is absent:
  - the SMPL prior masks and feature maps are present in the dataset path
  - depth and unproject losses are still being reweighted through the human-prior scale maps
- The explicit floor/presence auxiliary terms simply did not activate on the sampled batches pulled for this report.
- The local Windows shell can emit a harmless trailing error while streaming Modal volume content:
  - `'gbk' codec can't encode character '\u2713' ...`
  - this appears after the requested log content is already printed and does not indicate a remote training failure
- Capacity warning still applies on Modal:
  - `/mnt/data` inode usage was reported around `84.9%`
  - `/mnt/out` inode usage was reported around `91.3%`
  - future long runs may fail to create files if inode pressure keeps rising
