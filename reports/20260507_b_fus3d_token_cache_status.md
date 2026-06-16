# 2026-05-07 B-Fus3D Token Evidence Cache Status

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
formal cloud train/infer/export = blocked
teacher export = blocked
candidate export = blocked
```

This is a local-only B-Fus3D diagnostic skeleton. It does not train, does not call
cloud, does not write strict pass state, does not export a teacher, and does not
generate a candidate. VGGT depth, point, and normal predictions are not used as
hard teachers.

## Implemented

Added:

```text
tools/b_fus3d_token_cache.py
```

Dry-run output written locally:

```text
output/surface_research_preflight_local/B_Fus3D0_token_cache_dryrun/
  b_fus3d_token_cache_summary.json
  token_layer_stats.json
  roi_coverage_placeholders.json
```

The JSON contract explicitly records:

```text
research_only = true
local_only = true
no_train = true
no_cloud = true
no_teacher_export = true
no_candidate_export = true
no_strict_pass_write = true
uses_vggt_depth_point_normal_as_hard_teacher = false
```

## Local VGGT Entrypoint Audit

Token extraction entrypoint is clear:

- `vggt/models/aggregator.py:213` defines `Aggregator.forward`.
- `vggt/models/aggregator.py:242` creates patch tokens through `self.patch_embed(images)`.
- `vggt/models/aggregator.py:302` concatenates camera/register/patch tokens.
- `vggt/models/aggregator.py:416` records frame-attention intermediates.
- `vggt/models/aggregator.py:460` records global-attention intermediates.
- `vggt/models/aggregator.py:345` concatenates frame/global intermediates with `dim=-1`.
- `vggt/models/aggregator.py:351` returns `output_list, self.patch_start_idx`.

`vggt/models/vggt.py:128` calls the aggregator, then sends
`aggregated_tokens_list` to the camera/depth/point/normal/track heads. The public
`VGGT.forward` return dict exposes predictions, not the intermediate tokens, so
the B-Fus3D cache calls `model.aggregator(images)` directly instead of modifying
VGGT source.

Existing predictions/scene organization is also clear:

- `tools/run_local_vggt_inference.py:139` expects `scene_dir/images`.
- `tools/run_local_vggt_inference.py:199` builds the saved array dict.
- `tools/run_local_vggt_inference.py:212` writes `predictions.npz`.
- `tools/run_local_vggt_inference.py:251` writes `summary.json`.

The scanned predictions payload contains `pose_enc`, `extrinsic`, `intrinsic`,
`depth`, `depth_conf`, `world_points`, `world_points_conf`, `normal`, and
`normal_conf`, but the cache only scans these headers for scene organization.
They are not treated as teacher evidence.

## Dry-Run Readout

Command run:

```powershell
python tools\b_fus3d_token_cache.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --predictions output\local_inference_results\r34_raw518_r27_on6v_fullbody\predictions.npz `
  --output-dir output\surface_research_preflight_local\B_Fus3D0_token_cache_dryrun `
  --view-indices 0,10 `
  --target-size 518 `
  --overwrite
```

Result:

```text
status = metadata_only
source entrypoint = clear
scene images = 60 png files
scene masks = 60 png files
scene_manifest exported views = 60
selected dry-run views = 0,10
selected image size = 518x518
```

Expected token layout placeholder:

```text
patch_size = 14
patch_grid = 37 x 37
patch_tokens = 1369
special_tokens = 5
token_count = 1374
expected_layer_count = 24
expected_layer_shape = [1, 2, 1374, 2048]
```

Local asset scan also found a candidate local VGGT-size checkpoint:

```text
C:\Users\WINDOWS\.cache\torch\hub\checkpoints\model.pt
size = 5026874952 bytes
```

The dry-run did not use it because extraction requires an explicit
`--checkpoint` path.

## Sufficient Now

- Local source is enough to define the token extraction interface.
- The 60-view human-crop scene has images, masks, `scene_manifest.json`,
  `camera_params_sidecar.npz`, and `prior_maps.npz`.
- Header-only predictions scan is enough to document the current
  `predictions.npz` organization without loading arrays as teacher targets.
- The script can run without GPU or VGGT weights and still writes the required
  diagnostic JSON skeleton.

## Still Missing

- A verified `--extract` run with an explicit local VGGT checkpoint and compatible
  runtime memory/device.
- ROI-to-aggregator-token coverage mapping for full body, face, left hand, right
  hand, hair, and matting.
- Per-ROI mask provenance strong enough for later hard-gate diagnostics.
- Any strict coordinate pass or Open3D visual pass. Therefore full-body/hands hard
  gates remain placeholders only.

## Minimal B-Fus3D0 Commands

Metadata/dry-run:

```powershell
python tools\b_fus3d_token_cache.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --predictions output\local_inference_results\r34_raw518_r27_on6v_fullbody\predictions.npz `
  --output-dir output\surface_research_preflight_local\B_Fus3D0_token_cache_dryrun `
  --view-indices 0,10 `
  --target-size 518 `
  --overwrite
```

Local extraction/cache, only if the checkpoint/runtime is available:

```powershell
python tools\b_fus3d_token_cache.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --predictions output\local_inference_results\r34_raw518_r27_on6v_fullbody\predictions.npz `
  --output-dir output\surface_research_preflight_local\B_Fus3D0_token_cache_extract `
  --checkpoint C:\Users\WINDOWS\.cache\torch\hub\checkpoints\model.pt `
  --extract `
  --save-token-arrays `
  --cache-layers last `
  --view-indices 0,10 `
  --target-size 518 `
  --overwrite
```

If extraction fails, the concrete blocker should be one of: checkpoint load
mismatch, unavailable CUDA/runtime memory, import failure, or no selected scene
images. The token entrypoint itself is not currently blocked.
