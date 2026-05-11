# A5 External Dense Backend Adapter

Status: `adapter_contract_added_not_backend_success`

The A5 lane is a research-only contract for using an external same-frame dense
backend without repeating the old model-name chasing failure mode.

Added:

```text
tools/a5_external_dense_backend_preflight.py
```

Contract:

```text
input = raw same-frame scene with known cameras and masks
backend may provide = one shared 3D mesh, or one consistent multi-view depth set
forbidden = per-view unrelated patches, numeric-only coverage, visual-hull shell
required before teacher export = Open3D full/head/face/hairline/hands strict teacher gate
```

The adapter writes only research summaries and workspace manifests. It does not
call formal VGGT train/infer/export, does not export a teacher/candidate, and
does not write strict pass state.

Example no-backend smoke:

```powershell
python tools\a5_external_dense_backend_preflight.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --output-dir output\surface_research_preflight_local\A5_external_dense_adapter_contract `
  --backend custom `
  --view-indices 0,10,20,30,40,50 `
  --eval-view-indices 5,15 `
  --target-size 96 `
  --overwrite
```

Expected no-backend status:

```text
blocked_no_backend_output
```

This is the correct state until a real external dense backend returns one shared
3D surface or a calibrated consistent depth set.
