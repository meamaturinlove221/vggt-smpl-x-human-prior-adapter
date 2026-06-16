# A-Line Dense Reconstruction Preflight Stub

Status: `stub_implemented_research_artifact_only`

This is A-line dense teacher reconstruction research-preflight only. It did not
run dense reconstruction, did not export a teacher, did not export a candidate,
did not write a strict pass, and did not call formal VGGT cloud train/infer/export.

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
no_teacher_export = true
no_candidate_export = true
formal_vggt_cloud_train_infer_export = not_called
```

## Added Tool

```text
tools/a_line_dense_reconstruction_preflight_stub.py
```

Purpose:

```text
audit raw RGB/mask/known-camera/same-frame scene assets
bind an optional A3 visual-hull mesh seed as initialization only
probe A1/A2 runtime dependencies without importing heavy backends
write JSON/Markdown dry recipes only
```

The tool is intentionally a stub/wrapper plan. It does not call NeuS, 2DGS,
GS rasterizers, VGGT inference, VGGT training, export tools, or strict gate tools.

## Scene And Artifact Audit

Checked scene:

```text
output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop
```

Available assets:

```text
scene_manifest.json
camera_params_sidecar.npz
images/
masks/
```

Checked A3 seed:

```text
output/surface_research_preflight/A3_visual_hull_mesh_t96_g56_s4/A3_visual_hull_init
```

Seed metadata:

```text
mesh_vertices = 14208
mesh_faces = 28586
selected_views = 0,10,20,30,40,50
target_size = 96
grid_resolution = 56
```

The A3 seed is only a research initialization reference. It is not a teacher,
candidate, or pass signal.

## Verified Local Stub Run

Command:

```powershell
python tools\a_line_dense_reconstruction_preflight_stub.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --a3-seed-dir output\surface_research_preflight\A3_visual_hull_mesh_t96_g56_s4\A3_visual_hull_init `
  --output-dir output\surface_research_preflight_local\A1_A2_dense_stub_t96_a3seed `
  --view-indices 0,10,20,30,40,50 `
  --target-size 96 `
  --overwrite
```

Outputs:

```text
output/surface_research_preflight_local/A1_A2_dense_stub_t96_a3seed/a_line_dense_reconstruction_preflight_stub.json
output/surface_research_preflight_local/A1_A2_dense_stub_t96_a3seed/a_line_dense_reconstruction_preflight_stub.md
output/surface_research_preflight_local/A1_A2_dense_stub_t96_a3seed/a_line_dense_reconstruction_recipe.json
```

No mesh, prediction, teacher, candidate, strict registry, or training-case output
was written by this stub.

Key verified values:

```text
status = a_line_dense_reconstruction_preflight_stub_complete_not_reconstruction
asset_ready_for_a1_a2_research_wrapper = true
camera_source = camera_params_sidecar
camera_numeric_ok = true
selected_views = [0, 10, 20, 30, 40, 50]
mask_coverage_min = 0.0858289931
mask_coverage_mean = 0.1033166956
mask_coverage_max = 0.1278211806
a3_seed_ready_for_initialization = true
a3_seed_mesh_vertices = 14208
a3_seed_mesh_faces = 28586
strict_candidate_passes = 0
strict_teacher_passes = 0
formal_vggt_cloud_train_infer_export_called = false
```

Dependency probe:

```text
A1_neural_sdf: wrapper dependencies present for a local dry research wrapper
A2_gaussian_surface: blocked locally because neither gsplat nor diff_gaussian_rasterization is installed
nvdiffrast/open3d/trimesh/tinycudann/nerfstudio: not present locally
```

## A1/A2 Wrapper Plan

A1 minimal next implementation:

```text
input = raw RGB + mask + aligned known cameras + same-frame view set + optional A3 bbox/mesh seed
initialize = A3 mesh bbox / occupied surface as research-only SDF volume prior
run = future isolated A-line research backend only
output = research artifact directory only
stop = before teacher/candidate export and before strict pass accounting
```

A2 minimal next implementation:

```text
input = raw RGB + mask + aligned known cameras + same-frame view set + optional A3 mesh seed
initialize = sample research-only surfel/Gaussian carriers from A3 seed
run = future isolated A-line research backend only after gsplat or diff_gaussian_rasterization is available
output = research artifact directory only
stop = before teacher/candidate export and before strict pass accounting
```

## Blockers / Next Step

```text
A1: stub is ready; actual dense reconstruction backend is not implemented in this turn.
A2: local runtime is missing gsplat/diff_gaussian_rasterization, so only the wrapper recipe is ready.
Review: no Open3D full/head/face/hairline/hands review exists for any A1/A2 dense output because no dense output was produced.
Gate: strict_candidate_passes and strict_teacher_passes remain zero.
```

Do not pivot back to r-number tuning, VGGT shell recovery, Kinect/TSDF/signfix
teacher routes, support/threshold loops, or formal VGGT cloud train/infer/export
while strict passes remain zero.
