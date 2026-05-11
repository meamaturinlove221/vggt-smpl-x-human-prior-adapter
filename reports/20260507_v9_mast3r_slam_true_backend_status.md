# V9 MASt3R-SLAM True Backend Status

Status: `true_backend_completed_research_only_degenerate_geometry`

Verdict: official MASt3R-SLAM true backend smoke passed. It is not a usable geometry pass: the saved reconstruction PLY has `element vertex 0`.

## Artifacts

- Summary: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\summary.json`
- Run report: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\report.md`
- Trajectory: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\mast3r_slam_logs\input_png_sequence.txt`
- Reconstruction: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\mast3r_slam_logs\input_png_sequence.ply`
- Keyframe: `D:\vggt\vggt-main\output\surface_research_cloud_preflight\Cloud_B_V9\mast3r_slam_true_backend_smoke\mast3r_slam_logs\keyframes\input_png_sequence\0.0.png`

## Verified

- Official repo cloned with submodules.
- `mast3r`, `in3d`, and `MASt3R-SLAM` installed.
- Import probe passed: `torch 2.5.1+cu124`, CUDA `12.4`, CUDA available `True`.
- All three official Naver checkpoints downloaded:
  - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
  - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
  - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`
- 4 staged 4K4D PNGs were flattened from the remote Modal asset layout.
- Official `main.py --dataset <png-folder> --config config/base.yaml --save-as v9_mast3r_slam_true_backend_smoke --no-viz` returned `0`.

## Runtime Evidence

`main.py` loaded the MASt3R model and retrieval model, attempted relocalization, then completed:

```text
RELOCALIZING against kf  1  and  [0]
Failed to relocalize
Skipped frame 1
done
```

## Geometry Verdict

The backend produced its own logs, but they are degenerate:

- `input_png_sequence.txt`: one identity pose line.
- `input_png_sequence.ply`: `element vertex 0`.
- `keyframes/input_png_sequence/0.0.png`: one exported keyframe image.

No predictions, teacher, candidate, registry, or strict-pass artifact was written.

## Unblocker

The worker is not blocked for true backend decision smoke. The only remaining unblocker for non-empty geometry is input semantics: use a MASt3R-SLAM-compatible ordered RGB sequence, or a calibrated run matched to that sequence assumption. Do not spend more attempts on dependency or checkpoint fixes.
