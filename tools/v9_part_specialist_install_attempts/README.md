# V9 Part Specialist Real Dependency Landing Attempts

Date: 2026-05-07
Workspace: `D:\vggt\vggt-main`

Scope honored:

- No mainline training files were modified.
- No procedural tubes, dots, teacher, candidate, predictions, or strict-pass artifacts were generated.
- Writes were limited to this directory, plus pre-existing external dependency trees under `external/`.

## Executive Result

Real B-hand11 and B-hair4 modules are not landed as usable VGGT part specialists yet.

- B-hand11: `external/HGGT-main` is present, but the public HGGT checkout only contains README, media, and a synthetic data generation pipeline. It explicitly marks pretrained checkpoints, model inference code, and demo scripts as unreleased. No hand token decoder module, inference entrypoint, or checkpoint is present.
- B-hair4: `external/hair-gs-master` is present and contains real Hair-GS topology code, including `HairGaussianModel`, endpoint pairs, strand roots, loaders, training, merge, eval, and conversion entrypoints. It is not usable yet on this machine because the required environment, CUDA extension stack, FLAME/data assets, and PyTorch3D/manual dependencies are missing.

## Local Trees Found

- `external/HGGT-main`
- `external/hair-gs-master`
- `external/HGGT-main.zip`
- `external/hair-gs-master.zip`

No model/checkpoint assets were found under either external tree:

```powershell
Get-ChildItem -Recurse -File external\HGGT-main,external\hair-gs-master -Include *.pt,*.pth,*.ckpt,*.safetensors,*.onnx,*.npz,*.pkl,*.ply,*.obj,*.task,*.bin,*.tar
```

## B-hand11 HGGT / Hand Token Decoder

### Real Entry Points Present

- `external/HGGT-main/README.md`
- `external/HGGT-main/synthetic_pipeline/download_objaverse.py`
- `external/HGGT-main/synthetic_pipeline/graspxl_dataloader.py`
- `external/HGGT-main/synthetic_pipeline/render_multiprocess.py`
- `external/HGGT-main/synthetic_pipeline/utils/mano_utils.py`

### Positive Evidence

Syntax checks passed for the checked-in HGGT synthetic-pipeline code:

```powershell
.\.venv313\Scripts\python.exe -m py_compile `
  external\HGGT-main\synthetic_pipeline\download_objaverse.py `
  external\HGGT-main\synthetic_pipeline\graspxl_dataloader.py `
  external\HGGT-main\synthetic_pipeline\render_multiprocess.py `
  external\HGGT-main\synthetic_pipeline\utils\mano_utils.py `
  external\HGGT-main\synthetic_pipeline\utils\mesh_utils.py `
  external\HGGT-main\synthetic_pipeline\utils\objaverse.py `
  external\HGGT-main\synthetic_pipeline\utils\transform.py
```

Result: `py_compile_exit=0`.

`mano_utils` can import in existing torch-capable conda environments, but the full dataloader fails before construction:

```powershell
conda run -n g3splat python -c "import sys; sys.path.insert(0, r'D:/vggt/vggt-main/external/HGGT-main/synthetic_pipeline'); import utils.mano_utils as m; print('mano_utils ok', hasattr(m, 'seal_mano_mesh')); import graspxl_dataloader as g; print('graspxl ok', hasattr(g, 'GraspXLDataloader'))"
```

Observed result: `mano_utils ok True`, then `ModuleNotFoundError: No module named 'rootutils'`.

### Hard Gaps

- HGGT README marks these as unreleased:
  - pretrained model checkpoints
  - model inference code
  - demo scripts
- The local HGGT checkout has no model package, no hand token decoder, no checkpoint loader, no inference script, and no InterHand/HO3D/ARCTIC entrypoint.
- The present code is a synthetic generation pipeline around GraspXL, Objaverse, MANO, DART textures, and Blender rendering, not a real HGGT inference decoder.
- Required local assets are absent:
  - `external/HGGT-main/assets/mano_v1_2/models`
  - `external/HGGT-main/assets/handy/models/Right_Hand_Shape.pkl`
  - `external/HGGT-main/assets/graspxl_obj_ids.txt`
  - `external/HGGT-main/data/hggt_synthetic`
- The working `.venv313` lacks `torch`; existing conda envs lack at least `rootutils` and `smplx` for the full HGGT synthetic-pipeline import.

### Repro Commands

```powershell
# Source truth: confirms released/unreleased state in README.
Select-String -Path external\HGGT-main\README.md `
  -Pattern 'Release pretrained model checkpoints|Release model inference code|Release demo scripts|snapshot_download'

# Dependency import probe.
conda run -n g3splat python -c "import sys; sys.path.insert(0, r'D:/vggt/vggt-main/external/HGGT-main/synthetic_pipeline'); import utils.mano_utils as m; print('mano_utils ok', hasattr(m, 'seal_mano_mesh')); import graspxl_dataloader as g; print('graspxl ok', hasattr(g, 'GraspXLDataloader'))"

# Local asset probe.
Test-Path -LiteralPath external\HGGT-main\assets\mano_v1_2\models
Test-Path -LiteralPath external\HGGT-main\assets\handy\models\Right_Hand_Shape.pkl
Test-Path -LiteralPath external\HGGT-main\assets\graspxl_obj_ids.txt
Test-Path -LiteralPath external\HGGT-main\data\hggt_synthetic
```

## B-hair4 HairGS / Root-Strand Topology

### Real Entry Points Present

- `external/hair-gs-master/train.py`
- `external/hair-gs-master/merge.py`
- `external/hair-gs-master/eval.py`
- `external/hair-gs-master/scripts/convert_output.py`
- `external/hair-gs-master/scripts/download_parse_cy.py`
- `external/hair-gs-master/data/hair_data.py`
- `external/hair-gs-master/data/eval_data.py`
- `external/hair-gs-master/scene/hair_gaussian_model.py`
- `external/hair-gs-master/submodules/diff-gaussian-rasterization`
- `external/hair-gs-master/submodules/simple-knn`

### Positive Evidence

Syntax checks passed for selected core Hair-GS files:

```powershell
.\.venv313\Scripts\python.exe -m py_compile `
  external\hair-gs-master\data\hair_data.py `
  external\hair-gs-master\data\eval_data.py `
  external\hair-gs-master\scene\hair_gaussian_model.py `
  external\hair-gs-master\train.py `
  external\hair-gs-master\merge.py `
  external\hair-gs-master\eval.py
```

Result: `py_compile_exit=0`.

The code contains real root/strand topology primitives:

- `HairData`
- `strand_root_idx`
- `HairEvalData`
- `HairGaussianModel`
- `endpoint_pairs`
- `compute_strands_info`
- merge/update operations in `scene/hair_gaussian_model.py`

### Hard Gaps

- Current `.venv313` lacks `torch`; import fails immediately.
- Existing conda envs do not have the full Hair-GS stack. The most complete observed env, `g3splat`, has `torch`, `torchvision`, `cv2`, and `plyfile`, but lacks:
  - `pytorch3d`
  - `smplx`
  - `OpenGL`
  - `pyvista`
  - `dreifus`
  - `diff_gaussian_rasterization`
  - `simple_knn`
- Hair-GS README requires Windows MSVC for PyTorch CUDA extension compilation. `cl` is not currently on PATH.
- The machine has CUDA 13.1, while Hair-GS says it is tested with CUDA 11.x and 12.x.
- `conda env create --dry-run -f external\hair-gs-master\environment_win.yml` failed due SSL transport error before solve completed.
- Manual data/model assets are absent:
  - `external/hair-gs-master/dataset/FLAME/flame2023.pkl`
  - `external/hair-gs-master/dataset/FLAME/flame_static_embedding.pkl`
  - `external/hair-gs-master/dataset/FLAME/flame_dynamic_embedding.npy`
  - `external/hair-gs-master/dataset/FLAME/FLAME_masks.pkl`
  - `external/hair-gs-master/dataset/raw/usc_hairsalon`
  - `external/hair-gs-master/dataset/raw/cem_yuksel`
  - `external/hair-gs-master/dataset/raw/nersemble`
  - parsed COLMAP-format datasets under `external/hair-gs-master/dataset/parsed`

### Repro Commands

```powershell
# Dependency availability in existing envs.
conda run -n g3splat python -c "import importlib.util as u; mods=['torch','torchvision','pytorch3d','smplx','cv2','plyfile','OpenGL','pyvista','dreifus','diff_gaussian_rasterization','simple_knn']; print({m: bool(u.find_spec(m)) for m in mods})"

# Hair-GS import probe.
conda run -n g3splat python -c "import sys; sys.path.insert(0, r'D:/vggt/vggt-main/external/hair-gs-master'); import data.hair_data as hd; print('hair_data ok', sorted(hd.hair_data_load_callbacks.keys())); import scene.hair_gaussian_model as hgm; print('hair_gaussian_model ok', hasattr(hgm, 'HairGaussianModel'))"

# Windows env solve attempt.
conda env create --dry-run -f external\hair-gs-master\environment_win.yml

# Required data/model assets.
Test-Path -LiteralPath external\hair-gs-master\dataset\FLAME\flame2023.pkl
Test-Path -LiteralPath external\hair-gs-master\dataset\raw\usc_hairsalon
Test-Path -LiteralPath external\hair-gs-master\dataset\raw\cem_yuksel
Test-Path -LiteralPath external\hair-gs-master\dataset\parsed\usc_hairsalon
Test-Path -LiteralPath external\hair-gs-master\dataset\parsed\cem_yuksel
```

## Probe Logs

- `env_probe_20260507_203638.log`
- `asset_scan_20260507_203638.log`
- `hggt_probe_20260507_203638.log`
- `hggt_external_data_probe_20260507_203721.log`
- `hggt_pip_dryrun_20260507_203857.log`
- `hairgs_probe_20260507_203638.log`
- `hairgs_external_data_probe_20260507_203722.log`
- `hairgs_conda_dryrun_20260507_203857.log`
- `conda_env_import_probe_20260507_203718.log`
- `repo_source_truth_20260507_203857.log`

## Minimum Unblockers

B-hand11 can only land as a real module after obtaining a real HGGT implementation or equivalent hand decoder package with:

- model inference code,
- hand token decoder / mesh decoder entrypoint,
- checkpoint,
- documented input/output contract for uncalibrated multiview hand images,
- MANO or non-MANO mesh parameter contract,
- optionally InterHand/HO3D/ARCTIC loaders if those are intended as supervised/eval sources.

B-hair4 can land after:

- a Python 3.10 env matching Hair-GS,
- MSVC on PATH,
- compatible CUDA 11/12 or confirmed CUDA 13 build fixes,
- PyTorch3D from source,
- built `diff_gaussian_rasterization`, `simple_knn`, and `c_utils`,
- FLAME files,
- at least one parsed USC-HairSalon or Cem-Yuksel sample with COLMAP cameras, images, masks, orientations, `hair_eval_data.npz`, and `head_reconstruction_data.npz`.
