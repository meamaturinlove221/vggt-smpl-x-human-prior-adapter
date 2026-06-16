# V403 Agent G-Drive Dataset Inventory

Workspace: `D:\vggt\vggt-main`  
Mode: read-only/data inventory  
Outputs written: this Markdown report and `reports/V403_agent_gdrive_dataset_inventory.json`

No training was launched. No dataset files were deleted, moved, or modified. The only intended writes are the two V403 report files under `reports`.

## Usable Roots

| Root | Status | V50 rebuild readiness |
|---|---:|---|
| `G:\数据集\datasets\data_used_in_4K4D` | present | Ready raw 4K4D/DNA-derived subset root |
| `G:\数据集\datasets\data_used_in_4K4D\main\0012_11.smc` | present | Ready for RGB frames `0000/0001/0002` |
| `G:\数据集\datasets\data_used_in_4K4D\annotations\0012_11_annots.smc` | present | Ready for masks, cameras, SMPL-X params |
| `G:\数据集\datasets\data_used_in_4K4D\kinect\0012_11_kinect.smc` | present | Available for Kinect/temporal auxiliary rebuilds |
| `G:\数据集\datasets\smplx` | present | Ready SMPL-X model root |
| `G:\数据集\datasets\ZJU_MoCap\data\zju_mocap\CoreView_390` | present | Ready ZJU geometry/prior validation root |
| `F:\datasets\ZJU_MoCap\data\zju_mocap\CoreView_390` | present | Ready mirror |
| `F:\vggt\_zju_ascii_link` | present junction | Useful ASCII route to the G-drive ZJU root |
| `F:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_fullviews` | present | Usable full-view extracted reference scene |

## 4K4D / DNA Subset

Primary root: `G:\数据集\datasets\data_used_in_4K4D`

The local ReadMe says this folder contains files used in 4K4D and is not part of the official DNA-Rendering release. It is still the strongest local raw source for V50 rebuild because it has the extracted SMC files and manifests.

Observed contents:

- 8 main SMCs: `0012_11`, `0013_01`, `0013_03`, `0013_09`, `0013_11`, `0019_08`, `0021_03`, `0023_06`
- 8 annotation SMCs
- 8 Kinect SMCs
- 7 A-pose main SMCs and 7 A-pose Kinect SMCs
- 8 preview MP4s
- `data_used_in_4K4D_file_gid.json` with 48 expected entries
- `data_used_in_4K4D_rgb_cams.zip` with 8 RGB camera SMC members

I did not find a standalone official DNA-Rendering root under the scanned D/F/G roots. The useful DNA-related root here is the local 4K4D subset.

## `0012_11` Frame Readiness

`G:\数据集\datasets\data_used_in_4K4D\main\0012_11.smc`:

- `Camera_5mp`: 48 cameras
- `Camera_12mp`: 12 cameras
- sampled color frame keys include `0`, `1`, `2`
- sampled per-camera color frame count: 150

`G:\数据集\datasets\data_used_in_4K4D\annotations\0012_11_annots.smc`:

- `Camera_Parameter`: 60 cameras with `K`, `D`, `RT`, `Color_Calibration`
- `Mask`: 60 cameras
- sampled mask frame keys include `0`, `1`, `2`
- sampled per-camera mask frame count: 150
- `SMPLx` arrays: `betas (150,10)`, `expression (150,10)`, `fullpose (150,55,3)`, `transl (150,3)`, plus scalar `scale`

That is enough raw evidence for rebuilding `0012_11` frame0000/frame0001/frame0002 image, mask, camera, and SMPL-X prior inputs.

## SMPL-X Models

Primary model root: `G:\数据集\datasets\smplx`

Found:

- `SMPLX_NEUTRAL.npz` and `SMPLX_NEUTRAL.pkl`
- `SMPLX_MALE.npz` and `SMPLX_MALE.pkl`
- `SMPLX_FEMALE.npz` and `SMPLX_FEMALE.pkl`
- compact `smplx_npz\SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`

This root is ready for SMPL-X model loading.

## ZJU Roots

`G:\数据集\datasets\ZJU_MoCap\data\zju_mocap\CoreView_390` and `F:\datasets\ZJU_MoCap\data\zju_mocap\CoreView_390` both look complete enough for local validation:

- 23 camera directories
- 23 `mask` camera folders
- 23 `mask_cihp` camera folders
- 23 `keypoints2d` camera folders
- `params`, `new_params`, `vertices`, `new_vertices`: 1171 files each
- `annots.npy`, `annots_python2.npy`, `intri.yml`, `extri.yml`, `match_info.json` present

`F:\vggt\_zju_ascii_link` is a junction to `G:\数据集\datasets\ZJU_MoCap\data\zju_mocap`; use it if a local loader has trouble with Chinese paths.

## Existing D/F Project Outputs

Important distinction: the G-drive raw dataset is usable; several restored D-workspace derivative folders are not usable as-is.

`D:\vggt\vggt-main\output\4k4d_scenes` has matching scene directory names for `0012_11_frame0000`, `frame0001`, and `frame0002`, but sampled folders had no image/mask files and no `scene_manifest.json`.

`D:\vggt\vggt-main\output\training_cases` has 123 matching `0012_11_frame0000` directory names. Counts by view label: 6-view 82, 7-view 7, 8-view 8, 12-view 8, 13-view 7, 20-view 8, 60-view 3. Sampled V50-relevant case directories were empty or missing manifests/packages, so treat these as naming/layout hints, not ready training cases.

`F:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_fullviews` is a usable reference scene: 60 images, 60 masks, `scene_manifest.json`, RGB contact sheet, and mask contact sheet.

`F:\vggt\vggt-main\output\4k4d_fullview_rerun_20260416\modal_results\0012_11_frame0000_fullviews` has full-view preview/point-cloud result artifacts, including fused raw/masked PLYs and point-cloud summaries. `predictions.npz` was not present in that folder.

## V50 Rebuild Readiness

Overall: ready to rebuild from raw G-drive data; do not rely on the current D-workspace derivative shells as completed V50 inputs.

Ready inputs:

- raw 4K4D subset at `G:\数据集\datasets\data_used_in_4K4D`
- `0012_11` RGB/mask/camera/SMPL-X/Kinect SMCs
- SMPL-X model root at `G:\数据集\datasets\smplx`
- ZJU validation roots on G and F
- F-drive full-view extracted reference scene

Current caveats:

- `D:\vggt\vggt-main\output\frozen_candidates\V50_smplx_native_candidate_pass` exists but currently has zero files.
- `D:\vggt\vggt-main\output\surface_research_preflight_local\V50_final_promotion_transaction` exists but currently has zero files.
- `D:\vggt\vggt-main\output\frozen_candidates\V50R_rebuilt_after_artifact_loss` has `manifest.json` and `hash_manifest.json`.
- `D:\vggt\vggt-main\archive\V223_rebuilt_candidate_package.zip` exists and is about 505 MB with rebuilt package files, but its member names are `candidate_*_from_v42.npz` / `v16_*` rather than the original `candidate_files__*.npz` V50 naming.

Recommended rebuild source order:

1. Use `G:\数据集\datasets\data_used_in_4K4D` plus `G:\数据集\datasets\smplx` for the canonical rebuild.
2. Use `F:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_fullviews` as a reference full-view extracted scene.
3. Use `F:\vggt\_zju_ascii_link` for ZJU validation if non-ASCII paths fail.
4. Treat D-workspace `output\4k4d_scenes`, `output\training_cases`, and original V50 candidate directories as incomplete until regenerated or restored from an archive.
