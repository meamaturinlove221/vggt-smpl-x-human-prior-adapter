# 2026-04-22 HumanCrop Mainline Bootstrap

## Bottom line

The new `human_crop` mainline is now materially more complete than before:

- crop scene variants now exist for `6 / 8 / 12 / 20 / 60`
- resumed-strongfusion crop inference outputs now exist for `6 / 8 / 12 / 20`
- crop training cases now exist for `6 / 8 / 12 / 20`
- dedicated crop-mainline configs and a Modal helper script now exist

This is still **not** the mentor-final endpoint yet. What is now closed is the
`human_crop as default sparse-view base` engineering path. The remaining gap is
still final face/head quality, not whether the crop branch can be generated,
trained, or evaluated.

## What was generated

### Crop scene variants

- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_8views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_12views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_20views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop`

### Resume-strongfusion crop inference outputs

- `output/modal_results/20260422_6views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_8views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_12views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_20views_humancrop_from_resume_strongfusion_r1`

Note:

- the direct Modal download path corrupted the local `6v` and `12v` `predictions.npz`
- these two runs were repaired with `modal_4k4d_vggt_infer.py::download_prediction_chunks_rpc`
- all four local `predictions.npz` files are now readable and valid

### Crop training cases

- `output/training_cases/0012_11_frame0000_6views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_8views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_12views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_20views_sparseproto_humancrop_resume_r1`

### New configs / launcher

- `training/config/4k4d_prior_case_sparseproto_humancrop_resume_r1.yaml`
- `training/config/4k4d_prior_case_6view_focus_humancrop_resume_r1.yaml`
- `scripts/run_modal_4k4d_humancrop_sparseproto_strongfusion.ps1`

## ROI summary from the resumed strongfusion checkpoint

ROI counts below come from `modal_4k4d_vggt_infer.py::summarize_prediction_roi`
with `conf_percentile = 40.0`.

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `6v full` | `40,880` | `8,994` | `3,673` |
| `6v human_crop` | `111,082` | `24,438` | `9,634` |
| `8v human_crop` | `131,723` | `28,979` | `11,144` |
| `12v human_crop` | `210,913` | `46,401` | `17,395` |
| `20v human_crop` | `331,924` | `73,024` | `26,425` |

## Immediate reading

For the current resumed-strongfusion checkpoint:

- switching from `6v full` to `6v human_crop` lifts
  - full retained points from `40,880 -> 111,082`
  - head ROI from `8,994 -> 24,438`
  - face ROI from `3,673 -> 9,634`
- that means the crop base is already giving a large occupancy win before any new crop-specific training
- the new bottleneck is no longer "do we have a crop mainline?" but "does crop-trained geometry become visibly sharper at face/head ROI?"

## What is still not done

- no new crop-mainline training result has been validated yet from these newly generated crop cases
- no mentor-final claim should be made from the ROI counts alone
- `projected targetpatch` remains a negative-result branch
- `single-case preprocess overfit` remains a negative-result branch

## Recommended next action

Use the new crop launcher to run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_modal_4k4d_humancrop_sparseproto_strongfusion.ps1 -Mode family -RunEval
```

If a faster first check is needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_modal_4k4d_humancrop_sparseproto_strongfusion.ps1 -Mode focus6 -RunEval
```

The truthful current project line is now:

> `human_crop` has been promoted from a 6-view ablation into a real sparse-view mainline base, but the mentor-final quality gate still depends on the next crop-trained geometry result.
