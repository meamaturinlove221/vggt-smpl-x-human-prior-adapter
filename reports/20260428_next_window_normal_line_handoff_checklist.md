# 2026-04-28 next-window handoff: normal-only mentor checklist

This document is the migration packet for the next Codex conversation. It must be treated as the authoritative handoff for the current work. The active task is **normal line only**.

## 0. Non-negotiable operating rules

- Work directory: `D:\vggt\vggt-main`.
- Shell: PowerShell inside Codex only. Do **not** open external Windows terminals or GUI windows.
- Default every `shell_command` with `login:false`.
- Prefix Python / Modal commands:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
  ```
- Local Open3D Python:
  ```powershell
  D:\anaconda\envs\g3splat\python.exe
  ```
- For Open3D / NumPy-heavy local jobs, also set:
  ```powershell
  $env:KMP_DUPLICATE_LIB_OK='TRUE'
  ```
- If spawning agents, every agent must be `gpt-5.5` with `xhigh` reasoning. Do not use any other model.
- Use `apply_patch` tool for file edits. Do not run `apply_patch` through shell.
- Keep process hygiene. Check and clean only known rogue `python.exe`, `modal.exe`, `colmap.exe` jobs; do not kill Codex or unrelated WPS processes blindly.
- Do not claim "mentor-final passed" unless both:
  - quantitative same-protocol metrics pass;
  - Open3D / point-cloud visuals show real modeled face/head/full-body geometry, not only higher point count or a front-view texture shell.

## 1. Current scope decision

The mentor has explicitly paused other branches and asked to continue **normal** only.

Active scope:

- normal-depth-point geometry coupling;
- SMPL-X prior use, but not overreliance;
- multi-view deployment / evaluation;
- full-body point cloud and paired figures;
- face/head/hairline quality as the main bar;
- full-body integrity as the minimum bottom line.

Paused / do not spend mainline time on:

- projected targetpatch / summary-token patch;
- generic teacher fusion unless it directly supports normal-depth-point geometry;
- DepthPro / PSHuman / COLMAP / Kinect lines already judged negative unless a new gate clearly explains why it is not repeating prior failure;
- large sparse-view end-to-end training before a normal gate shows actual geometry improvement;
- report-only polishing that hides unresolved geometry failure.

## 2. Truthful current status

The normal line has **not** reached mentor-final quality.

Current reference best for the original 6-view headshoulder protocol remains:

- result: `output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz`
- scene: `output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop`
- face ROI: `16825`
- head ROI: `40527`
- full ROI: `184213`
- conf p40: `38.5067`

Known visual truth:

- the reference is still shell-like and not mentor-final;
- later candidates have not produced a clearly modeled face with eyes/nose/mouth/head structure;
- normal consistency improvements so far have not reliably translated into target-view point-cloud quality.

Do not say:

- "normal line passed";
- "6-view face/head reached mentor bar";
- "higher ROI count alone proves success";
- "front-only crop proves geometry";
- "SMPL-X coarse prior normal is high-quality predicted normal";
- "TSDF / confidence calibration / r2 / r9 / r10 / r11 / r12 / r13 / r15 is final".

## 3. Mentor's latest technical commands, normalized

### 3.1 Full-body point cloud first

Mentor asked:

- "全身效果我先看一下。"
- "你可以把点云结果发我一下，如果有最好配图一起发。"
- "不能一直只看上半身。"
- "全身几何结构也不能有明显遗漏，比如手断、身体大洞。"

Execution requirement:

- Always provide a full-body point cloud file and paired figures for any candidate.
- Face/head is the main target, but full-body is the minimum sanity gate.
- A candidate cannot pass if full-body has major holes, broken limbs, severe shell, or missing hands/body.

Current available full-body package:

- folder: `output\normal_line_delivery_20260428\targetcam30_multiview_fullbody_pointclouds`
- zip: `output\normal_line_delivery_20260428\targetcam30_multiview_fullbody_pointclouds.zip`
- best inspection candidate inside current package:
  - `output\normal_line_delivery_20260428\targetcam30_multiview_fullbody_pointclouds\r2_16v_full_pointcloud_open3d.ply`
- truthful status: inspection package only, **not pass**.

### 3.2 Real-photo self-consistency of SMPL-X source

Mentor's concern:

- In 4K4D / DNA data, SMPL-X pose and camera are available.
- In real photos, there may be only images, no SMPL-X parameters and no accurate camera parameters.
- The method must explain how SMPL-X prior is obtained in real scenes.

Correct technical position:

- Dataset stage: use provided SMPL-X pose/shape/expression and camera parameters.
- Real-data stage: use external SMPL-X estimator/fitter or silhouette/keypoint fitting to recover pose-aligned SMPL-X.
- Repo currently supports external bundle import and bridge:
  - `tools\run_realdata_smplx_driver.py`
  - `tools\build_scene_prior_from_external_bundle.py`
- This is **external estimator/fitter + repo import/alignment/scene prior bridge**, not yet fully in-repo raw image to SMPL-X regression.

Do not claim:

- "real image to SMPL-X is fully solved inside this repo."

### 3.3 SMPL-X prior cannot be too strong

Mentor warning:

- SMPL-X template features can be too strong.
- Face can become template-like instead of person-specific.
- Skirts, loose clothing, hair, and non-template geometry can be erased.
- Strong SMPL-X prior may suppress RGB / multi-view evidence.

Correct method framing:

- SMPL-X provides pose-aligned human topology and coarse body surface prior.
- It must be balanced with RGB and multi-view geometry evidence.
- Normal branch should refine geometry and preserve personal/non-template details, not force the output back to the SMPL-X template.

Practical consequence:

- Do not keep increasing coarse SMPL normal teacher weight blindly.
- Avoid presenting SMPL-X coarse prior normal as final high-detail normal.
- Prefer self-geometry / image-aligned residual / multi-view consistency constraints over pure template supervision.

### 3.4 Normal must be coupled to depth and point map

Mentor's key requirement:

- VGGT predicts camera, depth, point maps.
- A DPT dense branch can output normal and depth.
- Depth can be converted to normal by local finite differences / local surface gradients.
- Point maps can also produce normals through local neighborhood cross products.
- Network-output normal should not be an isolated pretty 2D map.
- Normal must constrain depth and point geometry.

Correct technical objective:

- `normal branch` is not for normal-map visualization alone.
- It must improve final 3D geometry through:
  - depth-to-normal consistency;
  - point-to-normal consistency;
  - depth-to-point consistency;
  - cross-view reprojection consistency.

Existing code support:

- model normal output: `vggt\models\vggt.py`
- normal / point-normal / depth-normal / depth-point losses: `training\loss.py`
- new r16 cross-view loss added in `training\loss.py`
- configs r2-r16 under `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_*.yaml`

### 3.5 Crop remains a valid input-processing line

Mentor acknowledged:

- cropping human region, removing background, and making human occupy more of the 518 input is reasonable.
- Sparse-view face detail suffers when the person is too small in 518 input.

Correct framing:

- `human crop` is a preprocessing base / ablation, not the main structural innovation.
- It improves effective human pixels and reduces background clutter.
- Main innovations should remain:
  - SMPL-X pose-aligned conditioning;
  - normal-depth-point coupling;
  - ROI-first local detail refinement / evaluation.

### 3.6 Multi-view must be deployed and checked

Mentor asked to compare view counts:

- 3-view / 6-view / 13-view / 16-view / 60-view or similar.
- More views do not automatically mean better face detail.
- Sometimes 6-view may be concentrated while more views distribute points more evenly.

Execution requirement:

- For every promising normal variant, evaluate:
  - original same-protocol 6-view headshoulder crop;
  - targetcam30 3/6/13/16/60 if available;
  - full/head/face;
  - depth-unprojection and world-points;
  - p40 and fixed signfix-threshold gates.

### 3.7 Innovation points must be concentrated

Recommended paper/story innovation points:

1. SMPL-X pose-aligned prior into VGGT:
   - not turning VGGT into an SMPL-X regressor;
   - converting SMPL-X into dense prior maps / summary tokens / layer-wise conditioning.
2. Normal-depth-point geometry consistency:
   - normal branch supports geometry, not just images;
   - depth-to-normal / point-to-normal / depth-point / cross-view coupling.
3. ROI-first sparse-view human refinement:
   - face/head/hairline as main target;
   - full-body sanity as bottom line;
   - Open3D ROI evidence and full-body PLY.

## 4. What has already been completed

### 4.1 Coarse prior normal advisor pack

Completed and synchronized:

- canonical: `output\normal_advisor_pack_20260421_coarseprior`
- legacy entry: `output\normal_advisor_pack_20260421`

Checklist already enforced:

- `prior normal` renamed/framed as `coarse prior normal`;
- `4v probe predicted normal` removed from main conclusion;
- failed probe archived as `failed_predicted_normal_probe`;
- main figures focus on:
  - 60v full-body RGB vs coarse prior normal;
  - 60v head ROI RGB vs coarse prior normal;
  - 60v face ROI RGB vs coarse prior normal;
  - 60v overview;
- 7v/13v are supplemental stability evidence only;
- text no longer claims high-quality VGGT predicted normal.

Truthful status:

- Coarse prior normal chain is closed as a visualization/prior story.
- It does not prove sparse-view point-cloud geometry is final.

### 4.2 Real-data SMPL-X bridge

Added:

- `tools\build_scene_prior_from_external_bundle.py`

Purpose:

- `scene-dir + external-prior-bundle -> scene-level prior_maps.npz`
- update `scene_manifest.json`
- allow prior-enabled scene to feed `modal_4k4d_vggt_infer.py`

Smoke completed:

- input scene: `output\smoke_external_bundle_case\scene`
- input bundle: `output\smoke_external_bundle_case\bundle`
- output scene: `output\smoke_external_bundle_case\scene_with_external_prior_bridge`
- summary values:
  - `prior_shape = [2, 30, 518, 518]`
  - `prior_summary_shape = [2, 16, 27]`

Still missing:

- full in-repo raw real image -> SMPL-X estimator/regressor.

### 4.3 Normal-line evidence package

Latest evidence package rebuilt and validated:

- manifest: `output\normal_line_delivery_20260428\normal_line_delivery_manifest_latest.json`
- zip: `output\normal_line_delivery_20260428\normal_line_latest_evidence_20260428.zip`
- validation result:
  - `items = 120`
  - `missing_count = 0`
  - required r10/r11/r13/r15 evidence included.

Main status report:

- `reports\20260428_normal_line_status.md`

Latest addendum:

- `output\normal_line_delivery_20260428\delivery_addendum_20260428_latest.md`

## 5. Negative / failed routes already established

Do not repeat these unless the new design explicitly addresses the failure mode.

### 5.1 Projected targetpatch / summary-token patch

- Already rejected.
- Do not return to it as mainline.

### 5.2 Small humancrop6v improvement

- `20260424_humancrop6v_ckpt0_on6v_headshoulder`
- face ROI `16842`
- reference signfix ckpt4 face ROI `16825`
- gain `+17`
- Open3D morphology essentially same.
- Not mentor-level.

### 5.3 Confidence-collapse pseudo positives

- Example: `20260424_sparseproto_humancrop_pointnormal_r1_open3d_face`
- high point count caused by `conf_threshold=1.0`
- face structure collapsed.
- Cannot be used.

### 5.4 TeacherGeom / ROI combo from old base

Old same-protocol results below signfix:

- `20260424_sparseproto_headshoulder_teachergeom_teachernormal_r1_on6v`: face `16617`
- `20260424_sparseproto_headshoulder_teachergeom_teachernormal_roi_combo_r1_on6v`: face `16494`
- `20260424_6view_focus_headshoulder_teachergeom_teachernormal_r1`: face `16636`
- `20260424_6view_focus_headshoulder_teachergeom_teachernormal_roi_combo_r1`: face `16559`

### 5.5 DepthPro / PSHuman / external teacher attempts

Several external teacher / fusion attempts were tried.

Truthful conclusion:

- 2D normal/depth/mesh-looking teachers did not produce a continuous aligned target-view face surface.
- Direct fusion often produced artifacts, side distortion, or shell.
- Do not launch large training from these unless a new teacher gate visibly passes.

### 5.6 r2/r9/r10/r11/r12/r13/r15 normal gates

Summary:

- r2 improved some normal consistency metrics but failed fixed threshold and visual geometry.
- r9 self-consensus depth residual made tiny sub-mm deltas; no morphology change.
- r10 fixed signfix-threshold gate exposed confidence drop.
- r11 confidence calibration showed geometry failure is not just confidence.
- r12 confidence-guard training smoke gave face ROI `15337`, below signfix.
- r13 TSDF fusion did not create face/head detail; side view remains shell.
- r15 self-geometry-only smoke gave face ROI `15338`, below signfix.

Do not continue r12/r15 by simply increasing epochs.

## 6. Current code changes important for next window

### 6.1 Modified / added source files

Modified:

- `training\loss.py`
- `training\config\4k4d_prior_case.yaml`

Added tools:

- `tools\normal_line_multiview_eval.py`
- `tools\refine_depth_from_normal_poisson.py`
- `tools\patch_predictions_mv_normal_depth_refine.py`
- `tools\fuse_prediction_depth_tsdf.py`

Added configs:

- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r2_depthpoint.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r3_normalstopgrad.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r4_roi_local.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r5_signed_depthnormal.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r6_nopriorgeom_clean6v.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r7_aligned_priormetric.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r8_headsonly_nopriorgeom.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r12_confguard_headsonly.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r15_selfgeom_headsonly.yaml`
- `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r16_xview_selfgeom.yaml`

### 6.2 r16 current state

New r16 idea:

- Add cross-view geometry consistency to the normal line.
- It directly implements mentor's "projection / normal / depth / backprojection" coupling.
- Unlike r12/r15, it does not only enforce single-view depth-normal/depth-point consistency.
- It adds cross-view reprojection consistency over depth, point, and normal.

Implemented in:

- `training\loss.py`
  - new `cross_view` hook in `compute_human_prior_loss`
  - new `compute_prior_cross_view_geometry_consistency_loss`
- config:
  - `training\config\4k4d_prior_case_sparseproto_humancrop_depthnormal_coupled_r16_xview_selfgeom.yaml`

Local smoke:

- `python -m py_compile training\loss.py` passed.
- synthetic three-view geometry loss smoke passed with zero loss on identical planes.

Cloud training:

- command timed out locally after 1 hour, but cloud training completed.
- local output:
  - `output\modal_training_results\20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4`
- `run_summary.json` status:
  - `completed`
  - GPU: `NVIDIA A100 80GB PCIe`
  - elapsed: `130.438s`
  - latest checkpoint: `vggt_4k4d_train/20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4/logs/ckpts/checkpoint_0.pt`
  - inference checkpoint relpath: `vggt_4k4d_train/20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4/inference_model.pt`

Next immediate action:

1. Run inference for r16 checkpoint on original same-protocol 6-view headshoulder crop.
2. Summarize ROI.
3. Render Open3D full/head/face from both `world_points` and `depth_unprojection`.
4. If r16 does not clearly beat signfix, record as failed and do not claim pass.

## 7. Exact next commands

### 7.1 Check process hygiene

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^(python|modal|colmap)\.exe$' } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
```

Do not kill WPS / Codex helper processes unless clearly rogue.

### 7.2 Inference r16 on original 6v headshoulder

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::run_scene_from_remote \
  --scene-subdir scenes/0012_11_frame0000_6views_sparseproto_headshoulder_crop \
  --output-subdir evals/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder \
  --checkpoint-relpath vggt_4k4d_train/20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4/logs/ckpts/checkpoint_0.pt \
  --download-local-dir output\modal_results\20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder
```

If `run_scene_from_remote` is not the actual CLI entry, inspect:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::run_scene_from_remote --help
modal run modal_4k4d_vggt_infer.py::run_scene_from_local --help
```

Known working local scene option pattern:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::run_scene_from_local `
  --local-scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --remote-scene-subdir scenes\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --output-subdir evals/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder `
  --checkpoint-relpath vggt_4k4d_train/20260428_normal_r16_xview_selfgeom_smoke20_from_ckpt4/logs/ckpts/checkpoint_0.pt `
  --download-local-dir output\modal_results\20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder
```

### 7.3 Summarize r16 same-protocol ROI

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::summarize_prediction_roi `
  --remote-output-subdir evals/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder `
  --scene-subdir scenes/0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --conf-percentile 40
```

Pass numeric minimum:

- face must exceed `16825` by a meaningful margin, not `+10` or `+17`;
- head/full must not regress;
- if face gain is small, visual must be obviously better or still fail.

### 7.4 Render r16 Open3D full/head/face

Use local Open3D Python:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
$py='D:\anaconda\envs\g3splat\python.exe'
$pred='D:\vggt\vggt-main\output\modal_results\20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder\predictions.npz'
$scene='D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop'
$out='D:\vggt\vggt-main\output\normal_line_multiview_20260428\r16_xview_selfgeom_open3d_on6v_headshoulder'
foreach($roi in @('full','head','face')){
  foreach($src in @('world_points','depth_unprojection')){
    & $py D:\vggt\vggt-main\tools\render_open3d_pointcloud.py `
      --predictions-npz $pred `
      --scene-dir $scene `
      --output-dir "$out\$($roi)_$($src)" `
      --point-source $src `
      --human-only `
      --roi $roi `
      --roi-source 2d `
      --conf-percentile 40 `
      --max-points 600000 `
      --width 1600 `
      --height 1200 `
      --point-size 2.0 `
      --camera-view-indices 3
  }
}
```

Also render signfix reference under the exact same settings if a side-by-side sheet is needed.

### 7.5 Compare fixed-threshold gate

If r16 looks promising under p40, do not trust p40 alone. Run fixed signfix threshold:

- original 6v signfix threshold: `38.5067`

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
$py='D:\anaconda\envs\g3splat\python.exe'
$pred='D:\vggt\vggt-main\output\modal_results\20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder\predictions.npz'
$scene='D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop'
$out='D:\vggt\vggt-main\output\normal_line_multiview_20260428\r16_xview_selfgeom_fixedthr_on6v_headshoulder'
foreach($roi in @('full','head','face')){
  foreach($src in @('world_points','depth_unprojection')){
    & $py D:\vggt\vggt-main\tools\render_open3d_pointcloud.py `
      --predictions-npz $pred `
      --scene-dir $scene `
      --output-dir "$out\$($roi)_$($src)_sfthr385067" `
      --point-source $src `
      --human-only `
      --roi $roi `
      --roi-source 2d `
      --conf-threshold 38.5067 `
      --max-points 600000 `
      --width 1600 `
      --height 1200 `
      --point-size 2.0 `
      --camera-view-indices 3
  }
}
```

Pass / fail:

- pass only if fixed-threshold point count and visual quality hold;
- fail if p40 looks okay but fixed threshold collapses.

## 8. If r16 fails

Do not continue by merely increasing epochs.

Recommended next normal-only thinking:

1. Check whether cross-view loss actually backpropagates into the branches that need geometry improvement.
   - If it only nudges normal/depth but point map stays split, add stronger direct point-map coupling or freeze/unfreeze different heads.
2. Use target-view full-body / head / face side-view diagnostics to identify whether the failure is:
   - depth confidence collapse;
   - world_points branch split;
   - camera/view selection issue;
   - ROI crop false-positive;
   - SMPL template over-constraint;
   - lack of image-aligned detail teacher.
3. Avoid external teacher training unless teacher gate provides a continuous, aligned target-view head/face surface.
4. Consider a small one-frame ROI overfit only if it directly optimizes final point cloud / depth, not only normal-map appearance.

## 9. Reporting / delivery requirements

Every candidate must produce:

- full-body PLY;
- full-body figure;
- head ROI figure;
- face ROI figure;
- p40 ROI counts;
- fixed-threshold ROI counts;
- normal consistency metrics;
- note whether `point-source` is `world_points` or `depth_unprojection`;
- source view protocol and target camera;
- truthful pass/fail statement.

Figures must not mix:

- coarse SMPL prior normal;
- predicted normal;
- refined normal;
- failed probe.

For normal maps, always label:

- RGB;
- coarse prior normal;
- predicted/refined normal;
- depth-derived normal;
- point-derived normal;
- diff / consistency if present.

For point clouds, always label:

- scene/protocol;
- checkpoint;
- point source;
- threshold;
- ROI source;
- target camera.

## 10. Final checklist before saying "pass"

Do not say pass unless all boxes are true:

- [ ] Same-protocol original 6v headshoulder face ROI clearly beats signfix `16825`.
- [ ] Improvement is not tiny jitter (`+10`, `+17`, `+30` are not enough).
- [ ] Full/head/face Open3D close-ups show real geometry improvement.
- [ ] Face is closer to a modeled human face, not merely a colored shell.
- [ ] Eyes/nose/mouth/face contour/head surface are more coherent.
- [ ] Hairline/head boundary is not worse.
- [ ] Full-body has no severe holes, broken hands/limbs, or ghost shell.
- [ ] `world_points` and `depth_unprojection` are both checked.
- [ ] Fixed signfix-threshold gate does not collapse.
- [ ] Multi-view behavior is documented for at least 3/6/13/16/60 or the available subset.
- [ ] Normal consistency metrics improve without destroying point-cloud quality.
- [ ] Results are packaged with PLY + figures + metrics.
- [ ] Report text remains truthful and does not hide failed gates.

If any box fails, report "not yet mentor-final" and continue.

## 11. One-sentence truth for next window

We are now on the mentor-directed **normal-only** line: use SMPL-X carefully as a coarse pose-aligned prior, but make normal useful through depth/point/cross-view geometry coupling; r2-r15 are negative, r16 cross-view self-geometry training has completed and must be evaluated next on original 6v headshoulder with ROI + Open3D + fixed-threshold full/head/face gates before any pass claim.
