# 2026-05-07 B-Hand Evidence Cache Status

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
formal cloud train/infer/export = blocked
teacher/candidate export = blocked
```

本次是 Agent C / B-hand 诊断前置，只新增 `tools/b_hand_evidence_cache.py`。没有修改 A5、Modal、D-line、B-Fus3D、strict registry 或 cloud guard。该脚本不训练、不推理、不 patch predictions、不生成 teacher/candidate、不写 strict pass。

## Local Source Readout

本地证据来源确认如下：

- hand ROI / visibility：`tools/audit_fullbody_hand_integrity.py` 提供 MediaPipe hand landmarker、skin-extremity fallback、2D/3D compactness gate；`tools/audit_smplx_weak_anchor.py` 使用 `smplx_visible & hand_mask` 做弱 prior 可见性；`tools/audit_smplx_raw_mesh_hand_anchor.py` 已有 left/right hand joint specs。
- scene crop / mask：`scene_manifest.json`、`images/`、`masks/` 可给 per-view RGB/mask；`normal_line_multiview_eval.py` 和 `augment_case_with_image_roi_masks.py` 已有 head/face/crop ROI 逻辑。
- camera：优先可从 `predictions.npz` 的 `intrinsic`/`extrinsic` 读取；也可从 `camera_params_sidecar.npz` 读取；需要真实 4K4D camera 时走 `resolve_scene_camera_params(...)`，但脚本默认不强制 materialize real cameras。
- SMPL-X prior：`prior_maps.npz` 中的 `smplx_visible_mask`、`smplx_posed_cam_{x,y,z}`、`prior_summary_tokens` 可做弱可见性诊断；annotation SMPLx 和 SMPL-X model 可进一步证明 left/right hand topology 可选，但不 rasterize 成 teacher。
- VGGT token hook：`run_local_vggt_inference.py` 已经能把 `prior_maps` / `prior_summary_tokens` 喂给 VGGT；B-hand B0 仍缺 side-aware hand ROI patch-token hook。

## Implemented Cache Skeleton

`tools/b_hand_evidence_cache.py` 当前输出：

- `visible_view_summary.left/right`：每侧 views_with_roi、ROI pixels、SMPL-X visible prior overlap、prediction support overlap。
- `hand_crop_metadata`：per-view/per-hand bbox、crop size、ROI density、component stats、image/mask paths。
- `camera_rays`：如果有 prediction/sidecar/real camera，则写 bbox center/corner ray samples；否则显式 `missing_camera_ray_metadata`。
- `vggt_token_hook`：按 14px patch grid 记录 ROI patch range 和 token id preview；标记为 placeholder，不修改 VGGT。
- `smplx_prior_availability`：记录 prior maps channels、summary token shape、SMPL-X visible/posed-cam availability、left/right topology count probe。
- `hggt_b0_contract` 和 `stop_conditions`：固定为 diagnostic-only stop contract。

为避免 Windows 本地 OpenMP DLL 冲突，脚本没有 runtime 依赖 `audit_fullbody_hand_integrity.py` / `normal_line_multiview_eval.py` 的 heavy import path；关键 scene loading、head/face ROI、skin fallback、connected components、3D box stats 都内联为轻量 PIL/NumPy 版本。参考来源仍按上面的本地脚本语义记录在 cache 的 `source_inventory`。

## Smoke

已做无仓库写入的临时 smoke：

```powershell
python -B tools\b_hand_evidence_cache.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_human_crop_softmatte `
  --output-dir $env:TEMP\vggt_b_hand_cache_smoke `
  --view-indices 0 `
  --disable-mediapipe-hands `
  --overwrite
```

结果：

```text
status = diagnostic_only_preflight_cache
left.views_with_roi = 1
right.views_with_roi = 1
smplx_visible_mask = available
smplx_posed_cam_xyz = available
camera_rays = missing unless predictions/sidecar/real camera is provided
VGGT hand-token hook = placeholder only
strict pass writes = 0
```

## HGGT-Style B0 Minimal Structure

B0 只允许作为 hand-token 前置诊断：

- side token：left/right identity + visible view histogram。
- wrist-arm anchor token：必须有和 forearm/body 支持相连的证据。
- palm/finger local tokens：只从 hand crop / patch range / weak prior overlap 建 skeleton，不宣称 residual 成功。
- confidence token：汇总 ROI support、SMPL-X overlap、prediction support、camera ray availability。

停止条件：

- strict registry 仍为 0，formal cloud train/infer/export blocked。
- 任一侧 hand ROI 缺失时 stop。
- camera ray metadata 缺失时 stop。
- VGGT side-aware hand token hook 未接入时 stop。
- side-specific SMPL-X hand anchor 只能是 weak prior，不能作为 pass。
- Open3D 手部视觉必须连臂；detached sheets/sticks、floating palm/finger clusters 一律 stop。

## Decision

B-hand 当前完成的是 evidence cache + HGGT-style B0 skeleton 前置诊断。它可以帮下一步定位 left/right hand evidence、crop、prior、ray、patch-token hook 的缺口，但不能 unblock teacher/candidate，也不能绕过 full-body/hands hard gates。
