# 2026-05-06 Surface Token Backend B1 Research Preflight

## Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes   = 0
teacher export          = blocked
candidate export        = blocked
formal VGGT cloud       = blocked
```

This report is B-line research-preflight only. It is not a strict pass, not a teacher, not a candidate, and not a cloud unblock signal.

## B0 Readout

Inspected `tools/optimize_surface_token_backend_b0.py` and existing B0 outputs under:

```text
output/surface_research_preflight_local/B0_surface_tokens_t96_step20
output/surface_research_preflight/B0_surface_tokens_t64_step2_gpu
```

B0 already moved beyond the old `image_mlp` lane by using quantized part-aware surface tokens, part-specific offset heads, fixed multi-view RGB/support features, silhouette losses, edge regularization, depth TV proxy, and photometric variance. Existing B0 local t96/step20 summary:

```text
avg_initial_iou = 0.7594788682
avg_final_iou   = 0.7605119822
avg_iou_delta   = 0.0010331140
token_count     = 400
mean_support    = 4.2675266266
visual_review   = fail / research_only
```

Decision: B0 is a useful smoke, but the small numeric IoU delta and failed visual review mean it cannot be promoted. The missing piece for the next B-line preflight is not another hidden/step/weight tweak; it is explicit token visibility accounting plus rendered diagnostic evidence.

## B1 Design

Added `tools/optimize_surface_token_backend_b1.py` as a new B-line-only research tool. It keeps B0's surface-token carrier but makes the backend contract more explicit:

- Surface tokens: canonical part-aware quantized tokens remain the learned carrier.
- Visibility aggregation: exports both projected mask-support per token and nvdiffrast raster-visible token support.
- Part-specialized heads: separate per-part offset heads and per-part residual-normal heads.
- Visibility gate: token-level visibility confidence gates offset magnitude and is exported as a diagnostic mesh/table.
- Rendered diagnostics: per-view mask, depth, normal, support heatmap, and photometric residual maps.
- Research outputs only: diagnostic PLY/PNG/JSON/MD artifacts; no teacher/candidate NPZ and no strict-pass registry write.

## Command Scaffold

Minimal local smoke:

```powershell
python tools\optimize_surface_token_backend_b1.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --template-payload output\normal_line_multiview_20260506\connected_surface_template_v28_semantic_detail_mouth_nose_fingers\connected_human_surface_template_payload.npz `
  --output-dir output\surface_research_preflight_local\B1_surface_tokens_t48_step1 `
  --view-indices 0,30 `
  --target-size 48 `
  --steps 1 `
  --token-grid 4 `
  --token-hidden 32 `
  --overwrite
```

Broader local research diagnostic, still not a pass command:

```powershell
python tools\optimize_surface_token_backend_b1.py `
  --scene-dir output\4k4d_preprocessed_scene_variants\0012_11_frame0000_60views_human_crop `
  --template-payload output\normal_line_multiview_20260506\connected_surface_template_v28_semantic_detail_mouth_nose_fingers\connected_human_surface_template_payload.npz `
  --output-dir output\surface_research_preflight_local\B1_surface_tokens_t96_step10 `
  --view-indices 0,10,20,30,40,50 `
  --target-size 96 `
  --steps 10 `
  --token-grid 5 `
  --token-hidden 64 `
  --overwrite
```

The command intentionally does not call `tools/package_normal_candidate_gate.py`, does not create a teacher/candidate bundle, and does not launch Modal/cloud.

## Current Verification

Static checks:

```text
python -m py_compile tools\optimize_surface_token_backend_b1.py
python tools\optimize_surface_token_backend_b1.py --help
```

Local smoke command executed and stopped safely with:

```text
status = blocked_no_nvdiffrast
nvdiffrast_import_error = ModuleNotFoundError("No module named 'nvdiffrast'")
strict_candidate_passes = 0
strict_teacher_passes = 0
```

Blocked summary was written to:

```text
output/surface_research_preflight_local/B1_surface_tokens_t48_step1/surface_token_b1_summary.json
output/surface_research_preflight_local/B1_surface_tokens_t48_step1/surface_token_b1_summary.md
```

## Blocker

This Windows Python environment does not currently import `nvdiffrast`. PyTorch also warns that the installed CUDA build supports up to `sm_90`, while the local RTX 5080 is `sm_120`. B1 is therefore implemented and command-scaffolded, but full rendered diagnostic generation needs a local environment with compatible PyTorch CUDA plus nvdiffrast.

## Next Step

Run the minimal B1 smoke in a compatible local CUDA/nvdiffrast environment. Review the per-token visibility table and rendered mask/depth/normal/photometric residual PNGs before considering any training design. Keep strict passes at zero unless a separate strict visual gate actually passes.
