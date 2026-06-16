# V23 Residual Surface V2

Status: `DONE_PASS`

Research-only/no formal pass. This run writes only V23 residual evidence, PLY, NPZ, audit, and report artifacts.

## Decision

V23 repaired the V17 zero head/face sampling bug by building per-region evidence from V16 ROI maps intersected with V15 raw silhouette support. The resulting research surface has nonempty body/head/face/left_hand/right_hand evidence and samples.

## Key Metrics

- evidence_pixels: `67218`
- sampled_points: `67218`
- mean_applied_residual_m: `0.0022218835074454546`
- max_applied_residual_m: `0.009395855478942394`

## Region Metrics

- body: pixels=`59474`, sampled=`59474`, raw_support=`67218`, native_overlap=`25046`, raw_mean_m=`0.12561164796352386`, applied_mean_m=`0.001958321314305067`, applied_p95_m=`0.0032420125789940357`
- head: pixels=`4114`, sampled=`4114`, raw_support=`6536`, native_overlap=`0`, raw_mean_m=`0.22453714907169342`, applied_mean_m=`0.0038778989110141993`, applied_p95_m=`0.005139493383467197`
- face: pixels=`2422`, sampled=`2422`, raw_support=`2422`, native_overlap=`0`, raw_mean_m=`0.18980033695697784`, applied_mean_m=`0.0036797290667891502`, applied_p95_m=`0.005187907721847296`
- left_hand: pixels=`736`, sampled=`736`, raw_support=`736`, native_overlap=`614`, raw_mean_m=`0.15346136689186096`, applied_mean_m=`0.0061115967109799385`, applied_p95_m=`0.008692210540175438`
- right_hand: pixels=`472`, sampled=`472`, raw_support=`485`, native_overlap=`130`, raw_mean_m=`0.3378547728061676`, applied_mean_m=`0.007451782003045082`, applied_p95_m=`0.009119796566665173`

## Outputs

- residual_surface_ply: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\v23_residual_surface_v2_points.ply`
- prior_sample_ply: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\v23_prior_region_sample_points.ply`
- residual_surface_npz: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\v23_residual_surface_v2_points.npz`
- mask_npz: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\v23_repaired_region_evidence_masks.npz`
- mask_summary_json: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\v23_repaired_region_evidence_masks_summary.json`
- summary_json: `D:\vggt\vggt-main\output\surface_research_preflight_local\V23_residual_surface_v2\summary.json`

## Guardrails

- research_only: `True`
- smplx_native_only: `True`
- no_mano: `True`
- no_flame: `True`
- no_hairgs: `True`
- no_predictions_write: `True`
- no_teacher_export: `True`
- no_candidate_export: `True`
- no_registry_write: `True`
- no_package_write: `True`
- no_strict_pass_claim: `True`

## Blockers

- none
