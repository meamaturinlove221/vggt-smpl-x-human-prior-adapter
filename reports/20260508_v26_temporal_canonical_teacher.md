# V26 Temporal Canonical SMPL-X Residual Teacher

Status: `DONE_PASS`

V26 constructed a three-frame research-only temporal residual accumulator. It is usable as research evidence but remains blocked for strict teacher promotion until normal evidence is available.

## Metrics

- frame_keys: `['frame0000', 'frame0001', 'frame0002']`
- shared_view_count: `6`
- canonical_support_pixels: `67218`
- canonical_residual_norm: `{'count': 67218, 'finite': 67218, 'min': 0.007028824649751186, 'mean': 0.373203307390213, 'median': 0.20886877179145813, 'p95': 0.8179078102111816, 'max': 1.4342215061187744}`
- temporal_variance: `{'count': 67218, 'finite': 67218, 'min': 8.939198714585928e-09, 'mean': 0.0009709821897558868, 'median': 3.565386919035518e-07, 'p95': 3.004518111993093e-05, 'max': 0.04053320735692978}`
- normal_available: `False`

## Regions

- body: canonical_support=`59474`, frames=`3`, temporal_variance_mean=`0.0009736426291055977`
- head: canonical_support=`4114`, frames=`3`, temporal_variance_mean=`0.0007855163421481848`
- face: canonical_support=`2422`, frames=`3`, temporal_variance_mean=`0.0012917023850604892`
- left_hand: canonical_support=`736`, frames=`3`, temporal_variance_mean=`0.0005766559625044465`
- right_hand: canonical_support=`472`, frames=`3`, temporal_variance_mean=`0.0012214314192533493`

## Outputs

- temporal_targets_npz: `D:\vggt\vggt-main\output\surface_research_preflight_local\V26_temporal_canonical_teacher\v26_temporal_canonical_teacher_targets.npz`
- summary_json: `D:\vggt\vggt-main\output\surface_research_preflight_local\V26_temporal_canonical_teacher\summary.json`
- report_json: `D:\vggt\vggt-main\reports\20260508_v26_temporal_canonical_teacher.json`
- report_md: `reports\20260508_v26_temporal_canonical_teacher.md`

## Blockers

- V25 loaded model has no normal_head; temporal normals unavailable and not fabricated
