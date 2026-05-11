# 2026-05-04 r47 Visibility-Aware Surface Backend Closure

## Truthful status

- `r47_visibility_surface_backend_diag`: **FAIL / not mentor-final**.
- Cloud remains blocked.
- This route was local-only and did not use 60-view, Kinect, LHM, Sapiens depth, external mesh teachers, or SMPL-X face/hair geometry truth.

## What r47 tested

r47 was introduced because r43/r44 already showed that simple canonical-bin surfel fusion is not enough. This variant is not a threshold/support rerun of r43. It adds:

- a shared canonical surfel set from VGGT `world_points` and `depth_unprojection` observations;
- SMPL-X prior maps only as visibility/canonical/body-part correspondence hints;
- graph Laplacian optimization over neighboring canonical bins;
- critical ROI support gating for face/hands;
- rasterization back to same-protocol `predictions.npz`;
- support heatmaps and shared-surface Open3D debug renders.

Implementation:

- `tools/optimize_visibility_aware_surface_predictions.py`

Outputs:

- Headshoulder: `output/normal_line_multiview_20260504/r47_visibility_surface_backend_diag_on6v_headshoulder`
- Fullbody: `output/normal_line_multiview_20260504/r47_visibility_surface_backend_diag_on6v_humancrop_fullbody`
- Candidate gate: `output/normal_line_multiview_20260504/candidate_gate_r47_visibility_surface_backend_diag`

## Surface diagnostic summary

Headshoulder diagnostic:

- accepted surfels: `3122`
- raster-eligible surfels: `2531`
- face surfels: `949`
- hands surfels: `216`
- p95 optimized-surface shift: `0.0086301025`
- face raster changed fraction: `0.5136`
- hands raster changed fraction: `0.0377`

Fullbody diagnostic:

- accepted surfels: `3458`
- raster-eligible surfels: `2739`
- face surfels: `1150`
- hands surfels: `199`
- p95 optimized-surface shift: `0.0070441687`
- face raster changed fraction: `0.4154`
- hands raster changed fraction: `0.0298`

The tool itself flags both `face_surface_sparse` and `hands_surface_sparse`, and the hand raster change is extremely low. This is already a strong warning that the shared surface is not producing robust hand geometry.

## Strict candidate gate result

Gate package:

- `output/normal_line_multiview_20260504/candidate_gate_r47_visibility_surface_backend_diag/report.md`
- `output/normal_line_multiview_20260504/candidate_gate_r47_visibility_surface_backend_diag/candidate_gate_summary.json`
- contact sheet: `output/normal_line_multiview_20260504/candidate_gate_r47_visibility_surface_backend_diag/candidate_gate_visual_sheet.png`
- explicit visual review: `output/normal_line_multiview_20260504/candidate_gate_r47_visibility_surface_backend_diag/visual_review_codex_fail.json`

Pass/fail:

- numeric gate: `False`
- normal consistency gate: `True`
- shape gate: `False`
- fullbody/hands gate: `False`
- explicit visual gate: `False`
- cloud upload blocked: `True`

Numeric face results:

- world_points p40 face: signfix `16825`, r47 `16830`, delta `+5`
- depth_unprojection p40 face: signfix `16764`, r47 `16857`, delta `+93`
- required meaningful face margin: `+500`

Fullbody/hands:

- all four fullbody audit modes fail:
  - `world_points p40`
  - `world_points fixed`
  - `depth_unprojection p40`
  - `depth_unprojection fixed`
- body fails on per-view body coverage.
- hands fail compact/support checks, with one implausible hand box in each mode.

Normal consistency:

- r47 improves many normal-depth-point consistency numbers, for example:
  - face pred-vs-point angle delta `-5.4566 deg`
  - head pred-vs-point angle delta `-6.8500 deg`
- This does **not** count as pass because the mentor gate is final 3D point cloud quality, not normal metric improvement.

## Open3D visual review

Explicit local visual review is fail:

- head/face still looks like a projected shell, not modeled facial geometry;
- facial relief, nose/eye/mouth structure, and hairline continuity are not recovered;
- full-body side/right/back/iso views still show slanted slab-like geometry rather than a normal volumetric human;
- hands are detached sheet fragments or sparse thin surfaces, not attached detailed hands;
- both world_points and depth_unprojection remain unacceptable.

## Decision

r47 is a useful negative result: a graph-optimized, visibility-aware shared surface can make normal consistency look better, but it still only reorganizes VGGT-observed 2.5D shell geometry. It does not create the continuous head/face/hair/hand/fullbody surface required by the mentor.

Freeze:

- do not tune r47 by `canonical_bin_size`, support count, raster radius, smoothing strength, alpha, confidence percentile, p40/fixed threshold, or more local variants;
- do not cloud-run r47;
- do not claim mentor-final based on normal consistency.

Implication:

- Any next attempt must introduce a genuinely stronger representation or signal than VGGT-observed shell rearrangement. Acceptable unblockers remain a strict-passing local candidate, a strict-passing teacher, a verified raw dataset target-frame dense surface asset, or a real learned human surface backend with supervision/training evidence. r47 did not provide that unblocker.
