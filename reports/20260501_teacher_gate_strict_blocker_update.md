# 2026-05-01 Teacher-Gate Strict Blocker Update

## Current Truth

No local candidate is mentor-final. Cloud upload remains blocked.

The active requirement is still teacher-gate first: a teacher must be a continuous, aligned head/face/hairline surface that can be projected back into the original 6-view headshoulder protocol. Numeric coverage alone is not enough; Open3D must show modeled geometry rather than a shell or texture billboard.

## r42 False Positive Closed

The previously tempting `r42_r24_visualhull_head_smplxhands_full_probe` has been re-run with the current strict package gate:

```text
output\normal_line_multiview_20260430\candidate_gate_r42_r24_visualhull_head_smplxhands_full_probe
```

Result: fail.

Although the same-protocol numeric, normal consistency, and shape gates pass, the explicit Open3D visual review fails. The face/head closeups remain shell-like with a large blank facial region instead of modeled nose/eyes/mouth/hairline. Full-body side/iso views contain slab/ghost geometry, and hand crops are sparse fragments rather than reliable hand/finger structure.

The updated full-body/hand compactness gate also fails:

- only 1 eligible hand view passes the compact 3D hand support check;
- 2 hand boxes are implausible;
- therefore the old automatic full-body/hands pass was a false positive.

## Old Teacher Pass Records Re-Audited

The following older teacher-gate `pass=True` records have been upgraded to the new strict `numeric_pass + visual_pass` format:

```text
output\normal_line_multiview_20260430\teacher_gate_hybrid_plus_r55_facecore_allviews
output\normal_line_multiview_20260430\teacher_gate_unified_hybrid_r49_smplx_allviews
output\normal_line_multiview_20260430\teacher_gate_unified_hybrid_r49_smplx_allviews_visual
output\normal_line_multiview_20260430\teacher_gate_unified_aligned_smplx_npz_view03
```

All are now strict failures.

Typical result:

```text
numeric_pass = true
visual_pass = false
pass = false
```

Reason: the targets are depth-compatible in 2D projection, but their Open3D renders are still sparse/self-derived shells or template-like patches. They are not continuous modeled personal face/head surfaces and cannot be used as mentor-level teachers.

## Source Data Recheck

The local 4K4D/DNA annotation SMC contains:

- camera calibration;
- masks;
- 2D and 3D keypoints;
- SMPL-X betas / expression / fullpose / scale / translation.

It does not contain dense ground-truth vertices, mesh, normal map, or point-surface teacher for `0012_11`.

The main SMC provides RGB streams. Real depth exists in the Kinect SMC, but the current calibrated Kinect teacher attempts still fail strict alignment / visibility / visual surface requirements and cannot serve as the head/face teacher.

## Blocker

The current blocker is not missing scripts or missing cloud training. The blocker is the absence of a locally verified, continuous, aligned, high-quality head/face/hairline surface teacher.

No training or cloud run is allowed until a teacher passes both:

- strict numeric projection / depth-compatible coverage;
- explicit Open3D visual review showing modeled head/face/hairline geometry.

## Next Safe Direction

Do not revive:

- HART-style PnP as camera replacement;
- r16/r18/r19 plus more epochs;
- r57/r58/r59/r60;
- COLMAP/MVS meshray;
- Kinect/external pointcloud projection patch;
- true-highres crop as a direct replacement for same-protocol headshoulder.

Possible next local-only probes must first produce a teacher candidate and run `tools/audit_headface_teacher_surface.py`. Training remains blocked until that teacher passes.
