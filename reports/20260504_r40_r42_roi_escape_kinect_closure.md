# r40-r42 ROI Escape / Kinect Closure

Date: 2026-05-04

## Current Truth

No local candidate satisfies the mentor gate. The refreshed strict registry is:

```text
reports/20260504_strict_gate_registry.json
reports/20260504_strict_gate_blocker_status.md
```

Counts:

- candidate gate summaries scanned: `97`
- teacher gate summaries scanned: `83`
- strict full mentor candidate passes: `0`
- strict teacher passes: `0`
- cloud upload: blocked

The active branch remains isolated on:

```text
codex/no-wall-selfgeom-bodyhand-next
```

## r40 / r41 / r42 Are Negative

### r40

Package:

```text
output/normal_line_multiview_20260504/candidate_gate_r40_partaware_softweight_bodyhand_smoke1
```

Result:

- world-points face p40: `15069 < 16825`, delta `-1756`
- depth-unprojection face p40: `16273 < 16764`, delta `-491`
- fixed-threshold counts rise, but this is not a pass
- full-body provenance is candidate-specific, but full-body p40 gate fails
- hand metric screen can pass in places, but Open3D hands remain thin fragments
- face/head Open3D remains shell-like, not modeled

Decision: freeze r40. Do not continue by more epochs, soft-weight tuning, or confidence thresholding.

### r41

Package:

```text
output/normal_line_multiview_20260504/candidate_gate_r41_mixed_headshoulder_fullbody_smoke1
```

Result:

- world-points face p40: `15145 < 16825`, delta `-1680`
- depth-unprojection face p40: `16356 < 16764`, delta `-408`
- normal gate fails
- full-body p40 gate fails
- shape gate fails
- face/head Open3D remains shell-like

Decision: freeze r41. Mixed headshoulder/fullbody distribution alignment did not create modeled face/head geometry.

### r42

Package:

```text
output/normal_line_multiview_20260504/candidate_gate_r42_roi_escape_headshoulder_probe_smoke1
```

Result:

- world-points face p40: `15114 < 16825`, delta `-1711`
- depth-unprojection face p40: `16031 < 16764`, delta `-733`
- normal gate fails
- full-body p40 gate fails
- shape gate fails on head/face depth-unprojection p40/fixed
- Open3D visual sheet shows shell-like face/head, tilted full-body sheets, and fragmented/amputated hand support

Decision: freeze r42. The ROI escape objective did not solve 3D ROI escape or produce a normal human-looking point cloud.

## ROI Escape Diagnostic

Tool:

```text
tools/audit_roi_3d_escape.py
```

Outputs:

```text
output/normal_line_multiview_20260504/roi_3d_escape_signfix_r40_r41_headshoulder
output/normal_line_multiview_20260504/roi_3d_escape_signfix_r42_headshoulder
```

Key r42 result:

| Entry | Source | Gate | ROI | 2D kept | In fused face | Face ratio |
|---|---|---|---|---:|---:|---:|
| signfix | world_points | p40 | face | 44963 | 14031 | 0.31206 |
| r42 | world_points | p40 | face | 47780 | 12973 | 0.27152 |
| signfix | depth_unprojection | p40 | face | 46378 | 13532 | 0.29178 |
| r42 | depth_unprojection | p40 | face | 48113 | 13374 | 0.27797 |

Interpretation:

The candidate can keep more 2D face pixels while fewer of them land inside the fused 3D face ROI. This is confidence/ROI churn, not a modeled face gain.

## P40 Ownership Geometry Diagnostic

Tool:

```text
tools/audit_p40_ownership_geometry.py
```

Outputs:

```text
output/normal_line_multiview_20260504/p40_ownership_geometry_r40_vs_signfix
output/normal_line_multiview_20260504/p40_ownership_geometry_r42_vs_signfix
```

Key r42 depth-unprojection face result:

| Group | Pixels | Target coverage | Target LCC | Target protrusion | Target thinness |
|---|---:|---:|---:|---:|---:|
| both | 39992 | 0.01045 | 0.99231 | 0.00091 | 0.06691 |
| new | 8121 | 0.09393 | 0.88794 | 0.00183 | 0.10039 |
| fixed_new | 12472 | 0.20537 | 0.78052 | 0.00386 | 0.15312 |

Interpretation:

New/fixed-new points do not become a convincing central face surface. They mostly change confidence ownership and visible-sheet support. Do not tune this into a pseudo-pass.

## Kinect Coordinate Pass Re-Checked

Coordinate-positive Kinect source:

```text
output/normal_line_multiview_20260502/kinect_teacher_60v_headface_camera_axes_s0005_gate
```

The 60-view coordinate audit passes only a coordinate-chain criterion:

- alignment source: `camera_axes`
- alignment residual p50: `0.06966881175102355`
- distance-to-base p50: `0.07487250864505768`
- per-view visibility: `49/60`

It does not prove original 6-view same-protocol teacher quality.

New strict re-gate:

```text
output/normal_line_multiview_20260504/teacher_gate_kinect60v_headface_s0005_original6v_subset_allviews
```

Result:

- overall pass: `False`
- numeric pass: `False`
- explicit visual review pass: `False`
- view 2 head_face coverage: `0.0927`, hole ratio `0.9073`
- view 3 head_face coverage: `0.2926`, hole ratio `0.7074`
- hairline fails all six views

Decision:

Kinect remains a coordinate diagnostic only. It cannot authorize patching or training unless a future genuinely new Kinect asset passes the same strict numeric plus Open3D visual teacher gate.

## Full-Body / Hand Gate Status

The mentor requirement is now encoded as a hard bottom-line:

- every candidate must provide candidate-specific full-body NPZ;
- full/head/face/hands Open3D renders must be generated for world-points and depth-unprojection;
- p40 and fixed-threshold views must both be checked;
- full-body cannot have large holes, slab/shell body, broken limbs, implausible proportions, or severe ghosting;
- hands cannot be missing, amputated, or only scattered sheet/noise support;
- passing support metrics is insufficient without explicit visual review JSON.

r42 contact sheet confirms why this matters: full-body front views can look superficially human, while side/back/iso reveal tilted shell geometry and the hands remain fragmented. This is not a normal 3D human point cloud.

## Answer To P2 / P3 / P4 / P5 Status

The proposed teacher-priority and teacher-gate plan is correct as a defensive filter, but it is not an active optimization route anymore:

- P2 priority ordering has effectively been executed across internal 60v, world-surface/TSDF/Poisson, SMPL-X, Kinect, and external teacher families. Strict teacher pass remains `0`.
- P3 unified teacher gate exists as `tools/audit_headface_teacher_surface.py` and is used for mesh / pointcloud / NPZ teacher audits.
- P4 one-frame ROI overfit is blocked because no teacher has passed strict teacher gate.
- P5 full-body / hand hard gate is implemented in `tools/audit_fullbody_hand_integrity.py` and `tools/package_normal_candidate_gate.py`, and remains mandatory.

Important decision:

Do not continue hard-teacher chasing as the main route. The gate stays, but the active work must avoid repeating old teacher / threshold / epoch loops.

## Current Allowed Direction

1. Keep human crop / softmatte as the input base, but report it as preprocessing support only.
2. Keep SMPL-X as weak body/hand topology and real-data prior bridge only; never as face/hair/clothing teacher.
3. Keep normal-depth-point coupling as the geometry principle, but do not repeat r16/r18/r19/r21/r37/r40-r42 variants without a new mechanism.
4. Any next local experiment must introduce a genuinely new geometry mechanism, not another confidence/ROI/threshold/epoch tweak.
5. Cloud upload remains blocked until the full local gate passes with explicit visual review showing a normal human-looking Open3D result.
