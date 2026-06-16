# Raw Kinect Asset Audit Closure

Date: 2026-05-04

## Current Truth

The raw 4K4D subset is locally available through the resolved non-ASCII `G:` path:

```text
G:\数据集\datasets\data_used_in_4K4D
```

Inventory report:

```text
output/normal_line_multiview_20260504/local_4k4d_inventory_G_20260504.json
```

It confirms `48/48` expected canonical files are present, including:

- `main/0012_11.smc`
- `annotations/0012_11_annots.smc`
- `kinect/0012_11_kinect.smc`
- `data_used_in_4K4D_rgb_cams.zip`

This corrects the older local note that the raw dataset path was unavailable.

## Raw Kinect Asset Audit

Raw Kinect TSDF was fused locally as an asset diagnostic:

```text
output/local_teacher_probes/0012_11_kinect_tsdf_asset_frame0_v01/kinect_smc_tsdf_summary.json
```

Result:

- selected Kinect cameras: `0..7`
- mesh vertices: `71899`
- mesh triangles: `143532`
- coarse full-body silhouette is visible in Open3D
- head/face and hand detail are noisy/coarse, not mentor-quality

This is not a teacher pass and not a candidate pass.

## Same-Protocol Teacher Gate

The existing camera-axis-aligned Kinect TSDF teacher mesh was re-gated under the current original 6-view headshoulder protocol:

```text
output/normal_line_multiview_20260504/teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews/teacher_gate_summary.json
```

Explicit visual review fail:

```text
output/normal_line_multiview_20260504/teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews/visual_review_codex_fail.json
```

Failure facts:

- strict teacher pass: `false`
- numeric pass: `false`
- explicit visual pass: `false`
- max face_core depth-compatible coverage: `0.0008`
- max head_face depth-compatible coverage: `0.1885`
- max hairline depth-compatible coverage: `0.0`
- face_core depth-compatible hits by view: `[0, 0, 0, 0, 0, 5]`
- hairline depth-compatible hits by view: `[0, 0, 0, 0, 0, 0]`

## Decision

Raw Kinect is not a strict-passing head/face/hairline teacher for:

```text
0012_11_frame0000_6views_sparseproto_headshoulder_crop
```

Do not continue with Kinect projection patches, Kinect-to-VGGT shell fitting, or teacher-supervised training from this mesh. It may remain a coarse raw-sensor/body diagnostic, but it does not unblock one-frame ROI overfit or cloud training.

## Still Blocked

The refreshed registry remains red:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud upload/run: blocked

## Allowed Non-Wall Next Actions

- Keep raw dataset availability as a corrected fact, not a pass claim.
- Use annotations/SMPL-X only as weak body/hand topology and real-data bridge, not face/hair/clothing truth.
- If a new raw-data route is attempted, it must produce one shared 3D head/face/hairline surface that passes strict teacher gate before any training.
- Continue only with a genuinely new surface/representation mechanism or read-only asset audit; do not repeat threshold, epoch, confidence, support-radius, or Kinect projection loops.
