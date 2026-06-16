# No-Wall Next Decision - Normal / Human Geometry Line

Date: 2026-05-03

## Current Truth

No local candidate satisfies the mentor gate. The latest strict registry scans
`90` candidate packages and `82` teacher packages:

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud upload: blocked

r34 true-700 token-grid is negative. It improves normal abs-angle metrics but
does not create modeled face/head/full-body/hands geometry.

## What r34 Proved

r34 ran real `[6, 3, 700, 700]` VGGT inference and mapped the result back to the
canonical 518 protocol. Therefore it is not the old fake-highres crop path.

Strict gate:

```text
output/normal_line_multiview_20260503/candidate_gate_r34_true700_tokengrid_r27_mapped518
```

Failure facts:

- world-points face p40: `15038 < 16825` signfix, delta `-1787`
- depth-unprojection face p40: `14896 < 16764` signfix, delta `-1868`
- fixed-threshold face counts rise, but this is not a pass
- normal metrics improve strongly, but signed predicted normals remain opposite
  to raw depth/point/SMPL-X derived normals
- depth/point camera-space disagreement worsens in median/p90 for head/face
- Open3D still shows shell-like head/face holes, slab-like full body side views,
  and fragmented hands

Decision: freeze r34. Do not continue by trying larger target sizes,
confidence boosts, fixed-threshold claims, or crop-consensus retuning.

## Routes Already Covered

- HART-style PnP camera replacement: negative; do not remove VGGT camera head.
- r16/r18/r19/r20 more epochs or same-config retry: normal/shape metrics can
  improve while face/head/full-body/hands remain invalid.
- r21/r23/r24/r25/r26/r27/r27b/r30: local self-geometry / weak body / photo /
  direct DPN family covered; no mentor pass.
- r28/r34 highres family: high-resolution/crop/token-grid alone does not create
  modeled face/head geometry.
- r33 SMPL-X canonical bins: useful correspondence diagnostic but not a geometry
  creator.
- r57-r68, Kinect/COLMAP/MVS/TSDF/Poisson/visual-hull/keypoint/MediaPipe/external
  teacher patches: blocked by strict teacher/candidate gates.

## Next Non-Wall Question

The remaining unresolved question is not "more normal metrics". It is:

> Can a single, shared 3D representation be formed from VGGT observations
> without treating SMPL-X or 60v/Kinect as a hard teacher, and can it pass the
> full/head/face/hands visual gate?

The only acceptable local experiment in this direction must obey:

- SMPL-X is used only as correspondence/index or weak body/hand topology.
- final points come from VGGT observations, not SMPL-X posed coordinates.
- no face/hair teacher from SMPL-X.
- no extra point-count credit from high-resolution grids.
- output must be packaged under the canonical 518 same-protocol gate.
- full-body and hands remain hard visual gates.

If this canonical/shared representation still renders as shell/slab/fragmented
hands, it is negative and should be frozen rather than tuned.
