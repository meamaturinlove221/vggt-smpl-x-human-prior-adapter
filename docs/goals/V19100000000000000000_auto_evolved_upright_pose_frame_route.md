# V19100000000000000000 Upright Posed-Frame Body Layout Route

Created: 2026-05-28T22:34:21+00:00

## Conclusion

V190 ran on Modal A10 and made pose-frame shell supervision active, but the mentor visual remains fail-closed.

The restored V950 `posed_world_xyz` and `world_points` are not grossly misaligned, so the current failure is not an asset-restoration or coordinate-frame hard block.

## Failure

- V190 still renders as a tilted / torn topology-volume cloud.
- Hard controls and V187/V186 priors remain close or better in several cases.
- The main figure is not a natural human-main full-scene RGB point cloud.

## Next Repair

V192 should enforce an upright posed-frame body layout before decoding:

1. derive a body-local vertical/forward/right frame from SMPL part anchors;
2. normalize each case into a mentor upright frame for training and render;
3. decode per-part occupancy in that frame, then transform back to world coordinates;
4. add explicit head-torso-limb order losses and limb continuity in body frame;
5. keep real VGGT environment insertion and same-scene controls.

Do not continue with render-only, thickness-only, V186 fallback, or free visible-anchor tuning.
