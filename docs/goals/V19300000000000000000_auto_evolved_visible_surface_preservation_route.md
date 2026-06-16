# V19300000000000000000 Visible Surface Preservation + Topology Infill Route

Created: 2026-05-28T22:46:11+00:00

## Conclusion

V192 ran on Modal A10 and used an upright body-local frame, but it remains fail-closed.

The key visual fact is that the VGGT baseline visible surface is more coherent than the learned shell outputs. V187/V190/V192 keep replacing too much of the visible human surface, producing torn point-cloud shells.

## New Route

V194 must preserve the source-visible VGGT RGB surface as the front visible layer and train topology-volume infill only in weak / hidden / back / disconnected regions.

Required changes:

1. Build a no-degrade visible-surface mask from V950 confidence, source visibility, and V536 no-change regions.
2. Keep VGGT baseline RGB points in visible high-confidence areas.
3. Decode SMPL-conditioned topology infill only for back shell, side shell, limb continuity, and weak regions.
4. Blend in infill without replacing the coherent visible front surface.
5. Compare against baseline, V186, V187, V190, V192, same-topology, shuffled, and thickness-only controls.
6. Mentor main evidence remains full-scene RGB point cloud with partial real environment.

Forbidden:

- whole-body shell replacement;
- render-only pass;
- metric-only pass;
- contour-only local detail;
- visual failure as external hard block.
