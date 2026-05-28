# V19500000000000000000 Surface-Locked Sparse Topology Completion Route

Created: 2026-05-28T23:01:20+00:00

## Conclusion

V194 is the first route that keeps the coherent visible VGGT human surface, but it is not mentor-ready.

The main figure is more human-readable than full shell replacement, yet the infill is a noisy cloud and hard controls / topology metrics still fail.

## Next Route

V196 should perform surface-locked sparse topology completion:

1. Keep 70-85% of visible VGGT baseline points unchanged.
2. Add only 8k-18k infill points, not broad 60k replacement clouds.
3. Infill must be connected to nearby baseline surface and SMPL body-part adjacency.
4. Penalize floating infill and points farther than a local radius from the visible surface.
5. Use part-specific infill bands for shoulder/neck, clothing boundary, arm endpoint, leg/foot, and back/side shell.
6. Generate mentor full-scene RGB board and same-scene controls with real environment.

Forbidden:

- whole-body shell replacement;
- free infill clouds;
- metric-only or render-only pass;
- claiming face details.
