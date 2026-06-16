# V20100000000000000000 Auto-Evolved Visible-Surface Non-Regression Route

Status: fail-closed continuation.

V198 source-upright render audit showed the real VGGT baseline is already the most mentor-readable full-scene human surface for the current back/side-back cases. V196/V197/V200 filtering routes preserved more baseline surface but did not outperform the baseline or hard controls, and in several views damaged leg/foot/clothing continuity.

Therefore the next route must not be another nearest/moderate-offset/fixed-ratio point selection pass.

Required next architecture:

```text
VGGT baseline visible RGB surface
        -> hard no-regression mask for high-confidence visible points
        -> SMPL-X posed adjacency and part graph
        -> connected weak-region infill decoder
        -> leg/foot/clothing continuity losses
        -> full-scene source-upright mentor board
```

Hard gates:

- visible baseline high-confidence points must stay unchanged;
- leg/foot and clothing continuity cannot regress relative to baseline;
- infill must be connected to weak regions, not a detached cloud;
- source-upright full-scene board is required, but render pass is auxiliary;
- true must beat baseline, same-topology, shuffled, and previous V194/V200 routes;
- face detail remains not applicable.

No external hard block: the current blocker is model objective/representation, not missing assets.
