# V950 Real VGGT SMPL Feature Adapter

```text
4K4D SMC RGB frames
        ->
current repo VGGT.forward / Aggregator.forward
        -> real VGGT tokens

SMPL-X surfel / voxel / graph / body-part / visibility / projection features
        -> SMPLFeatureEncoder
        -> SMPL prior tokens

real VGGT tokens + SMPL prior tokens
        -> RealVGGTTokenBinder (cross-attention + gated binding)
        -> DetailPreservingDecoder
        -> model-owned scene-space RGB human point cloud
```

Source labels are auxiliary only. TinyV330/synthetic scene tokens and posthoc
point composition are forbidden as final evidence.
