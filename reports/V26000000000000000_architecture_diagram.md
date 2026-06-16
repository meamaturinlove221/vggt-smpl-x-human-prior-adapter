# V260 Architecture Diagram

```text
RGB/mask/camera -> full VGGT forward outputs
SMPL-X surfel/voxel/graph feature bank
        -> VerifiedFullForwardTokenPath
        -> SMPLFeatureEncoderV5
        -> DetailDensificationHead
        -> human-main high-density full-scene RGB point cloud
```
