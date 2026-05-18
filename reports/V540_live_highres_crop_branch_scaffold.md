# V540 Live High-Res Crop Branch Scaffold

This branch adds a default-off, identity-initialized high-resolution crop geometry branch to VGGT.

What changed:

- `vggt.models.highres_crop_geometry.HighResCropGeometryBranch`
  - consumes per-crop human features and `(view, y, x)` source-pixel indices
  - predicts local point/depth/normal residuals plus gate/uncertainty
  - scatter-adds only into the provided source-pixel indices
  - initializes as strict identity
- `VGGT(..., enable_highres_crop_geometry=True, highres_crop_feature_dim=...)`
  - optional forward-path integration
  - default remains disabled and backward compatible
- `tools/smoke_highres_crop_geometry_branch.py`
  - verifies identity, gradient flow, optimizer step, and no outside-index changes

Smoke result:

```text
identity_l2 = 0
loss_start = 1.5999997e-07
loss_end = 3.8815813e-08
grad_nonzero = true
outside_changed_pixels = 0
pass = true
```

Boundary:

This branch is a production-code scaffold for the live high-res crop path. It does not generate a candidate package, mentor package, or strict registry entry. The active research candidate remains `V11700_gap_reduction_branch_520` until mentor-visible full/head/hair/hand gates pass.
