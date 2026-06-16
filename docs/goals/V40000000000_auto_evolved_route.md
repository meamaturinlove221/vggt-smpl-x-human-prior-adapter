# V40000000000 Auto-Evolved Route

## Failure Attribution

V304 solved the previous coordinate blocker by scanning all 8 SMC files, decoding PNG masks, using the correct `inv(RT)` camera convention, and finding a nonzero Sim(3)-style binding against `0021_03_annots.smc`.

V305 and V330 then showed that coordinate binding alone does not satisfy the mentor gate. Raw camera-bound metrics still rank `random_semantic` above `true_surface_transformer`. V350 can tune a local calibrator so the true sample wins by a small margin, and V360 can rank true first with an SDF-style score, but both are proxy/sample-level evidence rather than full-resolution trained camera-bound proof.

## New Architecture Hypothesis

Move from sample-level surface transport to full-resolution camera-bound point-transformer transport with differentiable silhouette supervision:

1. Use V304/V350 binding as camera initialization, not as final proof.
2. Train full-resolution true/control routes with a camera-bound loss.
3. Use SMPL-X surface graph tokens and point-transformer message passing.
4. Add differentiable silhouette/coverage/background leakage losses.
5. Preserve random/shuffled/noGraph/local smoothing/support/observation controls.

## Why Previous Route Failed

The prior route used existing sampled predictions (`65x65` and `130x130`) and camera-bound proxy scoring. It did not run a new full-resolution model that was optimized under camera constraints. As a result, random semantic and local smoothing controls remain too close under raw reprojection metrics.

## New Hard Gates

1. Full-resolution or explicitly sharded 518x518 camera-bound predictions.
2. True route beats random semantic, shuffled semantic, random graph, no graph, local smoothing, observation-only, support-only, and noSparse controls.
3. Raw camera-bound score, not only SDF-style proxy, ranks true first.
4. Learned normal residual remains nonzero.
5. Full-body/head/hair/hand visual boards are 3D scatter plus mask projection overlays.
6. Source manifests show no teacher/post-compose leakage.

## Training And Evaluation Matrix

Core groups:

- true_camera_bound_transport
- random_surface_semantic
- shuffled_surface_semantic
- random_surface_graph
- no_surface_graph
- local_knn_smoothing_surface
- observation_only
- support_only
- no_sparseconv_mlp
- no_teacher

Minimum seeds: 5 per group.

## Visual Proof Requirements

- full body projection overlay
- head-face close-up
- hairline close-up
- left hand close-up
- right hand close-up
- true/random/shuffled camera-bound comparison
- true/local smoothing/noGraph comparison
- learned normal residual visual

## Upload-Safe Packaging Rules

- core <= 50MB
- reports <= 50MB
- visuals <= 150MB
- selected predictions <= 250MB
- controls <= 250MB
- sidecar manifest with actual hashes
- internal npz readable

