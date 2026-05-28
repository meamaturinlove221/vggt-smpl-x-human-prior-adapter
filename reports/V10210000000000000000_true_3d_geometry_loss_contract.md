# V10210 True 3D Geometry Loss Contract

Created: 2026-05-28T17:47:15+00:00

Purpose: continue the V950100 visual-supervised residual route after V10200 failed closed.

Primary mentor gate remains a full-scene RGB point cloud with human as subject and partial real environment. Projection, metrics, source labels, and render-only repairs are auxiliary.

Training inputs:
- VGGT baseline human/environment points and RGB.
- V536 canonical/graph visible body binding.
- V161 visible target points as supervision target only.
- Same-scene hard controls for separation.

Forbidden inference inputs:
- raw Kinect depth
- teacher points
- dense V591/Kinect fusion

Loss terms:
1. `weak_region_residual_l1`: fit residuals only in visible weak regions.
2. `baseline_preservation_l1`: preserve high-confidence/no-change VGGT baseline zones.
3. `thickness_side_loss`: increase valid side-view thickness only when topology remains coherent.
4. `limb_continuity_loss`: keep hand/arm, leg/foot, shoulder/neck regions connected rather than noisy.
5. `part_topology_loss`: respect graph/body-part neighborhoods.
6. `environment_preservation_loss`: keep real VGGT environment unchanged and visible.
7. `control_separation_loss`: separate true from posthoc, same-topology, tiny, shuffled controls in 3D morphology, not by source labels.
8. `projection_aux_loss`: optional auxiliary only; cannot rescue 3D visual failure.

Face detail policy:
- facial detail target applicable: false
- face detail claim allowed: false
- allowed claim: head/face contour and hair region only
