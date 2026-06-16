# V10180 Auto-Evolved Depth-Cue Plus Geometry Repair Route

Created: 2026-05-28T17:38:53+00:00

The user's visual concern is valid: the V10150 board looked too 2D and lacked point-cloud depth feel.

Evidence:
- V10150 renderer used a direct `points[:, :2]` projection, so the original board was effectively 2D/orthographic.
- V10170 generated an oblique depth-shaded board: `D:\vggt\vggt-canonical-surfel-adapter\boards\V10170000000000000000_0012_11_frame001_oblique_depth_pointcloud_audit.png`.
- Candidate human PCA thickness ratio: 0.360287.
- Baseline human PCA thickness ratio: 0.358275.
- Candidate z-range gain over baseline: -0.003538.

Decision:
- Rendering must be repaired for all future mentor boards: oblique view, depth shading, side/local 3D views, same scene/same bounds.
- Rendering repair is auxiliary only. It cannot turn this candidate into mentor-ready.
- The candidate does not provide meaningfully stronger 3D morphology than the VGGT baseline, so route back to representation/geometry repair.

Next representation route:
1. Preserve real VGGT baseline high-confidence RGB/detail and real environment points.
2. Add canonical SMPL-X surfel/graph support only in visible weak regions.
3. Optimize explicit 3D thickness/side-view/limb-continuity objectives, not just front-view projection.
4. Produce human-main full-scene RGB point cloud plus same-scene controls.
5. Use projection only as auxiliary.
6. Face detail remains not applicable; allowed claim: head/face contour and hair region only.

Forbidden final claims:
- mentor-ready
- projection-only pass
- render-only pass
- facial detail improved
- metric-only pass
- route exhausted
