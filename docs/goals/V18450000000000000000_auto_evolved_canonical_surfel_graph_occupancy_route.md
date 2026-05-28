# V18450 Auto-Evolved Canonical SMPL-X Surfel Graph Occupancy Route

Created: 2026-05-28T21:08:26+00:00

This route continues the V13050-V600 anti-billboard topology-volume goal after V183 failed closed.

## Objective

Build a canonical SMPL-X surfel/graph occupancy student that produces a model-owned human-main full-scene RGB point cloud with partial real environment and same-scene controls.

## Required Model Route

1. Build or reuse a canonical SMPL-X surfel bank with body part, graph edges, normal, tangent, binormal, and local frame.
2. Sample real VGGT features/RGB/confidence/world-point support onto surfels.
3. Predict surfel occupancy, visibility, residual, local thickness, and RGB correction.
4. Use body-part graph continuity and distant-part exclusion losses.
5. Insert the occupied surfel result into real VGGT scene/environment points.
6. Compare against VGGT baseline, same-topology, shuffled, thickness-only, posthoc, and tiny controls.
7. Generate full-scene mentor board, controls board, turntable/cross-section board, local 3D morphology closeups, environment gate, viewer, report, bundles, and cleanup.

## Hard Gates

- No raw Kinect/teacher points at inference.
- Face detail is not applicable; claim only head/face contour and hair region.
- Metrics are auxiliary.
- If the main board is still billboard/sheet/torn, fail closed and continue.
- Do not return external hard block for visual failure.

## No Agent Rule

Do not launch agents/subagents unless the user explicitly re-authorizes them in the current turn.
