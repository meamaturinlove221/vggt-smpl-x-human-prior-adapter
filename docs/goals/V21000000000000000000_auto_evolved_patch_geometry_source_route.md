# V21000000000000000000 Auto-Evolved Patch Geometry Source Route

Current state:

V209 forced edit-budget selection ran on Modal A10 with 300 steps and 8192 points, but remains fail-closed.

Why V209 is not final:

1. V208 proved proposal-conditioned edit-budget logits still stay baseline-near.
2. V209 removed conservative selection as a confound by forcing proposal-supported weak regions to receive connected edit budget.
3. V209 increases connected infill in all four cases, but the added points appear as sparse local scatter rather than coherent 3D human morphology.
4. All four cases still fail the combined topology-volume gate.
5. Same-topology and shuffled controls still score higher than true, so this cannot be claimed as causal model improvement.
6. The mentor board still does not show a clear human-main full-scene improvement over VGGT baseline and hard controls.

Root cause:

The failure is no longer only loss pressure or selection conservatism. The proposal-supported candidate geometry lacks a coherent local patch source/target for body morphology. Forcing more points selects more weak-region scatter, but it does not create connected torso, shoulder/neck, leg/foot, clothing-boundary, or head/hair contour structure.

Next route:

V210 patch-geometry source and target reconstruction.

Required repair:

1. Build explicit local patch targets for proposal regions from real VGGT high-confidence points, SMPL-X posed frame support, and preserved baseline boundaries.
2. Decode patch-local geometry rather than selecting individual residual points from the current candidate pool.
3. Represent each patch with centerline, tangent/binormal/normal frame, radius/thickness profile, front/back shell samples, and boundary anchors.
4. Require patch continuity to neighboring visible baseline points and semantic part graph edges.
5. Preserve locked visible surface and real environment exactly as auxiliary context.
6. Keep face-detail claim disabled; current source views only support head/face contour and hair region.
7. Mentor evidence remains full-scene RGB point cloud with same-scene baseline/true/controls, turntable/cross-section, local 3D morphology closeups, and viewer.

Forbidden final states:

- forced edit budget pass
- scatter infill pass
- metric-only pass
- projection-only pass
- same-topology or shuffled stronger
- local contour-only overclaim
- visual failure as external hard block

Allowed final states:

- V210_PATCH_GEOMETRY_SOURCE_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
