# V21300000000000000000 Auto-Evolved Learned Skeleton-Volume Route

Current state:

V212 patch skeleton-volume diagnostic generated deterministic part-graph tubes and ran the full same-scene gate, but remains fail-closed.

Why V212 is not final:

1. V212 improves some cases over V211 by adding body-part skeleton-volume structure.
2. It is still deterministic geometry, not a learned/model-owned topology-volume student.
3. Same-topology and shuffled controls remain stronger than true.
4. The mentor board still shows local patch/tube additions rather than a natural, coherent full-body 3D point cloud.
5. All four cases still fail the combined topology-volume gate.

Root cause:

Part graph structure helps, but generic deterministic tubes do not bind enough to image-supported VGGT/SMPL features and can still be outcompeted by semantic-invalid controls. The next repair must train a learned skeleton-volume decoder with semantic edge conditioning and hard-control separation.

Next route:

V213 learned skeleton-volume decoder.

Required repair:

1. Use V210 patch sources and V212 skeleton edges as supervision inputs.
2. Condition decoding on real VGGT features, SMPL-X part graph, patch anchors, and semantic edge ids.
3. Predict skeleton-aware local occupancy and RGB-supported residual surfaces.
4. Add semantic edge consistency loss so same-topology/shuffled controls cannot win by generic thickness.
5. Preserve VGGT locked visible surface and real environment.
6. Generate full-scene mentor board, same-scene controls, turntable/cross-section, local morphology closeups, viewer, report, bundles, and cleanup only after hard gates pass.

Forbidden final states:

- deterministic tube pass
- metric-only pass
- same-topology or shuffled stronger
- local patch-only pass
- visual failure as external hard block

Allowed final states:

- V213_LEARNED_SKELETON_VOLUME_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
