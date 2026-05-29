# V21200000000000000000 Auto-Evolved Patch Skeleton-Volume Route

Current state:

V211 patch-local decoder ran on Modal A10 with 220 steps and 8192 points, but remains fail-closed.

Why V211 is not final:

1. V211 uses V210 patch sources and emits patch-local points, so it is a real step beyond pointwise scatter selection.
2. The full-scene board still does not show clear mentor-visible improvement over VGGT baseline.
3. Same-topology and shuffled controls still score higher than true.
4. The decoded patches mostly remain attached to visible clothing/back regions and do not form a coherent torso/limb/head volume.
5. All four cases still fail the combined topology-volume gate.

Root cause:

Patch source geometry is still built from visible proposal regions only. It lacks an explicit skeleton/part graph volume target that would connect patches into body-level torso, shoulder/neck, limb, leg/foot, and head/hair contour structure.

Next route:

V212 patch skeleton-volume decoder.

Required repair:

1. Build a part graph skeleton from SMPL-X/body-part anchors and VGGT visible surface boundaries.
2. Attach each patch to a skeleton edge or part node, not just a local visible surface patch.
3. Decode patch volume along skeleton directions with front/back/side shells and continuity to adjacent body parts.
4. Penalize same-topology and shuffled controls through semantic edge consistency, not generic thickness.
5. Preserve locked visible baseline surface and real environment.
6. Final evidence must remain full-scene RGB point cloud with same-scene controls and turntable/cross-section views.

Forbidden final states:

- patch-local decoder ran
- patch preview ready
- local patch scatter pass
- metric-only pass
- same-topology or shuffled stronger
- visual failure as external hard block

Allowed final states:

- V212_PATCH_SKELETON_VOLUME_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
