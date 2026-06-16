# V20800000000000000000 Auto-Evolved Edit-Budget Decoder Route

Current state:

V207 proposal-conditioned decoder ran on Modal A10 with 300 steps and 8192 points, but remains fail-closed.

Why V207 is not final:

1. V207 correctly injects V205 proposal score, visible lock, target seed, body part, and weak score into the decoder.
2. The path is model-owned and does not use raw Kinect or teacher points at inference.
3. The route still stays baseline-near in three cases: connected infill is only 2, 505, and 22 points for 0012_11, 0013_01, and 0021_03.
4. The current_v895_0021_03 case reaches 5200 connected infill points, but same-topology, shuffled, upright-pose, pose-frame, and prior surfel routes remain close or stronger.
5. All four cases still fail the combined topology-volume visual gate.
6. The mentor board does not show a clear human-main full-scene improvement over VGGT baseline and hard controls.

Root cause:

Proposal conditioning reaches the decoder, but the model has no hard edit budget or connected-output constraint that forces proposal-supported weak regions to produce visible, connected, semantic local infill. The non-regression pressure is useful for safety, but it overpowers the repair path and collapses most cases back to a conservative baseline-near variant.

Next route:

V208 edit-budget constrained proposal decoder.

Required repair:

1. Predict an explicit per-part edit budget before occupancy and residual decoding.
2. Enforce a minimum connected infill quota for proposal-supported weak regions, while preserving locked visible baseline points.
3. Add a connected-component output loss over proposed edit regions, not just pointwise occupancy.
4. Add a semantic part separation loss so same-topology and shuffled controls cannot win by producing generic thick structures.
5. Decode local patches around proposal seeds using part-specific heads for clothing/torso boundary, leg/foot, shoulder/neck, head/hair contour, and arm endpoint.
6. Keep projection and metrics auxiliary; final mentor evidence must remain full-scene RGB point cloud with human as subject and partial real environment.
7. Keep face-detail claim disabled. Only head/face contour and hair region may be claimed unless source face visibility changes.

Forbidden final states:

- training ran
- proposal-conditioned smoke
- baseline-near variant
- connected infill in only one case
- metric pass
- projection pass
- contour-only overclaim
- same-topology or shuffled stronger
- visual failure as external hard block

Allowed final states:

- V208_EDIT_BUDGET_DECODER_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
