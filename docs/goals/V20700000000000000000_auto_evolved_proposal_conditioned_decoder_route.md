# V20700000000000000000 Auto-Evolved Proposal-Conditioned Decoder Route

Current state:

V206 learned part proposal student ran on Modal A10 with 300 steps and 8192 points, but remains fail-closed.

Why V206 is not final:

1. V205 proposal diagnostics are useful and avoid locked visible surface contamination.
2. V206 consumes proposals only as training/selection weights after decoding.
3. The decoder still collapses to baseline-near output; three cases select almost no effective infill.
4. V206 does not beat V194/V202 or same-topology/shuffled controls.
5. The mentor board still lacks a clear visible improvement over VGGT baseline.

Root cause:

Proposal information is too late in the pipeline. It influences loss and final selection, but not the actual occupancy/residual decoding representation.

Next route:

V207 proposal-conditioned decoder.

Required repair:

1. Add proposal score, visible lock, target seed, body part, and weak score as decoder inputs.
2. Predict proposal-conditioned occupancy and residual before point selection.
3. Use a non-regression head to suppress changes on locked visible surface.
4. Use a local continuity head to strengthen clothing boundary, leg/foot edge, shoulder/neck, and head/hair contour only where source evidence exists.
5. Keep face-detail claim disabled.
6. Require full-scene mentor board and same-scene controls before any success claim.

Forbidden final states:

- proposal diagnostic ready
- training ran
- baseline-near variant
- metric pass
- projection pass
- contour-only overclaim
- external hard block for visual failure

Allowed final states:

- V207_PROPOSAL_CONDITIONED_DECODER_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
