# V20500000000000000000 Auto-Evolved Learned Part Proposal Route

Current state:

V20420 part-local target student ran on Modal A10 with 300 steps and 8192 points, but remains fail-closed.

Why V20420 is not final:

1. V20410 target masks are structurally useful but too narrow and mostly concentrated in clothing/torso bands.
2. V20420 preserved visible baseline surface, but the learned edit became too conservative.
3. Three cases selected only a tiny number of effective infill points after target filtering.
4. The current case selected more infill, but still did not beat V203/V202/V194 or hard controls.
5. The mentor board still looks like a baseline-near variant, not a clear adapter improvement.
6. Face detail remains not applicable; allowed claim is only head/face contour and hair region.

Failed gates:

- mentor visual gate
- true greater than baseline visual gate
- true greater than prior routes gate
- true greater than same-topology/shuffled controls gate
- local visible improvement gate

Root cause:

The route uses fixed part-local masks and strong preservation. This prevents contamination, but it also prevents the model from proposing visible improvements. The next model must learn part-local proposals rather than only obey fixed sparse masks.

Next route:

V205 learned part-local proposal plus non-regression ranking.

Required repair:

1. Keep V20410 target masks as hard safety anchors, not the full edit set.
2. Learn a part-local proposal score from VGGT confidence, weak score, body part, visible surface distance, SMPL feature support, and local continuity.
3. Use a ranking loss:
   - preserve visible baseline high-confidence points;
   - rank proposed edits higher only when they improve local continuity and do not contaminate RGB/body boundaries;
   - suppress edits that make same-topology/shuffled/thickness-only look better.
4. Decode only top-ranked proposals near visible weak regions.
5. Require local visual improvement in at least two visible regions before any mentor-ready claim.

Forbidden final states:

- metric pass
- mask pass
- target construction pass
- training ran
- baseline-near variant
- contour-only overclaim
- external hard block for visual failure

Allowed final states:

- V205_LEARNED_PART_PROPOSAL_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
