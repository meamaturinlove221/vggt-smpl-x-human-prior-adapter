# V21100000000000000000 Auto-Evolved Patch-Local Decoder Route

Current state:

V210 patch geometry source builder produced per-case patch centers, local frames, radii, thickness estimates, and boundary anchors from V205 proposal weak regions.

Why V210 is not final:

1. V210 is a source/target builder, not a model-owned full-scene student output.
2. The generated patch sources are mostly concentrated on visible clothing/back/body boundaries.
3. Patch centers do not yet decode coherent local point geometry or improve the full-scene mentor board.
4. V210 is useful because it replaces pointwise scatter selection with structured patch targets, but it cannot be claimed as mentor visual pass.

Root cause after V209:

The current candidate pool can be forced to output more points, but those points remain sparse scatter. The next repair needs a patch-local geometry decoder that emits coherent local surfaces from each patch source.

Next route:

V211 patch-local geometry decoder.

Required repair:

1. Use V210 patch centers, frames, radii, thickness, and boundary anchors as decoder inputs.
2. Emit local patch point sets in patch coordinates before inserting into the baseline full scene.
3. Preserve locked visible VGGT baseline points and real environment.
4. Enforce patch continuity with neighboring visible points and semantic body-part graph edges.
5. Compare against V209, V208, VGGT baseline, same-topology, shuffled, and prior routes.
6. Generate full-scene RGB mentor board, same-scene controls, turntable/cross-section, and local 3D morphology closeups before any success claim.

Forbidden final states:

- patch source ready
- patch preview ready
- scatter infill pass
- metric-only pass
- same-topology or shuffled stronger
- visual failure as external hard block

Allowed final states:

- V211_PATCH_LOCAL_DECODER_FAIL_CLOSED_WITH_NEXT_ROUTE
- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

Do not launch agents or subagents in this route.
