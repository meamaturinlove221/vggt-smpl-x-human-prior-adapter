# V17000000000000000000 Auto-Evolved Anti-Billboard Route

当前状态：

V137 Modal A10 training completed, but V138/V140 fail closed.

Failed gates:

- V137 anti-billboard training matrix: all 4 true cases still have `billboard_fail_v2=true`.
- V138 mentor visual gate: full-scene advisor board exists, but it is not mentor-ready because the trained true output still fails anti-billboard checks.
- V140 causality gate: same-topology and shuffled controls remain close to or stronger than true; thickness-only remains close on some cases.

Root cause:

The first trained topology-volume student learned to satisfy the weak-region pseudo-target and cross-section occupancy loss, but the learned separation still does not create stable front/back/side body volume. The controls reveal that topology/semantic causality is not isolated: same-topology and shuffled controls can produce equal or stronger anti-billboard scores.

Architecture repair:

1. Replace the weak pseudo-target with a stronger part-graph cross-section target derived from SMPL-X local frames and visible part masks.
2. Add adversarial control separation during training, not only after training:
   - same_topology_no_semantic must be penalized if it matches true;
   - shuffled_smpl_feature must be penalized if it matches true;
   - thickness_only_control must be penalized if it matches true.
3. Add explicit limb/torso continuity supervision:
   - head-neck-torso continuity;
   - shoulder-arm continuity;
   - torso-leg continuity;
   - clothing boundary continuity.
4. Decode multiple shell points per anchor instead of replacing only the original anchor point, while keeping the same point budget by balanced resampling.
5. Keep baseline high-confidence no-change zones fixed.

Data repair:

- Use V134 billboard weak regions as the only residual-edit mask.
- Keep V404/V393 face guard: no facial detail claim because source faces are not visible.
- Continue to preserve real environment points from VGGT baseline and keep same-scene controls.

Exact Modal plan:

1. Build V17100 part-graph cross-section target assets.
2. Build V17200 adversarial anti-billboard training runner.
3. Run Modal A10/A100:
   - 4 cases;
   - true 3 seeds;
   - controls 1-3 seeds;
   - checkpoint boards at 100/300/600/1000 steps;
   - anti-billboard metric v2 and mentor visual boards at every checkpoint.
4. Generate V17300 full-scene advisor board, same-scene controls, turntable/cross-section, local morphology boards.
5. Fail closed unless true beats baseline, same-topology, shuffled, and thickness-only in both metric v2 and 3D visual boards.

Allowed final states:

- V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED
- V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No-agent rule:

No agent/subagent launch in this run unless the user explicitly reauthorizes it in the current turn.

Forbidden returns:

- checkpoint
- metric pass
- projection-only
- render-only
- thickness-only
- procedural occupancy only
- weak baseline separation
- weak control separation
- shuffled/same-topology better
- local contour-only overclaim
- route exhausted
- limitation disclosed
- review ready
- visual failure as external hard block
