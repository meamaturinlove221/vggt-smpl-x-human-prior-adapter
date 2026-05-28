# V188 Auto-Evolved Asset Restoration Route

Current status:

`V18700_VISIBLE_ANCHOR_CANONICAL_SURFEL_FAIL_CLOSED_CONTINUE`

V187 used a fallback because local ignored training assets are missing. The fallback is diagnostic-only and still fails the mentor visual gate.

## Root Cause

The route has reached a point where further model changes require the original per-case training assets:

- V950 SMPL feature bank;
- V536 geometry part binding graph;
- V161 visible/detail target regions;
- V107/V173/V183 control matrices.

Without them, training is forced to use V186 predictions as surrogate surfel support, which creates circular fitting and cannot prove mentor-ready model output.

## Repair

V188 must restore or rebuild the missing assets before another mentor-readiness attempt.

Search priority:

1. local ignored output backups;
2. Modal volumes;
3. source feature-adapter repo;
4. scene-context evidence repo;
5. scripts that generated the assets.

After restoration:

1. verify all required NPZs are readable;
2. verify four-case coverage;
3. rerun visible-anchor canonical surfel training without fallback;
4. compare with V186/V187, baseline, same-topology, shuffled, thickness-only;
5. generate full-scene mentor board and fail closed if visual evidence is weak.

## Forbidden Success Claims

- fallback route ready;
- metric-only pass;
- visible-anchor score pass;
- asset-missing external hard block before recovery attempts;
- route exhausted;
- limitation disclosed;
- mentor-ready without full-scene visual evidence.

## Final Policy

Continue toward:

`V60000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED`

Only return:

`V60000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION`

after real recovery attempts prove the required assets cannot be restored or regenerated.
