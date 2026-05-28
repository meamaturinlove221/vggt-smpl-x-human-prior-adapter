# V20300000000000000000 Auto-Evolved Part-Specific Non-Regression Route

Status: fail-closed continuation.

V202 ran on Modal A10 with visible-surface non-regression and connected weak-region infill. It improved over V200 in score, but the mentor board still shows baseline as the cleanest visible human surface. V202 adds connected points but still contaminates clothing, leg, and foot boundaries and remains below V194/pose-frame/topology controls.

Do not continue global connected infill or selection-only postprocessing.

Next required route:

```text
baseline visible surface non-regression
    + part-specific weak-region heads
    + clothing boundary lock
    + leg/foot endpoint lock
    + small per-part infill quota
    + source-upright full-scene board
```

Hard gates:

- torso/clothing and leg/foot cannot regress against VGGT baseline;
- infill quota must be part-specific, not global;
- each local crop must show baseline / true / best control;
- true must beat baseline and hard controls in both source-upright visual board and topology metrics;
- face detail remains not applicable.

This is not an external hard block. It is a model objective/part-local decoder failure.
