# V13000 Auto-Evolved Volume Morphology Route

Created: 2026-05-28T18:44:09+00:00

Failed gate: V108/V114.

Root cause:
- The V107 tiny volume-aware student improves true thickness over baseline but does not beat shuffled/thickness-only controls.
- The visible local boards remain morphology/contour-level and do not prove fine detail.

Architecture repair:
- Build a multi-layer shell residual candidate that moves only V104 weak-volume visible body points.
- Preserve VGGT no-change/high-confidence zones.
- Keep real environment points unchanged.
- Fail closed if shuffled or thickness-only controls remain comparable.

No-agent rule: single-thread main run only.
