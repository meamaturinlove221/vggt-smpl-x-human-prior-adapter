# R40/R41 P40 Ownership And Full-Body Visual Closure

Date: 2026-05-04

## Current Truth

The strict registry remains red:

```text
reports/20260504_strict_gate_registry.json
```

- strict candidate passes: `0`
- strict teacher passes: `0`
- cloud upload/run: blocked

This note closes the remaining ambiguity around r40/r41 fixed-threshold
positives. They are not mentor-final candidates.

## Added Diagnostics

Read-only p40 ownership geometry audits:

```text
output/normal_line_multiview_20260504/p40_ownership_geometry_r40_vs_signfix_headshoulder/p40_ownership_geometry_summary.md
output/normal_line_multiview_20260504/p40_ownership_geometry_r41_vs_signfix_headshoulder/p40_ownership_geometry_summary.md
```

Read-only full-body cross-view overlap audits:

```text
output/normal_line_multiview_20260504/cross_view_overlap_r40_fullbody/cross_view_overlap_audit.md
output/normal_line_multiview_20260504/cross_view_overlap_r41_fullbody/cross_view_overlap_audit.md
```

Explicit Open3D visual fail payloads:

```text
output/normal_line_multiview_20260504/candidate_gate_r40_partaware_softweight_bodyhand_smoke1/visual_review_codex_fail.json
output/normal_line_multiview_20260504/candidate_gate_r41_mixed_headshoulder_fullbody_smoke1/visual_review_codex_fail.json
```

## What The Ownership Audit Shows

The fixed-threshold increases are not modeled-surface improvements.

r40 target-view depth-unprojection p40:

- face candidate-only coverage: `0.2379`;
- face lost coverage: `0.7621`;
- face new coverage: `0.0000`;
- face candidate-only central protrusion: `0.0035`;
- face normal abs angle: `5.1328 deg`.

r41 target-view depth-unprojection p40:

- face candidate-only coverage: `0.2170`;
- face lost coverage: `0.7830`;
- face new coverage: `0.0000`;
- face candidate-only central protrusion: `0.0042`;
- face normal abs angle: `10.7395 deg`.

So the candidates are not adding a coherent target-view face surface under the
strict p40 budget. They drop most of the baseline depth-unprojection face
support. Fixed-threshold blue/new pixels only mean more low-budget pixels enter
the mask; they do not prove face geometry.

## Full-Body / Hands Interpretation

The full-body overlap audits show moderate overlap for full/head/face but weak
hand support:

- r40 hands near fraction: `0.3480` world, `0.3892` depth;
- r41 hands near fraction: `0.3520` world, `0.3696` depth.

Same-view world-vs-depth is close, so this is not mainly a world/depth-sync
problem. The Open3D contact sheets reveal the real failure mode: side/back/iso
views remain multi-layer slabs and hands are detached sheet fragments.

## Visual Decision

After inspecting the r40/r41 contact sheets:

- head/face is not a modeled face;
- head/face/hairline contains large shell-like holes;
- full-body side/back/iso views do not look like a normal human surface;
- hands are not attached, complete, articulated hand geometry;
- both world_points and depth_unprojection are not acceptable.

Therefore r40/r41 remain negative even where fixed-threshold point counts are
high.

## Freeze Rule

Do not continue r40/r41-style routes by:

- only changing confidence ownership, ranking, threshold, or p40/fixed gates;
- treating fixed-threshold face count increases as progress;
- adding small body/hand weak-prior weights while the head/face surface remains
  a shell;
- using full-body overlap alone as a pass signal.

Any future body/hand route must create a normal-human Open3D full-body result
with attached hands and also preserve same-protocol head/face/hairline quality.
