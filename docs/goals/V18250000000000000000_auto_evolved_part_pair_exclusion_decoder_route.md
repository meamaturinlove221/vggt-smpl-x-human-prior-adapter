# V18250 Auto-Evolved Part-Pair Exclusion Decoder Route

Created: 2026-05-28T20:58:35+00:00

This route continues the anti-billboard topology-volume goal after V181 failed closed.

## Hard Constraint

Main mentor evidence remains a human-main full-scene RGB point cloud with partial real environment. Metrics, projection, render, thickness, adjacency scores, and local crops are auxiliary only.

## Repair Target

The model must stop decoding one global multi-shell that lets semantically distant parts overlap. It must decode body parts in a semantic graph frame with explicit part-pair occupancy/exclusion.

## Required Work

1. Build a part-pair exclusion decoder contract.
2. Add invalid-pair exclusion heads for head-foot, torso-foot, arm-leg, and left-right endpoints.
3. Add valid-contact heads for head-torso, torso-arm, torso-leg, and leg-foot.
4. Train or smoke a model-owned student without teacher/raw Kinect inference.
5. Compare against V173, V181, VGGT baseline, same-topology, shuffled, thickness-only, and posthoc.
6. Generate full-scene mentor board, same-scene controls, turntable/cross-section, local 3D morphology closeups, and environment gate.
7. Fail closed unless the visual gate and V180/V4-style causality gates both pass.

## No Agent Rule

Do not launch agents/subagents unless the user explicitly changes this run's permission.
