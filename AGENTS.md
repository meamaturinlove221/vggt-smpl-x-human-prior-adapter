# AGENTS.md

This repo is the canonical surfel adapter worktree for VGGT/SMPL-X mentor visual evidence.

Hard rules:

- Do not spawn agents or subagents unless the user explicitly re-authorizes it in the current turn.
- Do not promote, write registry entries, modify V50/V50R2, or replace the active candidate.
- Keep the active candidate as `V11700_gap_reduction_branch_520`.
- Treat `D:\vggt\vggt-feature-adapter` and `D:\vggt\vggt-scene-context-evidence` as historical/source repos unless a task explicitly says otherwise.
- The mentor main evidence must be a full-scene RGB point cloud where the human is the subject, partial environment remains visible, head/torso/limbs/hands are recognizable, and VGGT baseline / true adapter / controls are compared in the same scene, same bounds, same view, and same point size.
- Projection overlays, metrics, teacher/Kinect outputs, SMPL-only outputs, part-color views, canonical/T-pose diagnostics, isolated human scatter, heatmaps, and prototype baselines are auxiliary only. Never package them as mentor pass.
- Face detail claims require source-view evidence. If the source RGB/camera is back-view or side-back and eyes/nose/mouth are not visible, do not pursue or claim facial detail. The allowed claim is limited to head/face contour, back-head contour, and hair region; optimize the visible body morphology instead: shoulder/neck, torso/clothing boundary, hand/arm endpoint, leg/foot morphology, and the human-main full-scene RGB point cloud.
- Render repair, projection overlays, and thickness metrics are auxiliary checks only. A mentor board must use oblique/depth-cued 3D rendering rather than raw `points[:, :2]`, but render improvement alone is never mentor-ready.
- If a shuffled/random/same-topology control is thicker or visually comparable to the true route, fail closed and route back to volume-aware representation/training; do not claim causal success from thickness-only gain.
- If the human reads as a flat billboard, paper cutout, textured sprite, or single-layer shell in front/back/side/turntable views, fail closed even when thickness metrics improve. The next route must use anti-billboard topology-volume representation: front/back/side shell separation, cross-section occupancy, limb/torso continuity, part-aware local volume, and same-scene control separation.
- Procedural occupancy, shell offsets, global normal pushes, and thickness-only repairs are checkpoints/controls unless produced by a trained model-owned topology-volume student and verified against same-scene hard controls. Visual failure is not an external hard block when model/training repair can continue.
- If the main RGB point cloud is not human-main and recognizable, fail closed and route back to representation, transport, controls, or visual repair. Do not return `review ready`, `route exhausted`, `metric pass`, `visual pass`, or `limitation disclosed` as final.
- Report dirty worktrees honestly. Do not claim clean unless `git status` is actually clean.
