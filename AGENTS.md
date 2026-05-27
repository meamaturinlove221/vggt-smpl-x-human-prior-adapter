# AGENTS.md

This repo is the canonical surfel adapter worktree for VGGT/SMPL-X mentor visual evidence.

Hard rules:

- Do not spawn agents or subagents unless the user explicitly re-authorizes it in the current turn.
- Do not promote, write registry entries, modify V50/V50R2, or replace the active candidate.
- Keep the active candidate as `V11700_gap_reduction_branch_520`.
- Treat `D:\vggt\vggt-feature-adapter` and `D:\vggt\vggt-scene-context-evidence` as historical/source repos unless a task explicitly says otherwise.
- The mentor main evidence must be a full-scene RGB point cloud where the human is the subject, partial environment remains visible, head/torso/limbs/hands are recognizable, and VGGT baseline / true adapter / controls are compared in the same scene, same bounds, same view, and same point size.
- Projection overlays, metrics, teacher/Kinect outputs, SMPL-only outputs, part-color views, canonical/T-pose diagnostics, isolated human scatter, heatmaps, and prototype baselines are auxiliary only. Never package them as mentor pass.
- If the main RGB point cloud is not human-main and recognizable, fail closed and route back to representation, transport, controls, or visual repair. Do not return `review ready`, `route exhausted`, `metric pass`, `visual pass`, or `limitation disclosed` as final.
- Report dirty worktrees honestly. Do not claim clean unless `git status` is actually clean.
