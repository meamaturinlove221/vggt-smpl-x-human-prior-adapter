# PPT Outline: SMPL / SMPL-X into VGGT

Date: `2026-04-16`

## Slide 1. Problem

- VGGT can reconstruct geometry well, but articulated human regions remain difficult.
- Failure modes: self-occlusion, sparse support, weak head / hair detail, noisy depth confidence.
- Research question: how can a parametric human prior help VGGT recover cleaner human geometry?

## Slide 2. What SMPL Is

- SMPL is a parametric human body model.
- It combines shape parameters, pose parameters, pose-corrective blend shapes, and linear blend skinning.
- Key value: realistic, differentiable, compact, and reusable across vision and graphics.

## Slide 3. What SMPL-X Adds

- SMPL-X extends SMPL to body + hands + face + expression.
- It has richer articulation and semantics for expressive human modeling.
- Scientific implication: fine human geometry should not stop at body joints alone.

## Slide 4. Why VGGT Needs a Human Prior

- Multi-view geometry alone can under-reconstruct human regions.
- Human shape is highly structured and does not behave like arbitrary background geometry.
- A parametric prior can provide view-consistent support and plausible completion targets.

## Slide 5. Current Fusion Philosophy

- We do not make VGGT directly regress SMPL parameters.
- We use SMPL as an external geometric prior.
- Fusion principle: `soft prior + pseudo supervision`, not `hard template replacement`.

## Slide 6. Current Landed Pipeline

- Read SMPL vertices from the dataset side.
- Project vertices into each training camera view.
- Build:
  - `smpl_prior_masks`
  - `smpl_prior_feature_maps`
  - `human_prior_completion_*`
  - `head_hair_*`
- Feed these tensors into VGGT training losses.

## Slide 7. Loss Design

- Depth branch:
  - emphasize head / hair detail regions
  - add pseudo depth supervision in prior-completed regions
- Unprojection branch:
  - reweight loss in human completion regions
  - add pseudo world-point supervision
  - regularize depth confidence and depth presence

## Slide 8. What Is Already Landed

- Landed in `ZJU` geometry training code:
  - `training/data/datasets/zju_vggt_geom.py`
  - `training/loss.py`
  - `training/trainer.py`
  - `training/config/...smplprior_headhair_longrun.yaml`
  - `scripts/probe_zju_vggt_geom_dataset.py`
- Not yet fused into current `4K4D` inference scripts.

## Slide 9. Why This Design Is Scientifically Reasonable

- keeps VGGT as the main predictor
- avoids template over-constraint
- preserves explainability through intermediate artifacts
- supports controlled ablation studies

## Slide 10. Evidence Path

- Probe exports visual and point-cloud artifacts for:
  - prior mask
  - prior feature map
  - completion depth
  - completion world points
  - completed point cloud
- Summary statistics include:
  - prior coverage
  - completion point count
  - completed point count
  - added ratio

## Slide 11. Limitation

- current implementation is SMPL-prior-oriented, not full SMPL-X semantic fusion
- head / hair emphasis is still partly heuristic
- 4K4D full-view inference results are a separate line from the ZJU training fusion

## Slide 12. Next-Step Proposal

- keep current tensor interface
- upgrade prior source from SMPL-style vertices to richer SMPL-X parts
- add part-aware priors for face / hands / torso / limbs
- add finer pseudo supervision and optional surface consistency losses

## Slide 13. Closing Message

- SMPL is already integrated into VGGT as a structured supervision prior
- the current design is principled, modular, and probeable
- the next research step is to elevate it from body-support prior to SMPL-X part-aware human geometry prior
