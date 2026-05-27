# VGGT-SMPL-X Human Prior Adapter

<p align="center">
  <img src="docs/figures/vggt_smplx_human_prior_adapter_architecture.svg" alt="VGGT-SMPL-X Human Prior Adapter architecture" width="100%" />
</p>

## Route Position

This repository contains the **model-side route** of the VGGT + SMPL-X project.

The starting point is straightforward: vanilla VGGT can already estimate cameras, depth, point maps, and tracks from multi-view RGB, but it does not explicitly model human body topology. In practice, this means that full-body structure may appear, while head, face, hands, and other local regions remain weak or unstable. The purpose of this repository is to add a **human prior branch** without turning VGGT into a pure SMPL-X regressor.

In the current project split:

- **`VGGT-SMPL-X-Human-Prior-Adapter`** focuses on prior injection and model-side supervision.
- **`VGGT-ZJU-MoCap-Adapter`** prepares trusted dataset cases and alignment audits.
- **`vggt_for_4k_4d`** packages 4K4D-style cases, baseline/control comparisons, and advisor-facing evidence.

## What the Earlier Route Already Did

The earlier SMPL-X + VGGT route already had an important role in the project.

First, it provided a pose-aligned human structure prior. This gave the model a clearer notion of where the body, head, and hands should be.

Second, it made it possible to render **view-aligned prior evidence** under real cameras, such as silhouette-like signals, prior depth, prior points, and region cues. These are useful because they are not abstract body-model outputs; they are aligned to the same input views that VGGT sees.

Third, it improved region ownership. Instead of evaluating the human as a single undifferentiated point set, the route made it easier to reason about body, head, face, and hand regions separately.

At the same time, the earlier route also showed a clear boundary: if the SMPL-X prior is used too strongly, the output becomes template-driven. This can suppress details that should come from the images themselves, such as hair, clothing variation, and personal facial geometry.

## What This Repository Adds

The current repository keeps the main VGGT backbone intact and adds a **native human-prior path** on top of it.

The practical design is:

1. Build a posed SMPL-X mesh from the available body parameters.
2. Render view-aligned prior evidence under the real cameras.
3. Package this evidence into `prior_maps`, `prior_depths`, `prior_points`, and `prior_mask` style signals.
4. Inject the prior through a lightweight **HumanPriorAdapter**, instead of rewriting the whole VGGT architecture.
5. Keep the final geometry prediction owned by VGGT heads rather than by the body model alone.

This design choice is deliberate. The prior should help the model understand **where human structure is likely to be**, but the final details should still be decided by RGB evidence and geometric consistency.

## Geometric Logic of the Route

The route is not only about adding a body template. Its more important role is to connect multiple geometric outputs so that they do not contradict each other.

The underlying logic is:

- **Depth** describes visible surface distance in the target view.
- **Point maps** are the 3D expression of depth under camera geometry.
- **Normals** describe local surface orientation.
- A geometric normal can be derived from the predicted depth or point map.
- The predicted normal should therefore be consistent with the geometry implied by depth and points.

Even when this repository is used without the later normal-heavy branch, it still sits inside that same project logic: the human prior is useful only when it supports a more self-consistent 3D reconstruction.

## Experimental Closure

In this project, a model-side change is not treated as complete just because the loss improves.

The experimental loop for this repository is:

- prepare a trusted multi-view case with cameras, masks, and SMPL-X parameters;
- build prior inputs and prior supervision targets;
- run the vanilla VGGT baseline;
- run the prior-aware adapter;
- run controls such as no-prior / random-prior / shuffled-prior when applicable;
- export same-view visual comparisons for later evidence review.

This makes the repository useful not only for training, but also for controlled ablation.

## How We Judge the Result

Three levels are kept separate throughout the project:

1. **Metric pass**: losses or geometry metrics improve.
2. **Visual pass**: the 3D point cloud is visibly more human-structured.
3. **Advisor pass**: the improvement is visible in a **human-main full-scene RGB point cloud** under the same scene bounds and view.

This distinction matters. An isolated human scatter plot, a projection overlay, or an SMPL-X-only visualization may be useful during debugging, but none of them is the main success evidence.

## Repository Boundary

This repository should be understood as a **student-model route**.

- Dense teacher signals, raw depth fusion, or observation-verified references can be used for supervision or diagnosis.
- SMPL-X itself provides the structural prior.
- The final student candidate must still come from the VGGT-based route.

In other words, the repository is about **guiding VGGT with human priors**, not replacing VGGT with a body-model output.

## Current Status

At the present stage, this repository should be read as an **active research route** rather than a finished final system.

Its value is that it establishes a clean model-side place for SMPL-X prior injection, controlled supervision, and later comparison against baseline and controls. Whether a particular checkpoint qualifies as an advisor-pass result still depends on the evidence exported by the wider project pipeline.

## Figure

The architecture figure above is stored in:

```text
docs/figures/vggt_smplx_human_prior_adapter_architecture.svg
```
