# Prompt For Regenerating The Current Architecture Figure

## Main Prompt

Create a clean academic architecture diagram on a white background, in the style of a CVPR paper figure, visually inspired by the original VGGT architecture figure.

The figure should be wide, horizontal, and left-to-right. Use elegant serif labels, thin gray arrows, soft pastel module colors, rounded rectangles, and a neat research-paper layout. Keep the original VGGT core recognizable in the middle, but extend it with the current project modules.

Show the pipeline as follows:

1. Left side: four stacked input image thumbnails labeled:
   - anchor view
   - supervised view
   - source-only near view
   - uniform tail source

2. Immediately to the right, add a new orange-tinted module titled:
   - ZJU Geometry View Sampler
   Inside this module, list:
   - anchor selection
   - geom_plus_raw pool
   - supervised vs source-only
   - depth-conf quality filter
   - drop-worst supervised view
   - nearest_plus_uniform_tail

3. Center of the figure: preserve the original VGGT core structure:
   - DINO tokens
   - add camera token
   - Global Attention
   - Frame Attention
   - x L layers
   Wrap this middle part in a light dashed rounded rectangle titled:
   - Original VGGT Core

4. Right side of the core:
   - Camera Head -> Cameras
   - DPT -> Depth maps
   Keep these visually close to the original VGGT figure.

5. Below the main model path, add a blue module titled:
   - Reliability-aware Supervision
   Inside list:
   - conf_depth_point_masks + depth_conf_maps
   - drop-worst supervised view
   - gradconfmask + anchor-conditioned routing
   Connect this block with dashed arrows from the sampler and from the depth branch, to indicate train-time supervision.

6. Bottom-right branch:
   - Depth + Camera Unprojection
   - geometry teacher / render support / point cloud
   Use a green rounded rectangle.

7. To its right, add a purple module titled:
   - Teacher-fixed Visual Lift
   Inside list:
   - mask_hole_fill_plus_guided
   - better foreground render

8. Add a small result badge near the visual-lift module:
   - 20 / 20 local
   - 20 / 20 cloud

9. Add a top banner badge centered above the figure:
   - Promoted stable lead = nearest_plus_uniform_tail + confdepth_dropworst + gradconfmask

10. Add a subtle note near the bottom:
   - The original VGGT backbone is preserved. The current project mainly adds data-side view policy, reliability-aware supervision, and a frozen-teacher visual-lift branch on top of Depth + Camera.

Style requirements:

- white background
- publication-quality vector look
- pastel orange for the ZJU sampler
- blue/yellow for Global / Frame attention
- green for DPT and unprojection / teacher blocks
- purple for visual lift
- thin gray arrows
- clean serif typography
- no dark theme
- no photorealistic 3D rendering
- no UI screenshot look
- no excessive shading
- no cartoon style

The diagram should feel like an upgraded “current project version” of the original VGGT architecture figure, not a generic software flowchart.

## Shorter Prompt

Draw a CVPR-style architecture figure based on the original VGGT diagram, but extended for the current ZJU project pipeline: stacked multi-view inputs -> orange ZJU Geometry View Sampler (anchor selection, geom_plus_raw pool, supervised vs source-only, depth-conf quality filter, drop-worst supervised view, nearest_plus_uniform_tail) -> preserved original VGGT core (DINO, add camera token, Global Attention, Frame Attention, xL) -> Camera Head and DPT -> blue Reliability-aware Supervision block -> green Depth + Camera Unprojection / geometry teacher block -> purple Teacher-fixed Visual Lift block with mask_hole_fill_plus_guided -> result badge “20/20 local, 20/20 cloud”. White background, serif paper-figure typography, pastel colors, thin gray arrows, elegant academic layout.

## Negative Prompt

Do not generate:

- dark background
- glossy product infographic style
- UI dashboard style
- clip-art icons
- cartoon colors
- handwritten text
- photorealistic collage
- random extra modules not listed above
- messy overlapping arrows
