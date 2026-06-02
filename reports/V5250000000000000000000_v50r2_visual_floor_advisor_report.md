# V525 V50R2 Visual Floor Observation-Distillation Advisor Pack

## Architecture Diagram

```text
VGGT visible world points + RGB + confidence
        |
        |  SMPL-X surface prior / part binding
        v
V523 visible-anchor guarded student
        |
        |  residual clipped to preserve V521/V50R2 readability
        v
model-owned human-main full-scene RGB point cloud
        |
        +-- same-scene legacy VGGT/student controls
        +-- V50R2 visual floor reference
        +-- local fidelity / anti-2D boards
        +-- viewer / bundle / artifact audit
```

## Route Positioning

This pack is the not-promoted advisor evidence for the V50R2 visual-floor observation-distillation route. V50R2 is used only as visual floor, teacher, and evaluation reference. It is not used as the final student output.

The current candidate is V523: a VGGT-observation anchored, SMPL-part-bound student. It keeps the V521 visual recovery and separates visible-anchor preservation from the baseline/control comparison.

## Why Previous Route Failed

V519 and V520 improved automatic distance and thickness metrics but visually returned to smeared template-like or blob-like morphology. Those routes did not preserve the V50R2-style visible human readability.

V521 restored readable human shape and full-scene context, but it failed closed because it compared against the raw visible anchor as if that anchor were a baseline to beat. V523 corrected the source roles: visible anchor is an input guard, while prior VGGT/student artifacts are used as the mentor-facing controls.

## Current Change

V523 uses a visible-anchor guarded residual: the model can only make small, clipped residual changes on top of VGGT visible observations, preserving the readable head, torso, clothing, legs, and scene context. Body-part labels are derived from SMPL surface features plus geometry rather than the earlier coarse image-bin rule.

## Experiment Loop

- V519: canonical surfel graph training, failed manual morphology.
- V520: pose-aligned surfel graph repair, failed manual morphology.
- V521: observation-anchored visible student, visual recovery but controls/local gates incomplete.
- V523: separated visible-anchor guard from legacy controls and repaired part binding.
- V524: visibility-aware V509/V510/V511/V512 gate routing passed for advisor-pack assembly.

## VGGT Baseline / Controls Comparison

Best-view control distances against the V50R2 floor reference:

```text
V523 true:               0.001107
visible anchor guard:    0.001022  (nonregression guard, not counted as a baseline win)
V517 VGGT baseline:      0.107372
V517 no-SMPL:            0.109875
V520 shuffled semantic:  0.124221
V520 SMPL graph only:    0.117946
```

V512 gate status: `V512_MANUAL_MENTOR_GATE_V523_PASS_ADVISOR_PACK_REQUIRED_NOT_PROMOTED`.

## Point Cloud Visual Evidence

- Human-main full-scene board: `boards/V5230000000000000000000_human_main_full_scene.png`
- Same-scene baseline / true / controls: `boards/V5230000000000000000000_same_scene_controls.png`
- V50R2 visual floor comparison: `boards/V5230000000000000000000_v50r2_visual_floor_comparison.png`
- Local fidelity board: `boards/V5230000000000000000000_local_fidelity_part_binding.png`
- Anti-2D side/depth/cross-section board: `boards/V5230000000000000000000_turntable_side_depth_cross_section.png`
- Manual gate annotated board: `boards/V5120000000000000000000_manual_gate_annotated.png`

## Local Fidelity

Visibility-aware local gate:

```text
head/hair:       cam06 nn=0.038222
torso/clothing:  cam21 nn=0.006550
arm/hand:        cam06 nn=0.012013
leg/foot:        cam11 nn=0.071827
```

## Limitations

- This is not promotion and does not modify registry or active candidate state.
- V50R2 remains teacher/reference only.
- Visible anchor preservation is a guard, not a claimed baseline victory.
- Fine facial details are not claimed; face/head evidence is contour or region-level only.
- The final artifact is advisor-ready evidence, not a registry promotion.

## Next Plan

Use this pack for mentor review. If stricter improvement over the raw visible anchor is required, the next route should add a withheld-view or missing-region completion target while preserving V523 readability.

## File List

- `boards/V5230000000000000000000_human_main_full_scene.png`
- `boards/V5230000000000000000000_same_scene_controls.png`
- `boards/V5230000000000000000000_v50r2_visual_floor_comparison.png`
- `boards/V5230000000000000000000_local_fidelity_part_binding.png`
- `boards/V5230000000000000000000_turntable_side_depth_cross_section.png`
- `boards/V5120000000000000000000_manual_gate_annotated.png`
- `viewers/V5250000000000000000000_v50r2_visual_floor_viewer.html`
- `bundles/V5250000000000000000000_v50r2_visual_floor_advisor_pack.zip`
