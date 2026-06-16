# V4050000000000000000 Auto-Evolved Head-Contour Body-Morphology Route

V404 face visibility gate shows the current cases are back/side-back views. Facial landmarks are not visible in the source images, so the route must not chase or claim eye/nose/mouth-level face detail.

Corrected target:

- keep the mentor main evidence as human-main full-scene RGB point cloud with partial environment;
- improve body morphology, head/hair contour, hand/arm endpoint, clothing/torso boundary, and environment visibility;
- keep projection and face/head ROI as auxiliary diagnostics only;
- forbid facial detail claims until a future case has visible face ROI and explicit 3D landmark evidence.

Immediate next actions:

1. Stop routing current back-view cases into a facial-detail target.
2. Build head/hair contour and body-boundary residual targets instead.
3. Preserve or transport body_part_id for all baseline/true/control outputs.
4. Regenerate same-scene 3D controls and local 3D close-ups with the corrected claim boundary.
5. Continue fail-closed if true is not mentor-obvious better than baseline/controls.

This is an internal representation/model target correction, not an external user-action hard block.
