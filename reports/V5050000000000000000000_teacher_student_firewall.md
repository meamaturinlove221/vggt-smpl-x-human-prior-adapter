# V505 Teacher/Student Firewall

## Scope

This firewall applies to the V50R2 visual-floor observation-distillation route.

V50R2 may be used only as:

- visual_floor
- teacher
- reference
- training_loss
- evaluation
- report_reference

V50R2 must not be used as final model output.

## Allowed

- Teacher points may be used to compute training losses.
- Teacher points may be used for evaluation.
- Teacher RGB may be used as visual reference and RGB-consistency target.
- V50R2 boards may be shown as visual floor comparison.
- V50R2 teacher bank may be loaded by training and audit tools.

## Forbidden

- Teacher points entering final inference inputs.
- Final prediction copying teacher points.
- Final RGB copied directly from teacher crop.
- V50R2 treated as the final student.
- Kinect/RGB-D/raw depth used at final inference.
- Registry or promotion writes from this route.
- V50/V50R2 source package modification.

## Student Contract

A valid final student must be model-owned:

- inputs: VGGT world points, VGGT RGB, VGGT confidence, SMPL-X graph, normals/local frame, camera/mask, environment points
- forbidden inputs: V50R2 teacher points, V50R2 teacher RGB crop, Kinect depth, raw teacher depth, direct fusion geometry
- output: model-owned human points/RGB inserted into full-scene VGGT environment

## Detector Policy

`tools/V505_teacher_copy_detector.py` must flag:

- exact teacher point copy
- exact teacher RGB crop copy
- exact teacher mask passthrough when presented as student ownership
- candidate metadata that marks V50R2/teacher as final inference source

Smoke must include both:

- a known leak candidate that is detected
- a safe synthetic non-copy candidate that is not detected

Passing the smoke does not prove mentor readiness. It only proves the firewall can catch direct copy leakage.
