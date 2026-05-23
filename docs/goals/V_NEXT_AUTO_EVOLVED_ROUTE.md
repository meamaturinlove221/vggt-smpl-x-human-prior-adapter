# V29000000000 Camera-Bound Differentiable Renderer Repair Route

## Failure Attribution

V280 reprojection-aware scale search selected scale 0.0, meaning nonzero dense residuals are inconsistent with the current 4K4D K/RT binding. This blocks mentor-ready visual improvement under trusted reprojection.

## Architecture Hypothesis

Train a camera-bound differentiable residual where residuals are optimized jointly for topology score and reprojection trust. If the optimizer still selects zero residual, the failure is coordinate/calibration mismatch or the desired residual violates the camera-bound evidence.

## Hard Gates

- Use 4K4D SMC K/RT.
- Optimize nonzero residual with projection loss.
- Full-view visible improvement must remain nonzero.
- true must beat controls under combined topology/reprojection objective.
- If optimum remains zero, write user-action checklist asking for exact matched camera/calibration/coordinate convention for V11700/V920.
