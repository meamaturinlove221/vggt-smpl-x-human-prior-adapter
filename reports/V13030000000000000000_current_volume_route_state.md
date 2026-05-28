# V13030 Current Volume Route State

Status: `ACTIVE_FAIL_CLOSED_NOT_MENTOR_READY`

This route has made concrete progress but has not reached the final allowed state.

## Completed

- V10240 downgraded V10230 to checkpoint / internal visual hard block.
- V10250 audited current local artifacts and recovered V10170-V10230 evidence.
- V10260 standardized oblique/depth-cued 3D rendering and removed raw-XY mentor-board risk.
- V10300 rebuilt volume/thickness metrics and confirmed shuffled/thickness-only controls remain hard controls.
- V10400 generated weak-volume region masks for 4 cases.
- V105 implemented `VolumeAwareVisibleMorphologyStudent` with residual, front/back/side shell, occupancy, visibility, normal, RGB, and source heads.
- V106 wrote the volume supervision loss contract and smoke.
- V10700 ran a local tiny volume-aware matrix for 4 cases with model-owned outputs and no teacher/raw Kinect inference.
- V13010 and V13020 attempted stronger shell/topology volume repairs.

## Fail-Closed Findings

- V10700 true is thicker than baseline but does not beat shuffled/thickness-only controls.
- V13010 improves 3/4 cases against thickness controls, but `0013_01_frame001` still fails against `thickness_only_control`.
- V13020 improves topology coherence but still fails `0013_01_frame001` against `thickness_only_control`.
- Visual boards show better depth cues, but the human still has local shell tearing / sparse edge artifacts and cannot be called mentor-ready.
- Face detail remains not applicable; only `head/face contour and hair region` may be claimed.

## Next Route

Continue to a stronger V13040/V10700 Modal or local longer training route:

- train the volume-aware model longer instead of procedural shell pushing;
- add topology continuity and anti-tear regularization;
- make `thickness_only_control` a first-class adversarial control during training;
- preserve real VGGT environment and high-confidence baseline zones;
- generate full-scene human-main boards, controls, turntable/side-depth, and local 3D morphology close-ups;
- fail closed unless true visibly beats baseline, shuffled, same-topology, tiny, and thickness-only controls.

This is not an external hard block. It is an internal model/representation/training problem.
