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
- V13040 added an anti-billboard topology-volume gate after user visual review found that the current boards still read as 2D billboards / textured sprites.
- V13050 attempted a topology-volume occupancy candidate using weak-region front/back/side occupancy, part-local side offsets, and cross-section-aware replacement.
- V13060 added a trainable anti-billboard runner and Modal entrypoint. A local CPU smoke ran end-to-end after detecting that local CUDA is incompatible with the installed PyTorch build for RTX 5080 (`sm_120`).

## Fail-Closed Findings

- V10700 true is thicker than baseline but does not beat shuffled/thickness-only controls.
- V13010 improves 3/4 cases against thickness controls, but `0013_01_frame001` still fails against `thickness_only_control`.
- V13020 improves topology coherence but still fails `0013_01_frame001` against `thickness_only_control`.
- Visual boards show better depth cues, but the human still has local shell tearing / sparse edge artifacts and cannot be called mentor-ready.
- User visual review correctly identified that the current boards still read as a 2D billboard / textured sprite. V13040 formalized this as a hard gate: turntable, side-depth, and cross-section views must show topology-connected 3D body volume, not only a thicker sheet.
- V13050 improves the anti-billboard score on the inspected 0012 case, but it still fails: `topology_volume_occupancy_true` remains billboard-like in 0012/0013, and same-topology/shuffled controls remain close or stronger in multiple cases.
- V13050 visual boards show local tearing and multi-layer textured-sheet artifacts. This proves that procedural shell/occupancy offsets are not enough.
- V13060 local smoke proves the trainable path is wired, but the short CPU run is not final evidence and regresses anti-billboard score; final training must run on Modal A10/A100 with longer steps and adversarial controls.
- Face detail remains not applicable; only `head/face contour and hair region` may be claimed.

## Next Route

Continue to a stronger anti-billboard topology-volume route:

- stop procedural shell/occupancy pushing as the primary route;
- train or rebuild the representation around learned front/back/side shell occupancy rather than hand-tuned offsets;
- add topology continuity, cross-section occupancy, limb/torso continuity, and anti-billboard losses;
- make `thickness_only_control`, `same_topology_no_semantic`, and shuffled/random controls first-class adversarial controls during training;
- preserve real VGGT environment and high-confidence baseline zones;
- run the V13060 Modal entrypoint for a real GPU training matrix before any final mentor claim;
- generate full-scene human-main boards, controls, turntable/side-depth, and local 3D morphology close-ups;
- fail closed unless true visibly beats baseline, shuffled, same-topology, tiny, and thickness-only controls in oblique, side-depth, cross-section, and same-scene boards.

This is not an external hard block. It is an internal model/representation/training problem.
