# V13000 Auto-Evolved Anti-Billboard Topology-Volume Route

Status: `ACTIVE_FAIL_CLOSED_CONTINUE`

V13040 accepted the user visual correction: the current human point clouds still read as two-dimensional billboard / textured sprite geometry in turntable, side-depth, and cross-section views. This is not a renderer-only issue and not solved by increasing PCA thickness.

## Failed Gate

- `V13040_ANTI_BILLBOARD_FAIL_CLOSED_CONTINUE`
- `topology_volume_true` still billboard-fails on 3/4 cases.
- `same_topology_no_semantic` and shuffled controls can score close to or better than true on anti-billboard occupancy.
- Current `volume-aware` and `topology_volume` routes are therefore checkpoints, not mentor-ready.

## Root Cause

The model/candidate path is still mostly a single textured surface with procedural shell offsets. It increases thickness locally, but it does not create topology-connected front/back/side occupancy for torso, limbs, head/hair contour, shoulder/neck, clothing boundary, and leg/foot regions.

## Architecture Repair

Move from thickness-only repair to anti-billboard topology-volume:

```text
VGGT baseline high-confidence points
    + SMPL-X graph/body-part masks
    + weak-volume regions
    -> part-aware front/back/side shell occupancy
    -> cross-section occupancy regularization
    -> limb/torso continuity repair
    -> same-scene full-scene RGB point cloud
```

Hard constraints:

- no raw Kinect or teacher points at inference;
- preserve VGGT baseline high-confidence/no-change zones;
- environment stays from real VGGT scene points;
- face detail remains not applicable for back/side-back cases;
- projection and render are auxiliary only;
- success requires human-main full-scene 3D visual pass plus controls separation.

## Data Repair

Reuse current local evidence:

- `V536` geometry part graph;
- `V10400` weak-volume regions;
- `V10700` baseline and hard controls;
- `V13020` topology-volume checkpoint;
- `V13040` anti-billboard audit.

Do not reuse:

- renderer-only improvement as proof;
- thickness-only control as proof;
- contour-only local crops as proof;
- same-topology or shuffled pseudo-volume as final evidence.

## Exact Next Plan

1. Build `V13050` topology-volume occupancy candidate.
2. Generate model-owned NPZ/PLY for all 4 cases.
3. Run anti-billboard metrics against baseline, posthoc, same topology, tiny, shuffled, and thickness-only controls.
4. Generate turntable and cross-section boards.
5. Fail closed unless true beats controls in both visual boards and anti-billboard metrics.

## Final Allowed States

- `V30000000000000000000_ANTI_BILLBOARD_TOPOLOGY_VOLUME_MENTOR_READY_NOT_PROMOTED`
- `V30000000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION`

The second state is only allowed for real external blockers such as missing required files, unreadable assets, Modal/GPU outage, permission failure, or disk exhaustion. Visual failure, billboard geometry, weak control separation, or need for more training must continue auto-evolution.

## No-Agent Rule

No agent/subagent launch in this run.
