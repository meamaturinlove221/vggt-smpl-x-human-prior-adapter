# V6 Backend Execution Closure

Status: `research_only_backend_round_complete_strict_gate_red`

This round moved v6 beyond contract/preflight: B-Fus3D0-v2, B-GS1, B-hair1, and B-hand8 all produced local 3D backend artifacts or backend-smoke outputs. None are teacher/candidate exports, none write predictions, and none unlock formal cloud.

## Strict Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
teacher/candidate/predictions/registry = none
```

Final referee:

```text
research_artifact_referee.py --overwrite
status = research_only_referee_strict_gate_red
artifact_count = 68
verdict = strict gate red; research artifacts remain blocked from formal cloud/export
```

Cloud guard:

```text
candidate route: cloud_allowed=false, strict_candidate_passes is 0
teacher-supervised route: cloud_allowed=false, strict_candidate_passes is 0, strict_teacher_passes is 0
```

## B-Fus3D0-v2

Implemented and ran:

- `tools/b_fus3d0_v2_latent_grid_sdf_backend_smoke.py`
- `output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke`
- `reports/20260507_b_fus3d0_v2_latent_grid_sdf_backend_status.md`

Inputs used:

- VGGT layer23 tokens `[1, 6, 1374, 2048]`
- latent grid seed `5832 = 18^3` points
- query evidence `576` rows across full_body, face_core, hairline, left_hand, right_hand

Artifacts:

- real/shuffle/zero/random-view latent-grid SDF field PLYs
- `b_fus3d0_v2_latent_grid_sdf_fields.npz`
- `b_fus3d0_v2_open3d_contact_sheet.png`

Control result:

```text
real occupied points = 2832
shuffle occupied points = 2039
zero occupied points = 1654
random-view occupied points = 1347
real_minus_shuffle_query_occupied = 0.2375
real_minus_zero_query_occupied = 0.1766
real_confidence_minus_zero = 0.2577
real_beats_controls = true
```

Decision: real backend smoke progress, but not strict pass. It is still deterministic smoke, not trained Fus3D.

## B-GS1

Implemented and ran:

- `tools/b_gs1_visibility_aware_free_gaussian_backend.py`
- `tools/b_gs1_open3d_contact_sheet.py`
- `output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend`
- `reports/20260507_b_gs1_visibility_aware_gaussian_status.md`

Artifacts:

- constrained baseline PLY
- raw free candidate PLY
- visibility selected free PLY
- visibility-aware combined PLY
- random-control combined PLY
- Open3D contact sheets

Key metrics:

```text
constrained mean_iou = 0.6300106063
visibility-aware mean_iou = 0.6282780026
visibility_minus_constrained_iou = -0.0017326037
visibility_minus_constrained_overfill = 0.0019980460
visibility_minus_random_iou = 0.0242658063
visibility_minus_raw_capped_iou = 0.0379129620
selected free count = 2300
```

Decision: produced new 3D Gaussian artifacts and beats random/raw controls, but fails v6 acceptance because IoU is slightly worse than constrained and overfill is slightly higher. Freeze this scoring recipe.

## B-Hair1

Implemented and ran by subagent:

- `tools/b_hair0_backend_smoke.py`
- `output/surface_research_preflight_local/B_hair1_backend_smoke_v6`
- `reports/20260507_b_hair1_backend_status.md`
- `reports/20260507_b_hair1_backend_status.json`

Artifacts:

- real/shuffle/zero/mask-only strand/Gaussian-chain PLYs
- head/hairline/head-top review artifacts
- render diagnostics

Key metrics:

```text
roots = 320
chain steps = 7
chain points = 2240
real mean_iou = 0.049992
real recall = 0.059868
real overfill = 0.760449
real root_score mean = 0.484822
shuffle root_score mean = 0.493231
zero root_score mean = 0.496867
mask-only root_score mean = 0.550573
real_beats_controls = false
```

Decision: true hairline backend smoke exists, but real loses to controls. Gate remains red; freeze this exact evidence recipe unless improved real-token coupling is added.

## B-Hand8

Implemented and ran:

- `tools/b_hand8_connected_hand_arm_surface_backend_smoke.py`
- `output/surface_research_preflight_local/B_hand8_connected_hand_arm_surface_backend_smoke`
- `reports/20260507_b_hand8_connected_hand_arm_backend_status.md`

Artifacts:

- left/right/combined connected hand-arm mesh PLYs
- left/right/combined connected hand-arm pointcloud PLYs
- Open3D review renders
- `b_hand8_open3d_contact_sheet.png`

Open3D read check:

```text
left pointcloud points = 16284
right pointcloud points = 16714
combined pointcloud points = 32998
```

Hard checks:

```text
left/right wrist connected to forearm = true
left/right palm continuity proxy = true
left/right depth range proxy = true
largest component ratio high = true
left/right finger_structure_visible = false
not_smplx_scaffold_only = false
```

Decision: produced new connected hand-arm backend artifacts, but no B-hand7 pass artifact. It is still scaffold-derived and lacks learned finger reconstruction.

## D-Line

Implemented and ran:

- `tools/research_artifact_referee.py`
- `output/surface_research_preflight_local/research_artifact_referee_20260507`
- `reports/20260507_research_artifact_referee_status.md`

Independent D-line agent also ran guard tests and wrote:

- `reports/20260507_v6_dline_referee_refresh_agent.md`
- `reports/20260507_v6_dline_referee_refresh_agent.json`

Decision: D-line remains red. No formal cloud train/infer/export is allowed.

## Final Decision

V6 backend implementation round is complete locally:

- B-Fus3D0-v2 produced the strongest progress: real > shuffle/zero/random in this smoke and new SDF/occupancy 3D artifacts exist.
- B-GS1 produced visibility-aware Gaussian artifacts but fails anti-overfill acceptance.
- B-hair1 produced rooted strand-chain artifacts but real loses to controls.
- B-hand8 produced connected hand-arm artifacts but remains scaffold-only for fingers.
- D-line confirmed strict gate red and cloud blocked.

No strict pass was written.
