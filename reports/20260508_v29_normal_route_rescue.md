# V29 Normal Route Rescue

- status: `DONE_PASS`
- teacher_normal_available: `True`
- temporal_normal_available: `True`
- candidate_geometric_normal_available: `True`
- normal_depth_consistency_research_gate: `True`

## Region Support
- teacher: `{'body': 59474, 'face': 2422, 'head': 4114, 'left_hand': 736, 'right_hand': 472}`
- temporal: `{'body': 59341, 'face': 2417, 'head': 4103, 'left_hand': 731, 'right_hand': 464}`
- candidate: `{'body': 59474, 'face': 2422, 'head': 4114, 'left_hand': 736, 'right_hand': 472}`

## Sources
- teacher normals: V24 `teacher_normals_world`, propagated and normalized.
- temporal normals: geometric finite differences from V26 `target_frame_points`; V26 itself declared `normal_available=false`.
- candidate normals: geometric finite differences from V25 `research_points_world.npz`; marked `not_model_normal_head=true`.

## Guard
- No `predictions.npz`, candidate package, teacher package, strict registry, or strict pass was written.
