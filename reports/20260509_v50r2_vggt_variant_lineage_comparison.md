# V50R2 VGGT-derived point-cloud lineage comparison

- Created UTC: `2026-05-09T12:33:59Z`
- Scope: only VGGT-derived point-cloud routes are compared in the main table.
- Non-VGGT teacher/sensor routes such as Kinect, COLMAP/MVS, 2DGS, PSHuman, DepthPro, and Sapiens are treated as reference or failure-route evidence, not as VGGT baseline variants.
- Important evidence boundary: several April VGGT variant directories remain locally, but their original `predictions.npz` files are no longer present after cleanup. Those rows are therefore historical-report evidence, not newly recomputed point clouds.

## What this answers

The mentor asked to compare the current result against baseline VGGT and other VGGT-based point-cloud estimation variants before deciding the next improvement direction. The comparison below separates two questions:

1. Which VGGT route gave measurable occupancy or consistency gains?
2. Which route actually produced clearer, more stable human point-cloud geometry in Open3D?

The answer is not the same. Crop and softmatte raised point counts substantially. Normal/self-geometry improved several consistency metrics. The prior-enabled V42/V50R2 route added normal and package evidence. But none of these, by themselves, proves a large visible improvement in full head/face/hair/right-hand geometry over the best VGGT baseline.

## Main lineage table

| method | family | evidence | full | head | face | visual / mentor reading |
|---|---|---|---:|---:|---:|---|
| base_full_6v_vggt_preprocess_full | VGGT baseline / original full image | historical_report_only | 40882 | 8994 | 4177 | baseline weak; head/face ROI sparse |
| human_crop_6v_vggt | VGGT + human crop input preprocessing | historical_report_only | 111094 | 24441 | 11523 | large occupancy gain; not sufficient as final head/face quality |
| human_crop_hardmask_6v_vggt | VGGT + hard human mask/crop | historical_report_only | 111078 | 24437 | 10712 | occupancy improves, global geometry shifts more than plain crop |
| human_crop_softmatte_6v_vggt | VGGT + soft matte crop | historical_report_only | 151734 | 33382 | 15127 | densest ROI, but less stable than plain crop |
| normal_r16_xview_selfgeom | VGGT + normal/depth/point self-geometry | historical_report_only | 184213 | 40527 | 14981 | normal consistency partly improves, but face ROI is below signfix and Open3D remains shell-like |
| r32_selfgeom_crop_weakprior | VGGT + crop + weak SMPL-X prior + self-geometry | historical_report_only |  |  |  | negative strict visual review; shell/ghost head-face and fragmented hands |
| v25_base_vggt_research_prediction | VGGT base model research prediction | recomputed_current | mean valid 11575.5 per view in V50R2 comparison | mean valid 4019.17 per view | included in head_face region | basic full-body point-map output exists; no normal route |
| v42_prior_enabled_vggt | VGGT + SMPL-X prior-enabled prediction | recomputed_current | mean valid 11575.5 per view | mean valid 4019.17 per view | included in head_face region | normal evidence available; main point-map coordinate change from V25 is small |
| v50r2_candidate | VGGT candidate package with SMPL-X native evidence | recomputed_current | same main point map as V42 | tighter packaged head/face evidence, mean valid 2450.83 per view in comparison | packaged region evidence, not a larger raw point count | formal candidate closure and region evidence; not a new visibly sharper full-body point field |

## Key readings

- `human_crop` is the clearest early win over full-image baseline: full points rose from about `40.9k` to `111.1k`, head from `9.0k` to `24.4k`, and face from about `4.2k` / `3.7k` to roughly `9.6k-11.5k` depending on report source.
- `human_crop_softmatte` had the highest ROI point count, but also larger global deltas, so it is not automatically a better geometry result.
- `normal_r16_xview_selfgeom` implemented the mentor's depth/point/normal coupling idea and improved some consistency rows, but same-protocol face ROI stayed below signfix and the Open3D result remained shell-like.
- `r32_selfgeom_crop_weakprior` combined crop, self-geometry, and weak SMPL-X prior, but the visual gate still failed because head/face and hands remained shell/fragmented.
- `V42` / `V50R2` are the current prior-enabled VGGT path. They are useful for candidate closure and normal/region evidence, but the current full point-map coordinates are not visibly far from base VGGT: V42 vs V25 mean L2 is `0.00053544`, and V50R2 main candidate point map equals V42 (`max_abs = 0.0`).

## File audit for recomputation

| path | exists | size | mtime |
|---|---:|---:|---|
| `output/surface_research_cloud_preflight/V25_research_vggt_predictions/research_points_world.npz` | True | 105107011 | 2026-05-08T20:27:57Z |
| `output/surface_research_cloud_preflight/V42_prior_enabled_predictions/research_points_world.npz` | True | 105107787 | 2026-05-08T20:27:58Z |
| `output/frozen_candidates/V50R2_rebuilt_from_sessions_gdrive_modal/package_files/candidate_files__candidate_points.npz` | True | 17459703 | 2026-05-08T21:44:09Z |
| `output/modal_results/20260421_6views_preprocess_full_b40/predictions.npz` | False | None | None |
| `output/modal_results/20260421_6views_preprocess_crop_b40/predictions.npz` | False | None | None |
| `output/modal_results/20260428_normal_r16_xview_selfgeom_ckpt0_on6v_headshoulder/predictions.npz` | False | None | None |
| `output/local_inference_results/r32_confstable_geomonly1_on6v_fullbody/predictions.npz` | False | None | None |

## Mentor-facing conclusion

The strongest honest statement is: baseline VGGT already recovers a rough full-body point cloud; crop helps occupancy and image detail visibility; depth/point/normal self-consistency is implemented and technically active; SMPL-X prior-enabled V50R2 makes the result package stricter and supplies normals/region evidence. However, the current evidence does not yet show a decisive visible improvement in detailed human geometry over all VGGT baselines, especially for face/hairline/right hand. The next useful work should target local point-map-changing geometry, not only more reports or point-count increases.
