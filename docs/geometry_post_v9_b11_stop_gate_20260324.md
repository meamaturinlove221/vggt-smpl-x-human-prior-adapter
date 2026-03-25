# Geometry Post V9 B11 Stop Gate 2026-03-24

## Goal

- Run exactly one more post-`v9` residual cleanup round.
- Start from the highest-value repeated residual:
  - `Camera_B11 @ frame 1170`
  - with transfer checked across `frame 0 / 600 / 1080 / 1170`
- If this round still needs a narrower single-camera/single-frame patch pattern,
  stop adding overrides and do not open the next residual search family yet.

## Search Readout

- `B11 @ frame 1170` one-swap `uniform + nearest` search:
  - [summary.md](/f:/vggt/vggt-main/output/geometry_source_search_20260324/B11_frame1170_uniform_nearest_family_v1/summary.md)
- Best target-frame candidates:
  - `s1_011 = swap Camera_B6 -> Camera_B17`
  - `s1_017 = swap Camera_B2 -> Camera_B16`
- Best guard-pass target-frame readout:
  - `s1_011`
  - full delta: `-0.002776`
  - `fg_human` delta: `-0.005350`
  - `bg_far` delta: `-0.002645`
  - `bg_bottom_band` delta: `-0.000190`

## Controlled Local Contrast

- Four-frame `B11` probes:
  - [s1_017 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B11_s1_017_frames0_600_1080_1170_v1/summary.md)
  - [s1_011 summary.md](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/B11_s1_011_frames0_600_1080_1170_v1/summary.md)
- Comparisons against `v9`:
  - [v9 vs s1_017](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v9_vs_b11_s1_017_frames0_600_1080_1170/comparison.md)
  - [v9 vs s1_011](/f:/vggt/vggt-main/output/geometry_sweep_policy_transfer_20260324/v9_vs_b11_s1_011_frames0_600_1080_1170/comparison.md)

## Readout

- `s1_017` is not acceptable:
  - it flips `frame 1170` to `depth_unproject`
  - but it degrades the existing `frame 1080` depth win to `tie`
- `s1_011` is the better family:
  - `frame 1170`: `point_map -> depth_unproject`
  - `frame 600`: still `point_map`, but full and background metrics improve
  - `frame 1080`: stays `depth_unproject`
  - `frame 0`: stays `tie`
- But `s1_011` still does not qualify as the next main manifest upgrade:
  - it does not repair the repeated `frame 600` residual
  - and its four-frame transfer is already narrow enough that the next step
    would likely be another more specific patch family, not a cleaner reusable
    local source-policy rule
  - in the four-frame check, `frame 1080` also shows a positive `fg_human`
    delta (`+0.003748`) even though the full-frame decision remains depth-favored

## Decision

- Do **not** promote a `v10` override from this round.
- Freeze [zju_6src_hardcontrol_hybrid_v9_b1_b2_b4_b12_frameaware_b13_b15_b23_frames0_600_1080_1170.json](/f:/vggt/vggt-main/scripts/source_policy_overrides/zju_6src_hardcontrol_hybrid_v9_b1_b2_b4_b12_frameaware_b13_b15_b23_frames0_600_1080_1170.json)
  as the current local main manifest.
- Trigger the local stop condition:
  - stop manual residual patch collection after this round
  - do not immediately open `B16` or `B7`
  - do not start a new cloud run

## Current State

- Current local residual list remains the `v9` list:
  - `frame 0`: `Camera_B13`, `Camera_B23`
  - `frame 600`: `Camera_B11`, `Camera_B13`
  - `frame 1080`: `Camera_B23`
  - `frame 1170`: `Camera_B11`, `Camera_B16`, `Camera_B7`
- The main local conclusion is now:
  - `v9` is a safe local source-policy baseline
  - further progress is no longer best framed as more manual override cleanup
  - the next step should be a new explicit training/ablation question, still
    chosen locally before any cloud launch is reconsidered
