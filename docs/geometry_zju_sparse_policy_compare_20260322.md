# ZJU Sparse Policy Comparison

This note is the shortest answer to the question:

> after round 1 showed that sparse fixed subsets were weak, did target-aware sparse source selection actually help?

Artifacts:

- [round1 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_coreview390_v1/summary.md)
- [round2 summary](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round2_coreview390_targetaware_v1/summary.md)
- [automatic comparison](/f:/vggt/vggt-main/output/geometry_view_sweep_zju/round1_vs_round2_targetaware_v1/comparison.md)

## Short Answer

- For `6src_hist`: **yes, noticeably**
- For `12src_nested`: **only slightly, and not enough**

## Common-Case Comparison

Across the `252` sparse cases shared by both rounds:

- round 1 depth wins: `32`
- round 2 depth wins: `55`
- improved-to-depth: `49`
- regressed-from-depth: `26`

Average change on common cases:

- geometry gain delta: `+0.000309`
- coverage gain delta: `+0.013762`

That means the target-aware policy made `depth + camera` more favorable overall, but the effect is profile-dependent.

## Profile Split

### `6src_hist`

On the `153` shared `6src_hist` cases:

- depth wins: `28 -> 46`
- average geometry gain delta: `+0.000896`
- average coverage gain delta: `+0.024030`

Interpretation:

- the old fixed 6-view subset was indeed suppressing the geometry branch
- once the sparse subset follows the target camera, `depth + camera` becomes much more competitive

### `12src_nested`

On the `99` shared `12src_nested` cases:

- depth wins: `4 -> 9`
- average geometry gain delta: `-0.000598`
- average coverage gain delta: `-0.002107`

Interpretation:

- merely rotating the old 12-view template is not enough
- the 12-view sparse profile still behaves more like a `point_map`-leaning regime

## Decision

The correct conclusion after round 2 is:

1. Keep the geometry-chain mainline.
2. Do not go back to ghost objectives.
3. Treat sparse-view source selection as a real variable, not a detail.
4. For the next sparse baseline, keep `6src_hist` target-aware.
5. For the next difficult problem, focus on redesigning `12src_nested` or diagnosing sparse geometry quality.
