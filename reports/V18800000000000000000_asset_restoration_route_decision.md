# V188 Asset Restoration Route Decision

## Conclusion

V187 is a real Modal A10 run, but it remains fail-closed and diagnostic-only.

It used the fallback path because the local ignored training assets needed by the true canonical surfel route are currently missing:

- `output/V9500000000000000_smpl_feature_bank_v4`
- `output/V5360000000000000000_geometry_part_binding_repair`
- `output/V161000000000000_repaired_detail_regions`
- older hard-control matrices such as V107/V173/V183

The fallback used V186 model-owned predictions as surrogate surfel support. That allowed a useful diagnostic run, but it cannot be final mentor evidence. It also did not solve the visual gate: the output remains a floating visible-anchor volume cloud, not a human-main full-scene RGB point cloud that clearly beats baseline and controls.

## Evidence

- `reports/V18700000000000000000_runtime_environment.json`
- `reports/V18700000000000000000_training_decision.json`
- `reports/V18700000000000000000_training_manifest.csv`
- `reports/V18700000000000000000_visible_anchor_scores.csv`
- `boards/V18700000000000000000_visible_anchor_board.png`
- `boards/V18700000000000000000_visible_anchor_turntable_cross_section.png`

## Decision

Do not continue tuning V187 fallback. It is useful only as a diagnostic.

The next route must first restore or rebuild the original visible-anchor training assets, then rerun V187-style anchoring on real V950/V536/V161 inputs.

## Required V188 Work

1. Restore or rebuild:
   - V950 SMPL feature bank;
   - V536 geometry part binding graph;
   - V161 repaired detail / visible target regions;
   - V107/V173/V183 controls if available.
2. Verify NPZ readability and case coverage for:
   - `0012_11_frame001`;
   - `0013_01_frame001`;
   - `0021_03_frame001`;
   - `current_v895_0021_03`.
3. Rerun visible-anchor canonical surfel training without fallback.
4. Fail closed unless the full-scene mentor visual passes.

## Hard Fail Policy

This is not a `TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION` yet because there are still recoverable sources to inspect:

- Modal historical volumes;
- source repos;
- current diagnostic predictions;
- existing scripts that can regenerate V950/V536/V161.

Only if these sources are exhausted or inaccessible for three consecutive goal turns may this become a true external hard block.
