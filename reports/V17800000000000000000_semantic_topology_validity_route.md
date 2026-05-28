# V178 Semantic Topology Validity Route

## Conclusion

V177 front/back occupancy failed closed and should not replace V173.

Best current candidate remains V173 Modal multi-shell topology decoder:

- Beats VGGT baseline on all four cases.
- Beats same-topology / shuffled / thickness-only on three of four cases.
- Still fails `billboard_fail_v2` on all four cases.
- 0013_01 still loses to same-topology and shuffled.

## What V177 Proved

V177 tried to explicitly preserve thin-axis front/back extreme bins, but it reduced multi-layer and dense-section quality. This shows the route cannot be solved by hand-picking front/back extreme points or by more point sampling.

## New Root Cause

The anti-billboard metric v2 still rewards some semantically invalid controls:

- same-topology can score high because it creates many occupied layers and dense sections;
- shuffled can score high because it creates thickness-like spread;
- neither necessarily proves correct head-neck-torso / shoulder-arm / torso-leg morphology.

Therefore the next gate must add semantic topology validity:

1. body-part adjacency consistency;
2. expected part order along the main body axis;
3. limb endpoint plausibility;
4. torso/head/leg connectedness;
5. cross-part collision and impossible overlap penalties;
6. same-topology/shuffled semantic invalidity detection.

## Required Next

Build V178 metric v3 and gate:

- keep anti-billboard score v2 as geometry component;
- add semantic topology validity score;
- require true to beat baseline and hard controls on both geometry and semantic topology;
- do not use this as final success unless the full-scene visual board also passes.

Then V179 should train with this semantic topology validity pressure or select checkpoints against it.
