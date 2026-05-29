# V20400000000000000000 Auto-Evolved Part-Local Target Route

Status: fail-closed continuation.

V203 Modal A10 part-specific non-regression produced cleaner visible surfaces, but it nearly collapsed back to the VGGT baseline and still failed hard controls. V202 added more connected infill but contaminated clothing and leg/foot boundaries. The route is now trapped between:

- clean visible baseline with no clear improvement;
- noisy connected infill with boundary regression.

Next route must add explicit part-local targets instead of global infill or quota-only selection.

Required next architecture:

```text
VGGT visible baseline surface
    + strict no-regression mask
    + per-part weak target regions
    + SMPL-X adjacency-local target bands
    + clothing/leg/foot boundary target masks
    + part-local infill decoder
    + source-upright full-scene mentor board
```

Hard gates:

- true must not regress visible baseline in clothing/leg/foot/head-hair regions;
- true must add visible part-local improvement in at least two regions;
- same-topology/shuffled/thickness-only cannot be close or better;
- full-scene source-upright board remains primary visual evidence;
- face detail remains not applicable.

This is a model/data target definition problem, not an external hard block.
