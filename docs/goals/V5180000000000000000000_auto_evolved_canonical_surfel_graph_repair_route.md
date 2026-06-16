# V518 Auto-Evolved Canonical Surfel/Graph Repair Route

Trigger:
- V517 produced a full-scene model-owned candidate but V512 failed closed because the human remained blob-like and below the V50R2 visual floor.

Route:
- switch from free-point clarity composition to canonical SMPL-X graph/surfel representation;
- keep VGGT observation as scene/frame/RGB context;
- keep V50R2 as visual floor / teacher / reference only;
- generate same-scene controls: no SMPL graph, weak semantic, VGGT visible baseline;
- no promotion, no registry, no V50/V50R2 modification.

Next required step:
- train/adjudicate V519 with VGGT feature sampling on surfels, local body-part heads, anti-blob/anti-sheet losses, and strict controls before V509/V512 can pass.
