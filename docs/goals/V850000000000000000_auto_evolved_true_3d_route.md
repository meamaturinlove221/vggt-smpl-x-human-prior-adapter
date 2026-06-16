# V850000000000000000 Auto-Evolved True 3D Morphology Route

Current repo:

D:\vggt\vggt-canonical-surfel-adapter

No agent / subagent.
No promotion.
No registry.
No V50 / V50R2 change.
Active candidate remains:

V11700_gap_reduction_branch_520

============================================================
1. Failed Gate
============================================================

V300100-V900 could not truthfully return mentor-ready.

Failed gates:

- V300 final/audit/report consistency in the uploaded pack;
- V140/V420 3D main board still lacks convincing full-scene environment;
- local close-ups are not yet true, region-specific 3D detail proof;
- V190 matrix is old V740 predictions plus new scoring, not a new 3D morphology student run;
- projection and fair-score metrics cannot replace mentor 3D morphology judgment.

============================================================
2. Root Cause
============================================================

The current artifacts contain useful full VGGT.forward and SMPL-X feature evidence, but the final 3D student is still derived from an older detail-verified prediction path. The route needs a real 3D morphology generator that produces the model-owned student directly in the V410/V850 namespace, with real scene environment points and region-specific 3D close-ups.

============================================================
3. Architecture Repair
============================================================

Use canonical SMPL-X surfel / graph representation as the primary body topology:

RGB / mask / camera
    ->
full VGGT.forward world points / depth / confidence / tokens
    +
SMPL-X surfel / voxel / graph / local frame
    ->
3D morphology student
    ->
human-main full-scene RGB point cloud

Projection stays auxiliary only.

============================================================
4. Data Repair
============================================================

- Bind real environment points from VGGT full-scene outputs or scene-context assets.
- Build true 3D local regions from SMPL part labels and VGGT high-confidence local geometry.
- Keep true and controls in the same point budget, view, bounds, and environment budget.
- Do not use copied-prediction rescoring as final training evidence.

============================================================
5. Exact Next Modal Plan
============================================================

Run a new V850/V860 matrix:

- cases: current_v895_0021_03, 0021_03_frame001, 0012_11_frame001, 0013_01_frame001;
- configs: true_3d_morphology_detail, VGGT baseline, posthoc, same topology, tiny token, shuffled/random, source-label-only, scaffold-only;
- outputs: model-owned NPZ/PLY, 3D main board, 3D local close-ups, projection auxiliary, viewer;
- gates: mentor visual gate first, representation gate, teacher/student gate, scene-context gate, controls gate, artifact audit.

Allowed final states:

A. V900000000000000000_TRUE_3D_MORPHOLOGY_DETAIL_MENTOR_READY_NOT_PROMOTED

B. V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION

No checkpoint or projection-only return.
