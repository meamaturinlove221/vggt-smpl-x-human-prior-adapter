# V150 fair metric v2 definition

The old V740 score is not used. V150 fair score is config-agnostic and never reads the config name for a bonus or penalty.

Components:

- geometry coverage: same human and environment point budgets where applicable;
- mask inside ratio: projected points inside the real camera mask;
- silhouette edge alignment: projected points near the mask edge;
- RGB reprojection residual: absolute RGB residual against the real camera image;
- local crop visual score: mean projection score for head/hair, hand/arm, clothing;
- environment preservation: human ratio and environment budget.

No `detail_bonus`, `control_penalty`, true-only lift, active-count-only pass, or RGB-variance-only pass is allowed.
