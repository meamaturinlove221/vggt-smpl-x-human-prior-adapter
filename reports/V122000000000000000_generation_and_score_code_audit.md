# V122 generation and scoring code audit

Audited `tools/v74000000000000000_detail_verified_densifier.py` and `models/v780_detail_verified_vggt_smpl_adapter.py`.

Findings:

- `weighted_pick(..., replace=True)`: True
- config-specific weights: True
- config-specific RGB/contrast operations: True
- `detail_bonus`: True
- `control_penalty`: True
- score function reads config name: True

Decision: V120 fair-score and high-density detail claims are downgraded. V150 rebuilds scoring from projection/mask/RGB/edge metrics only, with no config-name bonus or control penalty.
