# V1800000000000000 Migration Handoff

## Final Status

`V1800000000000000_REAL_VGGT_SMPL_FEATURE_DETAIL_MENTOR_READY_NOT_PROMOTED`

This is not promotion. Registry was not modified. V50/V50R2 were not modified. Active candidate remains `V11700_gap_reduction_branch_520`.

## Repo

`D:\vggt\vggt-canonical-surfel-adapter`

No agent or subagent was launched in this route.

## Goal File

- `docs/goals/V900100000000000_V1800000000000000_real_vggt_smpl_feature_detail_goal.md`
- Manifest: `reports/V900100000000000_goal_file_manifest.json`
- Goal sha256: `c733ac57f179d8d88d6938dff10c802d61807c16912c60fc70fb4fb9aa80b525`
- Goal line count: `727`

## Main Evidence

- Advisor main board: `boards/V970000000000000_real_vggt_advisor_main_board.png`
- Same-scene controls: `boards/V970000000000000_same_scene_controls_board.png`
- Multi-sequence board: `boards/V970000000000000_cloudcompare_style_board.png`
- Viewer: `viewer/V1600000000000000_real_vggt_smpl_feature_viewer.html`

## Key Reports

- Final status: `reports/V1800000000000000_final_status.json`
- Final mentor gate: `reports/V1100000000000000_final_mentor_gate.json`
- Advisor report: `reports/V1400000000000000_real_vggt_smpl_feature_advisor_report.md`
- Bundle integrity: `reports/V1600000000000000_bundle_integrity.json`
- Cleanup: `reports/V1700000000000000_post_push_cleanup.json`

## Real VGGT Path

- V930 real token manifest: `reports/V930000000000000_real_vggt_token_manifest.json`
- V930 token shape audit: `reports/V930000000000000_token_shape_audit.json`
- V930 smoke: `reports/V930000000000000_vggt_forward_smoke.json`
- V940 SMPL feature schema: `reports/V940000000000000_smpl_feature_schema.json`
- V950 architecture contract: `reports/V950000000000000_architecture_contract.json`
- V950 gradient smoke: `reports/V950000000000000_forward_gradient_smoke.json`

## Modal Matrix

- Controller: `modal_v960_real_vggt_smpl_feature_matrix.py`
- Model: `models/v950_real_vggt_smpl_feature_adapter.py`
- Metrics: `reports/V960000000000000_seed_metrics.csv`
- Training manifest: `reports/V960000000000000_training_manifest.json`
- Downloaded outputs: `output/V960000000000000_real_vggt_matrix`

Modal result summary:

- Cases: `current_v895_0021_03`, `0021_03_frame001`, `0012_11_frame001`, `0013_01_frame001`
- Rows: `48`
- Failures: `0`
- GPU: `NVIDIA A10`
- True seeds: `0,1,2`
- Controls: baseline, no SMPL feature, random/shuffled SMPL feature, same topology no semantic, posthoc surfel, tiny synthetic token, SMPL-only template, source-label-only.

## Upload Bundles

All bundles are zip-clean, non-empty, and under 500 MB:

- `archive/V1600000000000000_core_bundle.zip`
- `archive/V1600000000000000_reports_bundle.zip`
- `archive/V1600000000000000_visuals_bundle.zip`
- `archive/V1600000000000000_viewer_bundle.zip`
- `archive/V1600000000000000_predictions_bundle.zip`
- `archive/V1600000000000000_controls_bundle.zip`
- `archive/V1600000000000000_real_vggt_tokens_bundle.zip`
- `archive/V1600000000000000_local_detail_bundle.zip`
- `archive/V1600000000000000_metrics_bundle.zip`
- `archive/V1600000000000000_multisequence_bundle.zip`

## Re-run Commands

```powershell
cd D:\vggt\vggt-canonical-surfel-adapter
python tools\V900100_V920_real_vggt_audit.py
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:OMP_NUM_THREADS='1'; python tools\V930_V940_real_vggt_token_and_smpl_features.py --case all --image-size 56 --embed-dim 32 --camera-ids 0,1 --device cpu
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:OMP_NUM_THREADS='1'; python tools\V950_real_vggt_smpl_adapter_smoke.py --case all --max-points 2048
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; python -m modal run modal_v960_real_vggt_smpl_feature_matrix.py --case-ids current_v895_0021_03,0021_03_frame001,0012_11_frame001,0013_01_frame001 --steps 300 --seeds 0,1,2 --max-points 4096
python tools\V970_V1800_real_vggt_smpl_finalizer.py
```

## Caveats

The repo worktree is dirty with many untracked research artifacts from this and previous routes. This was reported honestly in `reports/V1700000000000000_post_push_cleanup.json`.

V930 local smoke uses CPU-safe 56 px preprocessing to prove the real code path. V960 is the Modal GPU matrix used for the formal route evidence.
