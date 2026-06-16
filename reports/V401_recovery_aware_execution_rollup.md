# V401 Recovery-Aware Execution Rollup

- Final status: `ALL_BRANCHES_EXECUTED_BUT_STRICT_PROMOTION_BLOCKED_WITH_EVIDENCE`
- Original V50 restored: `False`
- Active candidate: `V50R_rebuilt_after_artifact_loss`
- Active candidate path: `output/frozen_candidates/V50R_rebuilt_after_artifact_loss`
- Active candidate hash locked: `True`
- strict_candidate_passes: `0`
- strict_teacher_passes: `0`
- formal_cloud_unblocked: `False`
- right_hand_status: `MERGE_FAIL_SOFT_REVIEW_ONLY`
- teacher_status: `FAIL_FROZEN`
- forbidden_scan_status: `PASS` hits=`0`
- process_scan_status: `PASS` modal_apps=`0` modal_containers=`0`

## Strict Promotion Blockers
- active candidate is V50R rebuild, not original V50; requires new mentor/D-line acceptance
- full_body strict visual not pass: PASS_WITH_RISK
- head_close strict visual not pass: PASS_WITH_RISK
- face_close strict visual not pass: PASS_WITH_RISK
- hairline_close strict visual not pass: SOFT_REVIEW_ONLY
- left_hand strict visual not pass: PASS_WITH_RISK
- right_hand strict visual not pass: SOFT_REVIEW_ONLY
- sixty_view_support strict visual not pass: PASS_WITH_RISK
- temporal_overlay strict visual not pass: PASS_WITH_RISK
- right hand hard merge not pass: MERGE_FAIL_SOFT_REVIEW_ONLY

## Required To Finish Mentor Strict Task
- restore original V50 package/hash/registry/visual_review files, or
- run a new route that upgrades V50R visual gates from PASS_WITH_RISK/SOFT_REVIEW_ONLY to PASS_VISUAL, especially right hand and hairline, then rerun V400 strict promotion
- obtain independent dense teacher source if strict_teacher_passes is required
