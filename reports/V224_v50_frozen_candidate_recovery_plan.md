# V224 V50 Frozen Candidate Recovery Plan

- Original V50 found locally: False
- Original V50 found in limited Modal paths: False
- Original V50 hash locked now: False
- Rebuilt package: D:\vggt\vggt-main\output\V223_rebuilt_candidate_package
- Rebuilt archive: D:\vggt\vggt-main\archive\V223_rebuilt_candidate_package.zip
- Rebuilt archive sha256: 310563c3b68b50a968ca12baf54c71c5e15ea518bfed81116980a72856ce9407

## Decision

Original V50 cannot be honestly restored from current local/Modal evidence. The available recovery path is to use the rebuilt V223 package from V42/V25/V16 cloud evidence and run a new strict promotion transaction. This must create a new pass if successful; it must not claim the lost V50 hash.

## Safe Next Steps
- If user provides original zip/registry/hash files, restore them read-only and verify hash invariants before any use.
- Otherwise treat output/V223_rebuilt_candidate_package as a new candidate package input, not V50.
- Run a fresh strict promotion transaction against the rebuilt package; only then write a new registry/pass if gates pass.
- Do not reassert strict_candidate_passes=1 from lost V50 artifacts.
- Keep V223_rebuilt_candidate_package.zip as recovery archive evidence.
