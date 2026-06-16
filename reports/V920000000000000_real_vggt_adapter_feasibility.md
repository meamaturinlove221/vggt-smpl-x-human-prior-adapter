# V920 Real VGGT Adapter Feasibility

Decision: enter V930 token extraction

Findings:

- Current repo has VGGT.forward and Aggregator: True / True
- Current repo VGGT.forward accepts sparse_prior_tokens: True
- Current repo Aggregator accepts sparse_prior_tokens: True
- Feature-adapter and vggt-main were also scanned.
- Checkpoint candidates found in scanned roots: 36

V930 must execute real VGGT.forward or real Aggregator.forward and save tokens with provenance. TinyV330 synthetic tokens are forbidden from final evidence.
