# V513 Auto-Evolved Route: Continue V508 Target Matrix

## Trigger

V512 manual mentor visual gate failed closed.

The failure is not an external hard block. It is an internal evidence gap:

- V50R2 visual floor is established.
- Current V352/V353/V300/V910 boards regress below V50R2.
- V504 teacher bank exists.
- V505 firewall passes direct-copy smoke.
- V506 model and V507 losses pass smoke.
- V508 A10G 300-step checkpoint has run.
- V509/V510/V511/V512 fail because there is no accepted model-owned student from the full target matrix.

## Executed Auto-Evolution Step

V508 A10G 300-step checkpoint was executed and synced locally:

- `reports/V5080000000000000000000_modal_a10g_300_result.json`
- `reports/V5080000000000000000000_training_manifest.csv`
- `reports/V5080000000000000000000_seed_metrics.csv`
- `reports/V5080000000000000000000_hash_reconciliation.json`

## Next Required Execution

Continue the V508 target matrix:

- A10G: 600, 1000, 2000, 4000
- A100: 300, 600, 1000, 2000, 4000

After each checkpoint:

- run V505 teacher-copy detector against candidate outputs
- run V503-style V50R2 regression scoring
- only then retry V509 full-scene insertion

## Hard Non-Promotion Rule

Do not claim:

- teacher-only success
- crop-only success
- metric-only success
- projection-only success
- route-created-only success
- visual failure as external hard block

## Preferred Repair Direction

If A10G/A100 checkpoints still fail the V50R2 floor:

1. reduce shell-only/control domination
2. strengthen visible RGB preservation
3. increase head/hair, shoulder/neck, hand/arm, leg/foot local losses
4. keep teacher outside model forward
5. retry model-owned full-scene insertion

Final allowed state remains:

`V9000000000000000000000_V50R2_DISTILLED_HUMAN_SCENE_POINTCLOUD_READY_NOT_PROMOTED`
