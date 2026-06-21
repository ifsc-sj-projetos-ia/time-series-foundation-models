# Full-shot Counterpart — Plan

**Status**: Draft — pending validation before execution.

## Motivation

The few-shot DOE (finding `07_ttm2_finetune_doe.md`) tested fewshot_fraction at `[0.05, 0.20]`. No config beat zero-shot. This tests whether **full training data** (`fewshot_fraction=1.0`) changes the outcome.

## Design

### Setup

- With stride=96 on `train_arr` (14,420 timesteps), full-shot gives ~143 windows
- 10 epochs, lr=0.001, batch_size=16, context=512
- Target: `target_import`, Unit 0

### Factor Changes from the Few-shot DOE

Factor C (fewshot_fraction) is **fixed at 1.0** — removed as a factor.
Factor B (freeze_backbone) now has more room to help: with ~143 windows, unfrozen backbone may not overfit as badly.

| Factor | Low (−1) | High (+1) |
|---|---|---|
| A — Revision | L2 (`512-96-ft-r2.1`) | L1 (`512-96-ft-l1-r2.1`) |
| B — Freeze backbone | True (frozen) | False (unfrozen) |

### Design Matrix — 2² Full Factorial + 2 Center Points = 6 runs

| Run | A (revision) | B (freeze) | Description |
|---|---|---|---|
| FS01 | L2 | True | Default config, full data |
| FS02 | L2 | False | Unfrozen backbone, full data |
| FS03 | L1 | True | L1 revision, frozen |
| FS04 | L1 | False | L1 revision, unfrozen (most aggressive) |
| FS05 | L2 | True | Center replicate (same as FS01) |
| FS06 | L2 | True | Center replicate (same as FS01) |

### Expected Timing

- 143 windows × batch 16 ≈ 9 steps per epoch
- 10 epochs × 0.1s per step ≈ 9s per run
- 6 runs ≈ ~1 minute total

### Execution

```bash
# FS01 — L2, frozen
python -m src.phase4_finetune.run_finetune \
  --unit 0 --target target_import \
  --revision 512-96-ft-r2.1 --freeze_backbone \
  --fewshot 1.0 --context 512 --epochs 10 --lr 0.001 --batch_size 16

# FS02 — L2, unfrozen
python -m src.phase4_finetune.run_finetune \
  --unit 0 --target target_import \
  --revision 512-96-ft-r2.1 --no_freeze \
  --fewshot 1.0 --context 512 --epochs 10 --lr 0.001 --batch_size 16

# FS03 — L1, frozen
python -m src.phase4_finetune.run_finetune \
  --unit 0 --target target_import \
  --revision 512-96-ft-l1-r2.1 --freeze_backbone \
  --fewshot 1.0 --context 512 --epochs 10 --lr 0.001 --batch_size 16

# FS04 — L1, unfrozen
python -m src.phase4_finetune.run_finetune \
  --unit 0 --target target_import \
  --revision 512-96-ft-l1-r2.1 --no_freeze \
  --fewshot 1.0 --context 512 --epochs 10 --lr 0.001 --batch_size 16

# FS05–06 — replicates (same as FS01)
```

### Expected Outcomes

| Scenario | Meaning | Action |
|---|---|---|
| Best FS config beats FlowState (SMAPE < 39) | Full-shot helps | Validate on all 69 units |
| Best FS config matches zero-shot (~46) | Still no improvement | Drop fine-tuning from report |
| Unfrozen beats frozen significantly | Bottleneck was freeze, not data | Consider unfrozen + full-shot on all units |
| Unfrozen is worse than frozen | Overfitting persists | Stick with frozen backbone |

### Output

Results saved to `results/phase4/fullshot/metrics.csv`.

---

**Decision point**: If full-shot doesn't beat FlowState, Phase 4 is done and we move to Phase 5 (final comparison). If full-shot beats FlowState, we expand to a multi-unit validation.
