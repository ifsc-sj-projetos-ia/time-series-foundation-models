# Full-shot Results

**Date**: 2026-06-17

## Design

2² factorial (A: revision, B: freeze_backbone) + 2 center replicates = 6 runs.
fewshot_fraction=1.0 (100% training data), 10 epochs, lr=0.001, batch=16, context=512.

## Results

| Config | Revision | Freeze | SMAPE | MAE | RMSE |
|---|---|---|---|---|---|
| **FS02** | L2 | **False** | **55.78** | 225.0 | 277.8 |
| FS05 | L2 | True | 64.05 | 245.7 | 283.6 |
| FS01 | L2 | True | 64.11 | 258.0 | 301.6 |
| FS06 | L2 | True | 64.73 | 253.2 | 291.8 |
| FS03 | L1 | True | 66.21 | 242.9 | 275.5 |
| FS04 | L1 | False | 75.74 | 297.4 | 330.7 |

## Key Findings

1. **Full-shot is worse than few-shot.** Best full-shot SMAPE=55.8 vs best few-shot SMAPE=46.5.

2. **Full-shot is worse than zero-shot.** All 6 configs are worse than TTM2 zero-shot (46.16) and far worse than FlowState (38.94).

3. **Unfrozen backbone helps slightly with full data** (FS02=55.8 vs FS01=64.1), unlike few-shot where unfrozen caused overfitting.

4. **Replicates are consistent** (FS01/FS05/FS06 SMAPE 64.1, 64.0, 64.7 → std=0.35).

5. **L2 beats L1** in full-shot mode — the opposite of the zero-shot LoadPrediction benchmarks, but consistent with the few-shot DOE.

## Final Verdict

TTM2 fine-tuning does not improve over zero-shot on this dataset regardless of:
- Data fraction (5%, 20%, 100%)
- Backbone freeze (frozen or unfrozen)
- Revision (L2 or L1)

**FlowState r1.0 at context=2048 is the best model** across all phases of this project.
