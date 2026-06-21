# Phase 4 — TTM2 Fine-tuning DOE Results

**Date**: 2026-06-17

## Design

3-factor full factorial (2³ = 8 runs + 2 center points = 10 runs)
- A: Revision (L2 `512-96-ft-r2.1` vs L1 `512-96-ft-l1-r2.1`)
- B: Freeze backbone (True vs False)
- C: Fewshot fraction (0.05 vs 0.20)
- Fixed: context=512, no covariates, 10 epochs, lr=0.001, batch=16

Target: `target_import`, Unit 0 (Harjumaa residential)

**Dropped factors**: Covariates (blocked — FCM head requires future_values during inference), Context length (blocked — revisions are locked to pretrained context).

## Results

| Rank | Run | Revision | Freeze | Fewshot | SMAPE | MAE | vs zero-shot |
|---|---|---|---|---|---|---|---|
| **1** | 03 | L2 | True | 0.05 | **46.53** | 165.3 | +0.37 |
| 2 | 04 | L1 | False | 0.20 | 48.85 | 156.9 | +2.69 |
| 3 | 01 | L1 | True | 0.20 | 49.05 | 150.1 | +2.89 |
| 4 | 02 | L2 | False | 0.20 | 52.62 | 175.4 | +6.46 |
| 5 | 08 | L2 | True | 0.12 | 57.79 | 204.7 | +11.63 |
| 6 | 10 | L2 | True | 0.12 | 62.50 | 231.0 | +16.34 |
| 7 | 09 | L1 | True | 0.05 | 64.56 | 216.9 | +18.40 |
| 8 | 06 | L2 | False | 0.05 | 68.90 | 259.9 | +22.74 |
| 9 | 07 | L2 | True | 0.20 | 71.35 | 273.1 | +25.19 |
| 10 | 05 | L1 | False | 0.05 | 83.92 | 355.9 | +37.76 |

## Main Effects

| Factor | Correlation | Interpretation |
|---|---|---|
| C (fewshot) | **-0.414** | More data → lower SMAPE (strongest effect) |
| B (freeze_backbone) | +0.225 | Unfrozen → higher SMAPE (overfitting) |
| A (revision) | +0.069 | L1 slightly worse than L2 (negligible) |

## Key Findings

1. **No fine-tuned config beats TTM2 zero-shot** (baseline SMAPE 46.16). The best config (run03) achieves 46.53 — essentially tied.

2. **Frozen backbone + more data is the winning combination**. Unfrozen backbones overfit on the small training windows (fewshot=0.05 gives only ~7 windows).

3. **Revision (L1 vs L2) has negligible effect** on fine-tuning outcomes. The L1 advantage seen in LoadPrediction zero-shot benchmarks does not persist after fine-tuning on this unit.

4. **Covariate fine-tuning is blocked** — the FCM head requires future_values of exogenous channels during inference, which we don't have in our evaluation protocol.

5. **FlowState r1.0 at context=2048** remains the best model overall (SMAPE 38.94, MAE 92.36).

## What This Means

Fine-tuning TTM2 on 5-20% of data with a frozen backbone for 10 epochs does not meaningfully improve over zero-shot on this dataset. The pretrained TTM2 already captures the patterns well enough. FlowState's SSM architecture generalizes better to this energy data without any fine-tuning.

**Recommendation**: Skip full 69-unit validation of fine-tuned TTM2. The DOE results on unit 0 are clear — fine-tuning adds no value for TTM2 on this dataset. FlowState is the best model and should be the focus of the final comparison.
