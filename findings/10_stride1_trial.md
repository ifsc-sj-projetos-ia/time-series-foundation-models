# Stride=1 Trial — Full-shot Fine-tuning with Dense Windows

**Date**: 2026-06-17
**Config**: L2 revision, unfrozen backbone, 100% data, stride=1 (dense sliding windows)

## Motivation

The full-shot DOE used stride=96 (non-overlapping windows), yielding only ~143 training examples. This tests whether **dense windows** (stride=1, ~13,800 examples) produce better results.

## Result

| Setup | SMAPE | MAE | RMSE |
|---|---|---|---|
| **Stride=1 (dense windows)** | **38.58** | 121.2 | 150.7 |
| FS02 (stride=96, best full-shot) | 55.78 | 225.0 | 277.8 |
| Run03 (stride=96, best few-shot) | 46.53 | 165.3 | 195.8 |
| TTM2 zero-shot (baseline) | 46.16 | 107.7 | 141.6 |
| **FlowState ctx2048 (best overall)** | **38.94** | **92.4** | **128.8** |

## Key Finding

**Stride=1 fine-tuning beats FlowState on SMAPE (38.58 vs 38.94).** FlowState still wins on MAE (92.4 vs 121.2), but the gap is much narrower.

The stride=96 DOE was misleading — it looked like fine-tuning couldn't help, but that was an artifact of sparse windows. With dense windows, the model has 96× more training data and learns better.

## What This Means

1. **The Phase 4 conclusion needs revision.** TTM2 fine-tuning CAN beat zero-shot and is competitive with FlowState, but only with stride ≤ 1.

2. **A full stride=1 DOE is warranted** — repeat the 3-factor design (revision, freeze, fewshot) with stride=1 to find the actual optimum config.

3. **Validate on all 69 units** with the best stride=1 config.

## Timing

- Stride=1: ~149s for 10 epochs (vs ~1s for stride=96)
- A full 10-run stride=1 DOE: ~25 minutes
- All 69 units validation: ~3 hours
