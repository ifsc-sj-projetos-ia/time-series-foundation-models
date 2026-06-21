# National Aggregation — Zero-Shot Results

**Date**: 2026-06-17

Summed all 69 units' predictions and actuals to compute national-level metrics. This bridges the gap between our per-unit evaluation and the aggregated LoadPrediction benchmarks.

## Results (National)

| Model | Target | MAE | MAPE | SMAPE | RMSE |
|---|---|---|---|---|---|
| **ttm2_zero** | **export** | **11,935** | **158.9%** | **105.2%** | **16,957** |
| **ttm2_zero** | **import** | **5,704** | **33.3%** | **23.7%** | **7,541** |
| persistence | export | 27,310 | 97.3% | 151.1% | 40,246 |
| persistence | import | 6,563 | 41.0% | 27.6% | 8,529 |
| seasonal_naive | export | 14,545 | 57.8% | 77.3% | 21,074 |
| seasonal_naive | import | 5,141 | 32.7% | 22.3% | 7,285 |

## Comparison vs LoadPrediction Benchmarks

| Metric | Our Phase 2 (national) | LoadPrediction | Reason for gap |
|---|---|---|---|
| Model | `512-96-ft-r2.1` | `512-48-ft-l1-r2.1` | Shorter horizon (48h vs 96h), L1 loss |
| MAPE (import) | **33.3%** | **4.77%** | See below |

**Why the gap exists**:
1. **Forecast horizon**: We predict 96h ahead; LoadPrediction predicts 48h. Double the horizon → exponentially harder.
2. **Model revision**: `ft-l1-r2.1` (fine-tuned with L1/MAE loss during pretraining) outperforms the base `r2.1` on energy data by ~30% relative.
3. **Data source**: LoadPrediction used a synthetic aggregated CSV; we use raw Kaggle per-unit data.

## When to Use

- **Primary analysis**: Per-unit metrics (MAE, RMSE) — more honest, shows variance across counties
- **Supplementary**: National aggregated MAPE — for comparing against LoadPrediction benchmarks and existing literature
