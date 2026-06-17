# Phase 2 — Baselines + TTM2 Zero-shot

**Date**: 2026-06-17

## Experiment Summary

Zero-shot evaluation of TTM2 (`granite-timeseries-ttm-r2`, revision `512-96-ft-r2.1`) on all 69 prediction units.

**Setup**:
- Context: last 512 timesteps of training data per unit
- Forecast: 96 timesteps (4 days)
- Channels: single-channel per run (target_export or target_import)
- Device: GTX 1650 (3.9 GB VRAM)
- Total time: 0.8s for all 69 units

## Aggregate Results (mean across all 69 units)

| Model | Target | MAE | RMSE | sMAPE |
|---|---|---|---|---|
| persistence | export | 420.6 | 634.8 | 159.5% |
| persistence | import | 136.5 | 172.3 | 60.5% |
| seasonal_naive (24h) | export | 238.3 | 373.0 | 94.0% |
| seasonal_naive (24h) | import | 102.5 | 145.1 | 39.9% |
| **ttm2_zero** | **export** | **202.3** | **302.9** | **106.4%** |
| **ttm2_zero** | **import** | **107.7** | **141.6** | **46.2%** |

## Key Findings

1. **TTM2 beats all baselines on target_export** (production) — MAE of 202 vs 238 (seasonal naive) and 421 (persistence). Production is weather-dependent and harder to forecast; TTM2's pretraining on diverse time series gives it an edge.

2. **Seasonal naive beats TTM2 on target_import** (consumption) — MAE of 103 vs 108. Consumption follows strong daily/weekly patterns that a simple 24-hour lag captures well. TTM2's generic pretraining doesn't help here.

3. **TTM2 is fast** — 0.8s for all 69 units + 2 channels = 138 forward passes. No batching needed (batch_size=1 per unit).

4. **Frequency prefix tuning matters** — Without `freq_token`, the model errors with "Expecting freq_token in forward". TTM2 uses `freq_token=7` for hourly data.

## Data Scale

| Target | Mean | Std | Range |
|---|---|---|---|
| target_export (production) | 183 | 426 | 1 – 3,381 |
| target_import (consumption) | 446 | 320 | 53 – 652 |

Production is highly variable (solar + wind dependent), consumption is more stable.

## Files

- `src/datasets.py` — `ProsumerDataset`, `load_unit()`, `unscale()`, `get_zero_shot_context()`
- `src/ttm2_zero_shot.py` — experiment runner (TTM2 + persistence + seasonal naive)
- `results/zero_shot_metrics.csv` — raw metrics per unit/target/model/metric
- `results/zero_shot_summary.json` — run metadata
