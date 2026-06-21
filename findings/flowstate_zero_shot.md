# Phase 3 — FlowState Zero-shot Results

**Date**: 2026-06-17
**Model**: FlowState r1.0 (default revision, 9M params)
**Device**: GTX 1650 (3.9 GB VRAM)

## Experimental Setup

- All 69 prediction units, single channel per run (target_export or target_import)
- Batch size: 1 (iterate units), context: last N timesteps of `train_arr`
- Forecast: 96 timesteps (4 days)
- Scale factor: 1.0 (hourly data, recommended default)
- Total time: 3–6s per full run across both context lengths

## Results vs TTM2 Zero-shot

| Model | Context | Export MAE | Import MAE | Export SMAPE | Import SMAPE |
|---|---|---|---|---|---|
| TTM2 L2 (`512-96-ft-r2.1`) | 512 | 202.33 | 107.73 | 106.4% | 46.2% |
| FlowState r1.0 | 512 | 201.94 | **90.08** | 103.3% | **38.9%** |
| FlowState r1.0 | **2048** | **181.71** | 92.36 | **99.6%** | 39.6% |

**Key findings:**

1. **FlowState beats TTM2 at equal context (512)** — lower MAE on both targets. Advantage is larger on import (107 → 90, -16%) than export (202 → 202, flat).

2. **More context (2048) helps export, not import** — export MAE drops from 202 to 182 (-10%), but import stays essentially flat (90 → 92). Solar production benefits from longer weather-pattern history; consumption's daily cycles are already captured at 512.

3. **FlowState is efficient** — 3–6s for all 69 units × 2 targets on GTX 1650, comparable to TTM2's 0.8s.

## Scale Factor Sweep

Tested 5 scale factors on 5 representative units (0, 3, 6, 30, 60) at context=2048.

| Scale Factor | Export MAE (avg) | Import MAE (avg) |
|---|---|---|
| 0.25 | 157.2 | 117.5 |
| 0.50 | 143.2 | **91.4** |
| **1.00** | **106.0** | 93.9 |
| 2.00 | 284.2 | 100.7 |
| 4.00 | 356.2 | 117.0 |

**Recommendation confirmed**: `scale_factor=1.0` is optimal for hourly energy data. The 0.5 and 1.0 are close for import. Values above 2.0 degrade performance significantly.

## Hardware

FlowState r1.0 runs comfortably on GTX 1650 (4GB VRAM) at batch=1 with both 512 and 2048 context. Peak VRAM usage ~3.2 GB during S5 FFT convolution. No Colab needed for r1.0.

## Files

- `src/phase3_flowstate/run_flowstate.py` — zero-shot runner
- `src/phase3_flowstate/scale_sweep.py` — scale factor experiment
- `results/phase3/flowstate_ctx512/metrics.csv`
- `results/phase3/flowstate_ctx2048/metrics.csv`
- `results/phase3/scale_sweep/sweep_results.csv`
