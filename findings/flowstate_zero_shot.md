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
| FlowState r1.0 | 2048 | **181.71** | 92.36 | **99.6%** | 39.6% |
| FlowState r1.1 | 4096 | 183.63 | 98.07 | 94.2% | 39.9% |

**Key findings:**

1. **FlowState beats TTM2 at equal context (512)** — lower MAE on both targets. Advantage is larger on import (107 → 90, -16%) than export (202 → 202, flat).

2. **More context (2048) helps export, not import** — export MAE drops from 202 to 182 (-10%), but import stays essentially flat (90 → 92). Solar production benefits from longer weather-pattern history; consumption's daily cycles are already captured at 512.

3. **r1.1 does not outperform r1.0** — at context=4096, r1.1 (18.5M params) gives marginally worse MAE than r1.0 (9M) at context=2048. The longer context may include irrelevant seasonal data, and the larger model adds no benefit for this dataset.

4. **FlowState is efficient** — 3–6s per full run on GTX 1650, comparable to TTM2's 0.8s. r1.1 takes 17s but offers no improvement.

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

Both r1.0 and r1.1 run comfortably on GTX 1650 (4GB VRAM). r1.0 at context=2048 uses ~3.2 GB peak VRAM; r1.1 at context=4096 uses only ~0.13 GB during the FFT convolution (PyTorch's memory management is more efficient than expected). No Colab needed for either revision.

## Files

- `src/phase3_flowstate/run_flowstate.py` — zero-shot runner
- `src/phase3_flowstate/scale_sweep.py` — scale factor experiment
- `results/phase3/flowstate_ctx512/metrics.csv`
- `results/phase3/flowstate_ctx2048/metrics.csv`
- `results/phase3/scale_sweep/sweep_results.csv`
