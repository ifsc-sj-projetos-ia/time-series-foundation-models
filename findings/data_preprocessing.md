# Phase 1 — Data Preprocessing

**Date**: 2026-06-17

## Pipeline Summary

`src/preprocess.py` processes all 69 prediction units into dual-format output.

## Output

```
data/processed/
├── unit_0.csv       (4.7 MB)   — merged long-format table, audit trail
├── unit_0.pt        (0.66 MB)  — PyTorch tensors + scaler + metadata
├── unit_1.csv
├── unit_1.pt
...
└── unit_68.pt

Total: 138 files, 385 MB
```

## Key Design Decisions

### Weather Station Assignment
The Kaggle-provided `weather_station_to_county_mapping.csv` only overlapped 13 of 112 weather stations exactly. Instead of modifying the mapping, we use **Haversine distance** to assign each of the 112 weather stations to the nearest county centroid (computed from the known mapped stations). All 15 counties with data get weather assigned.

### Channels (11 total)
```
target_export, target_import       # Production & consumption
temperature, dewpoint, rain,       # Historical weather
cloudcover_total, windspeed_10m, shortwave_radiation,
temperature_forecast,              # Forecast weather
cloudcover_total_forecast,
price                               # Electricity day-ahead price
```

### Split
- Train: `data_block_id` 0–600 (~20 months)
- Validation: `data_block_id` 601–633 (~1 month)
- Test: `data_block_id` 634–637 (4 days, 96 timesteps)

### .pt Tensor Format
Each `.pt` file contains:
- `train_arr` / `val_arr` / `test_arr` — continuous scaled time series (not windowed)
- `val_start_idx` / `test_start_idx` — where the eval windows begin
- `scaler_mean` / `scaler_std` — StandardScaler params (for manual inspection)
- `channels` / `n_channels` — feature channel names
- `context_length` / `forecast_length` — 512 / 96

### Memory Strategy
Processed one unit at a time with `del` + `gc.collect()` between iterations. Peak RAM usage during processing: ~3 GB.

## Timing
- Full pipeline (69 units): **27 seconds**
- Slowest step: the Haversine distance computation on 112 stations × 15 counties (runs once)
