# Project Plan: Comparing IBM Granite TSFM Models for Prosumer Energy Forecasting

## Overview

**Goal**: Compare **TTM2** (`ibm-granite/granite-timeseries-ttm-r2`) and **FlowState** (`ibm-granite/granite-timeseries-flowstate-r1`) on the Enefit prosumer energy dataset, following the ICD/ADS assignment structure.

**Dataset**: [Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data) (Kaggle)
- Hourly energy production + consumption across 69 prediction units in Estonia
- 21 months training (Sep 2021 - May 2023), 4 days test (May 28-31, 2023)
- Rich exogenous: weather (historical + forecasts), electricity/gas prices, installed capacity

### Models

| Property | TTM2 | FlowState r1.0 | FlowState r1.1 |
|---|---|---|---|
| Architecture | TinyTimeMixer (MLP) | SSM Encoder + Functional Basis Decoder | Same, improved |
| Params | ~1M | ~9M | ~18.5M |
| Context length | 512 / 1024 / 1536 | 2048 | 4096 |
| Modes | Zero-shot, Fine-tuning, Exogenous | Zero-shot only | Zero-shot only |
| Local (4GB VRAM) | ✓ Full batch=69 | ✓ Batch ≤ 8 | ✗ OOM |
| Colab (T4 16GB) | ✓ | ✓ | ✓ |

---

## Section-by-Section Plan

### 2.1 — Problem Definition & Requirements (1-2 pages)

**Problem**: Forecasting prosumer (consumer + producer) energy behavior is critical for grid balancing as distributed generation (e.g., rooftop solar) grows. Over/under-predicting leads to grid instability, wasted renewables, or blackouts.

**Human/Social Impact**:
- Grid operators need accurate forecasts to balance supply-demand in real time
- Affects energy equity (rural vs urban counties)
- Directly impacts renewable energy integration targets and consumer electricity costs in Estonia

**Stakeholders**: Enefit grid operators, Estonian energy policymakers, prosumer households/businesses

**Functional Requirements**:
- Predict hourly production AND consumption for 69 prediction units (county × business_type × product_type)
- Ingest exogenous signals: weather forecasts, electricity prices, installed capacity
- Produce forecasts up to 96 hours ahead (4 days)
- Support both zero-shot and fine-tuned inference

**Non-Functional Requirements**:
- Low compute footprint (both models under 20M params — aligns with "green AI")
- Apache 2.0 license on both models and dataset
- Fully reproducible via `granite-tsfm` and pinned dependencies
- Model selection must be practical for resource-constrained environments (e.g. smaller grid operators)

**Success Metrics**: MAE, RMSE, sMAPE — compared pairwise between TTM2, FlowState, and baselines

---

### 2.2 — Data & Preparation

**Source**: Kaggle Enefit competition (public, licensed for academic use)

**Files**:
| File | Rows | Description |
|---|---|---|
| `train.csv` | 2,018,352 | Targets: county, is_business, product_type, target, is_consumption, datetime |
| `client.csv` | 41,919 | Static metadata: installed_capacity, eic_count per (county, type, date) |
| `historical_weather.csv` | 1,710,803 | Actual weather: temp, dewpoint, rain, snow, cloudcover, wind, radiation |
| `forecast_weather.csv` | 3,424,513 | Forecast weather (1-48h ahead) same variables |
| `electricity_prices.csv` | 15,286 | Hourly day-ahead prices (EUR/MWh) |
| `gas_prices.csv` | 637 | Daily gas prices |
| `weather_station_to_county_mapping.csv` | 112 | Maps lat/lon stations to county IDs |

**Output per unit** — two files for auditability:
- `data/processed/unit_{id}.csv` — flat long-format CSV with a `split` column (`train`/`val`/`test`), suitable for inspection, debugging, and EDA
- `data/processed/unit_{id}.pt` — PyTorch dict with windowed tensors, used directly by the Dataset class

**Preprocessing Pipeline** (runs locally, 1 prediction unit at a time to fit 8GB RAM):

```
For each prediction_unit_id (0..68):
  1. Load subset of train.csv for this unit
  2. Separate is_consumption=0 (export) and is_consumption=1 (import) into 2 target columns
  3. Join client.csv by (county, is_business, product_type, date)
     → expand installed_capacity, eic_count as time-varying features
  4. Map county → nearest weather station via weather_station_to_county_mapping.csv
     → join historical_weather on nearest lat/lon + datetime
  5. Join electricity_prices on datetime
  6. Join forecast_weather (align by data_block_id, closest origin_datetime)
  7. Fit StandardScaler on train split; transform all splits
  8. Save merged table as data/processed/unit_{id}.csv  (with split column)
  9. Window into (context, forecast) sliding windows → PyTorch tensors
  10. Save tensors + scaler + metadata as data/processed/unit_{id}.pt
```

**Train/Val/Test Split**:
- Train: `data_block_id` 0–600 (∼20 months)
- Validation: `data_block_id` 601–633 (∼1 month)
- Test: `data_block_id` 634–637 (4 days, matches Kaggle test period)

**Bias & Limitations**:
- Data covers only Estonia — generalizability to other geographies is unproven
- Some counties have far fewer installations (e.g. Hiiumaa vs Harjumaa)
- Missing weather stations for some counties → nearest-station approximation
- Forecast weather has varying lead times (1-48h) — alignment is approximate

---

### 2.3 — GitHub Repository Structure

```
time-series-foundation-models/
├── README.md                          # Goal, deps, instructions, metrics, limitations
├── LICENSE                            # Apache-2.0
├── requirements.txt                   # granite-tsfm, torch, pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm
├── setup.py                           # Optional: editable install
│
├── data/
│   ├── download.sh                    # Kaggle API download script
│   └── processed/                     # .pt files (gitignored)
│
├── src/
│   ├── preprocess.py                  # Chunked merge + scaling → .pt files
│   ├── datasets.py                    # PyTorch Dataset for sliding windows
│   ├── ttm2_zero_shot.py              # Experiment 1a
│   ├── ttm2_finetune.py               # Experiment 2a (with exogenous)
│   ├── flowstate_zero_shot.py         # Experiment 1b
│   ├── flowstate_scale_sweep.py       # Experiment 3
│   └── eval.py                        # Unified metrics + comparison tables/plots
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Data exploration (energy patterns, correlations)
│   ├── 02_ttm2_experiments.ipynb      # TTM2 runs (local)
│   ├── 03_flowstate_experiments.ipynb # FlowState runs (Colab)
│   └── 04_final_comparison.ipynb      # Side-by-side tables, charts, discussion
│
├── models/                            # Fine-tuned checkpoints (gitignored)
├── results/                           # CSV tables, PNG plots (committed)
└── project_plan.md                    # This file
```

---

### 2.4 — Modeling & Experiments

#### Baseline Models
- **Persistence**: repeat last observed value
- **Seasonal Naive**: repeat value from 24 hours ago (daily seasonality)

#### Experiment 1: Zero-shot Comparison

| Factor | TTM2 | FlowState |
|---|---|---|
| Branch/Revision | `512-96-ft-mae-r2` | `r1.0` locally |
| Context | 512 timesteps (~21 days) | 2048 (r1.0) or 4096 (r1.1) timesteps |
| Forecast | 96 timesteps (4 days) | 96 timesteps |
| Channels | 1 per run (univariate) + repeat per unit | 1 per run (univariate) + repeat per unit |
| scale_factor | N/A | 1.0 (hourly) |
| Exogenous | None (zero-shot) | N/A (no exogenous support) |
| Batch | 69 | 1 (iterate units, fits 4GB VRAM) |

**Hypothesis**: FlowState will outperform TTM2 in zero-shot due to larger pretraining corpus and SSM architecture.

#### Experiment 2: TTM2 Fine-tuning vs FlowState Zero-shot

| Factor | TTM2 | FlowState |
|---|---|---|
| Setup | Fine-tune on 5-10% of training data | Best result from Exp 1 |
| Exogenous | Include weather forecast + price channels | N/A |
| Strategy | Channel-independent + channel-mix fine-tuning | Zero-shot (no training) |

**Hypothesis**: Fine-tuned TTM2 with exogenous signals will close or surpass the zero-shot FlowState gap for units with strong weather correlation.

#### Experiment 3: FlowState scale_factor Sensitivity

Test FlowState at multiple scale factors on a representative subset of 5 prediction units:

| Scale Factor | Meaning | Expected Behavior |
|---|---|---|
| 0.25 | Treat hourly data as 15-min cycles | Under-segment daily patterns |
| 0.5 | Treat as 30-min cycles | Slightly under-segment |
| 1.0 | Hourly (recommended default) | Baseline |
| 2.0 | Treat as 2-hourly | Over-segment, may alias daily |
| 3.43 | Daily (weekly seasonality) | Over-segment significantly |
| 4.0 | Extreme | Worst case |

**Why this matters**: FlowState's time-scale invariance is its headline innovation. Real hourly energy data has multiple seasonal cycles (daily, weekly, quarterly) — scale_factor tuning behavior on such data is an underexplored contribution.

---

### 2.5 — Ethical Evaluation

**Risks**:
- **County-level bias**: Rural counties (fewer prosumers) may have worse forecast accuracy, potentially leading to unequal grid service quality
- **Business vs residential**: If one group is systematically harder to predict, energy pricing or curtailment decisions could disproportionately affect them
- **Privacy**: Data is aggregated to county level — no individual re-identification risk. However, if this approach were applied to disaggregated smart meter data, privacy risks would be significant
- **Misuse**: Over-reliance on automated forecasts without human-in-the-loop could cause grid mismanagement during extreme weather events

**Mitigations**:
- Report per-county and per-group MAE to expose disparities
- Discuss contexts where models should NOT be used (e.g. extreme weather outliers)
- Both models are small (< 20M params) and can run on CPU — favorable carbon footprint compared to large transformer-based forecasters (e.g. TimesFM at 200M+ params)

**Limitations**:
- Only tested on Estonian data — transferability to other climates/grids is unproven
- FlowState is zero-shot only — no mechanism to adapt to domain-specific patterns
- TTM2 fine-tuning uses only 5-10% of data — results may improve with more training data

---

### 2.6 — Report Structure (4-8 pages)

1. **Introduction**: Energy forecasting problem, IBM Granite model family, comparison motivation
2. **Data & Methodology**: Enefit dataset description, preprocessing pipeline, TTM2 and FlowState architectures
3. **Experiments & Results**: 3 experiments with comparison tables and line plots
4. **Discussion**: Limitations, ethical considerations, tradeoffs between the two models
5. **Conclusion & Future Work**: Key findings, recommendations, what could be explored next
6. **GitHub link & reproducibility instructions**

---

### 2.7 — Presentation (10-15 min)

| Slide | Content |
|---|---|
| 1 | Title: "Forecasting Prosumer Energy with IBM Granite TSFMs" |
| 2 | Why prosumer forecasting matters (grid stability, Estonia's renewable targets) |
| 3-4 | Dataset: 69 prediction units, hourly data, weather + price signals |
| 5 | Model architectures: TTM2 (tiny MLP mixer) vs FlowState (SSM + FBD) |
| 6 | Experiment 1: Zero-shot comparison (table + bar chart) |
| 7 | Experiment 2: TTM2 fine-tuning closes the gap (line plot over forecast horizon) |
| 8 | Experiment 3: scale_factor sweep (sensitivity chart) |
| 9 | Key finding: Which model wins, and when? |
| 10 | Ethical considerations + limitations |
| 11 | Demo: forecast on a real prediction unit |
| 12 | What's next? |

---

## Technical Infrastructure

### Hardware Profile

| Resource | Your Laptop | Colab Free |
|---|---|---|
| GPU | GTX 1650 (4 GB VRAM) | T4 (16 GB VRAM) |
| RAM | 8 GB system | ~12 GB system |
| Disk | Local SSD | 100 GB ephemeral + Google Drive |

### VRAM Requirements (Inference, FP32)

| Scenario | Batch=69 | Batch=8 | Batch=1 |
|---|---|---|---|
| TTM2 (1M params, ctx=512) | ~0.5 GB | ~0.1 GB | <0.1 GB |
| FlowState r1.0 (9M, ctx=2048) | ~2.6 GB peak | ~0.4 GB | ~0.1 GB |
| FlowState r1.1 (18.5M, ctx=4096) | ~5+ GB (OOM on 4GB) | ~1.2 GB | ~0.3 GB |

**Why FlowState uses more VRAM**: The S5 encoder uses FFT-based convolution — the `modeling_flowstate.py` S5 layer creates temporary `(2*L, B, H)` complex64 tensors. For r1.1 at batch=69: `(8192, 69, 512) × 8 bytes × 2 buffers ≈ 5+ GB` peak.

### Execution Strategy

All experiments run locally on the GTX 1650 (4GB VRAM, 8GB RAM). FlowState and TTM2 both fit at batch=1 by iterating over the 69 prediction units individually.

```
LOCAL (GTX 1650, 8GB RAM)
├── Data preprocessing (1 unit at a time, ~2 GB peak)
├── TTM2 all experiments (batch=69, comfortable)
├── FlowState r1.0 inference (batch=1, iterate 69 units, ~6s)
├── FlowState r1.1 inference (batch=1, iterate 69 units, ~17s)
└── Scale sweep (5 units × 5 scale factors, ~3s)
```

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| FlowState OOM on GTX 1650 | Already confirmed — r1.0 at ctx=2048 and r1.1 at ctx=4096 both fit at batch=1 |
| 8GB RAM tight during preprocessing | Process one `prediction_unit_id` at a time; `del` + `gc.collect()` between units |
| Kaggle data download requires API key | Use `kagglehub` library (no API key needed for public datasets) |

---

## Repository Setup Commands

```bash
# Create repo structure
mkdir -p time-series-foundation-models/{src,notebooks,models,data/processed,results,docs}

# Init git
cd time-series-foundation-models
git init
echo -e "data/processed/\nmodels/\n*.pt\n*.pth\n*.joblib" > .gitignore
git add -A
git commit -m "Initial project scaffold"

# Install dependencies
pip install granite-tsfm torch pandas numpy scikit-learn matplotlib seaborn tqdm kagglehub
```
