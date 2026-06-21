# Implementation Plan — Comparing IBM Granite TSFM Models for Prosumer Energy Forecasting

## Phase 0 — Environment Setup

**Goal**: Everything runs before touching the dataset.

**Tasks**:
- [ ] Create repo scaffold: dirs, `.gitignore`, `requirements.txt`, `LICENSE` (Apache-2.0)
- [ ] `pip install` all dependencies locally
- [ ] Verify TTM2 loads:
  ```python
  from tsfm_public import TinyTimeMixerForPrediction
  model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")
  model(torch.randn(69, 512, 1))  # confirm no OOM, see output shape
  ```
- [ ] Verify FlowState r1.0 loads locally:
  ```python
  from tsfm_public import FlowStateForPrediction
  model = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1")
  model(past_values=torch.randn(8, 2048, 1), prediction_length=96, scale_factor=1.0)
  ```
- [ ] Download dataset into `data/raw/` via `kagglehub`

**Deliverable**: `requirements.txt` frozen, both models pass a dummy forward pass locally.

---

## Phase 1 — Data Preprocessing

**Goal**: Produce clean, scaled `.pt` tensors per prediction unit.

**File to create**: `src/preprocess.py`

**Output per unit** — two files for auditability:
- `data/processed/unit_{id}.csv` — flat long-format CSV with a `split` column (`train`/`val`/`test`)
- `data/processed/unit_{id}.pt` — PyTorch dict with windowed tensors + scaler + metadata

**Tasks**:
- [ ] Build `weather_station_to_county` mapping → nearest-station lookup per county
- [ ] Implement main loop over `prediction_unit_id` (0..68), one at a time:
  - Load `train.csv` subset → separate `is_consumption=0` (export) and `is_consumption=1` (import) into 2 target columns
  - Join `client.csv` by `(county, is_business, product_type, date)` → expand `installed_capacity`, `eic_count`
  - Map county → nearest weather station → join `historical_weather` and `forecast_weather` (closest origin_datetime, fixed hours_ahead)
  - Join `electricity_prices` on datetime
  - Fit `StandardScaler` on train split; transform val and test
  - Save merged long-format table as `data/processed/unit_{id}.csv` (with `split` column)
  - Window into `(context=512, forecast=96)` sliding windows → PyTorch tensors
  - Save tensors + scaler params + metadata as `data/processed/unit_{id}.pt`
- [ ] Create `01_eda.ipynb` with:
  - Target distribution per county, per business/residential
  - Weather-target correlation heatmap
  - Daily and weekly seasonal plots
**Select 5 representative units for local FlowState testing**:
| Unit | County | Type | Reason |
|---|---|---|---|
| 0 | 0 (Harjumaa) | residential, product=1 | Largest group, best-case |
| 3 | 0 (Harjumaa) | business, product=0 | Different consumption pattern |
| 6 | 1 (Hiiumaa) | residential, product=1 | Rural, low sample count |
| 30 | 11 (Tartumaa) | business, product=3 | High installed_capacity |
| 60 | 15 (Võrumaa) | residential, product=3 | Lowest eic_count, hardest case |

**Deliverable**: 69 `.csv` + 69 `.pt` files (dual-format), `01_eda.ipynb`, representative unit list documented.

---

## Phase 2 — Baselines + TTM2 Zero-shot

**Goal**: Establish performance floor and first model results.

**Files to create**: `src/datasets.py`, `src/eval.py`, `src/ttm2_zero_shot.py`

**Tasks**:
- [ ] Build `ProsumerDataset` (PyTorch Dataset) that:
  - Loads `.pt` for given unit
  - Yields sliding windows `(past_values, future_values)` with configurable `context_length` and `prediction_length`
  - Supports returning exogenous channels
- [ ] Implement `eval.py`:
  - `compute_metrics(y_true, y_pred)` → dict `{mae, rmse, smape}`
  - `aggregate_results(all_units_metrics)` → summary table
- [ ] Run persistence baseline for all 69 units
- [ ] Run seasonal naive (repeat -24h) baseline for all 69 units
- [ ] Run TTM2 zero-shot on all 69 units:
  - Model: `get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=512, prediction_length=96)`
  - Uses revision `512-96-ft-r2.1` (L2/MSE default)
- [ ] **Re-run with L1 revision**: swap to `512-96-ft-l1-r2.1` — same code, different pretrained weights (recommended for energy data, see `findings/loss_function.md`)
- [ ] Store results in `results/zero_shot_metrics.csv`

**Deliverable**: Baseline + TTM2 zero-shot metrics table in `02_ttm2_experiments.ipynb`.

---

## Phase 3 — FlowState Zero-shot + Scale Sweep

**Goal**: Evaluate FlowState r1.0 on all 69 units and understand scale_factor behavior.

**Files to create**:
- `src/phase3_flowstate/__init__.py`
- `src/phase3_flowstate/run_flowstate.py` — r1.0/r1.1 zero-shot runner
- `src/phase3_flowstate/scale_sweep.py` — scale_factor experiment

**Architecture note**: FlowState only supports zero-shot (no fine-tuning). Outputs 9 quantiles — we use `prediction_outputs` (mean) for point forecasts.

**Context comparison**: Run with two context lengths to understand tradeoffs:
- `context=512`: apples-to-apples with TTM2
- `context=2048`: FlowState's native pretraining length (gives it best chance)

**Scale sweep**: Test `[0.25, 0.5, 1.0, 2.0, 4.0]` on 5 representative units at context=2048. Hourly data recommends 1.0 by default.

**Tasks**:
- [ ] **Local — context=512**: FlowState r1.0 on all 69 units, scale_factor=1.0, batch=1 (iterate)
- [ ] **Local — context=2048**: FlowState r1.0 on all 69 units, scale_factor=1.0, batch=1 (iterate)
- [ ] **Scale sweep**: Run 5 scale factors on 5 representative units, store per-factor MAE
- [ ] Document in `03_flowstate_experiments.ipynb`

**Comparison against TTM2**: Read `results/phase2/l2_zero_shot/metrics.csv` for direct comparison.

**Deliverable**: FlowState metrics at both context lengths, scale sensitivity chart.

---

## Phase 4 — TTM2 Fine-tuning

**Goal**: Push TTM2 to its best possible performance.

**File to create**: `src/phase4_finetune/run_finetune.py`
**Directory**: `src/phase4_finetune/__init__.py`

**Model revision**: Use `512-96-ft-l1-r2.1` (L1/MAE pretrained) — see `findings/loss_function.md`

**Tasks**:
- [ ] Implement few-shot fine-tuning:
  - Use 5% of training data per unit (first ~1500 timesteps ≈ 2 months)
  - Channel-independent: fine-tune separate model per unit
  - Add exogenous channels (forecast weather + price) where available
- [ ] Experiment with channel-mix fine-tuning on 2–3 selected units
- [ ] Validate on val split, test on test split (blocks 634–637)
- [ ] Store fine-tuned checkpoints in `models/` (gitignored)
- [ ] Log metrics to `results/phase4/finetune/metrics.csv`

**Deliverable**: Fine-tuned TTM2 metrics, before/after comparison table.

---

## Phase 5 — Final Comparison + Writeup

**Goal**: Produce all tables, figures, report, and presentation.

**File to create**: `notebooks/04_final_comparison.ipynb`

**Tasks**:
- [ ] Aggregate all results into a single DataFrame
- [ ] Generate:
  - **Table 1**: Per-unit metrics — Persistence vs SeasonalNaive vs TTM2-zero (L2) vs TTM2-zero (L1) vs FlowState vs TTM2-finetuned
  - **Table 2**: National aggregation — sum all 69 units (bridges to LoadPrediction benchmarks)
  - **Table 3**: Per-county breakdown (ethics section)
  - **Figure 1**: Forecast horizon error plot (MAE vs hours-ahead)
  - **Figure 2**: FlowState scale_factor sensitivity chart
  - **Figure 3**: Per-county MAE bar chart
- [ ] Write `README.md`:
  - Project objective, dependencies, execution instructions
  - Main results table, limitations, future work
- [ ] Write report PDF following the 2.6 structure (4-8 pages)
- [ ] Prepare presentation (10–15 min, 12 slides)
- [ ] Final commit and push to GitHub

**Deliverable**: Complete GitHub repo, `relatorio.pdf`, presentation file.

---

## File Creation Order (Dependency Chain)

```
No deps
├── src/shared/eval.py                   ← all experiment scripts import this
├── src/shared/datasets.py               ← all experiment scripts import this
│
Phase 1
├── src/phase1_data/preprocess.py        ← produces data/processed/
│
Parallel blocks:
├── Phase 2
│   ├── src/phase2_baselines/baselines.py      ← persistence, seasonal_naive
│   ├── src/phase2_baselines/run_zero_shot_l2.py
│   ├── [future] src/phase2_baselines/run_zero_shot_l1.py
│   └── notebooks/02_ttm2_experiments.ipynb
│
├── Phase 3
│   ├── src/phase3_flowstate/run_flowstate.py
│   ├── src/phase3_flowstate/scale_sweep.py
│   └── notebooks/03_flowstate_experiments.ipynb
│
├── Phase 4
│   ├── src/phase4_finetune/run_finetune.py
│   └── models/                                ← checkpoints (gitignored)
│
Phase 5
└── notebooks/04_final_comparison.ipynb    ← reads all results/*/*.csv
```

## Repository

```
time-series-foundation-models/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                           ← project plan, implementation plan
├── src/
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── eval.py                  ← compute_metrics(), aggregate_results()
│   │   └── datasets.py              ← load_unit(), ProsumerDataset, unscale()
│   ├── phase1_data/
│   │   ├── __init__.py
│   │   └── preprocess.py            ← merge, scale, export CSV + PT
│   ├── phase2_baselines/
│   │   ├── __init__.py
│   │   ├── baselines.py             ← persistence, seasonal_naive
│   │   └── run_zero_shot_l2.py      ← TTM2 L2 zero-shot (512-96-ft-r2.1)
│   ├── phase3_flowstate/
│   │   ├── __init__.py
│   │   ├── run_flowstate.py         ← FlowState r1.0/r1.1 zero-shot
│   │   └── scale_sweep.py           ← scale_factor experiments
│   ├── phase4_finetune/
│   │   ├── __init__.py
│   │   └── run_finetune.py          ← TTM2 few-shot with exogenous
│   └── phase5_report/
│       ├── __init__.py
│       └── national_aggregation.py  ← summed national metrics
├── results/
│   ├── phase2/
│   │   └── l2_zero_shot/
│   │       ├── metrics.csv
│   │       └── summary.json
│   ├── phase3/
│   │   ├── flowstate_ctx512/
│   │   ├── flowstate_ctx2048/
│   │   ├── flowstate_ctx4096/
│   │   └── scale_sweep/
│   └── phase4/                      ← will contain fine-tuning results
├── findings/                        ← documented findings per phase
├── models/                          ← fine-tuned checkpoints (gitignored)
├── data/                            ← (gitignored)
└── notebooks/                       ← presentation notebooks
```

## Dependency Graph

```
                 phase1_data/preprocess.py
                            │
                    shared/datasets.py
                    ┌──────┴──────┐
              shared/eval.py      │
              ┌──────┼────────┐   │
    phase2_baselines  phase3_flowstate  phase4_finetune
    run_zero_shot_l2  run_flowstate     run_finetune
    baselines.py      scale_sweep.py
              │
              └──────────┬──────────┘
                         │
            phase5_report/national_aggregation.py
```

Phases 2, 3, and 4 are independent once `data/processed/` exists.
