# time-series-foundation-models

Comparing **TTM2** and **FlowState** (IBM Granite TSFM family) on the Enefit prosumer energy dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
├── __init__.py
└── eval.py           # Metrics: MAE, MSE, RMSE, MAPE, sMAPE
data/
├── processed/        # Preprocessed .pt tensors (gitignored)
models/               # Fine-tuned checkpoints (gitignored)
notebooks/            # EDA and experiment notebooks
results/              # CSV tables and PNG plots
```

## Model Verification (Phase 0)

| Model | Params | Batch | Context | Local (GTX 1650) |
|---|---|---|---|---|
| TTM2 | 805K | 69 | 512 | ✅ |
| FlowState r1.0 | 9M | 1 | 2048 | ✅ (batch=1) |
| FlowState r1.1 | 18.5M | — | 4096 | ❌ (needs Colab) |

See `project_plan.md` and `implementation_plan.md` for full details.
