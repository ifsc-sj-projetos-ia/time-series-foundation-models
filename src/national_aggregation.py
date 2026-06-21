import pandas as pd
import numpy as np
from pathlib import Path
from src.datasets import load_unit

RESULTS_DIR = Path("results")

models = ["ttm2_zero", "persistence", "seasonal_naive"]

all_actuals = {}
all_preds = {}

for model in models:
    all_actuals[model] = {t: [] for t in ["target_export", "target_import"]}
    all_preds[model] = {t: [] for t in ["target_export", "target_import"]}

for unit_id in range(69):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        continue
    fcst_len = data["forecast_length"]

    for target in ["target_export", "target_import"]:
        ch_idx = data["channels"].index(target)
        target_mean = data["scaler_mean"][ch_idx]
        target_std = data["scaler_std"][ch_idx]

        actual_scaled = data["test_arr"][:fcst_len, ch_idx].numpy()
        actual = actual_scaled * target_std + target_mean
        all_actuals["ttm2_zero"][target].append(actual)
        all_actuals["persistence"][target].append(actual)
        all_actuals["seasonal_naive"][target].append(actual)

from src.eval import compute_metrics

from src.ttm2_zero_shot import load_model, FREQ_TOKEN, persistence_baseline, seasonal_naive_baseline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model()

for unit_id in range(69):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        continue
    fcst_len = data["forecast_length"]

    for target in ["target_export", "target_import"]:
        ch_idx = data["channels"].index(target)
        target_mean = data["scaler_mean"][ch_idx]
        target_std = data["scaler_std"][ch_idx]

        ctx = data["train_arr"][-data["context_length"]:, ch_idx:ch_idx+1].unsqueeze(0).to(device)
        with torch.no_grad():
            freq_tensor = torch.tensor([FREQ_TOKEN], device=device)
            out = model(past_values=ctx, freq_token=freq_tensor)
        pred = out.prediction_outputs[0, :, 0].cpu().numpy() * target_std + target_mean
        all_preds["ttm2_zero"][target].append(pred)

        ctx_cpu = ctx.cpu()
        pred_p = persistence_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy() * target_std + target_mean
        all_preds["persistence"][target].append(pred_p)

        pred_sn = seasonal_naive_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy() * target_std + target_mean
        all_preds["seasonal_naive"][target].append(pred_sn)

print("=== National Aggregated Results (sum of all 69 units) ===\n")
for model in models:
    for target in ["target_export", "target_import"]:
        actual_sum = np.sum(np.stack(all_actuals[model][target]), axis=0)
        pred_sum = np.sum(np.stack(all_preds[model][target]), axis=0)
        m = compute_metrics(actual_sum, pred_sum)
        print(f"  {model:20s} | {target:15s} | MAE={m['mae']:>10.2f} | MAPE={m['mape']:>6.2f}% | SMAPE={m['smape']:>6.2f}% | RMSE={m['rmse']:>10.2f}")
