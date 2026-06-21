import numpy as np
import torch

from src.shared.datasets import load_unit
from src.shared.eval import compute_metrics
from src.phase2_baselines.baselines import persistence_baseline, seasonal_naive_baseline
from src.phase2_baselines.run_zero_shot_l2 import load_model, FREQ_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"
models = ["ttm2_zero", "persistence", "seasonal_naive"]

all_actuals = {m: {t: [] for t in ["target_export", "target_import"]} for m in models}
all_preds = {m: {t: [] for t in ["target_export", "target_import"]} for m in models}

for unit_id in range(69):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        continue
    fcst_len = data["forecast_length"]
    for target in ["target_export", "target_import"]:
        ch_idx = data["channels"].index(target)
        mean = data["scaler_mean"][ch_idx]
        std = data["scaler_std"][ch_idx]
        actual = data["test_arr"][:fcst_len, ch_idx].numpy() * std + mean
        for m in models:
            all_actuals[m][target].append(actual)

print("Loading TTM2 model for national prediction...")
model = load_model()
for unit_id in range(69):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        continue
    fcst_len = data["forecast_length"]
    for target in ["target_export", "target_import"]:
        ch_idx = data["channels"].index(target)
        mean = data["scaler_mean"][ch_idx]
        std = data["scaler_std"][ch_idx]
        ctx = data["train_arr"][-data["context_length"]:, ch_idx:ch_idx+1].unsqueeze(0).to(device)
        with torch.no_grad():
            freq_tensor = torch.tensor([FREQ_TOKEN], device=device)
            out = model(past_values=ctx, freq_token=freq_tensor)
        pred = out.prediction_outputs[0, :, 0].cpu().numpy() * std + mean
        all_preds["ttm2_zero"][target].append(pred)
        ctx_cpu = ctx.cpu()
        all_preds["persistence"][target].append(
            persistence_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy() * std + mean
        )
        all_preds["seasonal_naive"][target].append(
            seasonal_naive_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy() * std + mean
        )

print("\n=== National Aggregated Results (sum of all 69 units) ===\n")
for model in models:
    for target in ["target_export", "target_import"]:
        actual_sum = np.sum(np.stack(all_actuals[model][target]), axis=0)
        pred_sum = np.sum(np.stack(all_preds[model][target]), axis=0)
        m = compute_metrics(actual_sum, pred_sum)
        print(f"  {model:20s} | {target:15s} | MAE={m['mae']:>10.2f} | MAPE={m['mape']:>6.2f}% | SMAPE={m['smape']:>6.2f}% | RMSE={m['rmse']:>10.2f}")
