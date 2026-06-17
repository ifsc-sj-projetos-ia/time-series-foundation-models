import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets import load_unit, unscale, get_zero_shot_context
from src.eval import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_COLS = ["target_export", "target_import"]
UNIT_IDS = list(range(69))

SEASONAL_PERIOD = 24
FREQ_TOKEN = 7


def load_model():
    from tsfm_public.toolkit.get_model import get_model
    model = get_model(
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        prediction_length=96,
        freq_prefix_tuning=True,
        num_input_channels=1,
    )
    model.to(device)
    model.eval()
    return model


def persistence_baseline(ctx, fcst_len):
    last = ctx[:, -1:, :]
    return last.repeat(1, fcst_len, 1)


def seasonal_naive_baseline(ctx, fcst_len, period=SEASONAL_PERIOD):
    if ctx.shape[1] < period:
        return persistence_baseline(ctx, fcst_len)
    recent = ctx[:, -period:, :]
    repeats = (fcst_len + period - 1) // period
    tiled = recent.repeat(1, repeats, 1)
    return tiled[:, :fcst_len, :]


def run_unit(model, unit_id: int):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        return []
    fcst_len = data["forecast_length"]
    results = []

    for target in TARGET_COLS:
        ch_idx = data["channels"].index(target)
        ctx, tgt_scaled = get_zero_shot_context(data, channel_idx=ch_idx)
        ctx = ctx.to(device)

        target_mean = data["scaler_mean"][ch_idx]
        target_std = data["scaler_std"][ch_idx]

        tgt_unscaled = tgt_scaled * target_std + target_mean
        actual = tgt_unscaled[0, :, 0].cpu().numpy()

        with torch.no_grad():
            freq_tensor = torch.tensor([FREQ_TOKEN], device=ctx.device)
            out = model(past_values=ctx, freq_token=freq_tensor)
        pred_scaled = out.prediction_outputs
        pred_unscaled = pred_scaled * target_std + target_mean
        pred_ttm = pred_unscaled[0, :, 0].cpu().numpy()

        metrics_ttm = compute_metrics(actual, pred_ttm)
        for k, v in metrics_ttm.items():
            results.append({
                "unit_id": unit_id, "target": target, "model": "ttm2_zero",
                "metric": k, "value": v,
            })

        ctx_cpu = ctx.cpu()
        pred_persist = persistence_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy()
        pred_persist = pred_persist * target_std + target_mean
        metrics_persist = compute_metrics(actual, pred_persist)
        for k, v in metrics_persist.items():
            results.append({
                "unit_id": unit_id, "target": target, "model": "persistence",
                "metric": k, "value": v,
            })

        pred_snaive = seasonal_naive_baseline(ctx_cpu, fcst_len)[0, :, 0].numpy()
        pred_snaive = pred_snaive * target_std + target_mean
        metrics_snaive = compute_metrics(actual, pred_snaive)
        for k, v in metrics_snaive.items():
            results.append({
                "unit_id": unit_id, "target": target, "model": "seasonal_naive",
                "metric": k, "value": v,
            })

    return results


def main():
    print(f"Device: {device}")
    print(f"Units: {len(UNIT_IDS)}")
    print("Loading TTM2 model...")
    model = load_model()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    all_results = []
    start = time.time()

    for uid in tqdm(UNIT_IDS):
        try:
            unit_results = run_unit(model, uid)
            all_results.extend(unit_results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nError on unit {uid}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(UNIT_IDS):.1f}s per unit)")

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "zero_shot_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if not df.empty:
        pivot = df.pivot_table(
            index=["model", "target"], columns="metric", values="value", aggfunc="mean"
        ).round(4)
        print("\n=== Aggregate Results (mean across all units) ===")
        cols = [c for c in ["mae", "rmse", "smape"] if c in pivot.columns]
        print(pivot[cols].to_string())
    else:
        print("No results collected.")

    summary_path = RESULTS_DIR / "zero_shot_summary.json"
    summary = {
        "n_units": len(UNIT_IDS),
        "elapsed_seconds": round(elapsed, 1),
        "results_file": str(csv_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
