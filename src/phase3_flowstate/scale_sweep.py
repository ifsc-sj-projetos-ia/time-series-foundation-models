import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.shared.datasets import load_unit
from src.shared.eval import compute_metrics

TARGET_COLS = ["target_export", "target_import"]
REPRESENTATIVE_UNITS = [0, 3, 6, 30, 60]
SCALE_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
CONTEXT_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    from tsfm_public import FlowStateForPrediction

    model = FlowStateForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-flowstate-r1"
    )
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_unit(model, unit_id: int, scale_factor: float):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        return None
    fcst_len = data["forecast_length"]
    results = []

    for target in TARGET_COLS:
        ch_idx = data["channels"].index(target)
        mean = data["scaler_mean"][ch_idx]
        std = data["scaler_std"][ch_idx]

        ctx_raw = data["train_arr"][-CONTEXT_LENGTH:, ch_idx].unsqueeze(-1)
        ctx = ctx_raw.unsqueeze(0).to(DEVICE)
        tgt = data["test_arr"][:fcst_len, ch_idx].numpy() * std + mean

        with torch.no_grad():
            forecast = model(
                past_values=ctx,
                prediction_length=fcst_len,
                batch_first=True,
                scale_factor=scale_factor,
            )
        pred = forecast.prediction_outputs[0, :, 0].cpu().numpy() * std + mean

        m = compute_metrics(tgt, pred)
        results.append({
            "unit_id": unit_id, "target": target,
            "scale_factor": scale_factor,
            "mae": m["mae"], "rmse": m["rmse"], "smape": m["smape"],
        })

    return results


def main(
    output_dir: str = "results/phase3/scale_sweep",
    device: str = None,
):
    global DEVICE
    if device is not None:
        DEVICE = device

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Units: {REPRESENTATIVE_UNITS}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print("Loading FlowState model (r1.0)...")
    model = load_model()

    all_results = []
    start = time.time()

    for sf in SCALE_FACTORS:
        for uid in tqdm(REPRESENTATIVE_UNITS, desc=f"scale={sf}"):
            try:
                unit_results = evaluate_unit(model, uid, sf)
                if unit_results:
                    all_results.extend(unit_results)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\nError unit {uid} scale {sf}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    csv_path = output_path / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\n=== Scale Sweep Results (MAE) ===")
    pivot = df.pivot_table(
        index=["unit_id", "target"], columns="scale_factor",
        values="mae", aggfunc="mean"
    ).round(2)
    print(pivot.to_string())

    summary = {
        "model": "flowstate r1.0",
        "context_length": CONTEXT_LENGTH,
        "scale_factors": SCALE_FACTORS,
        "representative_units": REPRESENTATIVE_UNITS,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    main(device=args.device)
