import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.shared.datasets import load_unit
from src.shared.eval import compute_metrics
from src.phase2_baselines.baselines import persistence_baseline, seasonal_naive_baseline

TARGET_COLS = ["target_export", "target_import"]
UNIT_IDS = list(range(69))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(revision: str = "r1.0"):
    from tsfm_public import FlowStateForPrediction

    kwargs = {"pretrained_model_name_or_path": "ibm-granite/granite-timeseries-flowstate-r1"}
    if revision != "r1.0":
        kwargs["revision"] = revision

    model = FlowStateForPrediction.from_pretrained(**kwargs)
    model.to(DEVICE)
    model.eval()
    return model


def run_unit(model, unit_id: int, context_length: int, scale_factor: float):
    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        return []
    fcst_len = data["forecast_length"]
    results = []

    for target in TARGET_COLS:
        ch_idx = data["channels"].index(target)
        mean = data["scaler_mean"][ch_idx]
        std = data["scaler_std"][ch_idx]

        ctx_raw = data["train_arr"][-context_length:, ch_idx].unsqueeze(-1)
        ctx = ctx_raw.unsqueeze(0).to(DEVICE)
        tgt = data["test_arr"][:fcst_len, ch_idx]
        tgt_unscaled = tgt.numpy() * std + mean

        with torch.no_grad():
            forecast = model(
                past_values=ctx,
                prediction_length=fcst_len,
                batch_first=True,
                scale_factor=scale_factor,
            )
        pred_scaled = forecast.prediction_outputs
        pred = pred_scaled[0, :, 0].cpu().numpy() * std + mean

        metrics = compute_metrics(tgt_unscaled, pred)
        for k, v in metrics.items():
            results.append({
                "unit_id": unit_id, "target": target, "model": "flowstate",
                "metric": k, "value": v,
            })

    return results


def main(
    data_dir: str = "data/processed",
    output_dir: str = None,
    context_length: int = 2048,
    scale_factor: float = 1.0,
    revision: str = "r1.0",
    device: str = None,
):
    if device is None:
        device = DEVICE
    if output_dir is None:
        output_dir = f"results/phase3/flowstate_ctx{context_length}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Context: {context_length}, Scale factor: {scale_factor}, Revision: {revision}")
    print(f"Output: {output_path}")
    print(f"Loading FlowState model ({revision})...")
    model = load_model(revision=revision)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    all_results = []
    start = time.time()

    for uid in tqdm(UNIT_IDS):
        try:
            unit_results = run_unit(model, uid, context_length, scale_factor)
            all_results.extend(unit_results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nError on unit {uid}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(UNIT_IDS):.1f}s per unit)")

    df = pd.DataFrame(all_results)
    csv_path = output_path / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if not df.empty:
        pivot = df.pivot_table(
            index=["model", "target"], columns="metric", values="value", aggfunc="mean"
        ).round(4)
        print("\n=== Aggregate Results ===")
        cols = [c for c in ["mae", "rmse", "smape"] if c in pivot.columns]
        print(pivot[cols].to_string())

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "flowstate",
            "revision": revision,
            "context_length": context_length,
            "scale_factor": scale_factor,
            "n_units": len(UNIT_IDS),
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=2048, choices=[512, 2048, 4096])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--revision", default="r1.0", choices=["r1.0", "r1.1"])
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    main(
        context_length=args.context,
        scale_factor=args.scale,
        revision=args.revision,
        device=args.device,
    )
