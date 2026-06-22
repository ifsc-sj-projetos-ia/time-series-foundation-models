import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.shared.datasets import load_unit
from src.shared.eval import compute_metrics
from src.phase2_baselines.baselines import persistence_baseline, seasonal_naive_baseline
from src.phase2_baselines.run_zero_shot_l2 import load_model as load_ttm2
from src.phase2_baselines.run_zero_shot_l2 import FREQ_TOKEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = ["target_export", "target_import"]
UNIT_IDS = list(range(69))
DFLT_DIR = Path("results/phase5/national")


def _load_flowstate_model():
    from tsfm_public import FlowStateForPrediction
    model = FlowStateForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-flowstate-r1"
    )
    model.to(DEVICE)
    model.eval()
    return model


def _collect_actuals():
    all_actuals = {}
    for name in ["ttm2_zero", "flowstate", "persistence", "seasonal_naive"]:
        all_actuals[name] = {t: [] for t in TARGETS}
    for uid in UNIT_IDS:
        data = load_unit(uid)
        if data["test_timestamps"] == 0:
            continue
        fcst_len = data["forecast_length"]
        for tgt in TARGETS:
            ch = data["channels"].index(tgt)
            mean = data["scaler_mean"][ch]
            std = data["scaler_std"][ch]
            actual = data["test_arr"][:fcst_len, ch].numpy() * std + mean
            for name in all_actuals:
                all_actuals[name][tgt].append(actual)
    return all_actuals


def _run_ttm2(output_dir: Path, all_actuals: dict):
    print("  TTM2 zero-shot...")
    model = load_ttm2()
    preds = {t: [] for t in TARGETS}
    for uid in tqdm(UNIT_IDS, desc="  ttm2", leave=False):
        data = load_unit(uid)
        if data["test_timestamps"] == 0:
            continue
        fcst_len = data["forecast_length"]
        for tgt in TARGETS:
            ch = data["channels"].index(tgt)
            mean = data["scaler_mean"][ch]
            std = data["scaler_std"][ch]
            ctx = data["train_arr"][-data["context_length"]:, ch:ch+1].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                freq_tensor = torch.tensor([FREQ_TOKEN], device=DEVICE)
                out = model(past_values=ctx, freq_token=freq_tensor)
            pred = out.prediction_outputs[0, :, 0].cpu().numpy() * std + mean
            preds[tgt].append(pred)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save_and_report("ttm2_zero", all_actuals["ttm2_zero"], preds, output_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _run_flowstate(
    output_dir: Path, all_actuals: dict,
    context_length: int = 2048, scale_factor: float = 1.0, revision: str = "r1.0",
):
    label = f"flowstate_ctx{context_length}"
    print(f"  FlowState {revision} ctx={context_length}...")
    model = _load_flowstate_model()
    if revision == "r1.1":
        model = type(model).from_pretrained(
            "ibm-granite/granite-timeseries-flowstate-r1", revision="r1.1"
        ).to(DEVICE)
        model.eval()
    preds = {t: [] for t in TARGETS}
    for uid in tqdm(UNIT_IDS, desc=f"  flowstate", leave=False):
        data = load_unit(uid)
        if data["test_timestamps"] == 0:
            continue
        fcst_len = data["forecast_length"]
        for tgt in TARGETS:
            ch = data["channels"].index(tgt)
            mean = data["scaler_mean"][ch]
            std = data["scaler_std"][ch]
            ctx = data["train_arr"][-context_length:, ch].unsqueeze(-1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(
                    past_values=ctx, prediction_length=fcst_len,
                    batch_first=True, scale_factor=scale_factor,
                )
            pred = out.prediction_outputs[0, :, 0].cpu().numpy() * std + mean
            preds[tgt].append(pred)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save_and_report(label, all_actuals["flowstate"], preds, output_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _run_baselines(output_dir: Path, all_actuals: dict):
    print("  Baselines (persistence, seasonal_naive)...")
    for name, baseline_fn in [("persistence", persistence_baseline), ("seasonal_naive", seasonal_naive_baseline)]:
        preds = {t: [] for t in TARGETS}
        for uid in tqdm(UNIT_IDS, desc=f"  {name}", leave=False):
            data = load_unit(uid)
            if data["test_timestamps"] == 0:
                continue
            fcst_len = data["forecast_length"]
            for tgt in TARGETS:
                ch = data["channels"].index(tgt)
                mean = data["scaler_mean"][ch]
                std = data["scaler_std"][ch]
                ctx = data["train_arr"][-data["context_length"]:, ch].unsqueeze(-1)
                pred = baseline_fn(ctx.unsqueeze(0), fcst_len)[0, :, 0].numpy() * std + mean
                preds[tgt].append(pred)
        _save_and_report(name, all_actuals[name], preds, output_dir)
        torch.cuda.empty_cache()


def _save_and_report(model_name: str, actuals: dict, preds: dict, output_dir: Path):
    rows = []
    for tgt in TARGETS:
        actual_sum = np.sum(np.stack(actuals[tgt]), axis=0)
        pred_sum = np.sum(np.stack(preds[tgt]), axis=0)
        m = compute_metrics(actual_sum, pred_sum)
        m["model"] = model_name
        m["target"] = tgt
        rows.append(m)
        print(f"    {model_name:24s} | {tgt:15s} | MAE={m['mae']:>10.2f} | MAPE={m['mape']:>6.2f}% | SMAPE={m['smape']:>6.2f}% | RMSE={m['rmse']:>10.2f}")
    csv_path = output_dir / f"{model_name}.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"    saved: {csv_path}")


def main(
    output_dir: str = None,
    run_ttm2: bool = True,
    run_flowstate_512: bool = True,
    run_flowstate_2048: bool = True,
    run_baselines: bool = True,
):
    out = Path(output_dir or DFLT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("=== National Aggregation ===\nCollecting actual values...")
    all_actuals = _collect_actuals()

    start = time.time()
    if run_baselines:
        _run_baselines(out, all_actuals)
    if run_ttm2:
        _run_ttm2(out, all_actuals)
    if run_flowstate_512:
        _run_flowstate(out, all_actuals, context_length=512)
    if run_flowstate_2048:
        _run_flowstate(out, all_actuals, context_length=2048)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")

    summary = {
        "models": {
            "baselines": run_baselines,
            "ttm2_zero": run_ttm2,
            "flowstate_ctx512": run_flowstate_512,
            "flowstate_ctx2048": run_flowstate_2048,
        },
        "elapsed_seconds": round(elapsed, 1),
        "n_units": len(UNIT_IDS),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ttm2", action="store_true")
    parser.add_argument("--no-flowstate-512", action="store_true")
    parser.add_argument("--no-flowstate-2048", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()
    main(
        output_dir=args.output,
        run_ttm2=not args.no_ttm2,
        run_flowstate_512=not args.no_flowstate_512,
        run_flowstate_2048=not args.no_flowstate_2048,
        run_baselines=not args.no_baselines,
    )
