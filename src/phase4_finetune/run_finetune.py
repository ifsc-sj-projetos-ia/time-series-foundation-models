import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.shared.datasets import load_unit, ProsumerDataset
from src.shared.eval import compute_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(
    revision: str,
    context_length: int,
    forecast_length: int,
    num_input_channels: int,
    prediction_channel_indices: list,
    exogenous_channel_indices: list = None,
):
    from tsfm_public import TinyTimeMixerForPrediction

    kwargs = {
        "pretrained_model_name_or_path": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": revision,
        "context_length": context_length,
        "prediction_filter_length": forecast_length,
        "num_input_channels": num_input_channels,
        "prediction_channel_indices": prediction_channel_indices,
    }
    if exogenous_channel_indices:
        kwargs["exogenous_channel_indices"] = exogenous_channel_indices
        kwargs["enable_forecast_channel_mixing"] = True
        kwargs["decoder_mode"] = "mix_channel"

    model = TinyTimeMixerForPrediction.from_pretrained(**kwargs)
    return model


def _get_channels(target: str, covariates: list, all_channels: list):
    tgt_idx = all_channels.index(target)
    selected = [tgt_idx]
    cov_idxs = []
    for c in covariates:
        if c in all_channels:
            cov_idxs.append(all_channels.index(c))
    selected.extend(cov_idxs)
    prediction_channel_indices = list(range(len(selected)))
    prediction_channel_indices[0] = 0
    exogenous_channel_indices = list(range(1, len(selected))) if cov_idxs else None
    channel_names = [all_channels[tgt_idx]] + [all_channels[i] for i in cov_idxs]
    return selected, prediction_channel_indices, exogenous_channel_indices, channel_names


def fine_tune_unit(
    unit_id: int,
    target: str = "target_import",
    revision: str = "512-96-ft-r2.1",
    freeze_backbone: bool = True,
    fewshot_fraction: float = 0.05,
    covariates: list = None,
    context_length: int = 512,
    forecast_length: int = 96,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = None,
    output_dir: str = None,
    device: str = None,
):
    if covariates is None:
        covariates = []
    if device is None:
        device = DEVICE
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = load_unit(unit_id)
    if data["test_timestamps"] == 0:
        return {"error": "no test data"}

    channel_idxs, pred_ch_idxs, exog_ch_idxs, ch_names = _get_channels(
        target, covariates, data["channels"]
    )
    n_channels = len(channel_idxs)

    dataset = ProsumerDataset(
        unit_id=unit_id,
        split="train",
        channels=ch_names,
    )
    n_train = len(dataset)
    if fewshot_fraction < 1.0:
        n_keep = max(1, int(n_train * fewshot_fraction))
        indices = torch.randperm(n_train)[:n_keep].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    val_dataset = ProsumerDataset(
        unit_id=unit_id,
        split="val",
        channels=ch_names,
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    model = _build_model(
        revision=revision,
        context_length=context_length,
        forecast_length=forecast_length,
        num_input_channels=n_channels,
        prediction_channel_indices=pred_ch_idxs,
        exogenous_channel_indices=exog_ch_idxs,
    )
    model.to(device)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    if learning_rate is None:
        learning_rate = 0.001

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            freq_tensor = torch.full((batch_x.shape[0],), 7, dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(past_values=batch_x, future_values=batch_y, freq_token=freq_tensor)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        if val_loader:
            model.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    freq_tensor = torch.full((batch_x.shape[0],), 7, dtype=torch.long, device=device)
                    outputs = model(past_values=batch_x, future_values=batch_y, freq_token=freq_tensor)
                    vl = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]
                    val_loss += vl.item()
                    v_batches += 1
            val_loss_avg = val_loss / max(v_batches, 1)
        else:
            val_loss_avg = None

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_str = f"val={val_loss_avg:.4f}" if val_loss_avg else "no val"
            print(f"  epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  {val_str}")

    train_elapsed = time.time() - train_start

    ch_idx = data["channels"].index(target)
    target_mean = data["scaler_mean"][ch_idx]
    target_std = data["scaler_std"][ch_idx]

    model.eval()
    ctx_cols = [ch_idx] if exog_ch_idxs is None else channel_idxs
    ctx_raw = data["train_arr"][-context_length:, ctx_cols]
    if ctx_raw.ndim == 1:
        ctx_raw = ctx_raw.unsqueeze(-1)
    ctx = ctx_raw.unsqueeze(0).to(device)
    with torch.no_grad():
        freq_tensor = torch.full((1,), 7, dtype=torch.long, device=device)
        out = model(past_values=ctx, freq_token=freq_tensor)
    pred = out.prediction_outputs[0, :, 0].cpu().numpy() * target_std + target_mean

    tgt = data["test_arr"][:forecast_length, ch_idx].numpy() * target_std + target_mean
    metrics = compute_metrics(tgt, pred)

    result = {
        "unit_id": unit_id,
        "target": target,
        "revision": revision,
        "freeze_backbone": freeze_backbone,
        "fewshot_fraction": fewshot_fraction,
        "n_covariates": len(covariates),
        "covariates": covariates,
        "context_length": context_length,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_seconds": round(train_elapsed, 1),
        "n_train_windows": len(dataset),
        **metrics,
    }
    return result


def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--target", default="target_import")
    parser.add_argument("--revision", default="512-96-ft-r2.1")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_freeze", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--fewshot", type=float, default=0.05)
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = fine_tune_unit(
        unit_id=args.unit,
        target=args.target,
        revision=args.revision,
        freeze_backbone=args.freeze_backbone,
        fewshot_fraction=args.fewshot,
        covariates=args.covariates,
        context_length=args.context,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    print("\n=== Fine-tune Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output) / "result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
