import torch
from torch.utils.data import Dataset
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


_cache = {}


def load_unit(unit_id: int):
    if unit_id not in _cache:
        path = PROCESSED_DIR / f"unit_{unit_id}.pt"
        _cache[unit_id] = torch.load(path, weights_only=False)
    return _cache[unit_id]


def clear_cache():
    _cache.clear()


def unscale(tensor: torch.Tensor, unit_data: dict) -> torch.Tensor:
    if tensor.shape[-1] != len(unit_data["channels"]):
        return tensor
    mean = torch.tensor(unit_data["scaler_mean"]).to(tensor.device, tensor.dtype)
    std = torch.tensor(unit_data["scaler_std"]).to(tensor.device, tensor.dtype)
    return tensor * std + mean


def get_zero_shot_context(unit_data: dict, channel_idx: int = None):
    ctx_len = unit_data["context_length"]
    fcst_len = unit_data["forecast_length"]

    if channel_idx is not None:
        ctx = unit_data["train_arr"][-ctx_len:, channel_idx:channel_idx+1]
        tgt = unit_data["test_arr"][:fcst_len, channel_idx:channel_idx+1]
    else:
        ctx = unit_data["train_arr"][-ctx_len:]
        tgt = unit_data["test_arr"][:fcst_len]

    return ctx.unsqueeze(0), tgt.unsqueeze(0)


class ProsumerDataset(Dataset):
    def __init__(self, unit_id: int, split: str = "train", channels: list = None, stride: int = None):
        data = load_unit(unit_id)
        self.ctx_len = data["context_length"]
        self.fcst_len = data["forecast_length"]
        self.channels = channels or data["channels"]
        self.scaler_mean = torch.tensor(data["scaler_mean"])
        self.scaler_std = torch.tensor(data["scaler_std"])

        if split == "train":
            arr = data["train_arr"]
        elif split == "val":
            arr = data["val_arr"]
        elif split == "test":
            arr = data["test_arr"]
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.channels != data["channels"]:
            idxs = [data["channels"].index(c) for c in self.channels]
            arr = arr[:, idxs]

        self.x = []
        self.y = []
        stride = self.fcst_len if stride is None else stride
        for i in range(0, len(arr) - self.ctx_len - self.fcst_len + 1, stride):
            self.x.append(arr[i:i + self.ctx_len])
            self.y.append(arr[i + self.ctx_len:i + self.ctx_len + self.fcst_len])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
