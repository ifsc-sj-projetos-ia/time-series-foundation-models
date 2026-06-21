import torch

SEASONAL_PERIOD = 24


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
