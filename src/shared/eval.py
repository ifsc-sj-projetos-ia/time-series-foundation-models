import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0, "smape": 0.0}

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.sum() > 0 else 0.0

    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape}


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    numeric_cols = ["mae", "mse", "rmse", "mape", "smape"]
    available = [c for c in numeric_cols if c in df.columns]
    summary = df.groupby("model", dropna=False)[available].agg(["mean", "std"])
    summary.columns = [f"{a}_{s}" for a, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "metric", "avg")
    return summary
