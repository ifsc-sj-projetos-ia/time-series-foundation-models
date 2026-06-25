"""
Phase 6 — Preprocess v2
Generates consumption.csv and generation.csv (long format panel) for use with TTM2's id_columns.

Usage:
    python -m src.phase6_refactor.preprocess_v2
"""

import gc
import warnings
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.phase6_refactor.config import (
    RAW_DIR, OUT_DIR,
    EXOGENOUS_COLS, STATIC_COLS,
)

warnings.filterwarnings("ignore")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def build_county_station_map():
    mapping = pd.read_csv(RAW_DIR / "weather_station_to_county_mapping.csv")
    mapping = mapping.dropna(subset=["county"])
    mapping["county"] = mapping["county"].astype(int)
    centroids = mapping.groupby("county")[["latitude", "longitude"]].mean().to_dict("index")

    hw = pd.read_csv(RAW_DIR / "historical_weather.csv", usecols=["latitude", "longitude"])
    stations = hw.drop_duplicates().values

    assignments = {c: [] for c in centroids}
    for lat, lon in stations:
        best_c = min(centroids, key=lambda c: haversine(lat, lon, centroids[c]["latitude"], centroids[c]["longitude"]))
        assignments[best_c].append((float(lat), float(lon)))
    return assignments


def avg_weather(csv_path: str, station_map: dict, cols: list, dt_col: str, rename_map: dict = None):
    df = pd.read_csv(csv_path, parse_dates=[dt_col], low_memory=False)
    if "forecast" in csv_path:
        df = df.rename(columns={"forecast_datetime": "datetime"})
        df = df.sort_values("hours_ahead").groupby(
            ["latitude", "longitude", "datetime"], as_index=False
        ).first()

    groups = []
    for cid, stations in station_map.items():
        mask = False
        for lat, lon in stations:
            mask = mask | ((df["latitude"] == lat) & (df["longitude"] == lon))
        sub = df[mask]
        if sub.empty:
            continue
        avg = sub.groupby("datetime", as_index=False)[cols].mean()
        avg["county"] = cid
        groups.append(avg)

    result = pd.concat(groups, ignore_index=True)
    if rename_map:
        result = result.rename(columns=rename_map)
    return result


def main():
    print("Building county-station map...")
    station_map = build_county_station_map()

    print("Loading weather...")
    hist_cols = ["temperature", "dewpoint", "rain", "cloudcover_total",
                  "windspeed_10m", "shortwave_radiation"]
    hist = avg_weather(str(RAW_DIR / "historical_weather.csv"), station_map, hist_cols, "datetime")

    fcst_cols = ["temperature", "cloudcover_total"]
    fcst = avg_weather(str(RAW_DIR / "forecast_weather.csv"), station_map, fcst_cols, "forecast_datetime",
                       rename_map={c: f"{c}_forecast" for c in fcst_cols})

    print("Loading electricity prices...")
    prices = pd.read_csv(RAW_DIR / "electricity_prices.csv")
    prices["datetime"] = pd.to_datetime(prices["forecast_date"])
    prices = prices[["datetime", "euros_per_mwh"]].rename(columns={"euros_per_mwh": "price"})

    print("Loading client data...")
    client = pd.read_csv(RAW_DIR / "client.csv")
    client["date"] = pd.to_datetime(client["date"])
    client = client.sort_values("date")

    print("Loading train data...")
    chunks = []
    for chunk in pd.read_csv(RAW_DIR / "train.csv", chunksize=500000, parse_dates=["datetime"]):
        chunks.append(chunk)
    train = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    for target_name, is_cons in [("consumption", 1), ("generation", 0)]:
        print(f"\nProcessing {target_name} (is_consumption={is_cons})...")
        df = train[train["is_consumption"] == is_cons].copy()
        print(f"  Rows: {len(df)}")

        df = df.sort_values(["county", "is_business", "product_type", "datetime"]).reset_index(drop=True)
        client_sub = client[
            (client["county"].isin(df["county"].unique()))
            & (client["is_business"].isin(df["is_business"].unique()))
            & (client["product_type"].isin(df["product_type"].unique()))
        ].copy()

        df["date"] = df["datetime"].dt.date
        client_sub["date_merge"] = client_sub["date"].dt.date
        df = df.merge(
            client_sub[["date_merge", "county", "is_business", "product_type", "eic_count", "installed_capacity"]],
            left_on=["county", "is_business", "product_type", "date"],
            right_on=["county", "is_business", "product_type", "date_merge"],
            how="left",
        )
        df = df.drop(columns=["date", "date_merge"])

        df = df.merge(hist, on=["datetime", "county"], how="left")
        df = df.merge(fcst, on=["datetime", "county"], how="left")
        df = df.merge(prices, on="datetime", how="left")

        df["split"] = "train"
        sorted_blocks = sorted(df["data_block_id"].unique())
        n_blocks = len(sorted_blocks)
        n_train = int(n_blocks * 0.70)
        n_val = int(n_blocks * 0.15)
        train_blocks = set(sorted_blocks[:n_train])
        val_blocks = set(sorted_blocks[n_train:n_train + n_val])
        test_blocks = set(sorted_blocks[n_train + n_val:])
        df.loc[df["data_block_id"].isin(val_blocks), "split"] = "val"
        df.loc[df["data_block_id"].isin(test_blocks), "split"] = "test"

        keep_cols = ["datetime", "target", "prediction_unit_id", "split"] + STATIC_COLS
        for c in EXOGENOUS_COLS:
            if c in df.columns:
                keep_cols.append(c)

        out = df[keep_cols].sort_values(["prediction_unit_id", "datetime"])
        out_path = OUT_DIR / f"{target_name}.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(out)} rows)")
        split_counts = out["split"].value_counts().to_dict()
        split_counts = {k: int(v) for k, v in split_counts.items()}
        print(f"  Split: {split_counts}")
        del df, out, client_sub
        gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()

