import gc
import warnings
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/kaue/Documents/IFSC/IA/predict-energy-behavior-of-prosumers")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BLOCKS = (0, 600)
VAL_BLOCKS = (601, 633)
TEST_BLOCKS = (634, 637)

CONTEXT_LENGTH = 512
FORECAST_LENGTH = 96


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def build_county_station_map() -> dict:
    mapping = pd.read_csv(DATA_DIR / "weather_station_to_county_mapping.csv")
    mapping = mapping.dropna(subset=["county"])
    mapping["county"] = mapping["county"].astype(int)
    county_centroids = (
        mapping.groupby("county")[["latitude", "longitude"]].mean().to_dict("index")
    )

    hw = pd.read_csv(DATA_DIR / "historical_weather.csv", usecols=["latitude", "longitude"])
    station_coords = hw.drop_duplicates()[["latitude", "longitude"]].values

    station_assignments = {c: [] for c in county_centroids}
    for lat, lon in station_coords:
        best_county = None
        best_dist = float("inf")
        for cid, centroid in county_centroids.items():
            dist = haversine(lat, lon, centroid["latitude"], centroid["longitude"])
            if dist < best_dist:
                best_dist = dist
                best_county = cid
        station_assignments[best_county].append((lat, lon))
    return station_assignments


def average_weather(df: pd.DataFrame, station_map: dict, weather_cols: list, datetime_col: str) -> pd.DataFrame:
    groups = []
    for county_id, coords_list in station_map.items():
        mask = False
        for lat, lon in coords_list:
            mask = mask | ((df["latitude"] == lat) & (df["longitude"] == lon))
        sub = df[mask]
        if sub.empty:
            continue
        avg = sub.groupby(datetime_col, as_index=False)[weather_cols].mean()
        avg["county"] = county_id
        groups.append(avg)
    result = pd.concat(groups, ignore_index=True)
    return result


def load_historical_weather(station_map: dict) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "historical_weather.csv", parse_dates=["datetime"])
    cols = [
        "temperature", "dewpoint", "rain", "snowfall", "surface_pressure",
        "cloudcover_total", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
        "windspeed_10m", "winddirection_10m", "shortwave_radiation",
        "direct_solar_radiation", "diffuse_radiation",
    ]
    return average_weather(df, station_map, cols, "datetime")


def load_forecast_weather(station_map: dict) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "forecast_weather.csv", parse_dates=["forecast_datetime"], low_memory=False
    )
    df = df.rename(columns={"forecast_datetime": "datetime"})
    df = df.sort_values("hours_ahead").groupby(
        ["latitude", "longitude", "datetime"], as_index=False
    ).first()
    cols = [
        "temperature", "dewpoint",
        "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
        "direct_solar_radiation", "surface_solar_radiation_downwards",
        "snowfall", "total_precipitation",
    ]
    result = average_weather(df, station_map, cols, "datetime")
    result = result.rename(
        columns={c: f"{c}_forecast" for c in cols if c != "datetime"}
    )
    return result


HYBRID_COLS = [
    "temperature", "dewpoint",
    "cloudcover_total",
]


def preprocess_unit(
    unit_id: int,
    train_csv: pd.DataFrame,
    client_df: pd.DataFrame,
    hist_weather: pd.DataFrame,
    fcst_weather: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    unit = train_csv[train_csv["prediction_unit_id"] == unit_id].copy()
    if unit.empty:
        return None
    county_id = int(unit["county"].iloc[0])
    is_biz = int(unit["is_business"].iloc[0])
    prod_type = int(unit["product_type"].iloc[0])

    pivot = unit.pivot_table(
        index=["datetime", "data_block_id"],
        columns="is_consumption",
        values="target",
    ).reset_index()
    pivot.columns = ["datetime", "data_block_id", "target_export", "target_import"]

    client_sub = client_df[
        (client_df["county"] == county_id)
        & (client_df["is_business"] == is_biz)
        & (client_df["product_type"] == prod_type)
    ].copy()
    if not client_sub.empty:
        client_sub["date"] = pd.to_datetime(client_sub["date"])
        client_sub = client_sub.set_index("date")
        hourly_idx = pd.date_range(
            start=client_sub.index.min(),
            end=client_sub.index.max(),
            freq="h",
        )
        client_hourly = (
            client_sub[["eic_count", "installed_capacity"]]
            .reindex(hourly_idx)
            .ffill()
        )
        client_hourly.index.name = "datetime"
        client_hourly = client_hourly.reset_index()
        pivot = pivot.merge(client_hourly, on="datetime", how="left")

    hw = hist_weather[hist_weather["county"] == county_id].drop(columns=["county"])
    # Merge historical weather first (suffixes set to handle overlapping names with pivot)
    pivot = pivot.merge(hw, on="datetime", how="left")

    fw = fcst_weather[fcst_weather["county"] == county_id].drop(columns=["county"])
    pivot = pivot.merge(fw, on="datetime", how="left")

    pr = prices_df.rename(columns={"forecast_date": "datetime", "euros_per_mwh": "price"})
    pr["datetime"] = pd.to_datetime(pr["datetime"])
    pivot = pivot.merge(pr[["datetime", "price"]], on="datetime", how="left")

    pivot["data_block_id"] = pivot["data_block_id"].astype(int)
    pivot["split"] = "train"
    pivot.loc[
        (pivot["data_block_id"] >= VAL_BLOCKS[0]) & (pivot["data_block_id"] <= VAL_BLOCKS[1]), "split"
    ] = "val"
    pivot.loc[
        (pivot["data_block_id"] >= TEST_BLOCKS[0]) & (pivot["data_block_id"] <= TEST_BLOCKS[1]), "split"
    ] = "test"

    pivot["unit_id"] = unit_id
    pivot["county"] = county_id
    pivot["is_business"] = is_biz
    pivot["product_type"] = prod_type

    return pivot


FEATURE_COLS = [
    "target_export", "target_import",
    "temperature", "dewpoint", "rain", "cloudcover_total",
    "windspeed_10m", "shortwave_radiation",
    "temperature_forecast", "cloudcover_total_forecast",
    "price",
]
FALLBACK_COLS = ["target_export", "target_import", "price"]


def scale_and_export(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    available = []
    for c in FEATURE_COLS:
        if c in df.columns:
            n_valid = df[c].notna().sum()
            if n_valid > 0:
                available.append(c)
    if not available:
        available = FALLBACK_COLS

    values = df[available].values.astype(np.float32)
    mask = np.isnan(values)
    values[mask] = 0.0

    if fit:
        scaler = StandardScaler()
        scaler.fit(values)
    scaled = scaler.transform(values)
    scaled[mask] = 0.0

    return scaled, scaler, available


def main():
    print("Building county-to-station map...")
    station_map = build_county_station_map()

    print("Loading historical weather...")
    hist_weather = load_historical_weather(station_map)

    print("Loading forecast weather...")
    fcst_weather = load_forecast_weather(station_map)

    print("Loading electricity prices...")
    prices = pd.read_csv(DATA_DIR / "electricity_prices.csv")
    prices["forecast_date"] = pd.to_datetime(prices["forecast_date"])

    print("Loading client data...")
    client = pd.read_csv(DATA_DIR / "client.csv")

    print("Loading train data...")
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / "train.csv", chunksize=200000, parse_dates=["datetime"]):
        chunks.append(chunk)
    train_all = pd.concat(chunks, ignore_index=True)

    unit_ids = sorted(train_all["prediction_unit_id"].unique())
    print(f"Processing {len(unit_ids)} prediction units...")

    for unit_id in tqdm(unit_ids):
        result = preprocess_unit(unit_id, train_all, client, hist_weather, fcst_weather, prices)
        if result is None or result.empty:
            continue

        csv_path = OUT_DIR / f"unit_{unit_id}.csv"
        result.to_csv(csv_path, index=False)

        train_part = result[result["split"] == "train"]
        val_part = result[result["split"] == "val"]
        test_part = result[result["split"] == "test"]

        train_arr, scaler, channels = scale_and_export(train_part, fit=True)
        n_feat = train_arr.shape[1]
        val_arr = scale_and_export(val_part, scaler, fit=False)[0] if len(val_part) > 0 else np.empty((0, n_feat), dtype=np.float32)
        test_arr = scale_and_export(test_part, scaler, fit=False)[0] if len(test_part) > 0 else np.empty((0, n_feat), dtype=np.float32)

        pt_path = OUT_DIR / f"unit_{unit_id}.pt"
        torch.save(
            {
                "unit_id": unit_id,
                "county": int(result["county"].iloc[0]),
                "is_business": int(result["is_business"].iloc[0]),
                "product_type": int(result["product_type"].iloc[0]),
                "channels": channels,
                "n_channels": len(channels),
                "context_length": CONTEXT_LENGTH,
                "forecast_length": FORECAST_LENGTH,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_std": scaler.scale_.tolist(),
                "train_timestamps": len(train_arr),
                "val_timestamps": len(val_arr),
                "test_timestamps": len(test_arr),
                "train_arr": torch.tensor(train_arr),
                "val_arr": torch.tensor(val_arr),
                "test_arr": torch.tensor(test_arr),
                "val_start_idx": max(0, len(train_arr) - CONTEXT_LENGTH),
                "test_start_idx": max(0, len(train_arr) + len(val_arr) - CONTEXT_LENGTH),
            },
            pt_path,
        )

        del result, train_part, val_part, test_part, train_arr, val_arr, test_arr
        gc.collect()

    print("Done!")


if __name__ == "__main__":
    main()
