"""
Generate all figures for the report:
1. Comparison bar chart (MAE, 6 models x 2 targets)
2. Estonia choropleth map (SMAPE by county)
3. FlowState scale factor sensitivity
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_NAMES = {
    0: "Harjumaa", 1: "Hiiumaa", 2: "Ida-Virumaa", 3: "Järvamaa",
    4: "Jõgevamaa", 5: "Lääne-Virumaa", 6: "Läänemaa", 7: "Pärnumaa",
    8: "Põlvamaa", 9: "Raplamaa", 10: "Saaremaa", 11: "Tartumaa",
    12: "Unknown", 13: "Valgamaa", 14: "Viljandimaa", 15: "Võrumaa",
}
GEO_URL = "https://raw.githubusercontent.com/buildig/EHAK/master/geojson/maakond.json"


def _unit_county_map() -> dict:
    mapping = {}
    for uid in range(69):
        path = DATA_DIR / f"unit_{uid}.pt"
        if path.exists():
            d = torch.load(path, weights_only=False)
            mapping[uid] = int(d["county"])
    return mapping


def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "final_comparison.csv")


def fig1_comparison_mae():
    """Grouped bar chart: MAE for all models x 2 targets."""
    df = _load_metrics()
    targets = [("target_export", "Geração (export)"), ("target_import", "Consumo (import)")]
    models = ["flowstate", "persistence", "seasonal_naive", "ttm2_zero"]
    model_labels = ["FlowState r1.0", "Persistence", "Seasonal Naive", "TTM2 zero-shot"]
    colors = ["#55a868", "#888888", "#aaaaaa", "#4c72b0"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, (target, tlabel) in zip(axes, targets):
        vals = []
        for m in models:
            row = df[(df["model"] == m) & (df["target"] == target)]
            v = row["mae"].values[0] if not row.empty else 0
            vals.append(v)

        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_title(tlabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Comparação de MAE entre modelos", fontsize=12, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "01_comparison_mae.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig2_estonia_map():
    """Choropleth map: SMAPE by county for FlowState (target_import)."""
    unit_county = _unit_county_map()

    flowstate = pd.read_csv(RESULTS_DIR / "phase3/flowstate_ctx2048/metrics.csv")
    fs_import = flowstate[(flowstate["model"] == "flowstate") & (flowstate["target"] == "target_import")]
    fs_pivot = fs_import.pivot_table(index="unit_id", columns="metric", values="value")
    county_smape = {}
    for uid in range(69):
        c = unit_county.get(uid)
        if c is not None and uid in fs_pivot.index:
            county_smape[c] = county_smape.get(c, []) + [fs_pivot.loc[uid, "smape"]]

    county_avg = {c: np.mean(v) for c, v in county_smape.items()}

    try:
        estonia_map = gpd.read_file(GEO_URL)

        def normalize(name):
            n = name.lower().replace(" maakond", "").replace("maa", "")
            return n.strip()

        map_data = pd.DataFrame({
            "county_id": list(COUNTY_NAMES.keys()),
            "county_name": list(COUNTY_NAMES.values()),
        })
        map_data["smape"] = map_data["county_id"].map(county_avg)
        map_data["merge_key"] = map_data["county_name"].apply(normalize)
        estonia_map["merge_key"] = estonia_map["MNIMI"].apply(normalize)
        merged = estonia_map.merge(map_data, on="merge_key", how="left")

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        merged.plot(column="smape", ax=ax, legend=True,
                    legend_kwds={"label": "SMAPE (%)", "orientation": "horizontal"},
                    cmap="OrRd", edgecolor="black", linewidth=0.5,
                    missing_kwds={"color": "lightgrey"})

        for _, row in merged.iterrows():
            if pd.notnull(row.get("smape")):
                plt.annotate(f"{row['smape']:.0f}%",
                             xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                             ha="center", fontsize=7, color="black",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        plt.title("SMAPE por condado — FlowState r1.0 (consumo)", fontsize=12)
        plt.axis("off")
        fig.tight_layout()
        path = OUTPUT_DIR / "02_estonia_map.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Map skipped: {e}")


def fig3_scale_sensitivity():
    """FlowState error vs scale_factor."""
    path = RESULTS_DIR / "phase3/scale_sweep/sweep_results.csv"
    df = pd.read_csv(path)
    targets = [("target_export", "Geração"), ("target_import", "Consumo")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for (target, tlabel), ax in zip(targets, [ax1, ax2]):
        sub = df[df["target"] == target]
        pivot = sub.pivot_table(index="scale_factor", values="mae", aggfunc="mean")
        ax.plot(pivot.index, pivot["mae"], marker="o", color="#55a868", linewidth=1.5)
        ax.set_xlabel("Scale factor")
        ax.set_ylabel("MAE")
        ax.set_title(tlabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.4, label="ótimo")
        ax.legend(fontsize=8)

    fig.suptitle("Sensibilidade ao scale factor — FlowState r1.0", fontsize=12)
    fig.tight_layout()
    path = OUTPUT_DIR / "03_scale_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    fig1_comparison_mae()
    fig3_scale_sensitivity()
    fig2_estonia_map()
    print("\nAll figures generated in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
