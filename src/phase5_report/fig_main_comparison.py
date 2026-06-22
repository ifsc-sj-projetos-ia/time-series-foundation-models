"""
Figure for section 3.2 — Main Comparison (6 models × 2 targets × 2 metrics)
Bar chart with MAE and SMAPE side by side.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = {
    "model": [
        "Persistence", "Seasonal Naive", "TTM2 zero-shot",
        "FlowState ctx512", "FlowState ctx2048", "FlowState r1.1"
    ],
    "export_mae": [420.6, 238.3, 202.3, 201.9, 181.7, 183.6],
    "export_smape": [159.5, 94.0, 106.4, 103.3, 99.6, 94.2],
    "import_mae": [136.5, 102.5, 107.7, 90.1, 92.4, 98.1],
    "import_smape": [60.5, 39.9, 46.2, 38.9, 39.6, 39.9],
}

df = pd.DataFrame(data)
models = df["model"].tolist()

colors = ["#888888", "#aaaaaa", "#4c72b0", "#55a868", "#c44e52", "#937860"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configs = [
    (axes[0, 0], "export_mae", "Geração — MAE"),
    (axes[0, 1], "export_smape", "Geração — SMAPE (%)"),
    (axes[1, 0], "import_mae", "Consumo — MAE"),
    (axes[1, 1], "import_smape", "Consumo — SMAPE (%)"),
]

for ax, col, title in configs:
    vals = df[col].values
    best_idx = np.argmin(vals)
    bar_colors = ["#c44e52" if i == 0 else "#4c72b0" if i == 2 else colors[i] for i in range(len(models))]

    bars = ax.bar(range(len(models)), vals, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE" if "mae" in col else "SMAPE (%)", fontsize=9)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.015,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.15)

fig.suptitle("Comparação Principal — MAE e SMAPE por modelo e por variável-alvo",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
path = OUTPUT_DIR / "01b_main_comparison.png"
fig.savefig(path, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {path}")
