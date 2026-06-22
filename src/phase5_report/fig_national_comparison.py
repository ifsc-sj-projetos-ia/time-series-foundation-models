"""
Figure for section 3.4 — National Aggregation (4 models × 2 targets × 2 metrics)
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = {
    "model": ["Persistence", "Seasonal Naive", "TTM2 zero-shot", "FlowState ctx2048"],
    "export_mae": [27310, 14545, 11935, 9862],
    "export_smape": [151.1, 77.3, 105.2, 80.5],
    "import_mae": [6563, 5141, 5704, 4542],
    "import_smape": [27.6, 22.3, 23.7, 20.0],
}

df = pd.DataFrame(data)
models = df["model"].tolist()
best_idx = {"export_mae": 3, "export_smape": 1, "import_mae": 3, "import_smape": 3}

colors_base = ["#888888", "#aaaaaa", "#4c72b0", "#c44e52"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configs = [
    (axes[0, 0], "export_mae", "Geração — MAE"),
    (axes[0, 1], "export_smape", "Geração — SMAPE (%)"),
    (axes[1, 0], "import_mae", "Consumo — MAE"),
    (axes[1, 1], "import_smape", "Consumo — SMAPE (%)"),
]

for ax, col, title in configs:
    vals = df[col].values
    bar_colors = []
    for i in range(len(models)):
        if i == best_idx.get(col):
            bar_colors.append("#c44e52")
        elif i == 0:
            bar_colors.append("#888888")
        elif i == 1:
            bar_colors.append("#aaaaaa")
        elif i == 2:
            bar_colors.append("#4c72b0")
        else:
            bar_colors.append(colors_base[i])

    bars = ax.bar(range(len(models)), vals, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE" if "mae" in col else "SMAPE (%)", fontsize=9)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.015,
                f"{v:,.0f}" if "mae" in col else f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.15)

fig.suptitle("Agregação Nacional — MAE e SMAPE por modelo (soma das 69 unidades)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
path = OUTPUT_DIR / "03_national_comparison.png"
fig.savefig(path, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {path}")
