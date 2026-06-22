"""
Generate population comparison figure for the ethics/disparity section.
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

population = {
    "Harjumaa": 638076,
    "Hiiumaa": 8474,
    "Ida-Virumaa": 133358,
    "Järvamaa": 30072,
    "Jõgevamaa": 27739,
    "Läänemaa": 20688,
    "Lääne-Virumaa": 59608,
    "Põlvamaa": 24036,
    "Pärnumaa": 87418,
    "Raplamaa": 34038,
    "Saaremaa": 31919,
    "Tartumaa": 162390,
    "Valgamaa": 28114,
    "Viljandimaa": 45637,
    "Võrumaa": 34317,
}

county_ids = {
    "Harjumaa": 0, "Hiiumaa": 1, "Ida-Virumaa": 2, "Järvamaa": 3,
    "Jõgevamaa": 4, "Läänemaa": 6, "Lääne-Virumaa": 5, "Põlvamaa": 8,
    "Pärnumaa": 7, "Raplamaa": 9, "Saaremaa": 10, "Tartumaa": 11,
    "Valgamaa": 13, "Viljandimaa": 14, "Võrumaa": 15,
}

df = pd.DataFrame({
    "county": list(population.keys()),
    "county_id": [county_ids[c] for c in population.keys()],
    "population": list(population.values()),
}).sort_values("population", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#c44e52" if c == "Harjumaa" else "#4c72b0" if c == "Tartumaa" else "#55a868" if p < 30000 else "#888888" for c, p in zip(df["county"], df["population"])]
bars = ax.barh(range(len(df)), df["population"], color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["county"], fontsize=9)
ax.set_xlabel("População")
ax.set_title("População por condado — Estônia (2023)", fontsize=12)
for bar, v in zip(bars, df["population"]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
            f"{v:,}", ha="left", va="center", fontsize=8)
ax.set_xlim(0, max(df["population"]) * 1.25)
fig.tight_layout()
path = OUTPUT_DIR / "04_population_by_county.png"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

csv_path = OUTPUT_DIR / "estonia_population.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
