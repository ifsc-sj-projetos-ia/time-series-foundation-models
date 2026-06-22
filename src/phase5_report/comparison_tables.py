"""
Comparison Tables — reads all results/phase/*/metrics.csv and prints side-by-side tables.

Usage:
    python -m src.phase5_report.comparison_tables

Output:
    - Prints comparison tables to stdout
    - Saves results/phase5/comparison/all_metrics.csv
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "phase5" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("Persistence",           RESULTS_DIR / "phase2/l2_zero_shot/metrics.csv",           "persistence"),
    ("Seasonal Naive",        RESULTS_DIR / "phase2/l2_zero_shot/metrics.csv",           "seasonal_naive"),
    ("TTM2 zero-shot (L2)",   RESULTS_DIR / "phase2/l2_zero_shot/metrics.csv",           "ttm2_zero"),
    ("FlowState r1.0 ctx512", RESULTS_DIR / "phase3/flowstate_ctx512/metrics.csv",       "flowstate"),
    ("FlowState r1.0 ctx2048",RESULTS_DIR / "phase3/flowstate_ctx2048/metrics.csv",      "flowstate"),
    ("FlowState r1.1 ctx4096",RESULTS_DIR / "phase3/flowstate_ctx4096/metrics.csv",      "flowstate"),
]

NATIONAL = [
    ("Persistence",           RESULTS_DIR / "phase5/national/persistence.csv"),
    ("Seasonal Naive",        RESULTS_DIR / "phase5/national/seasonal_naive.csv"),
    ("TTM2 zero-shot (L2)",   RESULTS_DIR / "phase5/national/ttm2_zero.csv"),
    ("FlowState r1.0 ctx2048",RESULTS_DIR / "phase5/national/flowstate_ctx2048.csv"),
]


def _load_aggregated(path: Path, model_filter: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model_filter:
        df = df[df["model"] == model_filter]
    pivot = df.pivot_table(
        index=["target"], columns="metric", values="value", aggfunc="mean"
    ).reset_index()
    return pivot


def _agg_metric(df: pd.DataFrame, target: str, metric: str) -> float:
    row = df[df["target"] == target]
    if row.empty or metric not in row.columns:
        return None
    return row[metric].values[0]


def main():
    print("=== Comparison Tables ===\n")

    rows = []
    for label, path, model_filter in SOURCES:
        if path.exists():
            pivot = _load_aggregated(path, model_filter)
            for target in ["target_export", "target_import"]:
                mae = _agg_metric(pivot, target, "mae")
                rmse = _agg_metric(pivot, target, "rmse")
                smape = _agg_metric(pivot, target, "smape")
                mape = _agg_metric(pivot, target, "mape")
                rows.append({
                    "model": label, "target": target,
                    "mae": mae, "rmse": rmse, "smape": smape, "mape": mape,
                })
        else:
            print(f"  [SKIP] {path} not found")

    df = pd.DataFrame(rows)

    print("--- Per-Unit Aggregate (mean across 69 units) ---\n")
    for target in ["target_export", "target_import"]:
        print(f"  [{target}]\n")
        sub = df[df["target"] == target].copy()
        sub = sub.sort_values("smape")
        for _, r in sub.iterrows():
            smape_str = f"{r['smape']:.2f}%" if pd.notna(r['smape']) else "N/A"
            mae_str = f"{r['mae']:.2f}" if pd.notna(r['mae']) else "N/A"
            rmse_str = f"{r['rmse']:.2f}" if pd.notna(r['rmse']) else "N/A"
            print(f"    {r['model']:30s}  SMAPE={smape_str:>8s}  MAE={mae_str:>10s}  RMSE={rmse_str:>10s}")
        print()

    csv_path = OUT_DIR / "all_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("--- National Aggregation (summed across 69 units) ---\n")
    nat_rows = []
    for label, path in NATIONAL:
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        ndf = pd.read_csv(path)
        for _, r in ndf.iterrows():
            nat_rows.append({
                "model": label, "target": r["target"],
                "mae": r["mae"], "rmse": r["rmse"],
                "smape": r["smape"], "mape": r["mape"],
            })
    ndf = pd.DataFrame(nat_rows)
    for target in ["target_export", "target_import"]:
        print(f"  [{target}]\n")
        sub = ndf[ndf["target"] == target].sort_values("smape")
        for _, r in sub.iterrows():
            print(f"    {r['model']:30s}  SMAPE={r['smape']:>8.2f}%  MAE={r['mae']:>10.2f}  RMSE={r['rmse']:>10.2f}")
        print()
    ndf.to_csv(OUT_DIR / "national_metrics.csv", index=False)

    summary = {
        "n_models": df["model"].nunique(),
        "n_targets": df["target"].nunique(),
        "sources": [str(p) for _, p, _ in SOURCES],
    }
    import json
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
