"""
Phase 4 DOE Screening — TTM2 Fine-tuning on Unit 0

Screening phase 1 revealed:
- Factor D (covariates): blocked — FCM head requires future_values during inference
- Factor E (context_length): blocked — revision constrains context to pretrained length
- Remaining feasible factors: A (revision), B (freeze_backbone), C (fewshot_fraction)

Design: 2³ full factorial (8 runs) + 2 center points = 10 runs
Target: target_import (consumption) on unit 0 (Harjumaa residential)

After screening, validate the best config on all 69 units.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.phase4_finetune.run_finetune import fine_tune_unit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LENGTH = 512


def _decode_row(row: dict) -> dict:
    return {
        "revision": "512-96-ft-l1-r2.1" if row["A"] > 0 else "512-96-ft-r2.1",
        "freeze_backbone": row["B"] <= 0,
        "fewshot_fraction": round(0.05 + (row["C"] + 1) / 2 * (0.20 - 0.05), 4),
        "context_length": CONTEXT_LENGTH,
        "covariates": [],
    }


def build_design_matrix() -> pd.DataFrame:
    n = 8
    design = np.ones((n, 3))
    for i in range(n):
        design[i, 0] = 1 if (i & 1) else -1
        design[i, 1] = 1 if (i & 2) else -1
        design[i, 2] = 1 if (i & 4) else -1

    center = np.zeros((2, 3))

    full = np.vstack([design, center])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(full))
    full = full[idx]

    df = pd.DataFrame(full, columns=["A", "B", "C"])
    df["run_id"] = range(1, len(df) + 1)
    return df


def main(
    unit_id: int = 0,
    target: str = "target_import",
    output_dir: str = "results/phase4/doe_screening",
    device: str = None,
):
    if device is None:
        device = DEVICE
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_design = build_design_matrix()
    df_design.to_csv(output_path / "design_matrix.csv", index=False)

    print(f"=== Phase 4 DOE Screening (v2 — 3 factors) ===")
    print(f"Target: {target}, Unit: {unit_id}, Device: {device}")
    print(f"Design: {len(df_design)} runs (8 factorial + {len(df_design) - 8} center)")
    print(f"Output: {output_path}")
    print()

    all_results = []
    start = time.time()

    for _, row in df_design.iterrows():
        run_id = int(row["run_id"])
        config = _decode_row(row)

        cfg_str = (
            f"run{run_id:02d} "
            f"rev={'L1' if 'l1' in config['revision'] else 'L2'} "
            f"fb={'frz' if config['freeze_backbone'] else 'unf'} "
            f"ff={config['fewshot_fraction']:.2f}"
        )
        print(f"[{run_id:02d}/{len(df_design)}] {cfg_str} ...")

        run_start = time.time()
        try:
            result = fine_tune_unit(
                unit_id=unit_id,
                target=target,
                device=device,
                **config,
            )
            elapsed = time.time() - run_start
            result["run_id"] = run_id
            result["A"] = row["A"]
            result["B"] = row["B"]
            result["C"] = row["C"]
            result["elapsed_seconds"] = round(elapsed, 1)
            all_results.append(result)
            print(f"  -> SMAPE={result.get('smape', 'ERR'):.2f}  MAE={result.get('mae', 'ERR'):.2f}  ({elapsed:.0f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  -> FAILED: {e}")
            all_results.append({
                "run_id": run_id, "error": str(e),
                "A": row["A"], "B": row["B"], "C": row["C"],
            })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - start

    df_results = pd.DataFrame(all_results)
    csv_path = output_path / "screening_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / len(df_design):.0f}s per run)")
    print(f"Saved: {csv_path}")

    valid = df_results.dropna(subset=["smape"]).copy()
    if not valid.empty:
        print("\n=== Rankings by SMAPE ===")
        cols = ["run_id", "revision", "freeze_backbone", "fewshot_fraction", "smape", "mae", "rmse"]
        ranked = valid.sort_values("smape")[cols]
        print(ranked.to_string(index=False))

        print("\n=== Relative to TTM2 Zero-shot Baseline ===")
        baseline_smape = 46.16
        for _, r in ranked.iterrows():
            delta = r["smape"] - baseline_smape
            sign = "+" if delta > 0 else ""
            print(f"  run{r['run_id']:02d}  SMAPE={r['smape']:.2f}  ({sign}{delta:.2f} vs zero-shot)")

        print("\n=== Main Effect Correlations with SMAPE ===")
        coded = valid[["A", "B", "C", "smape"]].copy()
        corr = coded.corr()["smape"].drop("smape").sort_values()
        print(corr.to_string())

        with open(output_path / "analysis.json", "w") as f:
            json.dump({
                "top_configs": ranked.head(5).to_dict(orient="records"),
                "main_effects": corr.to_dict(),
                "n_success": len(valid),
                "n_failed": len(df_results) - len(valid),
                "total_seconds": round(total_elapsed, 1),
            }, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--target", default="target_import")
    parser.add_argument("--output", default="results/phase4/doe_screening")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    main(unit_id=args.unit, target=args.target, output_dir=args.output, device=args.device)

