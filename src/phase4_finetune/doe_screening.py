"""
Phase 4 DOE Screening — TTM2 Fine-tuning on Unit 0

Design: 2⁵⁻¹ Resolution V fractional factorial (16 runs) + 2 center points = 18 runs
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
UNIT_IDS = list(range(69))


def _decode_row(row: dict) -> dict:
    """Convert coded factor levels to real configuration values."""
    return {
        "revision": "512-96-ft-l1-r2.1" if row["A"] > 0 else "512-96-ft-r2.1",
        "freeze_backbone": row["B"] <= 0,
        "fewshot_fraction": round(0.05 + (row["C"] + 1) / 2 * (0.20 - 0.05), 4),
        "covariates": ["temperature", "price"] if row["D"] > 0 else [],
        "context_length": int(round(512 + (row["E"] + 1) / 2 * (1024 - 512))),
    }


def build_design_matrix() -> pd.DataFrame:
    n = 16
    design = np.ones((n, 5))
    for i in range(n):
        design[i, 0] = 1 if (i & 1) else -1
        design[i, 1] = 1 if (i & 2) else -1
        design[i, 2] = 1 if (i & 4) else -1
        design[i, 3] = 1 if (i & 8) else -1
    design[:, 4] = design[:, 0] * design[:, 1] * design[:, 2] * design[:, 3]

    center = np.zeros((2, 5))

    full = np.vstack([design, center])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(full))
    full = full[idx]

    df = pd.DataFrame(full, columns=["A", "B", "C", "D", "E"])
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

    print(f"=== Phase 4 DOE Screening ===")
    print(f"Target: {target}, Unit: {unit_id}, Device: {device}")
    print(f"Design: {len(df_design)} runs (16 factorial + {len(df_design) - 16} center)")
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
            f"ff={config['fewshot_fraction']:.2f} "
            f"cov={len(config['covariates'])} "
            f"ctx={config['context_length']}"
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
            result["D"] = row["D"]
            result["E"] = row["E"]
            result["elapsed_seconds"] = round(elapsed, 1)
            all_results.append(result)
            print(f"  -> SMAPE={result.get('smape', 'ERR'):.2f}  MAE={result.get('mae', 'ERR'):.2f}  ({elapsed:.0f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  -> FAILED: {e}")
            all_results.append({
                "run_id": run_id, "error": str(e),
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"], "E": row["E"],
            })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - start
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / len(df_design):.0f}s per run)")

    df_results = pd.DataFrame(all_results)
    csv_path = output_path / "screening_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\n=== Top-5 by SMAPE ===")
    valid = df_results[df_results.get("smape", pd.NA).notna()].copy()
    if not valid.empty:
        top5 = valid.nsmallest(5, "smape")[
            ["run_id", "revision", "freeze_backbone", "fewshot_fraction",
             "n_covariates", "context_length", "smape", "mae", "rmse"]
        ]
        print(top5.to_string(index=False))

        print("\n=== Main Effect Correlations with SMAPE ===")
        coded = valid[["A", "B", "C", "D", "E", "smape"]].copy()
        corr = coded.corr()["smape"].drop("smape").sort_values()
        print(corr.to_string())

        with open(output_path / "analysis.json", "w") as f:
            analysis = {
                "top5_by_smape": top5.to_dict(orient="records"),
                "main_effects": corr.to_dict(),
                "n_runs": len(df_results),
                "n_success": len(valid),
                "n_failed": len(df_results) - len(valid),
                "total_seconds": round(total_elapsed, 1),
            }
            json.dump(analysis, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--target", default="target_import")
    parser.add_argument("--output", default="results/phase4/doe_screening")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    main(unit_id=args.unit, target=args.target, output_dir=args.output, device=args.device)
