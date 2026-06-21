"""
Colab launcher — run from a Google Colab notebook cell.

Usage:
    1. Upload colab_bundle.zip + data/processed/ to Google Drive
    2. Run this script cell-by-cell in a Colab notebook:

    # Cell 1 — mount Drive, install deps
    !pip install granite-tsfm

    # Cell 2 — run this script
    !python colab_run.py --mode zero_shot --context 2048
"""

import argparse
import sys
from pathlib import Path


def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")


def extract_bundle(bundle_path: str, dest: str = "."):
    import zipfile
    with zipfile.ZipFile(bundle_path, "r") as z:
        z.extractall(dest)
    print(f"Extracted {bundle_path} to {dest}")


def run_zero_shot(
    data_dir: str = "/content/drive/MyDrive/tsfm-data",
    output_dir: str = "/content/drive/MyDrive/tsfm-results/phase3",
    context_length: int = 2048,
    scale_factor: float = 1.0,
    device: str = "cuda",
):
    from phase3_flowstate.run_flowstate import main
    main(
        data_dir=data_dir,
        output_dir=output_dir,
        context_length=context_length,
        scale_factor=scale_factor,
        device=device,
    )


def run_scale_sweep(
    data_dir: str = "/content/drive/MyDrive/tsfm-data",
    output_dir: str = "/content/drive/MyDrive/tsfm-results/phase3",
    device: str = "cuda",
):
    from phase3_flowstate.scale_sweep import main
    main(data_dir=data_dir, output_dir=output_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowState Colab launcher")
    parser.add_argument("--mode", choices=["zero_shot", "scale_sweep"], required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--data_dir", default="/content/drive/MyDrive/tsfm-data")
    parser.add_argument("--output_dir", default="/content/drive/MyDrive/tsfm-results/phase3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bundle", help="Path to colab_bundle.zip (optional)")
    args = parser.parse_args()

    if args.bundle:
        extract_bundle(args.bundle)

    sys.path.insert(0, ".")

    if args.mode == "zero_shot":
        run_zero_shot(args.data_dir, args.output_dir, args.context, args.scale, args.device)
    elif args.mode == "scale_sweep":
        run_scale_sweep(args.data_dir, args.output_dir, args.device)
