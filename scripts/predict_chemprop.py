"""Run inference on the blinded test set using a trained Chemprop model.

Usage:
  source .venv/bin/activate
  python scripts/predict_chemprop.py \
    --model-path runs/chemeleon_baseline/model_0/best.pt \
    --test-csv data/activity_test_blinded.csv \
    --preds-csv runs/chemeleon_baseline/test_preds.csv
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a .pt file (e.g., runs/.../model_0/best.pt) or a directory of .pt files.",
    )
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--preds-csv", type=Path, required=True)
    ap.add_argument("--smiles-col", type=str, default="SMILES")
    ap.add_argument(
        "--molecule-featurizers",
        nargs="+",
        default=None,
        help=(
            "Optional list of Chemprop molecule featurizers to use at predict time. "
            "If you trained with extra features (e.g. rdkit_2d / morgan_count), you should "
            "pass the same list here."
        ),
    )
    ap.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Optional Lightning accelerator for chemprop predict (e.g. mps, cpu).",
    )
    ap.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional Lightning devices for chemprop predict (e.g. 1).",
    )
    args = ap.parse_args()

    args.preds_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "chemprop",
        "predict",
        "--test-path",
        str(args.test_csv),
        "--smiles-columns",
        args.smiles_col,
        "--model-path",
        str(args.model_path),
        "--output",
        str(args.preds_csv),
        "--num-workers",
        "0",
    ]

    if args.molecule_featurizers:
        cmd += ["--molecule-featurizers", *args.molecule_featurizers]
    if args.accelerator:
        cmd += ["--accelerator", args.accelerator]
    if args.devices:
        cmd += ["--devices", args.devices]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
