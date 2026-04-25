#!/usr/bin/env python
"""CLI: run Chemprop inference on the blind test set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.models.chemprop import predict


def main():
    parser = argparse.ArgumentParser(description="Predict pEC50 with a trained Chemprop model")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing best.pt files")
    parser.add_argument("--test-csv", type=str, default="data/activity_test_blinded.csv")
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--featurizers", nargs="+", default=None)
    parser.add_argument("--accelerator", type=str, default="mps")
    args = parser.parse_args()

    df = predict(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        molecule_featurizers=args.featurizers,
        accelerator=args.accelerator,
    )
    print(f"Predictions written to {args.out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
