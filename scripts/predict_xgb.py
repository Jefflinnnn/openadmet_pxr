#!/usr/bin/env python
"""CLI: run XGBoost inference on the blind test set using saved split models."""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.data.load import load_test
from openadmet_pxr.features.fingerprints import combined_features


def main():
    parser = argparse.ArgumentParser(description="Predict with XGBoost on test set")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Directory containing XGBoost model files")
    parser.add_argument("--morgan", action="store_true")
    parser.add_argument("--rdkit2d", action="store_true")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-nbits", type=int, default=2048)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    if not args.morgan and not args.rdkit2d:
        parser.error("Specify at least one of --morgan or --rdkit2d")

    run_dir = Path(args.run_dir)
    test_df = load_test()
    smiles = list(test_df["SMILES"])
    X_test = combined_features(
        smiles, use_morgan=args.morgan, use_rdkit_2d=args.rdkit2d,
        morgan_radius=args.morgan_radius, morgan_n_bits=args.morgan_nbits,
    )

    model_files = sorted(
        glob.glob(str(run_dir / "*.json"))
        + glob.glob(str(run_dir / "*.ubj"))
        + glob.glob(str(run_dir / "*.pkl"))
    )
    if not model_files:
        raise FileNotFoundError(f"No XGBoost model files found in {run_dir}")

    all_preds = []
    for mf in model_files:
        model = xgb.XGBRegressor()
        model.load_model(mf)
        all_preds.append(model.predict(X_test))
    preds = np.mean(all_preds, axis=0)

    out = pd.DataFrame({"SMILES": smiles, "pEC50": preds})
    out.to_csv(args.out_csv, index=False)
    print(f"Predictions written to {args.out_csv}  ({len(out)} rows)")
    print(f"pEC50 range: [{preds.min():.3f}, {preds.max():.3f}]")


if __name__ == "__main__":
    main()
