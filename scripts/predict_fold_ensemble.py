#!/usr/bin/env python
"""Predict test set using all CV fold models and average predictions (fold ensembling)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.models.chemprop import run_chemprop
from openadmet_pxr.submission.validate import make_submission

SUBMISSIONS_DIR = Path(__file__).parents[1] / "submissions"


def main():
    parser = argparse.ArgumentParser(description="Fold ensemble prediction from CV models")
    parser.add_argument("--run-dir", required=True, help="CV run directory (e.g. runs/C13_d4h600_ens5_analog)")
    parser.add_argument("--test-path", default="data/activity_test_blinded.csv")
    parser.add_argument("--submission-name", required=True)
    parser.add_argument("--molecule-featurizers", nargs="+", default=None)
    parser.add_argument("--accelerator", default="mps")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_dir = run_dir / "fold_ensemble_preds"
    out_dir.mkdir(exist_ok=True)

    all_preds = []
    for i in range(args.n_splits):
        split_dir = run_dir / f"split_{i}"
        model_paths = sorted(split_dir.rglob("best.pt"))
        if not model_paths:
            print(f"  Split {i}: no best.pt found, skipping")
            continue

        preds_file = out_dir / f"split_{i}_test_preds.csv"
        predict_cmd = [
            "predict",
            "--test-path", args.test_path,
            "--smiles-columns", "SMILES",
            "--model-paths", *[str(p) for p in model_paths],
            "--output", str(preds_file),
            "--accelerator", args.accelerator,
        ]
        if args.molecule_featurizers:
            predict_cmd += ["--molecule-featurizers", *args.molecule_featurizers]

        print(f"  Split {i}: predicting with {len(model_paths)} models...", flush=True)
        run_chemprop(predict_cmd)

        preds_df = pd.read_csv(preds_file)
        pred_col = next((c for c in preds_df.columns if "pEC50" in c or "predicted" in c.lower()), preds_df.columns[-1])
        all_preds.append(preds_df[pred_col].values)
        print(f"  Split {i}: done, pred range [{preds_df[pred_col].min():.3f}, {preds_df[pred_col].max():.3f}]")

    if not all_preds:
        raise RuntimeError("No predictions collected — check run_dir and split structure")

    ensemble_preds = np.mean(all_preds, axis=0)
    print(f"\nFold ensemble ({len(all_preds)} folds averaged):")
    print(f"  pEC50 range: [{ensemble_preds.min():.3f}, {ensemble_preds.max():.3f}]")
    print(f"  pEC50 mean:  {ensemble_preds.mean():.3f}")

    sub_path = SUBMISSIONS_DIR / f"submission_{args.submission_name}.csv"
    make_submission(args.test_path, ensemble_preds, sub_path)
    print(f"\nSubmission saved: {sub_path}")


if __name__ == "__main__":
    main()
