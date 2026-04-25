#!/usr/bin/env python
"""CLI: create and validate a final submission CSV from prediction files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from openadmet_pxr.models.ensemble import blend_predictions, rae_weighted_blend
from openadmet_pxr.submission.validate import make_submission, validate_submission

SUBMISSIONS_DIR = Path(__file__).parents[1] / "submissions"


def main():
    parser = argparse.ArgumentParser(description="Create submission CSV")
    parser.add_argument("--pred-files", nargs="+", required=True,
                        help="Prediction CSV files to blend")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Blend weights (default: equal)")
    parser.add_argument("--rae-scores", nargs="+", type=float, default=None,
                        help="CV RAE scores for inverse-RAE weighting")
    parser.add_argument("--test-csv", type=str, default="data/activity_test_blinded.csv")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: submissions/submission_<name>.csv)")
    parser.add_argument("--name", type=str, default="ensemble",
                        help="Name tag for output file")
    args = parser.parse_args()

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else SUBMISSIONS_DIR / f"submission_{args.name}.csv"

    pred_files = [Path(f) for f in args.pred_files]

    if args.rae_scores:
        blended = rae_weighted_blend(pred_files, args.rae_scores)
    else:
        blended = blend_predictions(pred_files, weights=args.weights)

    sub = make_submission(args.test_csv, blended.values, out_path)
    print(f"Submission saved to {out_path}")
    print(sub.describe())


if __name__ == "__main__":
    main()
