"""Create an Activity Track submission file from predictions.

The challenge requires exactly 513 rows and these columns:
  SMILES, Molecule Name, pEC50

This script merges the blinded test metadata CSV with a Chemprop predictions CSV.

Usage:
  source .venv/bin/activate
  python scripts/make_submission.py \
    --test-csv data/activity_test_blinded.csv \
    --preds-csv runs/chemeleon_baseline/test_preds.csv \
    --out-csv submissions/chemeleon_baseline.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["SMILES", "Molecule Name", "pEC50"]
N_REQUIRED_ROWS = 513


def _find_pred_col(preds_df: pd.DataFrame) -> str:
    # chemprop predict usually outputs a single column with the target name,
    # or something like "pEC50" / "pEC50_pred".
    candidates = [
        c
        for c in preds_df.columns
        if c.lower() in {"pec50", "pec50_pred", "prediction", "pred"}
        or c.lower().endswith("_pred")
    ]
    if "pEC50" in preds_df.columns:
        return "pEC50"
    # Multitask models in this repo often use an explicit PXR column name.
    if "pxr_pEC50" in preds_df.columns:
        return "pxr_pEC50"
    if len(candidates) == 1:
        return candidates[0]
    # fallback: last column
    return preds_df.columns[-1]


def validate_submission(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) != N_REQUIRED_ROWS:
        raise ValueError(f"Submission must have {N_REQUIRED_ROWS} rows; got {len(df)}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Submission contains NaNs")
    if not np.isfinite(df["pEC50"].to_numpy(dtype=float)).all():
        raise ValueError("Submission pEC50 contains inf/-inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--preds-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument(
        "--pred-col",
        type=str,
        default=None,
        help=(
            "Optional explicit column name to read from --preds-csv. "
            "Useful for multitask predictions (e.g., 'pxr_pEC50')."
        ),
    )
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_csv)
    preds_df = pd.read_csv(args.preds_csv)

    pred_col = args.pred_col or _find_pred_col(preds_df)
    if pred_col not in preds_df.columns:
        raise ValueError(f"Prediction column {pred_col!r} not found in preds CSV")
    if len(preds_df) != len(test_df):
        raise ValueError(
            f"Preds/test row mismatch: preds={len(preds_df)} test={len(test_df)}. "
            "Ensure you predicted on the same test CSV in the same order."
        )

    sub_df = test_df[["SMILES", "Molecule Name"]].copy()
    sub_df["pEC50"] = preds_df[pred_col].astype(float)

    # final ordering
    sub_df = sub_df[REQUIRED_COLUMNS]

    validate_submission(sub_df)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(args.out_csv, index=False)
    print(f"Wrote submission: {args.out_csv} rows={len(sub_df)}")


if __name__ == "__main__":
    main()
