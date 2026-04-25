"""Ensemble Chemprop prediction CSVs.

Chemprop `predict` writes a CSV with columns like:
  SMILES, Molecule Name, pEC50

This utility averages multiple such prediction files (optionally weighted) and
writes a single prediction CSV with the same schema.

Typical usage:
  python scripts/ensemble_preds.py \
    --preds-csv runs/run1/test_preds.csv runs/run2/test_preds.csv \
    --out-csv runs/ensemble/test_preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument(
        "--target-col",
        type=str,
        default="pEC50",
        help="Prediction column to ensemble (default: pEC50)",
    )
    ap.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional weights (same length as --preds-csv). If omitted, uniform average.",
    )
    args = ap.parse_args()

    if args.weights is not None and len(args.weights) != len(args.preds_csv):
        raise SystemExit("--weights must have the same length as --preds-csv")

    dfs: list[pd.DataFrame] = []
    for p in args.preds_csv:
        df = pd.read_csv(p)
        for col in ("SMILES", "Molecule Name", args.target_col):
            if col not in df.columns:
                raise SystemExit(f"{p} missing required column: {col}")
        dfs.append(df[["SMILES", "Molecule Name", args.target_col]].copy())

    base = dfs[0][["SMILES", "Molecule Name"]].copy()

    # Ensure alignment.
    for df in dfs[1:]:
        if not base["SMILES"].equals(df["SMILES"]) or not base["Molecule Name"].equals(df["Molecule Name"]):
            raise SystemExit(
                "Prediction CSVs are not aligned (SMILES / Molecule Name differ or order differs). "
                "Re-run predict with identical input ordering."
            )

    preds = np.stack([df[args.target_col].astype(float).to_numpy() for df in dfs], axis=0)
    if args.weights is None:
        weights = np.ones(preds.shape[0], dtype=float) / preds.shape[0]
    else:
        w = np.asarray(args.weights, dtype=float)
        if np.any(w < 0):
            raise SystemExit("--weights must be non-negative")
        if w.sum() == 0:
            raise SystemExit("--weights must not sum to 0")
        weights = w / w.sum()

    y = (preds.T @ weights).astype(float)

    out = base.copy()
    out[args.target_col] = y

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote ensembled predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
