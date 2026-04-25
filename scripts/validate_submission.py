"""Validate an OpenADMET PXR Activity Track submission file.

Usage:
  source .venv/bin/activate
  python scripts/validate_submission.py submissions/whatever.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["SMILES", "Molecule Name", "pEC50"]
N_REQUIRED_ROWS = 513


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    path: Path = args.path
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"FAIL: missing columns: {sorted(missing)}")
    if len(df) != N_REQUIRED_ROWS:
        raise SystemExit(f"FAIL: expected {N_REQUIRED_ROWS} rows, got {len(df)}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise SystemExit("FAIL: contains NaNs")
    if not np.isfinite(df["pEC50"].to_numpy(dtype=float)).all():
        raise SystemExit("FAIL: pEC50 contains inf/-inf")

    print(
        f"OK: {path} rows={len(df)} cols={list(df.columns)} "
        f"pEC50_range=({df['pEC50'].min():.3f}, {df['pEC50'].max():.3f})"
    )


if __name__ == "__main__":
    main()
