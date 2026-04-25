"""Validate and format submission CSVs for the PXR challenge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = {"SMILES", "Molecule Name", "pEC50"}
EXPECTED_ROWS = 513


def validate_submission(path: Path | str) -> None:
    """Raise if the submission CSV fails any pre-submission check."""
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")

    nan_count = df["pEC50"].isna().sum()
    if nan_count > 0:
        raise ValueError(f"pEC50 has {nan_count} NaN values")

    inf_count = np.isinf(df["pEC50"].values).sum()
    if inf_count > 0:
        raise ValueError(f"pEC50 has {inf_count} Inf values")

    dup_smiles = df["SMILES"].duplicated().sum()
    if dup_smiles > 0:
        raise ValueError(f"Duplicate SMILES: {dup_smiles}")

    print(f"Submission valid: {len(df)} rows, pEC50 range [{df['pEC50'].min():.3f}, {df['pEC50'].max():.3f}]")


def make_submission(
    test_csv: Path | str,
    predictions: np.ndarray | list[float],
    out_path: Path | str,
) -> pd.DataFrame:
    """Merge test metadata with predictions and write submission CSV."""
    test_df = pd.read_csv(test_csv)
    if len(test_df) != EXPECTED_ROWS:
        raise ValueError(f"Test CSV has {len(test_df)} rows, expected {EXPECTED_ROWS}")

    sub = test_df[["SMILES", "Molecule Name"]].copy()
    sub["pEC50"] = predictions
    sub.to_csv(out_path, index=False)
    validate_submission(out_path)
    return sub
