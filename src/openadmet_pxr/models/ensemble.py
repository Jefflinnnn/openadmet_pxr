"""Ensemble multiple prediction CSVs by weighted average."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def blend_predictions(
    pred_files: list[Path | str],
    weights: list[float] | None = None,
    smiles_col: str = "SMILES",
    pred_col: str = "pEC50",
) -> pd.Series:
    """Weighted average of prediction CSVs, aligned on SMILES."""
    dfs = [pd.read_csv(f).set_index(smiles_col)[pred_col] for f in pred_files]
    if weights is None:
        weights = [1.0] * len(dfs)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Align on common SMILES index
    aligned = pd.concat(
        [s.rename(f"pred_{i}") for i, s in enumerate(dfs)], axis=1
    )
    return (aligned * weights).sum(axis=1).rename(pred_col)


def rae_weighted_blend(
    pred_files: list[Path | str],
    rae_scores: list[float],
    smiles_col: str = "SMILES",
    pred_col: str = "pEC50",
) -> pd.Series:
    """Inverse-RAE weighted blend: better models get higher weight."""
    inv_rae = [1.0 / r for r in rae_scores]
    return blend_predictions(pred_files, weights=inv_rae, smiles_col=smiles_col, pred_col=pred_col)
