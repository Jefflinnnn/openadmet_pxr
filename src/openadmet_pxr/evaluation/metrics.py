"""Regression metrics for the PXR activity prediction task."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rae(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    """Relative Absolute Error relative to a train-mean baseline."""
    numerator = np.sum(np.abs(y_pred - y_true))
    denominator = np.sum(np.abs(y_train_mean - y_true))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: float | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics (mae, rmse, rae, r2, spearman, kendall)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_train_mean is None:
        y_train_mean = float(np.mean(y_true))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rae": rae(y_true, y_pred, y_train_mean),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
    }


def aggregate_cv_metrics(split_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-split metric dicts into mean ± std summary."""
    keys = split_metrics[0].keys()
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in split_metrics])
        result[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return result
