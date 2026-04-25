"""XGBoost regression wrapper with scaffold CV evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from openadmet_pxr.evaluation.metrics import score, aggregate_cv_metrics
from openadmet_pxr.evaluation.splits import scaffold_cv_splits, save_splits
from openadmet_pxr.evaluation.tracking import log_cv_run as _log_cv_run
from openadmet_pxr.features.fingerprints import combined_features

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "mae",
}


def train_cv(
    smiles: list[str],
    targets: list[float],
    splits: list[dict[str, list[int]]],
    xgb_params: dict[str, Any] | None = None,
    use_morgan: bool = True,
    use_rdkit_2d: bool = False,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    run_name: str = "xgb_cv",
    extra_tracking_params: dict | None = None,
    out_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    params = {**(xgb_params or DEFAULT_XGB_PARAMS)}
    smiles = list(smiles)
    targets = np.array(targets, dtype=float)

    X = combined_features(
        smiles, use_morgan=use_morgan, use_rdkit_2d=use_rdkit_2d,
        morgan_radius=morgan_radius, morgan_n_bits=morgan_n_bits,
    )

    split_metrics, split_results = [], []
    model = None

    for i, split in enumerate(splits):
        train_idx = np.array(split["train"])
        val_idx = np.array(split["val"])
        X_train, y_train = X[train_idx], targets[train_idx]
        X_val, y_val = X[val_idx], targets[val_idx]
        y_train_mean = float(np.mean(y_train))

        model_params = {k: v for k, v in params.items()
                        if k not in ("early_stopping_rounds", "eval_metric")}
        model = xgb.XGBRegressor(
            **model_params,
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_preds = model.predict(X_val)
        m = score(y_val, val_preds, y_train_mean=y_train_mean)
        split_metrics.append(m)
        split_results.append({
            "split": i, "n_train": len(train_idx), "n_val": len(val_idx),
            "best_iteration": model.best_iteration, "metrics": m,
            "val_idx": val_idx.tolist(), "val_preds": val_preds.tolist(), "val_true": y_val.tolist(),
        })

        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "SMILES": [smiles[j] for j in val_idx],
                "pEC50_true": y_val, "pEC50_pred": val_preds,
            }).to_csv(out_dir / f"split_{i}_preds.csv", index=False)

        print(f"  Split {i:2d}: MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
              f"R2={m['r2']:.4f}  Spearman={m['spearman']:.4f}")

    cv_summary = aggregate_cv_metrics(split_metrics)

    tracking_params = {
        "model": "xgboost", "use_morgan": use_morgan, "use_rdkit_2d": use_rdkit_2d,
        "morgan_radius": morgan_radius, "morgan_n_bits": morgan_n_bits,
        "n_splits": len(splits), "n_estimators": params.get("n_estimators"),
        "max_depth": params.get("max_depth"), "learning_rate": params.get("learning_rate"),
        **(extra_tracking_params or {}),
    }

    all_val_idx = set(idx for s in split_results for idx in s["val_idx"])
    train_idx_all = [i for i in range(len(smiles)) if i not in all_val_idx]
    train_df = pd.DataFrame({"SMILES": [smiles[i] for i in train_idx_all], "pEC50": targets[train_idx_all]})

    _log_cv_run(
        run_name=run_name, params=tracking_params, cv_summary=cv_summary,
        train_df=train_df, model_object=model,
        artifacts=list(Path(out_dir).glob("*.csv")) if out_dir else None,
    )

    return cv_summary, split_results


def print_cv_summary(run_name: str, cv_summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{'='*55}")
    print(f"  {run_name}")
    print(f"{'='*55}")
    for metric in ["mae", "rae", "r2", "spearman", "kendall"]:
        s = cv_summary[metric]
        print(f"  {metric:<10} {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    print(f"{'='*55}\n")
