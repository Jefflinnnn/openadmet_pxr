"""XGBoost regression wrapper with scaffold CV and live W&B tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
import xgboost as xgb

from openadmet_pxr.evaluation.metrics import score, aggregate_cv_metrics
from openadmet_pxr.evaluation.tracking import setup, WANDB_ENTITY, WANDB_PROJECT
from openadmet_pxr.features.fingerprints import combined_features
from openadmet_pxr.models.chemprop import ACTIVE_THRESHOLD, _active_metrics

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


class _WandbCallback(xgb.callback.TrainingCallback):
    """Log val MAE per boosting round live to the active W&B run."""
    def __init__(self, split_idx: int):
        self.split_idx = split_idx

    def after_iteration(self, model, epoch, evals_log):
        log = {f"boosting_round/split_{self.split_idx}": epoch}
        for data, metrics in evals_log.items():
            for metric, values in metrics.items():
                log[f"split_{self.split_idx}/{data}_{metric}"] = values[-1]
        if wandb.run:
            wandb.log(log)
        return False


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
    cv_strategy: str = "scaffold",
    extra_tracking_params: dict | None = None,
    out_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Train XGBoost with CV, logging per-round metrics live to W&B."""
    params = {**(xgb_params or DEFAULT_XGB_PARAMS)}
    smiles = list(smiles)
    targets = np.array(targets, dtype=float)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    X = combined_features(
        smiles, use_morgan=use_morgan, use_rdkit_2d=use_rdkit_2d,
        morgan_radius=morgan_radius, morgan_n_bits=morgan_n_bits,
    )

    config = {
        "model": "xgboost",
        "use_morgan": use_morgan,
        "use_rdkit_2d": use_rdkit_2d,
        "morgan_radius": morgan_radius,
        "morgan_n_bits": morgan_n_bits,
        "n_splits": len(splits),
        "n_estimators": params.get("n_estimators"),
        "max_depth": params.get("max_depth"),
        "learning_rate": params.get("learning_rate"),
        "cv_strategy": cv_strategy,
        **(extra_tracking_params or {}),
    }

    tags = [
        "model:xgboost",
        f"cv:{cv_strategy}",
        f"morgan:{use_morgan}",
        f"rdkit2d:{use_rdkit_2d}",
        f"depth:{params.get('max_depth')}",
    ]

    setup()
    split_metrics: list[dict] = []
    split_results: list[dict] = []
    all_true: list[float] = []
    all_pred: list[float] = []
    model = None

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
        tags=tags,
        reinit=True,
    )

    # Define per-split step axes for clean loss curves
    for i in range(len(splits)):
        wandb.define_metric(f"split_{i}/*", step_metric=f"boosting_round/split_{i}")

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
            callbacks=[_WandbCallback(i)],
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

        all_true.extend(y_val.tolist())
        all_pred.extend(val_preds.tolist())

        # Per-split summary + active-specific metrics
        split_log = {f"split_{i}/{k}": v for k, v in m.items()}
        active_m = _active_metrics(y_val, val_preds, y_train_mean)
        if active_m:
            split_log.update({f"split_{i}/{k}": v for k, v in active_m.items()})
        wandb.log(split_log)

        if out_dir:
            pd.DataFrame({
                "SMILES": [smiles[j] for j in val_idx],
                "pEC50_true": y_val, "pEC50_pred": val_preds,
            }).to_csv(out_dir / f"split_{i}_preds.csv", index=False)

        print(f"  Split {i:2d}: MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
              f"R2={m['r2']:.4f}  Spearman={m['spearman']:.4f}")

    # Aggregated CV summary
    cv_summary = aggregate_cv_metrics(split_metrics)
    summary_log = {}
    for k, v in cv_summary.items():
        summary_log[f"cv/{k}_mean"] = v["mean"]
        summary_log[f"cv/{k}_std"] = v["std"]

    # Pooled active-specific metrics across all val folds
    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)
    y_train_mean_global = float(np.nanmean(targets))
    active_pooled = _active_metrics(all_true_arr, all_pred_arr, y_train_mean_global)
    if active_pooled:
        summary_log.update({f"cv/{k}": v for k, v in active_pooled.items()})
        summary_log["cv/n_actives_in_val"] = int((all_true_arr >= ACTIVE_THRESHOLD).sum())

    wandb.log(summary_log)
    for k, v in summary_log.items():
        run.summary[k] = v

    if out_dir:
        for f in out_dir.glob("*.csv"):
            wandb.save(str(f))

    run.finish()
    return cv_summary, split_results


def print_cv_summary(run_name: str, cv_summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{'='*55}")
    print(f"  {run_name}")
    print(f"{'='*55}")
    for metric in ["mae", "rae", "r2", "spearman", "kendall"]:
        s = cv_summary[metric]
        print(f"  {metric:<10} {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    print(f"{'='*55}\n")
