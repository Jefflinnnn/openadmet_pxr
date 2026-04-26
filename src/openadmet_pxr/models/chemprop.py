"""Chemprop v2 training and prediction wrapper with W&B tracking."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from openadmet_pxr.evaluation.metrics import score, aggregate_cv_metrics
from openadmet_pxr.evaluation.tracking import setup, WANDB_ENTITY, WANDB_PROJECT

ACTIVE_THRESHOLD = 6.0

PROJECT_ROOT = Path(__file__).parents[3]


def _find_chemprop_bin() -> str:
    venv_bin = PROJECT_ROOT / ".venv" / "bin" / "chemprop"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("chemprop")
    if found:
        return found
    raise FileNotFoundError("chemprop binary not found in .venv or PATH")


def run_chemprop(args: list[str]) -> None:
    cmd = [_find_chemprop_bin()] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"chemprop exited with code {result.returncode}")


def _log_epoch_curves(split_out: Path, split_idx: int) -> None:
    """Log per-epoch train/val loss from Chemprop's metrics.csv to W&B."""
    metrics_files = list(split_out.rglob("metrics.csv"))
    if not metrics_files or not wandb.run:
        return
    df = pd.read_csv(metrics_files[0])
    if "epoch" not in df.columns:
        return
    df = df.dropna(subset=["epoch"])
    for _, row in df.iterrows():
        epoch = int(row["epoch"])
        log = {}
        for col in df.columns:
            if col in ("epoch", "step") or pd.isna(row[col]):
                continue
            log[f"split_{split_idx}/{col}"] = float(row[col])
        if log:
            # Use a per-split step namespace so curves don't overlap across splits
            wandb.log({f"epochs/split_{split_idx}": epoch, **log})


def _active_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: float,
    threshold: float = ACTIVE_THRESHOLD,
) -> dict[str, float] | None:
    """Score only on active molecules (pEC50 >= threshold)."""
    mask = (y_true >= threshold) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return None
    m = score(y_true[mask], y_pred[mask], y_train_mean=y_train_mean)
    return {f"active_{k}": v for k, v in m.items()}


def train_cv(
    data_path: Path | str,
    target_columns: list[str],
    splits: list[dict[str, list[int]]],
    out_dir: Path | str,
    run_name: str,
    epochs: int = 50,
    warmup_epochs: int = 2,
    batch_size: int = 64,
    depth: int = 3,
    hidden_dim: int = 300,
    dropout: float = 0.0,
    ffn_num_layers: int = 1,
    task_weights: list[float] | None = None,
    molecule_featurizers: list[str] | None = None,
    from_foundation: str | None = "CheMeleon",
    accelerator: str = "mps",
    pytorch_seed: int | None = None,
    ensemble_size: int = 1,
    weight_column: str | None = None,
    loss_function: str | None = None,
    cv_strategy: str = "scaffold",
    extra_tracking_params: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Train Chemprop across all CV splits, predict val set, log to W&B live."""
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    primary_col = target_columns[0]

    config = {
        "model": "chemprop",
        "foundation": from_foundation or "scratch",
        "target_columns": ",".join(target_columns),
        "n_targets": len(target_columns),
        "task_weights": str(task_weights) if task_weights else "uniform",
        "molecule_featurizers": str(molecule_featurizers or []),
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "batch_size": batch_size,
        "depth": depth,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "ffn_num_layers": ffn_num_layers,
        "ensemble_size": ensemble_size,
        "weight_column": weight_column or "none",
        "loss_function": loss_function or "mse",
        "n_splits": len(splits),
        "cv_strategy": cv_strategy,
        **(extra_tracking_params or {}),
    }

    # Structured tags for easy filtering in W&B dashboard
    tags = [
        f"model:chemprop",
        f"foundation:{from_foundation or 'scratch'}",
        f"cv:{cv_strategy}",
        f"depth:{depth}",
        f"hidden:{hidden_dim}",
        f"dropout:{dropout}",
        f"ensemble:{ensemble_size}",
        f"weights:{weight_column or 'none'}",
    ]

    setup()
    split_metrics: list[dict] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
        tags=tags,
        reinit=True,
    )

    # Define custom axes so per-split loss curves are independent
    for i in range(len(splits)):
        wandb.define_metric(f"split_{i}/*", step_metric=f"epochs/split_{i}")

    y_train_mean_global = float(df[primary_col].dropna().mean())

    for i, split in enumerate(splits):
        split_out = out_dir / f"split_{i}"
        split_out.mkdir(exist_ok=True)

        splits_file = split_out / "splits.json"
        with open(splits_file, "w") as f:
            json.dump([split], f)

        train_cmd = [
            "train",
            "--data-path", str(data_path),
            "--smiles-columns", "SMILES",
            "--target-columns", *target_columns,
            "--task-type", "regression",
            "--output-dir", str(split_out),
            "--epochs", str(epochs),
            "--warmup-epochs", str(warmup_epochs),
            "--batch-size", str(batch_size),
            "--depth", str(depth),
            "--message-hidden-dim", str(hidden_dim),
            "--dropout", str(dropout),
            "--ffn-num-layers", str(ffn_num_layers),
            "--splits-file", str(splits_file),
            "--accelerator", accelerator,
            "--patience", "10",
        ]
        if from_foundation:
            train_cmd += ["--from-foundation", from_foundation]
        if molecule_featurizers:
            train_cmd += ["--molecule-featurizers", *molecule_featurizers]
        if task_weights:
            train_cmd += ["--task-weights", *[str(w) for w in task_weights]]
        if pytorch_seed is not None:
            train_cmd += ["--pytorch-seed", str(pytorch_seed)]
        if ensemble_size > 1:
            train_cmd += ["--ensemble-size", str(ensemble_size)]
        if weight_column:
            train_cmd += ["--weight-column", weight_column]
        if loss_function:
            train_cmd += ["--loss-function", loss_function]

        print(f"  Training split {i}...", flush=True)
        run_chemprop(train_cmd)

        _log_epoch_curves(split_out, i)

        # Predict val set
        val_idx = np.array(split["val"])
        val_csv = split_out / "val_input.csv"
        df.iloc[val_idx][["SMILES"]].to_csv(val_csv, index=False)

        preds_file = split_out / "val_predictions.csv"
        model_paths = list(split_out.rglob("best.pt"))
        predict_cmd = [
            "predict",
            "--test-path", str(val_csv),
            "--smiles-columns", "SMILES",
            "--model-paths", *[str(p) for p in model_paths],
            "--output", str(preds_file),
            "--accelerator", accelerator,
        ]
        if molecule_featurizers:
            predict_cmd += ["--molecule-featurizers", *molecule_featurizers]
        run_chemprop(predict_cmd)

        preds_df = pd.read_csv(preds_file)
        pred_col = primary_col if primary_col in preds_df.columns else next(
            (c for c in preds_df.columns if "predicted" in c.lower()),
            preds_df.columns[-1],
        )

        y_true = df.iloc[val_idx][primary_col].values
        y_pred = preds_df[pred_col].values
        y_train_mean = float(df.iloc[split["train"]][primary_col].dropna().mean())

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        m = score(y_true[valid], y_pred[valid], y_train_mean=y_train_mean)
        split_metrics.append(m)

        all_true.extend(y_true[valid].tolist())
        all_pred.extend(y_pred[valid].tolist())

        pd.DataFrame({
            "SMILES": df.iloc[val_idx]["SMILES"].values,
            "pEC50_true": y_true, "pEC50_pred": y_pred,
        }).to_csv(split_out / "val_preds.csv", index=False)

        # Per-split metrics (all metrics + active-specific)
        split_log = {f"split_{i}/{k}": v for k, v in m.items()}
        active_m = _active_metrics(y_true[valid], y_pred[valid], y_train_mean)
        if active_m:
            split_log.update({f"split_{i}/{k}": v for k, v in active_m.items()})
        wandb.log(split_log)

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
    active_pooled = _active_metrics(all_true_arr, all_pred_arr, y_train_mean_global)
    if active_pooled:
        summary_log.update({f"cv/{k}": v for k, v in active_pooled.items()})
        n_actives = int((all_true_arr >= ACTIVE_THRESHOLD).sum())
        summary_log["cv/n_actives_in_val"] = n_actives

    wandb.log(summary_log)
    # Also set as run summary so they appear in the W&B runs table
    for k, v in summary_log.items():
        run.summary[k] = v

    run.finish()
    return cv_summary


def predict(
    model_dir: Path | str,
    test_csv: Path | str,
    out_csv: Path | str,
    molecule_featurizers: list[str] | None = None,
    accelerator: str = "mps",
) -> pd.DataFrame:
    """Run Chemprop inference on a test CSV."""
    model_paths = list(Path(model_dir).rglob("best.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No best.pt found under {model_dir}")

    cmd = [
        "predict",
        "--test-path", str(test_csv),
        "--smiles-columns", "SMILES",
        "--model-paths", *[str(p) for p in model_paths],
        "--output", str(out_csv),
        "--accelerator", accelerator,
    ]
    if molecule_featurizers:
        cmd += ["--molecule-featurizers", *molecule_featurizers]
    run_chemprop(cmd)
    return pd.read_csv(out_csv)
