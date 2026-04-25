"""Chemprop v2 training and prediction wrapper with MLflow tracking."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openadmet_pxr.evaluation.metrics import score, aggregate_cv_metrics
from openadmet_pxr.evaluation.tracking import log_cv_run as _log_cv_run

PROJECT_ROOT = Path(__file__).parents[3]
CHEMPROP_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "chemprop")


def _run_chemprop(args: list[str]) -> None:
    cmd = [CHEMPROP_BIN] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"chemprop exited with code {result.returncode}")


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
    extra_tracking_params: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Train Chemprop across all CV splits, predict val set, log to MLflow."""
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    primary_col = target_columns[0]
    split_metrics = []

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

        print(f"  Training split {i}...", flush=True)
        _run_chemprop(train_cmd)

        # predict val set
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
        _run_chemprop(predict_cmd)

        preds_df = pd.read_csv(preds_file)
        pred_col = next(
            (c for c in preds_df.columns if "predicted" in c.lower()),
            preds_df.columns[-1],
        )

        y_true = df.iloc[val_idx][primary_col].values
        y_pred = preds_df[pred_col].values
        y_train_mean = float(df.iloc[split["train"]][primary_col].dropna().mean())

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        m = score(y_true[valid], y_pred[valid], y_train_mean=y_train_mean)
        split_metrics.append(m)

        pd.DataFrame({
            "SMILES": df.iloc[val_idx]["SMILES"].values,
            "pEC50_true": y_true, "pEC50_pred": y_pred,
        }).to_csv(split_out / "val_preds.csv", index=False)

        print(f"  Split {i:2d}: MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
              f"R2={m['r2']:.4f}  Spearman={m['spearman']:.4f}")

    cv_summary = aggregate_cv_metrics(split_metrics)

    tracking_params = {
        "model": "chemprop",
        "foundation": from_foundation or "scratch",
        "target_columns": ",".join(target_columns),
        "n_targets": len(target_columns),
        "task_weights": str(task_weights) if task_weights else "uniform",
        "molecule_featurizers": str(molecule_featurizers or []),
        "epochs": epochs, "depth": depth, "hidden_dim": hidden_dim,
        "dropout": dropout, "n_splits": len(splits),
        **(extra_tracking_params or {}),
    }

    train_df = df[df.index.isin(
        [idx for s in splits for idx in s["train"]]
    )][["SMILES", primary_col]].rename(columns={primary_col: "pEC50"}).dropna()

    _log_cv_run(
        run_name=run_name, params=tracking_params, cv_summary=cv_summary,
        train_df=train_df,
        artifacts=list(out_dir.rglob("val_preds.csv")),
    )

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
    _run_chemprop(cmd)
    return pd.read_csv(out_csv)
