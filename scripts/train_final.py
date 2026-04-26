#!/usr/bin/env python
"""Train final model on all training data (no val split) and predict test set."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.models.chemprop import run_chemprop
from openadmet_pxr.submission.validate import make_submission

PROJECT_ROOT = Path(__file__).parents[1]
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main():
    parser = argparse.ArgumentParser(description="Train on full data and produce submission CSV")
    parser.add_argument("--data-path", default="data/activity_train.csv")
    parser.add_argument("--test-path", default="data/activity_test_blinded.csv")
    parser.add_argument("--target-columns", nargs="+", default=["pEC50"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn-layers", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--weight-column", type=str, default=None)
    parser.add_argument("--loss-function", type=str, default=None)
    parser.add_argument("--no-foundation", action="store_true")
    parser.add_argument("--accelerator", default="mps")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--submission-name", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(args.data_path)
    n = len(train_df)
    val_size = max(50, int(0.05 * n))
    np.random.seed(42)
    val_idx = np.random.choice(n, val_size, replace=False).tolist()
    train_idx = [i for i in range(n) if i not in set(val_idx)]
    splits_file = out_dir / "full_train_split.json"
    with open(splits_file, "w") as f:
        json.dump([{"train": train_idx, "val": val_idx, "test": []}], f)

    print(f"Training on {len(train_idx)} molecules ({val_size} held out for early stopping only)...")

    train_cmd = [
        "train",
        "--data-path", args.data_path,
        "--smiles-columns", "SMILES",
        "--target-columns", *args.target_columns,
        "--task-type", "regression",
        "--output-dir", str(out_dir),
        "--epochs", str(args.epochs),
        "--warmup-epochs", str(args.warmup_epochs),
        "--batch-size", str(args.batch_size),
        "--depth", str(args.depth),
        "--message-hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--ffn-num-layers", str(args.ffn_layers),
        "--ensemble-size", str(args.ensemble_size),
        "--splits-file", str(splits_file),
        "--accelerator", args.accelerator,
        "--patience", "10",
    ]
    if not args.no_foundation:
        train_cmd += ["--from-foundation", "CheMeleon"]
    if args.weight_column:
        train_cmd += ["--weight-column", args.weight_column]
    if args.loss_function:
        train_cmd += ["--loss-function", args.loss_function]

    run_chemprop(train_cmd)
    print("Training complete.")

    model_paths = list(out_dir.rglob("best.pt"))
    print(f"Found {len(model_paths)} model(s) for prediction.")
    preds_file = out_dir / "test_predictions.csv"
    predict_cmd = [
        "predict",
        "--test-path", args.test_path,
        "--smiles-columns", "SMILES",
        "--model-paths", *[str(p) for p in model_paths],
        "--output", str(preds_file),
        "--accelerator", args.accelerator,
    ]
    run_chemprop(predict_cmd)

    preds_df = pd.read_csv(preds_file)
    primary_col = args.target_columns[0]
    pred_col = primary_col if primary_col in preds_df.columns else preds_df.columns[-1]
    predictions = preds_df[pred_col].values

    sub_path = SUBMISSIONS_DIR / f"submission_{args.submission_name}.csv"
    make_submission(args.test_path, predictions, sub_path)
    print(f"\nSubmission saved: {sub_path}")


if __name__ == "__main__":
    main()
