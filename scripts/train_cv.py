"""
Run Suiren-ConfAvg 5-fold CV for PXR pEC50 (Stage 0 — no sample weighting).

For each fold:
  1. Invokes Suiren main.py from the fold's working directory so that
     data/pxr_pec50/{name}_train.csv and data/pxr_pec50/{name}_valid.csv
     are found at the expected relative paths.
  2. Finds the best checkpoint saved by Suiren.
  3. Runs inference on the validation set.
  4. Saves per-compound predictions to runs/fold_{i}/val_predictions.csv.

Usage:
    python scripts/train_cv.py [--folds 0 1 2 3 4] [--epochs 100] [--lr 2e-4]
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUIREN_MAIN = os.path.join(PROJECT_ROOT, "models/suiren/repo/main.py")
SUIREN_INFER = os.path.join(PROJECT_ROOT, "models/suiren/repo/inference.py")
PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "models/suiren/weights/model.pt")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "runs/analog_mimic_5fold.json")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TRAIN.csv")
ACTIVE_THRESHOLD = 6.0
NAME = "pxr_pec50"


def find_best_checkpoint(fold_dir):
    """Return path to the best regression checkpoint saved by Suiren."""
    pattern = os.path.join(fold_dir, "checkpoints", NAME, "*", f"{NAME}_regression.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    # Suiren saves only the best checkpoint per run; take the most recent dir.
    return matches[-1]


def run_training(fold_idx, fold_run_dir, epochs, lr, seed):
    """Invoke Suiren training from the fold's run directory."""
    run_name = f"stage0_cv_fold{fold_idx}_e{epochs}_lr{lr}_s{seed}"
    cmd = [
        sys.executable, SUIREN_MAIN,
        "--checkpoint-pretrain", PRETRAIN_CKPT,
        "--mode", "regression",
        "--loss", "l1",
        "--name", NAME,
        "--data-mode", "smiles_defined",
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed),
        "--batch-size", "8",
        "--workers", "0",  # avoid multiprocessing issues on macOS
        "--wandb-run-name", run_name,
    ]
    print(f"\n=== Fold {fold_idx}: training ===")
    print("CWD:", fold_run_dir)
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=fold_run_dir, check=True)
    return result


def run_inference(fold_idx, fold_run_dir, val_csv_path, checkpoint_path, output_path):
    """Run Suiren inference on the validation set."""
    cmd = [
        sys.executable, SUIREN_INFER,
        "--task", NAME,
        "--checkpoint", checkpoint_path,
        "--input", val_csv_path,
        "--output", output_path,
    ]
    print(f"\n=== Fold {fold_idx}: inference ===")
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=fold_run_dir, check=True)


def compute_metrics(val_predictions_path, train_df, val_indices):
    """Compute overall MAE and active-subgroup MAE from Suiren's output CSV."""
    preds_df = pd.read_csv(val_predictions_path)
    val_df = train_df.iloc[val_indices].reset_index(drop=True)

    # Suiren inference appends a 'value' column with predictions.
    # The input val CSV already has a 'value' column (true labels), so Suiren
    # overwrites it. Re-join on position.
    true = val_df["pEC50"].values
    pred = preds_df["value"].values

    if len(true) != len(pred):
        raise ValueError(f"Length mismatch: true={len(true)}, pred={len(pred)}")

    abs_err = np.abs(true - pred)
    n_nan = np.isnan(pred).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} NaN predictions excluded from metrics")
    overall_mae = float(np.nanmean(abs_err))

    active_mask = true >= ACTIVE_THRESHOLD
    active_mae = float(np.nanmean(abs_err[active_mask])) if active_mask.sum() > 0 else float("nan")

    return {
        "overall_mae": float(overall_mae),
        "active_mae": float(active_mae),
        "n_val": int(len(true)),
        "n_active_val": int(active_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                        help="Which folds to run (default: all 5)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(SPLITS_JSON):
        sys.exit(f"Splits file not found: {SPLITS_JSON}\nRun scripts/make_cv_splits.py first.")
    if not os.path.exists(PRETRAIN_CKPT):
        sys.exit(f"Pretrained weights not found: {PRETRAIN_CKPT}\nDownload via: huggingface-cli download ajy112/Suiren-ConfAvg --local-dir models/suiren/weights")

    with open(SPLITS_JSON) as fh:
        splits = json.load(fh)

    train_df = pd.read_csv(TRAIN_CSV)
    results = []

    for fold_idx in args.folds:
        fold_run_dir = os.path.join(PROJECT_ROOT, f"runs/fold_{fold_idx}")
        val_csv = os.path.join(fold_run_dir, "data", NAME, "raw", f"{NAME}_valid.csv")
        pred_output = os.path.join(fold_run_dir, "val_predictions.csv")

        run_training(fold_idx, fold_run_dir, args.epochs, args.lr, args.seed)

        checkpoint = find_best_checkpoint(fold_run_dir)
        print(f"Fold {fold_idx}: best checkpoint → {checkpoint}")

        run_inference(fold_idx, fold_run_dir, val_csv, checkpoint, pred_output)

        metrics = compute_metrics(pred_output, train_df, splits[fold_idx]["val_indices"])
        metrics["fold"] = fold_idx
        results.append(metrics)
        print(f"Fold {fold_idx}: overall_mae={metrics['overall_mae']:.4f}  active_mae={metrics['active_mae']:.4f}  "
              f"n_val={metrics['n_val']}  n_active={metrics['n_active_val']}")

    if results:
        overall_maes = [r["overall_mae"] for r in results]
        active_maes = [r["active_mae"] for r in results if not pd.isna(r["active_mae"])]
        print(f"\n=== CV Summary ({len(results)} folds) ===")
        print(f"Overall MAE:  mean={np.mean(overall_maes):.4f}  std={np.std(overall_maes):.4f}")
        print(f"Active  MAE:  mean={np.mean(active_maes):.4f}  std={np.std(active_maes):.4f}")

        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(PROJECT_ROOT, "runs/cv_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
