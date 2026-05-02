"""
Train Suiren-ConfAvg on all 4,139 PXR training compounds and generate a
submission CSV for the leaderboard.

Usage:
    python scripts/train_final.py [--epochs 100] [--lr 2e-4] [--seed 0]

Output:
    runs/final/submission.csv  -- 513 rows: Molecule Name, pEC50
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUIREN_MAIN = os.path.join(PROJECT_ROOT, "models/suiren/repo/main.py")
SUIREN_INFER = os.path.join(PROJECT_ROOT, "models/suiren/repo/inference.py")
PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "models/suiren/weights/model.pt")
TEST_CSV = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TEST_BLINDED.csv")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TRAIN.csv")
NAME = "pxr_pec50"
FINAL_RUN_DIR = os.path.join(PROJECT_ROOT, "runs/final")

PRED_MIN = 1.0
PRED_MAX = 8.5
TRAIN_MEAN = 4.65
MEAN_TOLERANCE = 0.3
ACTIVE_FRAC_MIN = 0.05
ACTIVE_FRAC_MAX = 0.50


def find_best_checkpoint(run_dir):
    pattern = os.path.join(run_dir, "checkpoints", NAME, "*", f"{NAME}_regression.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    return matches[-1]


def run_training(epochs, lr, seed):
    run_name = f"stage0_final_e{epochs}_lr{lr}_s{seed}"
    cmd = [
        sys.executable, SUIREN_MAIN,
        "--checkpoint-pretrain", PRETRAIN_CKPT,
        "--mode", "regression",
        "--loss", "l1",
        "--name", NAME,
        "--data-mode", "smiles_random",  # no held-out val needed; Suiren random splits internally
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed),
        "--batch-size", "8",
        "--workers", "0",
        "--wandb-run-name", run_name,
    ]
    print("=== Training on full dataset ===")
    print("CWD:", FINAL_RUN_DIR)
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=FINAL_RUN_DIR, check=True)


def run_inference(checkpoint_path, input_csv, output_csv):
    cmd = [
        sys.executable, SUIREN_INFER,
        "--task", NAME,
        "--checkpoint", checkpoint_path,
        "--input", input_csv,
        "--output", output_csv,
    ]
    print("\n=== Inference on test set ===")
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, cwd=FINAL_RUN_DIR, check=True)


def sanity_checks(pred_series):
    """Raise on any submission-blocking issue; print warnings otherwise."""
    errors = []
    warnings = []

    nan_count = pred_series.isna().sum()
    if nan_count > 0:
        errors.append(f"{nan_count} NaN predictions")

    inf_count = np.isinf(pred_series.fillna(0)).sum()
    if inf_count > 0:
        errors.append(f"{inf_count} inf predictions")

    out_of_range = ((pred_series < PRED_MIN) | (pred_series > PRED_MAX)).sum()
    if out_of_range > 0:
        warnings.append(f"{out_of_range} predictions outside [{PRED_MIN}, {PRED_MAX}]")

    mean_pred = pred_series.mean()
    if abs(mean_pred - TRAIN_MEAN) > MEAN_TOLERANCE:
        warnings.append(f"Mean prediction {mean_pred:.3f} deviates from train mean {TRAIN_MEAN} by >{MEAN_TOLERANCE}")

    active_frac = (pred_series >= 6.0).mean()
    if active_frac < ACTIVE_FRAC_MIN or active_frac > ACTIVE_FRAC_MAX:
        warnings.append(f"Predicted active fraction {active_frac:.1%} outside expected range "
                        f"[{ACTIVE_FRAC_MIN:.0%}, {ACTIVE_FRAC_MAX:.0%}]")

    if errors:
        raise ValueError(f"Submission blocked — {'; '.join(errors)}")
    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print(f"All sanity checks passed (n={len(pred_series)}, mean={mean_pred:.3f}, active_frac={active_frac:.1%})")


def prepare_test_csv():
    """Write test SMILES in Suiren's expected format for inference."""
    test_infer_dir = os.path.join(FINAL_RUN_DIR, "data", NAME)
    os.makedirs(test_infer_dir, exist_ok=True)
    test_df = pd.read_csv(TEST_CSV)
    # Suiren inference only needs SMILES; value column is optional.
    infer_csv = os.path.join(test_infer_dir, f"{NAME}_test_infer.csv")
    test_df[["SMILES"]].to_csv(infer_csv, index=False)
    return infer_csv, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and use existing checkpoint")
    args = parser.parse_args()

    if not os.path.exists(PRETRAIN_CKPT):
        sys.exit(f"Pretrained weights not found: {PRETRAIN_CKPT}")

    os.makedirs(FINAL_RUN_DIR, exist_ok=True)

    if not args.skip_training:
        # smiles_random mode expects data/{name}/raw/{name}.csv
        raw_dir = os.path.join(FINAL_RUN_DIR, "data", NAME, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        train_csv_dest = os.path.join(raw_dir, f"{NAME}.csv")
        if not os.path.exists(train_csv_dest):
            src = os.path.join(raw_dir, f"{NAME}_train.csv")
            if not os.path.exists(src):
                sys.exit(f"Full training CSV not found: {src}\nRun scripts/make_cv_splits.py first.")
            shutil.copy(src, train_csv_dest)
        run_training(args.epochs, args.lr, args.seed)

    checkpoint = find_best_checkpoint(FINAL_RUN_DIR)
    print(f"\nBest checkpoint: {checkpoint}")

    infer_csv, test_df = prepare_test_csv()
    raw_output = os.path.join(FINAL_RUN_DIR, "test_raw_predictions.csv")
    run_inference(checkpoint, infer_csv, raw_output)

    preds_df = pd.read_csv(raw_output)
    if "value" not in preds_df.columns:
        sys.exit("Inference output missing 'value' column — check Suiren inference output format.")

    sanity_checks(preds_df["value"])

    submission = test_df[["Molecule Name", "SMILES"]].copy()
    submission["pEC50"] = preds_df["value"].values
    submission_path = os.path.join(FINAL_RUN_DIR, "submission.csv")
    submission[["SMILES", "Molecule Name", "pEC50"]].to_csv(submission_path, index=False)
    print(f"\nSubmission written to {submission_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
