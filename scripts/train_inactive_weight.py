"""
Train Suiren with inactive_weight=2.0 (tail_weight) to reduce very-inactive overprediction.

Same config as best leaderboard model but with --tail-weight 2.0 --tail-threshold 3.0.
This upweights 695 training compounds with pEC50 < 3.0, forcing the model to reduce
overprediction at the low tail (our biggest error source: 30.9% of total error).

Usage:
    python scripts/train_inactive_weight.py                 # train all 3 seeds
    python scripts/train_inactive_weight.py --seeds 0       # train seed 0 only
    python scripts/train_inactive_weight.py --skip-training # inference + averaging only
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
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TRAIN.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TEST_BLINDED.csv")
NAME = "pxr_pec50"
RUN_BASE = os.path.join(PROJECT_ROOT, "runs/inactive_weight")

SAVE_EPOCHS = list(range(17, 24))  # 17-23 inclusive
SEEDS = [0, 1, 2]

CONFIG = {
    "lr": 2e-4,
    "batch_size": 32,
    "weight_decay": 0.05,
    "drop_path": 0.2,
    "active_weight": 2.0,
    "tail_weight": 2.0,
    "tail_threshold": 3.0,
    "warmup_epochs": 5,
    "epochs": 25,
    "clip_grad": 1.0,
}

PRED_MIN = 1.0
PRED_MAX = 8.5


def seed_run_dir(seed):
    return os.path.join(RUN_BASE, f"seed_{seed}")


def prepare_data(run_dir):
    """Full-data training: all 4139 in train, 50 inactives in val for monitoring."""
    raw_dir = os.path.join(run_dir, "data", NAME, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    train_out = os.path.join(raw_dir, f"{NAME}_train.csv")
    val_out = os.path.join(raw_dir, f"{NAME}_valid.csv")

    if os.path.exists(train_out) and os.path.exists(val_out):
        return

    df = pd.read_csv(TRAIN_CSV)
    data = df[["SMILES", "pEC50"]].rename(columns={"pEC50": "value"})

    train_df = data.copy()

    rng = np.random.default_rng(42)
    inactives = data[data["value"] < 6.0]
    val_idx = rng.choice(len(inactives), size=50, replace=False)
    val_df = inactives.iloc[val_idx]

    train_df[["SMILES", "value"]].to_csv(train_out, index=False)
    val_df[["SMILES", "value"]].to_csv(val_out, index=False)

    print(f"  Data prepared: {len(train_df)} train, {len(val_df)} val (monitoring only)")


def run_training(seed):
    run_dir = seed_run_dir(seed)
    os.makedirs(run_dir, exist_ok=True)

    processed_dir = os.path.join(run_dir, "data", NAME, "processed")
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    prepare_data(run_dir)

    run_name = f"iw2_seed{seed}"
    cmd = [
        sys.executable, SUIREN_MAIN,
        "--checkpoint-pretrain", PRETRAIN_CKPT,
        "--mode", "regression",
        "--loss", "l1",
        "--name", NAME,
        "--main-metric", "MAE",
        "--data-mode", "smiles_defined",
        "--epochs", str(CONFIG["epochs"]),
        "--lr", str(CONFIG["lr"]),
        "--seed", str(seed),
        "--batch-size", str(CONFIG["batch_size"]),
        "--opt", "adamw",
        "--weight-decay", str(CONFIG["weight_decay"]),
        "--clip-grad", str(CONFIG["clip_grad"]),
        "--sched", "cosine",
        "--warmup-epochs", str(CONFIG["warmup_epochs"]),
        "--warmup-lr", "1e-6",
        "--min-lr", "1e-7",
        "--early-stopping", "0",
        "--drop-path", str(CONFIG["drop_path"]),
        "--active-weight", str(CONFIG["active_weight"]),
        "--tail-weight", str(CONFIG["tail_weight"]),
        "--tail-threshold", str(CONFIG["tail_threshold"]),
        "--workers", "0",
        "--print-freq", "100",
        "--wandb-run-name", run_name,
        "--run-type", "final",
        "--save-epochs", ",".join(str(e) for e in SAVE_EPOCHS),
    ]

    print(f"\n{'='*60}")
    print(f"  TRAINING seed={seed}")
    print(f"{'='*60}")
    print(f"  Config: lr={CONFIG['lr']}, bs={CONFIG['batch_size']}, "
          f"aw={CONFIG['active_weight']}, tw={CONFIG['tail_weight']}, "
          f"tail_thresh={CONFIG['tail_threshold']}")
    print(f"  Epochs: {CONFIG['epochs']}, saving at: {SAVE_EPOCHS}")

    log_path = os.path.join(run_dir, "training.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, cwd=run_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log_f.write(result.stdout)

    if result.returncode != 0:
        print(f"  ERROR: Training failed. See {log_path}")
        print(result.stdout[-2000:])
        return False

    print(f"  Training complete. Log: {log_path}")
    return True


def find_epoch_checkpoints(seed):
    run_dir = seed_run_dir(seed)
    ckpts = {}
    for epoch in SAVE_EPOCHS:
        pattern = os.path.join(run_dir, "checkpoints", NAME, "*",
                               f"{NAME}_regression_epoch{epoch}.pt")
        matches = glob.glob(pattern)
        if matches:
            ckpts[epoch] = sorted(matches)[-1]
    if not ckpts:
        pattern = os.path.join(run_dir, "checkpoints", NAME, "*",
                               f"{NAME}_regression.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            ckpts["best"] = matches[-1]
    return ckpts


def run_inference_single(checkpoint_path, output_path):
    cmd = [
        sys.executable, SUIREN_INFER,
        "--task", NAME,
        "--checkpoint", checkpoint_path,
        "--input", TEST_CSV,
        "--output", output_path,
        "--batch-size", "64",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Inference failed: {result.stderr[:500]}")
        return False
    return True


def run_all_inference(seeds=None):
    if seeds is None:
        seeds = SEEDS
    pred_dir = os.path.join(RUN_BASE, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_preds = []

    for seed in seeds:
        ckpts = find_epoch_checkpoints(seed)
        if not ckpts:
            print(f"  WARNING: No checkpoints found for seed {seed}")
            continue

        for epoch_key, ckpt_path in sorted(ckpts.items()):
            out_path = os.path.join(pred_dir, f"pred_seed{seed}_ep{epoch_key}.csv")

            if os.path.exists(out_path):
                print(f"  Using cached: seed={seed}, epoch={epoch_key}")
            else:
                print(f"  Inferring: seed={seed}, epoch={epoch_key}")
                ok = run_inference_single(ckpt_path, out_path)
                if not ok:
                    continue

            df = pd.read_csv(out_path)
            if "pEC50" in df.columns:
                preds = df["pEC50"].values
            elif "prediction" in df.columns:
                preds = df["prediction"].values
            elif "value" in df.columns:
                preds = df["value"].values
            else:
                print(f"  WARNING: No prediction column in {out_path}")
                continue

            all_preds.append(preds)
            print(f"    mean={np.nanmean(preds):.3f}, "
                  f"active_frac={(preds >= 6.0).sum()/len(preds):.1%}")

    return all_preds


def create_submission(all_preds):
    if not all_preds:
        print("ERROR: No predictions to average")
        return None

    print(f"\n{'='*60}")
    print(f"  AVERAGING {len(all_preds)} prediction sets")
    print(f"{'='*60}")

    pred_array = np.array(all_preds)
    avg_pred = np.nanmean(pred_array, axis=0)

    n_nan = np.isnan(avg_pred).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN predictions — filling with train mean")
        avg_pred = np.where(np.isnan(avg_pred), 4.321, avg_pred)

    avg_pred = np.clip(avg_pred, PRED_MIN, PRED_MAX)

    print(f"  Averaged predictions:")
    print(f"    mean={avg_pred.mean():.3f}, std={avg_pred.std():.3f}")
    print(f"    active_frac={(avg_pred >= 6.0).sum() / len(avg_pred):.1%}")
    print(f"    range=[{avg_pred.min():.3f}, {avg_pred.max():.3f}]")

    # Compare to best
    best_sub_path = os.path.join(PROJECT_ROOT, "submissions/fulldata_3seed_ep17-23.csv")
    if os.path.exists(best_sub_path):
        best_sub = pd.read_csv(best_sub_path)
        best_preds = best_sub["pEC50"].values
        diff = avg_pred - best_preds
        print(f"\n  vs fulldata baseline (MAE 0.474):")
        print(f"    mean diff={diff.mean():+.4f}, abs diff={np.abs(diff).mean():.4f}")
        print(f"    correlation={np.corrcoef(avg_pred, best_preds)[0,1]:.4f}")

    test_df = pd.read_csv(TEST_CSV)
    submission = test_df[["SMILES", "Molecule Name"]].copy()
    submission["pEC50"] = avg_pred

    submissions_dir = os.path.join(PROJECT_ROOT, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    final_path = os.path.join(submissions_dir, "iw2_3seed_ep17-23.csv")
    submission.to_csv(final_path, index=False)
    print(f"\n  Submission: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Suiren with inactive_weight=2.0")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds
    os.makedirs(RUN_BASE, exist_ok=True)

    if not args.skip_training:
        for seed in seeds:
            success = run_training(seed)
            if not success:
                print(f"  Seed {seed} failed. Continuing.")

    all_preds = run_all_inference(seeds)
    create_submission(all_preds)


if __name__ == "__main__":
    main()
