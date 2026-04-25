"""Train a baseline Activity Track regressor using Chemprop + CheMeleon foundation.

This is a thin wrapper around the Chemprop v2 CLI to keep the repo reproducible.

Usage:
  source .venv/bin/activate
  python scripts/train_chemeleon_chemprop.py \
    --train-csv data/activity_train.csv \
    --out-dir runs/chemeleon_baseline
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smiles-col", type=str, default="SMILES")
    ap.add_argument(
        "--target-col",
        type=str,
        default="pEC50",
        help=(
            "Single target column name (backwards-compatible). Ignored if --target-cols is provided."
        ),
    )
    ap.add_argument(
        "--target-cols",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more target column names for multitask training. "
            "These are passed through to chemprop as --target-columns."
        ),
    )
    ap.add_argument(
        "--task-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional per-task loss weights (same length/order as --target-cols). "
            "Passed through to chemprop as --task-weights."
        ),
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Chemprop warmup epochs (must be < total epochs).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--pytorch-seed", type=int, default=0)

    # Experiment knobs (kept minimal, but covers most performance levers)
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Chemprop split strategy, e.g. SCAFFOLD_BALANCED, RANDOM, KMEANS. "
            "If provided, overrides the default random split."
        ),
    )
    ap.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help=(
            "Path to a Chemprop splits JSON file. If set, Chemprop will use these exact "
            "train/val/test indices. Useful for analog-mimic validation."
        ),
    )
    ap.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="If > 1, run K-fold CV with this many folds (Chemprop -k/--num-folds).",
    )
    ap.add_argument(
        "--loss",
        type=str,
        default=None,
        help="Loss function, e.g. mae, rmse, mse. If omitted, Chemprop default is used.",
    )
    ap.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="If > 1, train an ensemble of this size inside Chemprop.",
    )
    ap.add_argument(
        "--molecule-featurizers",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional extra molecule features to concatenate, e.g. rdkit_2d_normalized morgan_count. "
            "(See `chemprop train --help` for allowed values.)"
        ),
    )
    ap.add_argument(
        "--accelerator",
        type=str,
        default="mps",
        help="Passed to Lightning Trainer. For Apple Silicon, use 'mps'.",
    )
    ap.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Passed to Lightning Trainer (e.g. '1').",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience. If omitted, Chemprop default is used.",
    )

    # Allow passing through additional chemprop CLI flags without having to keep
    # expanding this wrapper.
    # Example:
    #   python scripts/train_chemeleon_chemprop.py ... \
    #     --chemprop-args --depth 4 --message-hidden-dim 600 --dropout 0.2
    ap.add_argument(
        "--chemprop-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended verbatim to `chemprop train`.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_epochs = args.warmup_epochs
    if args.epochs <= warmup_epochs:
        # Chemprop requires warmup < epochs.
        warmup_epochs = max(0, args.epochs - 1)

    target_cols = list(args.target_cols) if args.target_cols else [args.target_col]
    if args.task_weights is not None:
        if args.target_cols is None:
            raise SystemExit("--task-weights requires --target-cols (multitask training)")
        if len(args.task_weights) != len(target_cols):
            raise SystemExit(
                f"--task-weights length ({len(args.task_weights)}) must match --target-cols length ({len(target_cols)})"
            )

    # Note: chemprop CLI binary is installed into the venv as `chemprop`.
    cmd = [
        "chemprop",
        "train",
        "--data-path",
        str(args.train_csv),
        "--smiles-columns",
        args.smiles_col,
        "--target-columns",
        *target_cols,
        "--task-type",
        "regression",
        "--save-dir",
        str(out_dir),
        "--from-foundation",
        "CheMeleon",
        "--accelerator",
        str(args.accelerator),
        "--devices",
        str(args.devices),
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--batch-size",
        str(args.batch_size),
        "--data-seed",
        str(args.data_seed),
        "--pytorch-seed",
        str(args.pytorch_seed),
        "--num-workers",
        "0",
    ]

    if args.task_weights is not None:
        cmd += ["--task-weights", *[str(w) for w in args.task_weights]]

    # Split control (priority: splits-file > num-folds > split > default split-sizes)
    if args.splits_file is not None:
        cmd += ["--splits-file", str(args.splits_file)]
        # When using a splits-file, split sizes are implied by indices.
    elif args.num_folds and args.num_folds > 1:
        cmd += ["-k", str(args.num_folds)]
        if args.split is not None:
            cmd += ["--split", str(args.split)]
    else:
        if args.split is not None:
            cmd += ["--split", str(args.split)]
        # Keep a small validation split for sanity checks.
        cmd += ["--split-sizes", "0.8", "0.2", "0.0"]

    if args.loss is not None:
        cmd += ["--loss", str(args.loss)]

    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]

    if args.ensemble_size and args.ensemble_size > 1:
        cmd += ["--ensemble-size", str(args.ensemble_size)]

    if args.molecule_featurizers:
        cmd += ["--molecule-featurizers", *args.molecule_featurizers]

    if args.chemprop_args:
        cmd += list(args.chemprop_args)

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
