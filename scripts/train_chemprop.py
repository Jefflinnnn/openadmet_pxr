#!/usr/bin/env python
"""CLI: train Chemprop GNN with scaffold CV and MLflow tracking."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.evaluation.splits import load_splits, scaffold_cv_splits, analog_mimic_splits, save_splits
from openadmet_pxr.models.chemprop import train_cv
from openadmet_pxr.models.xgb import print_cv_summary

RUNS_DIR = Path(__file__).parents[1] / "runs"


def main():
    parser = argparse.ArgumentParser(description="Train Chemprop GNN with CV")
    parser.add_argument("--data-path", type=str, default="data/activity_train.csv")
    parser.add_argument("--target-columns", nargs="+", default=["pEC50"])
    parser.add_argument("--task-weights", nargs="+", type=float, default=None)
    parser.add_argument("--splits-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ffn-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--featurizers", nargs="+", default=None)
    parser.add_argument("--no-foundation", action="store_true")
    parser.add_argument("--accelerator", type=str, default="mps")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    splits = load_splits(args.splits_file)
    print(f"Loaded {len(splits)} splits")

    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / args.run_name
    foundation = None if args.no_foundation else "CheMeleon"

    cv_summary = train_cv(
        data_path=args.data_path, target_columns=args.target_columns,
        splits=splits, out_dir=out_dir, run_name=args.run_name,
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size, depth=args.depth, hidden_dim=args.hidden_dim,
        dropout=args.dropout, ffn_num_layers=args.ffn_layers,
        task_weights=args.task_weights, molecule_featurizers=args.featurizers,
        from_foundation=foundation, accelerator=args.accelerator,
        extra_tracking_params={"splits_file": Path(args.splits_file).name},
    )
    print_cv_summary(args.run_name, cv_summary)


if __name__ == "__main__":
    main()
