#!/usr/bin/env python
"""CLI: train XGBoost with scaffold CV and MLflow tracking."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.data.load import load_train, load_chembl_pxr, load_chembl_cyp3a4
from openadmet_pxr.evaluation.splits import scaffold_cv_splits, analog_mimic_splits, save_splits, load_splits
from openadmet_pxr.models.xgb import train_cv, print_cv_summary, DEFAULT_XGB_PARAMS

RUNS_DIR = Path(__file__).parents[1] / "runs"


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with scaffold CV")
    parser.add_argument("--morgan", action="store_true")
    parser.add_argument("--rdkit2d", action="store_true")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-nbits", type=int, default=2048)
    parser.add_argument("--add-chembl-pxr", action="store_true")
    parser.add_argument("--add-chembl-cyp3a4", action="store_true")
    parser.add_argument("--split-type", choices=["scaffold", "analog"], default="scaffold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--splits-file", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.morgan and not args.rdkit2d:
        parser.error("Specify at least one of --morgan or --rdkit2d")

    train = load_train()
    smiles = list(train["SMILES"])
    targets = list(train["pEC50"])

    if args.add_chembl_pxr:
        df = load_chembl_pxr().rename(columns={"chembl_pxr_pchembl": "pEC50"})
        smiles.extend(df["SMILES"].tolist()); targets.extend(df["pEC50"].tolist())
    if args.add_chembl_cyp3a4:
        df = load_chembl_cyp3a4().rename(columns={"chembl_cyp3a4_pchembl": "pEC50"})
        smiles.extend(df["SMILES"].tolist()); targets.extend(df["pEC50"].tolist())

    print(f"Dataset: {len(smiles)} molecules")

    if args.splits_file and Path(args.splits_file).exists():
        splits = load_splits(args.splits_file)
    elif args.split_type == "scaffold":
        splits = scaffold_cv_splits(smiles, n_folds=args.n_folds, n_repeats=args.n_repeats)
        out_path = RUNS_DIR / f"scaffold_cv{args.n_folds}x{args.n_repeats}.json"
        save_splits(splits, out_path); print(f"Saved splits to {out_path}")
    else:
        splits = analog_mimic_splits(smiles, targets, n_folds=args.n_folds)
        out_path = RUNS_DIR / f"analog_mimic_{args.n_folds}fold.json"
        save_splits(splits, out_path); print(f"Saved splits to {out_path}")
    print(f"Using {len(splits)} CV splits")

    xgb_params = {**DEFAULT_XGB_PARAMS, "n_estimators": args.n_estimators,
                  "max_depth": args.max_depth, "learning_rate": args.learning_rate}
    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / args.run_name

    cv_summary, _ = train_cv(
        smiles=smiles, targets=targets, splits=splits, xgb_params=xgb_params,
        use_morgan=args.morgan, use_rdkit_2d=args.rdkit2d,
        morgan_radius=args.morgan_radius, morgan_n_bits=args.morgan_nbits,
        run_name=args.run_name,
        extra_tracking_params={"split_type": args.split_type,
                                "add_chembl_pxr": args.add_chembl_pxr,
                                "add_chembl_cyp3a4": args.add_chembl_cyp3a4},
        out_dir=out_dir,
    )
    print_cv_summary(args.run_name, cv_summary)


if __name__ == "__main__":
    main()
