"""Score predictions across *all* split entries in a Chemprop splits file.

This is a multi-split companion to `scripts/score_preds.py`.

It produces:
  1) a per-split metrics CSV (one row per split entry)
  2) a small JSON summary (mean/std across splits)

The intent is to create a distribution of performance values for downstream
method-comparison statistics (repeated-measures ANOVA / paired tests).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def rae(y_true: np.ndarray, y_pred: np.ndarray, baseline: float) -> float:
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(np.sum(np.abs(y_true - baseline)))
    return float("nan") if den == 0 else num / den


def _metric_summary(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else float("nan"),
        "n": int(np.sum(~np.isnan(arr))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, required=True)
    ap.add_argument(
        "--preds-csv",
        type=Path,
        default=None,
        help=(
            "Single predictions CSV to reuse for every split. "
            "(Only valid if you're intentionally evaluating the *same* predictions across multiple splits.)"
        ),
    )
    ap.add_argument(
        "--preds-csv-template",
        type=str,
        default=None,
        help=(
            "Template for per-split prediction CSVs. Must include '{split_idx}'. Example: "
            "runs/method_x/split_{split_idx}/train_preds.csv"
        ),
    )
    ap.add_argument("--splits-file", type=Path, required=True)
    ap.add_argument("--smiles-col", type=str, default="SMILES")
    ap.add_argument("--name-col", type=str, default="Molecule Name")
    ap.add_argument("--target-col", type=str, default="pEC50")
    ap.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help=(
            "Optional: if provided, adds (repeat_idx, fold_idx) columns assuming split_idx = repeat*k + fold"
        ),
    )
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method name label to store in outputs (e.g., chemeleon_depth3).",
    )
    ap.add_argument(
        "--out-per-split-csv",
        type=Path,
        required=True,
        help="Path to write per-split metrics (CSV).",
    )
    ap.add_argument(
        "--out-summary-json",
        type=Path,
        default=None,
        help="Optional path to write mean/std summary JSON.",
    )
    args = ap.parse_args()

    if (args.preds_csv is None) == (args.preds_csv_template is None):
        raise SystemExit("Provide exactly one of --preds-csv or --preds-csv-template")
    if args.preds_csv_template is not None and "{split_idx}" not in args.preds_csv_template:
        raise SystemExit("--preds-csv-template must include '{split_idx}'")

    df = pd.read_csv(args.data_csv)

    if args.target_col not in df.columns:
        raise SystemExit(f"--data-csv missing target column {args.target_col!r}")

    splits = json.loads(args.splits_file.read_text())
    if not isinstance(splits, list) or not splits:
        raise SystemExit("splits-file must be a JSON list with at least one split")

    y = df[args.target_col].astype(float).to_numpy()

    # Optional: load single preds once.
    preds_single: pd.DataFrame | None = None
    yhat_single: np.ndarray | None = None
    if args.preds_csv is not None:
        preds_single = pd.read_csv(args.preds_csv)

        if len(df) != len(preds_single):
            raise SystemExit(
                f"Row count mismatch: data has {len(df)} rows, preds has {len(preds_single)} rows. "
                "Score script assumes predictions were generated on the same CSV."
            )

        # Basic alignment checks.
        if args.smiles_col in df.columns and args.smiles_col in preds_single.columns:
            if not df[args.smiles_col].astype(str).reset_index(drop=True).equals(
                preds_single[args.smiles_col].astype(str).reset_index(drop=True)
            ):
                raise SystemExit("SMILES column does not align between --data-csv and --preds-csv")
        if args.name_col in df.columns and args.name_col in preds_single.columns:
            if not df[args.name_col].astype(str).reset_index(drop=True).equals(
                preds_single[args.name_col].astype(str).reset_index(drop=True)
            ):
                raise SystemExit(
                    "Molecule Name column does not align between --data-csv and --preds-csv"
                )

        if args.target_col not in preds_single.columns:
            raise SystemExit(f"--preds-csv missing prediction column {args.target_col!r}")

        yhat_single = preds_single[args.target_col].astype(float).to_numpy()

    rows: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        if args.preds_csv_template is not None:
            preds_path = Path(args.preds_csv_template.format(split_idx=split_idx))
            if not preds_path.exists():
                raise SystemExit(f"Missing preds CSV for split {split_idx}: {preds_path}")
            preds = pd.read_csv(preds_path)
            if len(df) != len(preds):
                raise SystemExit(
                    f"Row count mismatch for split {split_idx}: data has {len(df)} rows, preds has {len(preds)} rows."
                )
            if args.target_col not in preds.columns:
                raise SystemExit(
                    f"split {split_idx} preds-csv missing prediction column {args.target_col!r}: {preds_path}"
                )
            yhat = preds[args.target_col].astype(float).to_numpy()
        else:
            assert yhat_single is not None
            yhat = yhat_single

        train_idx = np.asarray(split.get("train", []), dtype=int)
        val_idx = np.asarray(split.get("val", []), dtype=int)

        if train_idx.size == 0 or val_idx.size == 0:
            raise SystemExit(f"split {split_idx} must contain non-empty 'train' and 'val'")

        y_train = y[train_idx]
        y_val = y[val_idx]
        yhat_val = yhat[val_idx]
        yhat_train = yhat[train_idx]

        # Handle multitask CSVs: many rows may have NaN labels for the target task.
        train_mask = (~np.isnan(y_train)) & (~np.isnan(yhat_train))
        val_mask = (~np.isnan(y_val)) & (~np.isnan(yhat_val))

        y_train_l = y_train[train_mask]
        y_val_l = y_val[val_mask]
        yhat_val_l = yhat_val[val_mask]

        baseline = float(np.mean(y_train_l)) if y_train_l.size else float("nan")

        if y_val_l.size >= 2:
            sp = stats.spearmanr(y_val_l, yhat_val_l)
            kt = stats.kendalltau(y_val_l, yhat_val_l)
            sp_rho = float(sp.correlation) if sp is not None else float("nan")
            kt_tau = float(kt.correlation) if kt is not None else float("nan")
        else:
            sp_rho = float("nan")
            kt_tau = float("nan")

        rae_val = (
            rae(y_val_l, yhat_val_l, baseline=baseline) if y_val_l.size else float("nan")
        )

        rows.append(
            {
                "method": args.method,
                "split_idx": int(split_idx),
                "repeat_idx": int(split_idx // args.k_folds) if args.k_folds else None,
                "fold_idx": int(split_idx % args.k_folds) if args.k_folds else None,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "n_train_labeled": int(y_train_l.size),
                "n_val_labeled": int(y_val_l.size),
                "mae": mae(y_val_l, yhat_val_l) if y_val_l.size else float("nan"),
                "rae": rae_val,
                "rae_lift": float("nan") if np.isnan(rae_val) else 1.0 - float(rae_val),
                "r2": r2(y_val_l, yhat_val_l) if y_val_l.size else float("nan"),
                "spearman_rho": sp_rho,
                "kendall_tau": kt_tau,
                "baseline_train_mean": baseline,
            }
        )

    out_df = pd.DataFrame(rows)
    args.out_per_split_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_per_split_csv, index=False)
    print(f"Wrote per-split metrics to {args.out_per_split_csv}")

    summary = {
        "method": args.method,
        "n_splits": int(len(out_df)),
        "metrics": {
            m: _metric_summary(out_df[m].astype(float).to_list())
            for m in ["rae", "mae", "r2", "spearman_rho", "kendall_tau"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_summary_json is not None:
        args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote summary to {args.out_summary_json}")


if __name__ == "__main__":
    main()
