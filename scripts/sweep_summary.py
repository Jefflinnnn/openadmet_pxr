#!/usr/bin/env python
"""Summarize CV results from all Chemprop runs that have val_preds."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from openadmet_pxr.evaluation.metrics import score, aggregate_cv_metrics

RUNS_DIR = Path(__file__).parents[1] / "runs"
TRAIN_CSV = Path(__file__).parents[1] / "data" / "activity_train.csv"


def summarize_run(run_dir: Path, df_train: pd.DataFrame) -> dict | None:
    split_metrics = []
    for i in range(20):
        preds_file = run_dir / f"split_{i}" / "val_preds.csv"
        splits_file = run_dir / f"split_{i}" / "splits.json"
        if not preds_file.exists() or not splits_file.exists():
            break
        preds = pd.read_csv(preds_file)
        y_true = preds["pEC50_true"].values
        y_pred = preds["pEC50_pred"].values
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        split_data = json.load(open(splits_file))
        train_idx = [t for t in split_data[0]["train"] if t < len(df_train)]
        y_train_mean = df_train["pEC50"].iloc[train_idx].mean()
        m = score(y_true[valid], y_pred[valid], y_train_mean=y_train_mean)
        split_metrics.append(m)

    if not split_metrics:
        return None

    cv = aggregate_cv_metrics(split_metrics)
    return {
        "run": run_dir.name,
        "splits": len(split_metrics),
        "MAE": cv["mae"]["mean"],
        "MAE_std": cv["mae"]["std"],
        "RAE": cv["rae"]["mean"],
        "RAE_std": cv["rae"]["std"],
        "R2": cv["r2"]["mean"],
        "Spearman": cv["spearman"]["mean"],
    }


def main():
    df_train = pd.read_csv(TRAIN_CSV)
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "split_0" / "val_preds.csv").exists():
            continue
        result = summarize_run(run_dir, df_train)
        if result:
            rows.append(result)

    if not rows:
        print("No completed runs found.")
        return

    res = pd.DataFrame(rows).sort_values("MAE")
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(res[["run", "splits", "MAE", "MAE_std", "RAE", "RAE_std", "R2", "Spearman"]].to_string(index=False))
    best = res.iloc[0]
    print(f"\nBest (by MAE): {best['run']}  MAE={best['MAE']:.4f}  RAE={best['RAE']:.4f}")


if __name__ == "__main__":
    main()
