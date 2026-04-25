"""Benchmark statistics for *practical* method comparison.

This script consumes one or more per-split metrics CSVs (from
`scripts/score_preds_all.py`) and produces:

  - an overall repeated-measures ANOVA (method effect) per metric
  - pairwise *paired* t-tests between methods with multiple-comparison control
  - effect sizes (Cohen's dz for paired differences)

The primary intent is to support the *distributional* benchmarking guidance in:
  "Practically significant method comparison protocols for machine learning in
   small molecule drug discovery" (JCIM, 2025)

Dependency note
---------------
This script uses `statsmodels` if available. If it is not installed, the script
will still run pairwise paired tests (SciPy), but will skip RM-ANOVA.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _holm_adjust(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (step-down)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        p = float(pvals[idx])
        adj = (m - rank) * p
        adj = min(1.0, max(prev, adj))
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def _paired_cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = float(np.std(d, ddof=1))
    return float("nan") if sd == 0 else float(np.mean(d) / sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more per-split CSVs from scripts/score_preds_all.py",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="rae",
        help="Metric column to analyze (default: rae).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write stats tables.",
    )
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.results_csv]
    df = pd.concat(dfs, ignore_index=True)

    required_cols = {"method", "split_idx", args.metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in results: {sorted(missing)}")

    metric = args.metric
    df = df[["method", "split_idx", metric]].copy()
    df["split_idx"] = df["split_idx"].astype(int)
    df[metric] = df[metric].astype(float)

    # Ensure we only compare methods on *common* split indices.
    methods = sorted(df["method"].unique().tolist())
    split_sets = {
        m: set(df.loc[df["method"] == m, "split_idx"].astype(int).tolist()) for m in methods
    }
    common_splits = set.intersection(*(split_sets[m] for m in methods)) if methods else set()
    if not common_splits:
        raise SystemExit("No common split_idx across methods; cannot do paired comparisons")
    df = df[df["split_idx"].isin(sorted(common_splits))].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pivot to [n_splits x n_methods] for paired stats.
    pivot = df.pivot_table(index="split_idx", columns="method", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2:
        raise SystemExit("Need at least 2 splits with complete method coverage")

    # RM-ANOVA (if statsmodels is available)
    anova_path = args.out_dir / f"anova_rm_{metric}.csv"
    try:
        from statsmodels.stats.anova import AnovaRM

        long_df = pivot.reset_index().melt(id_vars=["split_idx"], var_name="method", value_name=metric)
        aov = AnovaRM(long_df, depvar=metric, subject="split_idx", within=["method"]).fit()
        aov_table = aov.anova_table.reset_index().rename(columns={"index": "term"})
        aov_table.to_csv(anova_path, index=False)
        print(f"Wrote RM-ANOVA table to {anova_path}")
    except ModuleNotFoundError:
        print(
            "statsmodels not installed; skipping RM-ANOVA. "
            "Install with: uv pip install statsmodels"
        )

    # Pairwise paired t-tests + Holm correction.
    pairs = list(itertools.combinations(methods, 2))
    rows: list[dict[str, object]] = []
    pvals: list[float] = []
    for a, b in pairs:
        x = pivot[a].to_numpy(dtype=float)
        y = pivot[b].to_numpy(dtype=float)
        t = stats.ttest_rel(x, y, nan_policy="omit")
        p = float(t.pvalue) if t is not None else float("nan")
        pvals.append(p)
        rows.append(
            {
                "metric": metric,
                "method_a": a,
                "method_b": b,
                "n": int(len(x)),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "mean_diff_a_minus_b": float(np.mean(x - y)),
                "t": float(t.statistic) if t is not None else float("nan"),
                "p": p,
                "cohens_dz": _paired_cohens_dz(x, y),
            }
        )

    p_adj = _holm_adjust(pvals)
    for r, adj in zip(rows, p_adj):
        r["p_holm"] = float(adj)

    pairwise_df = pd.DataFrame(rows).sort_values("p_holm", ascending=True)
    pairwise_path = args.out_dir / f"pairwise_paired_t_{metric}.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Wrote pairwise stats to {pairwise_path}")

    # Method summary.
    summary = (
        df.groupby("method")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_splits"})
        .sort_values("mean", ascending=True if metric in {"rae", "mae"} else False)
    )
    summary_path = args.out_dir / f"summary_{metric}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
