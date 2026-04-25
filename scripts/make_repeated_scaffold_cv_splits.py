"""Generate repeated scaffold-balanced CV splits for Chemprop.

This implements a *repeated* K-fold cross-validation protocol using
Bemis–Murcko scaffolds as the grouping key.

Why this exists
--------------
Chemprop can do K-fold CV ("-k") and repeated splits ("--num-replicates"),
but for *method comparison* workflows it's often convenient to:
  1) generate a fixed set of splits once,
  2) run many methods against the *same* splits, and
  3) aggregate + statistically compare the per-split results.

Output format
-------------
Chemprop expects a JSON file that is a list of objects like:
  [{"train": [...], "val": [...], "test": [...]}]

Indices are 0-based row indices into the input CSV.

Notes
-----
- "test" is left empty; we treat the held-out fold as "val".
- We only write the fields Chemprop requires (train/val/test) to avoid
  compatibility issues with Chemprop's splits-file parser.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class SplitStats:
    n_total: int
    n_scaffolds: int
    k_folds: int
    n_repeats: int
    n_splits: int
    min_fold_size: int
    max_fold_size: int


def _murcko_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES with RDKit: {smiles!r}")
    # Returns an empty string for molecules without a Murcko scaffold (rare).
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _assign_groups_to_folds(
    groups: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Greedy bin packing to balance fold sizes."""
    groups = sorted(groups, key=len, reverse=True)
    fold_groups: list[list[list[int]]] = [[] for _ in range(k)]
    fold_sizes = [0] * k

    i = 0
    while i < len(groups):
        j = i
        while j < len(groups) and len(groups[j]) == len(groups[i]):
            j += 1
        block = groups[i:j]
        rng.shuffle(block)
        for g in block:
            idx = int(np.argmin(fold_sizes))
            fold_groups[idx].append(g)
            fold_sizes[idx] += len(g)
        i = j

    folds: list[list[int]] = []
    for gs in fold_groups:
        flat: list[int] = []
        for g in gs:
            flat.extend(g)
        folds.append(flat)
    return folds


def make_repeated_scaffold_cv_splits(
    df: pd.DataFrame,
    smiles_col: str,
    k_folds: int,
    n_repeats: int,
    data_seed: int,
) -> tuple[list[dict[str, list[int]]], SplitStats]:
    if k_folds < 2:
        raise ValueError("k-folds must be >= 2")
    if n_repeats < 1:
        raise ValueError("repeats must be >= 1")
    if smiles_col not in df.columns:
        raise ValueError(f"CSV missing smiles-col {smiles_col!r}")

    smiles_list = df[smiles_col].astype(str).tolist()
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        scaf = _murcko_scaffold_smiles(s)
        scaffolds.setdefault(scaf, []).append(i)

    groups = list(scaffolds.values())
    n_total = len(df)
    n_scaffolds = len(groups)

    if n_scaffolds < k_folds:
        raise ValueError(
            f"Not enough unique scaffolds ({n_scaffolds}) for k_folds={k_folds}. "
            "Reduce --k-folds."
        )

    all_indices = np.arange(n_total, dtype=int)
    splits: list[dict[str, list[int]]] = []
    fold_sizes_seen: list[int] = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(int(data_seed) + rep)
        folds = _assign_groups_to_folds(groups=groups, k=k_folds, rng=rng)

        # Sanity checks.
        flat = sorted([i for f in folds for i in f])
        if flat != list(all_indices):
            raise RuntimeError("Internal error: folds do not cover the dataset exactly once")

        for fold_idx in range(k_folds):
            val_set = set(folds[fold_idx])
            train = [int(i) for i in all_indices if int(i) not in val_set]
            val = sorted(int(i) for i in val_set)
            if not train or not val:
                raise RuntimeError(
                    f"Internal error: empty fold produced (rep={rep}, fold={fold_idx})."
                )
            fold_sizes_seen.append(len(val))
            splits.append({"train": train, "val": val, "test": []})

    stats = SplitStats(
        n_total=n_total,
        n_scaffolds=n_scaffolds,
        k_folds=k_folds,
        n_repeats=n_repeats,
        n_splits=len(splits),
        min_fold_size=int(min(fold_sizes_seen)) if fold_sizes_seen else 0,
        max_fold_size=int(max(fold_sizes_seen)) if fold_sizes_seen else 0,
    )
    return splits, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--smiles-col", type=str, default="SMILES")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    splits, stats = make_repeated_scaffold_cv_splits(
        df=df,
        smiles_col=args.smiles_col,
        k_folds=args.k_folds,
        n_repeats=args.repeats,
        data_seed=args.data_seed,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(splits))

    print(
        "Wrote splits to",
        args.out_path,
        f"n_total={stats.n_total}",
        f"n_scaffolds={stats.n_scaffolds}",
        f"k_folds={stats.k_folds}",
        f"repeats={stats.n_repeats}",
        f"n_splits={stats.n_splits}",
        f"fold_size_range=[{stats.min_fold_size}, {stats.max_fold_size}]",
        sep=" ",
    )


if __name__ == "__main__":
    main()
