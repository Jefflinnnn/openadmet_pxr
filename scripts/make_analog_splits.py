"""Generate a Chemprop splits.json that mimics the challenge's analog-expansion test design.

Motivation
----------
The blinded test set was constructed by:
  1) selecting potent / selective hits
  2) doing similarity search (ECFP4 Tanimoto > 0.4)
  3) assaying the resulting analog expansion set

Random splits can overestimate performance. This script creates a validation set
made of *analog neighborhoods* around potent seeds, using the same similarity notion.

Output format
-------------
Chemprop expects a JSON file that is a list of objects like:
  [{"train": [...], "val": [...], "test": [...]}]

Indices are 0-based row indices into the input CSV.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# RDKit emits a deprecation warning for Morgan fingerprints on every call.
# Silence these so split generation doesn't spam the console.
RDLogger.DisableLog("rdApp.warning")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass(frozen=True)
class SplitStats:
    n_total: int
    n_seeds: int
    n_components: int
    n_active_components: int
    val_size: int


def _ecfp4(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def _build_components(
    smiles_list: list[str],
    seed_indices: list[int],
    tanimoto_threshold: float,
) -> tuple[list, UnionFind]:
    fps = []
    for s in smiles_list:
        fp = _ecfp4(s)
        if fp is None:
            # We rely on exact row indices. If any SMILES fail RDKit parsing,
            # we can't safely create a splits file.
            raise ValueError(f"Failed to parse SMILES with RDKit: {s!r}")
        fps.append(fp)

    uf = UnionFind(len(smiles_list))
    for si in seed_indices:
        sims = DataStructs.BulkTanimotoSimilarity(fps[si], fps)
        for j, sim in enumerate(sims):
            if sim >= tanimoto_threshold:
                uf.union(si, j)
    return fps, uf


def _components_from_union_find(uf: UnionFind) -> dict[int, list[int]]:
    comps: dict[int, list[int]] = {}
    for i in range(len(uf.parent)):
        r = uf.find(i)
        comps.setdefault(r, []).append(i)
    return comps


def _assign_components_to_folds(
    component_lists: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Greedy bin packing to balance fold sizes."""
    # largest-first
    component_lists = sorted(component_lists, key=len, reverse=True)
    fold_comps: list[list[list[int]]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    # shuffle within equal sizes for reproducibility
    i = 0
    while i < len(component_lists):
        j = i
        while j < len(component_lists) and len(component_lists[j]) == len(component_lists[i]):
            j += 1
        block = component_lists[i:j]
        rng.shuffle(block)
        for comp in block:
            idx = int(np.argmin(fold_sizes))
            fold_comps[idx].append(comp)
            fold_sizes[idx] += len(comp)
        i = j

    folds: list[list[int]] = []
    for comps in fold_comps:
        flat: list[int] = []
        for c in comps:
            flat.extend(c)
        folds.append(flat)
    return folds


def make_splits(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    seed_threshold: float,
    tanimoto_threshold: float,
    val_frac: float,
    n_splits: int,
    data_seed: int,
) -> tuple[list[dict[str, list[int]]], SplitStats]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val-frac must be in (0, 1)")
    if n_splits < 1:
        raise ValueError("n-splits must be >= 1")

    smiles_list = df[smiles_col].astype(str).tolist()
    y = df[target_col].astype(float).to_numpy()
    seed_indices = [int(i) for i in np.where(y >= seed_threshold)[0].tolist()]
    if not seed_indices:
        raise ValueError(
            f"No seeds found with {target_col} >= {seed_threshold}. "
            "Lower --seed-threshold or use a top-k seed strategy (not implemented yet)."
        )

    rng = np.random.default_rng(data_seed)

    _, uf = _build_components(
        smiles_list=smiles_list,
        seed_indices=seed_indices,
        tanimoto_threshold=tanimoto_threshold,
    )
    comps = _components_from_union_find(uf)

    # Only hold out components that contain at least one seed ("active neighborhoods")
    seed_roots = {uf.find(i) for i in seed_indices}
    active_components = [comps[r] for r in seed_roots]

    n_total = len(df)
    target_val = int(round(val_frac * n_total))

    splits: list[dict[str, list[int]]] = []

    if n_splits == 1:
        # Choose a subset of active components to reach ~val_frac
        active_components = sorted(active_components, key=len, reverse=True)
        rng.shuffle(active_components)
        val: list[int] = []
        for comp in active_components:
            if len(val) >= target_val:
                break
            val.extend(comp)
        val_set = set(val)
        train = [i for i in range(n_total) if i not in val_set]

        splits.append({"train": train, "val": sorted(val_set), "test": []})

        stats = SplitStats(
            n_total=n_total,
            n_seeds=len(seed_indices),
            n_components=len(comps),
            n_active_components=len(active_components),
            val_size=len(val_set),
        )
        return splits, stats

    # K-fold: partition active components into K folds; each fold is val once
    folds = _assign_components_to_folds(active_components, k=n_splits, rng=rng)
    for fold_idx in range(n_splits):
        val_set = set(folds[fold_idx])
        train = [i for i in range(n_total) if i not in val_set]
        splits.append({"train": train, "val": sorted(val_set), "test": []})

    stats = SplitStats(
        n_total=n_total,
        n_seeds=len(seed_indices),
        n_components=len(comps),
        n_active_components=len(active_components),
        val_size=len(splits[0]["val"]),
    )
    return splits, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--smiles-col", type=str, default="SMILES")
    ap.add_argument("--target-col", type=str, default="pEC50")
    ap.add_argument(
        "--seed-threshold",
        type=float,
        default=6.0,
        help="Potency threshold for selecting seed actives (default: 6.0 ~ 1uM).",
    )
    ap.add_argument(
        "--tanimoto",
        type=float,
        default=0.4,
        help="ECFP4 Tanimoto threshold for defining analog neighborhoods (default: 0.4).",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data to place into validation (only for n-splits=1).",
    )
    ap.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Number of folds to create (default: 1). If >1, creates K fold-like splits.",
    )
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)

    splits, stats = make_splits(
        df=df,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        seed_threshold=args.seed_threshold,
        tanimoto_threshold=args.tanimoto,
        val_frac=args.val_frac,
        n_splits=args.n_splits,
        data_seed=args.data_seed,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(splits))

    print(
        "Wrote splits to",
        args.out_path,
        f"n_total={stats.n_total}",
        f"n_seeds={stats.n_seeds}",
        f"n_components={stats.n_components}",
        f"n_active_components={stats.n_active_components}",
        f"val_size~{stats.val_size}",
        sep=" ",
    )


if __name__ == "__main__":
    main()
