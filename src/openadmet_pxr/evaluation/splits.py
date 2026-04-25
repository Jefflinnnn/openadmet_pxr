"""CV split generation: scaffold-stratified and analog-mimic."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _murcko_scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _morgan_fp(smi: str, radius: int = 2, nbits: int = 1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def scaffold_cv_splits(
    smiles: list[str],
    n_folds: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Bemis-Murcko scaffold-stratified K-fold CV (n_repeats × n_folds splits)."""
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        sc = _murcko_scaffold(smi) or "__invalid__"
        scaffolds[sc].append(i)

    scaffold_groups = list(scaffolds.values())
    splits = []

    for repeat in range(n_repeats):
        rng = random.Random(seed + repeat)
        rng.shuffle(scaffold_groups)
        folds: list[list[int]] = [[] for _ in range(n_folds)]
        fold_sizes = [0] * n_folds
        for group in scaffold_groups:
            smallest = int(np.argmin(fold_sizes))
            folds[smallest].extend(group)
            fold_sizes[smallest] += len(group)
        for fold_idx in range(n_folds):
            val = folds[fold_idx]
            train = [idx for k, f in enumerate(folds) if k != fold_idx for idx in f]
            splits.append({"train": sorted(train), "val": sorted(val)})

    return splits


def analog_mimic_splits(
    smiles: list[str],
    pec50: list[float],
    seed_threshold: float = 6.0,
    tanimoto_threshold: float = 0.4,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Analog-neighborhood splits mirroring the challenge's test design."""
    fps = [_morgan_fp(s) for s in smiles]
    n = len(smiles)
    component_of = list(range(n))

    def find(x):
        while component_of[x] != x:
            component_of[x] = component_of[component_of[x]]
            x = component_of[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            component_of[rx] = ry

    seed_indices = [i for i, v in enumerate(pec50) if v >= seed_threshold]
    for si in seed_indices:
        if fps[si] is None:
            continue
        other = [(j, fps[j]) for j in range(n) if fps[j] is not None and j != si]
        if not other:
            continue
        js, ofps = zip(*other)
        sims = DataStructs.BulkTanimotoSimilarity(fps[si], list(ofps))
        for j, sim in zip(js, sims):
            if sim >= tanimoto_threshold:
                union(si, j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    component_list = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(component_list)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for comp in component_list:
        smallest = int(np.argmin(fold_sizes))
        folds[smallest].extend(comp)
        fold_sizes[smallest] += len(comp)

    splits = []
    for fold_idx in range(n_folds):
        val = folds[fold_idx]
        train = [idx for k, f in enumerate(folds) if k != fold_idx for idx in f]
        splits.append({"train": sorted(train), "val": sorted(val)})

    return splits


def save_splits(splits: list[dict[str, list[int]]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)


def load_splits(path: Path | str) -> list[dict[str, list[int]]]:
    with open(path) as f:
        return json.load(f)
