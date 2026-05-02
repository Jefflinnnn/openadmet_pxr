"""
Enumerate Matched Molecular Pairs (MMPs) from a training CSV.

An MMP is a pair of molecules that differ by a single structural
transformation (one bond cut, different R-groups on the same core).

Output: JSON list of {idx_a, idx_b, smiles_a, smiles_b, pec50_a, pec50_b, delta_pec50}

Usage:
    python scripts/enumerate_mmps.py [--input data/pxr-challenge_TRAIN.csv] [--output data/pxr_train_mmps.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMMPA
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def fragment_molecule(smi, idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    fragments = []
    try:
        cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        return []

    for core_smi, chains_smi in cuts:
        if core_smi is None or chains_smi is None:
            continue
        fragments.append((core_smi, chains_smi, idx))

    return fragments


def enumerate_mmps(df, smiles_col="SMILES", value_col="pEC50", min_tanimoto=0.5):
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    mols = []
    fps = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
        fps.append(fpgen.GetFingerprint(mol) if mol else None)

    core_to_frags = defaultdict(list)
    for idx, smi in enumerate(df[smiles_col]):
        frags = fragment_molecule(smi, idx)
        for core_smi, chains_smi, mol_idx in frags:
            core_to_frags[core_smi].append((mol_idx, chains_smi))

    pairs = []
    seen = set()

    for core_smi, members in core_to_frags.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_a, chain_a = members[i]
                idx_b, chain_b = members[j]
                if chain_a == chain_b:
                    continue
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if key in seen:
                    continue

                if fps[idx_a] is None or fps[idx_b] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[idx_a], fps[idx_b])
                if sim < min_tanimoto:
                    continue

                seen.add(key)
                pec50_a = float(df[value_col].iloc[idx_a])
                pec50_b = float(df[value_col].iloc[idx_b])
                pairs.append({
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "smiles_a": df[smiles_col].iloc[idx_a],
                    "smiles_b": df[smiles_col].iloc[idx_b],
                    "pec50_a": pec50_a,
                    "pec50_b": pec50_b,
                    "delta_pec50": pec50_a - pec50_b,
                    "tanimoto": round(sim, 4),
                })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pxr-challenge_TRAIN.csv")
    parser.add_argument("--output", default="data/pxr_train_mmps.json")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--value-col", default="pEC50")
    parser.add_argument("--min-tanimoto", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} compounds from {args.input}")

    pairs = enumerate_mmps(df, args.smiles_col, args.value_col, args.min_tanimoto)

    unique_compounds = set()
    for p in pairs:
        unique_compounds.add(p["idx_a"])
        unique_compounds.add(p["idx_b"])

    deltas = [abs(p["delta_pec50"]) for p in pairs]
    active_involved = sum(1 for p in pairs if p["pec50_a"] >= 6 or p["pec50_b"] >= 6)

    print(f"MMP pairs found: {len(pairs)}")
    print(f"Unique compounds in pairs: {len(unique_compounds)}")
    print(f"Pairs involving an active: {active_involved}")
    if deltas:
        print(f"|ΔpEC50| — mean: {np.mean(deltas):.3f}, median: {np.median(deltas):.3f}, "
              f"max: {np.max(deltas):.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
