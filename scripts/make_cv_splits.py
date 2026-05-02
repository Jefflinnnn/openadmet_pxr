"""
Generate analog-mimic 5-fold CV splits for PXR pEC50 training.

Each fold holds out one cluster of active analogues (pEC50 >= 6, Tanimoto >= 0.4
on ECFP4) plus a balanced sample of inactives. Splits are saved to:
  runs/analog_mimic_5fold.json   -- integer indices into the training CSV
  runs/fold_{i}/pxr_pec50_train.csv
  runs/fold_{i}/pxr_pec50_valid.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

TRAIN_CSV = "data/pxr-challenge_TRAIN.csv"
SPLITS_JSON = "runs/analog_mimic_5fold.json"
N_FOLDS = 5
ACTIVE_THRESHOLD = 6.0
TANIMOTO_THRESHOLD = 0.4
ECFP_RADIUS = 2
ECFP_NBITS = 2048
SEED = 42


def ecfp4(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_NBITS)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(gen.GetFingerprint(mol))
    return fps


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(TRAIN_CSV)
    df = df.reset_index(drop=True)
    n = len(df)

    print(f"Loaded {n} training compounds")

    fps = ecfp4(df["SMILES"].tolist())
    invalid = [i for i, fp in enumerate(fps) if fp is None]
    if invalid:
        print(f"WARNING: {len(invalid)} compounds have invalid SMILES and will always be in train: {invalid}")

    active_idx = df.index[df["pEC50"] >= ACTIVE_THRESHOLD].tolist()
    inactive_idx = df.index[df["pEC50"] < ACTIVE_THRESHOLD].tolist()
    print(f"Actives: {len(active_idx)}  Inactives: {len(inactive_idx)}")

    # Build analogue clusters: seed each cluster from an active, then pull in
    # any other compound (active or inactive) with Tanimoto >= threshold.
    # Each compound is assigned to at most one cluster (first active it's
    # similar to); singletons (no neighbours) form their own size-1 cluster.
    assigned = {}  # compound_idx -> cluster_id
    clusters = []  # list of sets of compound indices

    for act_i in active_idx:
        if act_i in assigned:
            continue
        fp_i = fps[act_i]
        if fp_i is None:
            continue
        cluster = {act_i}
        for j in range(n):
            if j == act_i or j in assigned:
                continue
            fp_j = fps[j]
            if fp_j is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
            if sim >= TANIMOTO_THRESHOLD:
                cluster.add(j)
        cid = len(clusters)
        clusters.append(cluster)
        for idx in cluster:
            assigned[idx] = cid

    print(f"Analogue clusters: {len(clusters)}  (covering {len(assigned)} compounds)")
    cluster_sizes = [len(c) for c in clusters]
    print(f"Cluster size — min: {min(cluster_sizes)}  max: {max(cluster_sizes)}  mean: {np.mean(cluster_sizes):.1f}")

    # Assign clusters to folds (round-robin by descending size so large
    # clusters are spread evenly).
    order = np.argsort(cluster_sizes)[::-1]
    fold_cluster_ids = [[] for _ in range(N_FOLDS)]
    for rank, cid in enumerate(order):
        fold_cluster_ids[rank % N_FOLDS].append(cid)

    # Build val index sets from cluster assignments.
    val_sets = []
    for f in range(N_FOLDS):
        val_idx = set()
        for cid in fold_cluster_ids[f]:
            val_idx.update(clusters[cid])
        val_sets.append(val_idx)

    # Remaining (unassigned) inactives are distributed across folds to
    # keep fold sizes roughly balanced.
    unassigned_inactive = [i for i in inactive_idx if i not in assigned]
    rng.shuffle(unassigned_inactive)
    chunks = np.array_split(unassigned_inactive, N_FOLDS)
    for f in range(N_FOLDS):
        val_sets[f].update(chunks[f].tolist())

    # Serialize splits.
    splits = []
    for f in range(N_FOLDS):
        val = sorted(val_sets[f])
        train = sorted(set(range(n)) - val_sets[f])
        n_active_val = int((df.iloc[val]["pEC50"] >= ACTIVE_THRESHOLD).sum())
        print(f"Fold {f}: train={len(train)}  val={len(val)}  actives_in_val={n_active_val}")
        splits.append({"train_indices": train, "val_indices": val})

    os.makedirs("runs", exist_ok=True)
    with open(SPLITS_JSON, "w") as fh:
        json.dump(splits, fh)
    print(f"\nSaved splits to {SPLITS_JSON}")

    # Write per-fold CSVs in Suiren's expected format.
    # PyG InMemoryDataset expects files under data/{name}/raw/{name}_{split}.csv
    for f, split in enumerate(splits):
        fold_data_dir = f"runs/fold_{f}/data/pxr_pec50/raw"
        os.makedirs(fold_data_dir, exist_ok=True)

        train_df = df.iloc[split["train_indices"]][["SMILES", "pEC50"]].rename(columns={"pEC50": "value"})
        val_df = df.iloc[split["val_indices"]][["SMILES", "pEC50"]].rename(columns={"pEC50": "value"})

        train_df.to_csv(f"{fold_data_dir}/pxr_pec50_train.csv", index=False)
        val_df.to_csv(f"{fold_data_dir}/pxr_pec50_valid.csv", index=False)
        print(f"Fold {f}: wrote train ({len(train_df)}) and valid ({len(val_df)}) CSVs")

    # Also write full-data CSV for final model training.
    full_data_dir = "runs/final/data/pxr_pec50/raw"
    os.makedirs(full_data_dir, exist_ok=True)
    full_df = df[["SMILES", "pEC50"]].rename(columns={"pEC50": "value"})
    full_df.to_csv(f"{full_data_dir}/pxr_pec50_train.csv", index=False)
    print(f"\nFull training data ({len(full_df)} rows) written to {full_data_dir}/pxr_pec50_train.csv")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
