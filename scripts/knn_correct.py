#!/usr/bin/env python
"""Post-hoc k-NN similarity correction on top of existing model predictions.

For each test molecule, blends the model prediction toward the pEC50 of its
k nearest training neighbors, weighted by Tanimoto similarity. This corrects
the systematic underprediction of active-like compounds without retraining.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.submission.validate import make_submission

SUBMISSIONS_DIR = Path(__file__).parents[1] / "submissions"


def morgan_fps(smiles: list[str], radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def knn_blend(
    test_smiles: list[str],
    model_preds: np.ndarray,
    train_smiles: list[str],
    train_targets: np.ndarray,
    k: int = 5,
    alpha: float = 0.3,
    sim_threshold: float = 0.4,
) -> np.ndarray:
    """
    Blend model predictions with k-NN estimate.

    final = (1 - w) * model_pred + w * knn_pred
    where w = alpha * mean_sim (scales blend weight by how similar neighbors are).
    Only blends when at least one neighbor exceeds sim_threshold.
    """
    from rdkit.DataStructs import BulkTanimotoSimilarity

    print("Computing Morgan fingerprints...")
    test_fps  = morgan_fps(test_smiles)
    train_fps = morgan_fps(train_smiles)
    valid_train = [(fp, t) for fp, t in zip(train_fps, train_targets) if fp is not None]
    tr_fps, tr_targets = zip(*valid_train)
    tr_targets = np.array(tr_targets)

    corrected = model_preds.copy()
    n_blended = 0

    for i, (tfp, base_pred) in enumerate(zip(test_fps, model_preds)):
        if tfp is None:
            continue
        sims = np.array(BulkTanimotoSimilarity(tfp, list(tr_fps)))
        top_k_idx = np.argsort(sims)[-k:][::-1]
        top_k_sims = sims[top_k_idx]
        top_k_targets = tr_targets[top_k_idx]

        if top_k_sims[0] < sim_threshold:
            continue

        # Similarity-weighted k-NN prediction
        weights = top_k_sims / top_k_sims.sum()
        knn_pred = float(np.dot(weights, top_k_targets))

        # Blend weight scales with mean similarity of top-k neighbors
        w = alpha * float(top_k_sims.mean())
        corrected[i] = (1 - w) * base_pred + w * knn_pred
        n_blended += 1

    print(f"Blended {n_blended}/{len(test_smiles)} test molecules (sim>{sim_threshold})")
    print(f"Mean correction: {(corrected - model_preds).mean():+.4f}")
    print(f"Max correction:  {(corrected - model_preds).max():+.4f}")
    return corrected


def main():
    parser = argparse.ArgumentParser(description="k-NN post-hoc correction on model predictions")
    parser.add_argument("--base-submission", required=True,
                        help="Existing submission CSV to correct (must have SMILES, Molecule Name, pEC50)")
    parser.add_argument("--train-csv", default="data/activity_train.csv")
    parser.add_argument("--test-csv", default="data/activity_test_blinded.csv")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Blend strength (0=model only, 1=kNN only)")
    parser.add_argument("--sim-threshold", type=float, default=0.4,
                        help="Min similarity to nearest neighbor to apply correction")
    parser.add_argument("--submission-name", required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base_submission)
    train = pd.read_csv(args.train_csv)

    model_preds = base["pEC50"].values
    test_smiles = base["SMILES"].tolist()
    train_smiles = train["SMILES"].tolist()
    train_targets = train["pEC50"].values

    print(f"Base predictions: mean={model_preds.mean():.3f}, "
          f"range=[{model_preds.min():.3f}, {model_preds.max():.3f}]")

    corrected = knn_blend(
        test_smiles=test_smiles,
        model_preds=model_preds,
        train_smiles=train_smiles,
        train_targets=train_targets,
        k=args.k,
        alpha=args.alpha,
        sim_threshold=args.sim_threshold,
    )

    print(f"Corrected predictions: mean={corrected.mean():.3f}, "
          f"range=[{corrected.min():.3f}, {corrected.max():.3f}]")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"submission_{args.submission_name}.csv"
    make_submission(args.test_csv, corrected, sub_path)
    print(f"\nSaved: {sub_path}")


if __name__ == "__main__":
    main()
