"""
Assemble ens_cm_lr3e-04_3seed_sur_w0.3.csv from its two components:
  1. Average chemeleon_mt_lr3e-04_s{0,1,2}.csv  →  chemeleon_mt_lr3e-04_3seed.csv
  2. Blend (w=0.3 CheMeleon + w=0.7 Suiren)     →  ens_cm_lr3e-04_3seed_sur_w0.3.csv

Run train_chemeleon.py for seeds 0-2 first, then call this script.

Usage:
    .venv/bin/python3 scripts/build_ensemble.py [--w-cm 0.3] [--evaluate]
"""

import argparse
import os

import numpy as np
import pandas as pd

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV      = os.path.join(PROJECT_ROOT, "data/pxr-challenge_TEST_BLINDED.csv")
UNBLINDED_CSV = os.path.join(PROJECT_ROOT, "data/phase1_unblinded.csv")
SUIREN_CSV    = os.path.join(PROJECT_ROOT, "submissions/iw2_3seed_ep17-23.csv")
SUBMISSIONS   = os.path.join(PROJECT_ROOT, "submissions")

PRED_MIN, PRED_MAX = 1.0, 8.5


def load_seed(seed):
    path = os.path.join(SUBMISSIONS, f"chemeleon_mt_lr3e-04_s{seed}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run: "
            f".venv/bin/python3 scripts/train_chemeleon.py --seed {seed} --max-lr 3e-4"
        )
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-cm", type=float, default=0.3,
                        help="CheMeleon weight in final blend (default: 0.3)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score against phase1_unblinded.csv if available")
    args = parser.parse_args()

    os.makedirs(SUBMISSIONS, exist_ok=True)

    # --- Step 1: average 3 CheMeleon seeds ---
    seeds = [load_seed(s) for s in [0, 1, 2]]
    mol_names = seeds[0]["Molecule Name"].values
    smiles    = seeds[0]["SMILES"].values
    cm_avg = np.mean([df["pEC50"].values for df in seeds], axis=0)

    cm3_path = os.path.join(SUBMISSIONS, "chemeleon_mt_lr3e-04_3seed.csv")
    pd.DataFrame({"SMILES": smiles, "Molecule Name": mol_names, "pEC50": cm_avg}).to_csv(
        cm3_path, index=False
    )
    print(f"Saved: {cm3_path}")

    # --- Step 2: blend with Suiren ---
    if not os.path.exists(SUIREN_CSV):
        raise FileNotFoundError(f"Suiren component not found: {SUIREN_CSV}")

    sur = pd.read_csv(SUIREN_CSV).set_index("Molecule Name")["pEC50"]
    cm  = pd.DataFrame({"Molecule Name": mol_names, "SMILES": smiles, "pEC50": cm_avg}) \
            .set_index("Molecule Name")["pEC50"]

    w = args.w_cm
    ens = np.clip((w * cm + (1 - w) * sur).reindex(mol_names).values, PRED_MIN, PRED_MAX)

    test_df = pd.read_csv(TEST_CSV)
    sub = test_df[["SMILES", "Molecule Name"]].copy()
    sub["pEC50"] = ens
    out_path = os.path.join(SUBMISSIONS, f"ens_cm_lr3e-04_3seed_sur_w{w:.1f}.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  (w_cm={w}, w_sur={1-w})")

    if args.evaluate and os.path.exists(UNBLINDED_CSV):
        ub = pd.read_csv(UNBLINDED_CSV)[["Molecule Name", "pEC50"]].rename(
            columns={"pEC50": "true"})
        merged = ub.merge(sub[["Molecule Name", "pEC50"]], on="Molecule Name")
        mae = np.abs(merged["true"] - merged["pEC50"]).mean()
        print(f"Unblinded MAE: {mae:.4f}  (n={len(merged)})")


if __name__ == "__main__":
    main()
