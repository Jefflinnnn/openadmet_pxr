#!/usr/bin/env python
"""OOF error analysis for top models — scaffold, MW, logP, activity cliff breakdown."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from openadmet_pxr.evaluation.metrics import score

RUNS_DIR = Path(__file__).parents[1] / "runs"
TRAIN_CSV = Path(__file__).parents[1] / "data" / "activity_train.csv"

MODELS = [
    "C13_d4h600_ens5_analog",
    "C11_weighted5_analog",
    "C10_ensemble5",
]

def load_oof(run_name):
    dfs = []
    for i in range(5):
        f = RUNS_DIR / run_name / f"split_{i}" / "val_preds.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df["abs_err"] = (df["pEC50_pred"] - df["pEC50_true"]).abs()
    df["err"] = df["pEC50_pred"] - df["pEC50_true"]
    return df

def get_props(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"mw": np.nan, "logp": np.nan, "hbd": np.nan, "hba": np.nan,
                         "tpsa": np.nan, "rotbonds": np.nan, "rings": np.nan, "scaffold": ""})
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except:
            scaf = ""
        rows.append({
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "scaffold": scaf,
        })
    return pd.DataFrame(rows)

print("Loading OOF predictions...")
oofs = {name: load_oof(name) for name in MODELS}

# Use C13 as primary
df = oofs["C13_d4h600_ens5_analog"].copy()
print(f"C13 OOF: {len(df)} molecules, MAE={df['abs_err'].mean():.4f}")

print("Computing RDKit properties...")
props = get_props(df["SMILES"].tolist())
df = pd.concat([df.reset_index(drop=True), props], axis=1)

# ── 1. Activity range bins ──────────────────────────────────────────────────
df["activity_bin"] = pd.cut(df["pEC50_true"],
    bins=[0, 4, 5, 6, 7, 10],
    labels=["<4 (inactive)", "4-5", "5-6", "6-7 (active)", ">7 (potent)"])
print("\n=== Error by activity bin ===")
print(df.groupby("activity_bin", observed=True)["abs_err"]
      .agg(["mean","median","count"]).round(3).to_string())

# ── 2. MW bins ──────────────────────────────────────────────────────────────
df["mw_bin"] = pd.cut(df["mw"], bins=[0,300,400,500,600,900],
    labels=["<300","300-400","400-500","500-600",">600"])
print("\n=== Error by MW bin ===")
print(df.groupby("mw_bin", observed=True)["abs_err"]
      .agg(["mean","median","count"]).round(3).to_string())

# ── 3. logP bins ─────────────────────────────────────────────────────────────
df["logp_bin"] = pd.cut(df["logp"], bins=[-5,0,2,4,6,12],
    labels=["<0","0-2","2-4","4-6",">6"])
print("\n=== Error by logP bin ===")
print(df.groupby("logp_bin", observed=True)["abs_err"]
      .agg(["mean","median","count"]).round(3).to_string())

# ── 4. Scaffold frequency ────────────────────────────────────────────────────
scaf_counts = df["scaffold"].value_counts()
df["scaf_freq"] = df["scaffold"].map(scaf_counts)
df["scaf_bin"] = pd.cut(df["scaf_freq"], bins=[0,1,2,5,20,9999],
    labels=["singleton","2","3-5","6-20",">20"])
print("\n=== Error by scaffold frequency ===")
print(df.groupby("scaf_bin", observed=True)["abs_err"]
      .agg(["mean","median","count"]).round(3).to_string())

# ── 5. Worst scaffolds ───────────────────────────────────────────────────────
scaf_err = df.groupby("scaffold")["abs_err"].agg(["mean","count"])
worst = scaf_err[scaf_err["count"] >= 3].sort_values("mean", ascending=False).head(10)
print("\n=== Worst scaffolds (≥3 members, by mean abs error) ===")
print(worst.round(3).to_string())

# ── 6. Activity cliffs: high true, low pred ──────────────────────────────────
actives = df[df["pEC50_true"] >= 6.0].copy()
print(f"\n=== Actives (pEC50≥6): n={len(actives)}, MAE={actives['abs_err'].mean():.4f} ===")
print(f"  Underpredicted (err<-0.5): {(actives['err'] < -0.5).sum()}")
print(f"  Overpredicted  (err> 0.5): {(actives['err'] >  0.5).sum()}")
print(f"  Well predicted (|err|≤0.5): {(actives['abs_err'] <= 0.5).sum()}")

# ── 7. Cross-model comparison on worst molecules ─────────────────────────────
print("\n=== Cross-model comparison on top-30 hardest molecules (C13) ===")
hard_idx = df.nlargest(30, "abs_err").index
hard_smiles = set(df.loc[hard_idx, "SMILES"])
rows = []
for mname, oof in oofs.items():
    sub = oof[oof["SMILES"].isin(hard_smiles)]
    rows.append({"model": mname, "n": len(sub),
                 "MAE_hard": sub["abs_err"].mean(),
                 "MAE_overall": oof["abs_err"].mean()})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ── 8. Save enriched OOF for further analysis ────────────────────────────────
out = Path(__file__).parents[1] / "runs" / "C13_oof_analysis.csv"
df.to_csv(out, index=False)
print(f"\nSaved enriched OOF to {out}")
