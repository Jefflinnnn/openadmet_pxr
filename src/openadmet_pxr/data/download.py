"""Download all datasets needed for the PXR activity prediction pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parents[3] / "data"
EXTERNAL_DIR = DATA_DIR / "external"

HF_DATASET = "openadmet/pxr-challenge-train-test"

CHEMBL_TARGETS = {
    "cyp3a4": "CHEMBL340",
    "pxr":    "CHEMBL3401",
    "ahr":    "CHEMBL3201",
    "car":    "CHEMBL5503",
}

ADME_URL = (
    "https://raw.githubusercontent.com/molecularinformatics/"
    "Computational-ADME/main/ADME_public_set_3521.csv"
)


def download_pxr(out_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET)
    train = ds["train"].to_pandas()[["SMILES", "Molecule Name", "pEC50"]].copy()
    test  = ds["test"].to_pandas()[["SMILES", "Molecule Name"]].copy()
    train_path = out_dir / "activity_train.csv"
    test_path  = out_dir / "activity_test_blinded.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"PXR train: {len(train)} rows → {train_path}")
    print(f"PXR test:  {len(test)} rows  → {test_path}")
    return train_path, test_path


def download_chembl_target(
    target_id: str,
    value_field: str = "pchembl_value",
    out_dir: Path = EXTERNAL_DIR,
    sleep: float = 0.5,
) -> Path:
    from rdkit.Chem import MolToSmiles, MolFromSmiles
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://www.ebi.ac.uk/chembl/api/data/activity"
    params = {"target_chembl_id": target_id, "pchembl_value__isnull": "false",
              "limit": 1000, "format": "json", "assay_type": "B"}
    rows, offset = [], 0
    while True:
        params["offset"] = offset
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for act in data["activities"]:
            smi = act.get("canonical_smiles") or act.get("molecule_smiles")
            val = act.get(value_field) or act.get("pchembl_value")
            if smi and val:
                rows.append({"SMILES": smi, f"chembl_{value_field}": float(val)})
        if not data["page_meta"]["next"]:
            break
        offset += len(data["activities"])
        time.sleep(sleep)

    if not rows:
        print(f"ChEMBL {target_id}: 0 rows returned from API")
        out_path = out_dir / f"{target_id.lower().replace('chembl', 'chembl_')}_{value_field}.csv"
        pd.DataFrame(columns=["SMILES", f"chembl_{value_field}"]).to_csv(out_path, index=False)
        return out_path
    df = pd.DataFrame(rows).drop_duplicates(subset="SMILES")
    canonical = []
    for smi in df["SMILES"]:
        mol = MolFromSmiles(smi)
        canonical.append(MolToSmiles(mol) if mol else None)
    df["SMILES"] = canonical
    df = df.dropna(subset=["SMILES"]).drop_duplicates(subset="SMILES")

    name = target_id.lower().replace("chembl", "chembl_")
    out_path = out_dir / f"{name}_{value_field}.csv"
    df.to_csv(out_path, index=False)
    print(f"ChEMBL {target_id}: {len(df)} rows → {out_path}")
    return out_path


def download_adme(out_dir: Path = EXTERNAL_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ADME_public_set_3521.csv"
    resp = requests.get(ADME_URL, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"ADME public: {len(pd.read_csv(out_path))} rows → {out_path}")
    return out_path


def download_all(targets: list[str] | None = None) -> None:
    download_pxr()
    for name in (targets or list(CHEMBL_TARGETS.keys())):
        if name not in CHEMBL_TARGETS:
            raise ValueError(f"Unknown target '{name}'. Options: {list(CHEMBL_TARGETS)}")
        download_chembl_target(CHEMBL_TARGETS[name])
    download_adme()
