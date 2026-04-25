"""Build multitask CSVs by outer-joining PXR data with external datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from openadmet_pxr.data.load import (
    DATA_DIR, load_train, load_chembl_cyp3a4, load_chembl_pxr,
    load_chembl_ahr, load_chembl_car, load_adme,
)

MULTITASK_DIR = DATA_DIR / "multitask"

ADME_RENAME = {
    "LOG HLM_CLint (mL/min/kg)": "adme_log_hlm_clint",
    "LOG MDR1-MDCK ER (B-A/A-B)": "adme_log_mdr1_er",
    "LOG SOLUBILITY PH 6.8 (ug/mL)": "adme_log_solubility",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)": "adme_log_ppb_human",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)": "adme_log_ppb_rat",
    "LOG RLM_CLint (mL/min/kg)": "adme_log_rlm_clint",
}


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def _canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMILES"] = df["SMILES"].map(_canonical)
    return df.dropna(subset=["SMILES"]).drop_duplicates(subset="SMILES")


def build_multitask_csv(
    include_cyp3a4: bool = True,
    include_chembl_pxr: bool = False,
    include_ahr: bool = False,
    include_car: bool = False,
    include_adme: bool = False,
    out_path: Path | str | None = None,
) -> Path:
    """Outer-join PXR training data with selected external datasets."""
    MULTITASK_DIR.mkdir(parents=True, exist_ok=True)

    pxr = _canonicalize_df(load_train()[["SMILES", "Molecule Name", "pEC50"]])
    merged = pxr.set_index("SMILES")
    extras: dict[str, pd.Series] = {}

    if include_cyp3a4:
        df = _canonicalize_df(load_chembl_cyp3a4()).rename(columns={"chembl_pchembl_value": "chembl_cyp3a4_pchembl"})
        extras["chembl_cyp3a4_pchembl"] = df.set_index("SMILES")["chembl_cyp3a4_pchembl"]
    if include_chembl_pxr:
        df = _canonicalize_df(load_chembl_pxr()).rename(columns={"chembl_pchembl_value": "chembl_pxr_pchembl"})
        extras["chembl_pxr_pchembl"] = df.set_index("SMILES")["chembl_pxr_pchembl"]
    if include_ahr:
        df = _canonicalize_df(load_chembl_ahr()).rename(columns={"chembl_pchembl_value": "chembl_ahr_pchembl"})
        extras["chembl_ahr_pchembl"] = df.set_index("SMILES")["chembl_ahr_pchembl"]
    if include_car:
        df = _canonicalize_df(load_chembl_car()).rename(columns={"chembl_pchembl_value": "chembl_car_pchembl"})
        extras["chembl_car_pchembl"] = df.set_index("SMILES")["chembl_car_pchembl"]
    if include_adme:
        df = _canonicalize_df(load_adme())
        df = df.rename(columns=ADME_RENAME)
        for col in [c for c in ADME_RENAME.values() if c in df.columns]:
            extras[col] = df.set_index("SMILES")[col]

    for col, series in extras.items():
        extra_df = series.reset_index()
        extra_df.columns = ["SMILES", col]
        merged = merged.join(extra_df.set_index("SMILES"), how="outer")

    result = merged.reset_index()
    mask = result["Molecule Name"].isna()
    result.loc[mask, "Molecule Name"] = result.loc[mask, "SMILES"].str[:20]

    if out_path is None:
        parts = ["pxr"]
        if include_cyp3a4:     parts.append("cyp3a4")
        if include_chembl_pxr: parts.append("chembl_pxr")
        if include_ahr:        parts.append("ahr")
        if include_car:        parts.append("car")
        if include_adme:       parts.append("adme")
        out_path = MULTITASK_DIR / ("_".join(parts) + ".csv")

    result.to_csv(out_path, index=False)
    n_tasks = len([c for c in result.columns if c not in ("SMILES", "Molecule Name")])
    print(f"Saved: {out_path}  ({len(result)} rows, {n_tasks} target columns)")
    return Path(out_path)
