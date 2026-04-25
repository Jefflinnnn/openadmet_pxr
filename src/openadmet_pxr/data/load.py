"""Convenience loaders for all PXR challenge datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[3] / "data"
EXTERNAL_DIR = DATA_DIR / "external"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "activity_train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "activity_test_blinded.csv")


def load_chembl_cyp3a4() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / "chembl_340_pchembl_value.csv")


def load_chembl_pxr() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / "chembl_3401_pchembl_value.csv")


def load_chembl_ahr() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / "chembl_3201_pchembl_value.csv")


def load_chembl_car() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / "chembl_5503_pchembl_value.csv")


def load_adme() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / "ADME_public_set_3521.csv")


def dataset_summary() -> pd.DataFrame:
    loaders = {
        "pxr_train": load_train, "pxr_test": load_test,
        "chembl_cyp3a4": load_chembl_cyp3a4, "chembl_pxr": load_chembl_pxr,
        "chembl_ahr": load_chembl_ahr, "chembl_car": load_chembl_car,
        "adme_public": load_adme,
    }
    return pd.DataFrame([
        {"dataset": name, "rows": len(loader()), "columns": list(loader().columns)}
        for name, loader in loaders.items()
    ])
