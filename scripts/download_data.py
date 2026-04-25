"""Download OpenADMET PXR Activity Track data from Hugging Face.

Writes out train/test CSVs with the expected columns.

Usage:
  source .venv/bin/activate
  python scripts/download_data.py --out-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_ID = "openadmet/pxr-challenge-train-test"


def _to_df(ds) -> pd.DataFrame:
    return ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default config contains activity TRAIN and blinded activity TEST
    dset = load_dataset(DATASET_ID, name="default")

    train_df = _to_df(dset["train"])
    test_df = _to_df(dset["test"])

    # Keep only relevant columns if present
    # Train should include pEC50; test will not.
    train_cols = [c for c in ["SMILES", "Molecule Name", "pEC50"] if c in train_df.columns]
    test_cols = [c for c in ["SMILES", "Molecule Name"] if c in test_df.columns]

    train_df = train_df[train_cols].copy()
    test_df = test_df[test_cols].copy()

    train_path = out_dir / "activity_train.csv"
    test_path = out_dir / "activity_test_blinded.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Wrote: {train_path} rows={len(train_df)} cols={list(train_df.columns)}")
    print(f"Wrote: {test_path} rows={len(test_df)} cols={list(test_df.columns)}")


if __name__ == "__main__":
    main()
