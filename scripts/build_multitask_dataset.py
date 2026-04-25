"""Build a multitask CSV for Chemprop training.

This script combines:
  1) OpenADMET PXR Activity Track training data (pEC50)
  2) Optional external auxiliary datasets (currently: Computational-ADME public set)

It outputs a single CSV with:
  - SMILES
  - optional Molecule Name
  - target columns (one per task)

Additional tasks can be provided via `--extra-task-csv`, which should point to a CSV
containing at least:
  - SMILES
  - exactly one numeric target column (or specify it with --extra-task-target-col)

Targets may be missing (NaN) for some molecules/tasks; Chemprop v2 supports
missing labels for multitask regression by masking those losses.

Usage:
  source .venv/bin/activate
  python scripts/build_multitask_dataset.py \
    --pxr-train-csv data/activity_train.csv \
    --adme3521-csv data/external/ADME_public_set_3521.csv \
    --out-csv data/multitask/pxr_plus_adme3521.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def _canon_smiles(smiles: str) -> str | None:
    """Returns canonical (isomeric) SMILES or None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _prep_pxr(pxr_train_csv: Path, *, smiles_col: str, name_col: str, target_col: str) -> pd.DataFrame:
    df = _read_csv(pxr_train_csv)
    missing = [c for c in [smiles_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"PXR train CSV missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "SMILES": df[smiles_col].map(_canon_smiles),
            "Molecule Name": df[name_col] if name_col in df.columns else None,
            "pxr_pEC50": pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["SMILES", "pxr_pEC50"]).copy()
    return out


def _prep_adme3521(adme_csv: Path) -> pd.DataFrame:
    df = _read_csv(adme_csv)
    if "SMILES" not in df.columns:
        raise ValueError(f"ADME 3521 CSV missing SMILES column. Found: {list(df.columns)}")

    task_map = {
        "LOG HLM_CLint (mL/min/kg)": "adme_log_hlm_clint",
        "LOG RLM_CLint (mL/min/kg)": "adme_log_rlm_clint",
        "LOG MDR1-MDCK ER (B-A/A-B)": "adme_log_mdr1_mdck_er",
        "LOG SOLUBILITY PH 6.8 (ug/mL)": "adme_log_solubility_ph68",
        "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)": "adme_log_ppb_human_unbound",
        "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)": "adme_log_ppb_rat_unbound",
    }
    present = [c for c in task_map if c in df.columns]
    if not present:
        raise ValueError(
            "ADME 3521 CSV has no recognized task columns. "
            f"Expected one of: {list(task_map)}. Found: {list(df.columns)}"
        )

    out = pd.DataFrame({"SMILES": df["SMILES"].map(_canon_smiles)})
    for c in present:
        out[task_map[c]] = pd.to_numeric(df[c], errors="coerce")
    out = out.dropna(subset=["SMILES"]).copy()
    return out


def _prep_extra_task_csv(
    path: Path,
    *,
    target_col: str | None,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Load a generic task CSV with SMILES + one numeric label column."""
    df = _read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Extra task CSV {path} missing {smiles_col}. Found: {list(df.columns)}")

    if target_col is None:
        # infer: first non-SMILES numeric-like column
        candidates = [c for c in df.columns if c != smiles_col]
        if len(candidates) != 1:
            raise ValueError(
                f"Extra task CSV {path} must have exactly 1 non-{smiles_col} column "
                f"or you must pass --extra-task-target-col. Found: {list(df.columns)}"
            )
        target_col = candidates[0]

    if target_col not in df.columns:
        raise ValueError(f"Extra task CSV {path} missing target col {target_col}. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "SMILES": df[smiles_col].map(_canon_smiles),
            target_col: pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["SMILES"]).copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pxr-train-csv", type=Path, required=True)
    ap.add_argument("--pxr-smiles-col", type=str, default="SMILES")
    ap.add_argument("--pxr-name-col", type=str, default="Molecule Name")
    ap.add_argument("--pxr-target-col", type=str, default="pEC50")

    ap.add_argument("--adme3521-csv", type=Path, default=None)

    ap.add_argument(
        "--extra-task-csv",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Additional task CSV(s) with at least SMILES + 1 target column. "
            "If the CSV has >1 non-SMILES column, pass --extra-task-target-col."
        ),
    )
    ap.add_argument(
        "--extra-task-target-col",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional target column name(s) corresponding to --extra-task-csv entries. "
            "Either provide 0 (infer) or the same count as --extra-task-csv."
        ),
    )

    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument(
        "--dedup",
        type=str,
        default="mean",
        choices=["mean", "median", "first"],
        help="How to aggregate duplicate SMILES within each source dataset.",
    )
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pxr = _prep_pxr(
        args.pxr_train_csv,
        smiles_col=args.pxr_smiles_col,
        name_col=args.pxr_name_col,
        target_col=args.pxr_target_col,
    )
    if args.dedup in ("mean", "median"):
        agg = "mean" if args.dedup == "mean" else "median"
        # Keep Molecule Name as first non-null, aggregate targets.
        tgt_cols = [c for c in pxr.columns if c not in {"SMILES", "Molecule Name"}]
        pxr = (
            pxr.groupby("SMILES", as_index=False)
            .agg({"Molecule Name": "first", **{c: agg for c in tgt_cols}})
            .copy()
        )
    else:
        pxr = pxr.drop_duplicates(subset=["SMILES"]).copy()

    dfs: list[pd.DataFrame] = [pxr]
    if args.adme3521_csv is not None:
        adme = _prep_adme3521(args.adme3521_csv)
        if args.dedup in ("mean", "median"):
            agg = "mean" if args.dedup == "mean" else "median"
            tgt_cols = [c for c in adme.columns if c != "SMILES"]
            adme = adme.groupby("SMILES", as_index=False).agg({c: agg for c in tgt_cols}).copy()
        else:
            adme = adme.drop_duplicates(subset=["SMILES"]).copy()
        dfs.append(adme)

    extra_targets: list[str | None]
    if args.extra_task_target_col:
        if len(args.extra_task_target_col) not in (1, len(args.extra_task_csv)):
            raise ValueError(
                "--extra-task-target-col must have 0, 1 (broadcast), or the same count as --extra-task-csv"
            )
        if len(args.extra_task_target_col) == 1 and len(args.extra_task_csv) > 1:
            extra_targets = [args.extra_task_target_col[0]] * len(args.extra_task_csv)
        else:
            extra_targets = list(args.extra_task_target_col)
    else:
        extra_targets = [None] * len(args.extra_task_csv)

    for path, tgt in zip(args.extra_task_csv, extra_targets, strict=False):
        extra = _prep_extra_task_csv(path, target_col=tgt)
        if args.dedup in ("mean", "median"):
            agg = "mean" if args.dedup == "mean" else "median"
            tgt_cols = [c for c in extra.columns if c != "SMILES"]
            extra = extra.groupby("SMILES", as_index=False).agg({c: agg for c in tgt_cols}).copy()
        else:
            extra = extra.drop_duplicates(subset=["SMILES"]).copy()
        dfs.append(extra)

    # Outer-join on SMILES so we keep union of molecules across tasks.
    out = dfs[0]
    for other in dfs[1:]:
        out = out.merge(other, on="SMILES", how="outer")

    # Basic reporting
    task_cols = [c for c in out.columns if c not in {"SMILES", "Molecule Name"}]
    print(f"Rows (unique SMILES): {len(out)}")
    for c in task_cols:
        n = int(out[c].notna().sum())
        print(f"  {c}: labeled {n} ({n/len(out):.1%})")

    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} cols={list(out.columns)}")


if __name__ == "__main__":
    main()
