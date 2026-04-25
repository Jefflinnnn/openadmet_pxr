"""Download and clean ChEMBL activity records for a target into a simple CSV.

This is intended for building multitask auxiliary labels (e.g., CAR/AhR/CYP3A4)
to improve PXR pEC50 predictions.

We intentionally keep the output lightweight and model-friendly:
  - canonicalized SMILES (RDKit canonical/isomeric)
  - a single numeric label column (typically pChEMBL value)

Default choice: pchembl_value
  - It is already a standardized -log10 molar potency-like value (higher=more potent).
  - Exists across IC50/EC50/Ki/Kd etc.
  - Avoids unit conversion edge cases.

Usage:
  source .venv/bin/activate
  python scripts/download_chembl_target_activity.py \
    --target-chembl-id CHEMBL340 \
    --out-csv data/external/chembl_CHEMBL340_cyp3a4_pchembl.csv \
    --label-col chembl_cyp3a4_pchembl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem


def _canon_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _fetch_json(url: str, *, timeout_s: int = 60, sleep_s: float = 0.0) -> dict:
    if sleep_s:
        time.sleep(sleep_s)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "openadmet_pxr/chembl-downloader",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as f:
        return json.load(f)


def download_activity(
    *,
    target_chembl_id: str,
    limit_per_page: int = 1000,
    sleep_s: float = 0.05,
    max_pages: int | None = None,
) -> list[dict]:
    """Downloads all ChEMBL activity records for a target.

    Uses ChEMBL's REST paging (`page_meta.next`).
    """

    base = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    origin = "https://www.ebi.ac.uk"
    params = {
        "target_chembl_id": target_chembl_id,
        "limit": str(limit_per_page),
    }
    url = base + "?" + urllib.parse.urlencode(params)

    out: list[dict] = []
    page = 0
    while True:
        page += 1
        data = _fetch_json(url, sleep_s=sleep_s)
        out.extend(data.get("activities", []))
        next_url = (data.get("page_meta") or {}).get("next")
        if not next_url:
            break
        # ChEMBL may return either absolute URLs or site-relative paths.
        url = (origin + next_url) if isinstance(next_url, str) and next_url.startswith("/") else next_url
        if max_pages is not None and page >= max_pages:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chembl-id", type=str, required=True)
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path (will be overwritten).",
    )
    ap.add_argument(
        "--label-col",
        type=str,
        required=True,
        help="Name of the output label column.",
    )
    ap.add_argument(
        "--value-field",
        type=str,
        default="pchembl_value",
        help="Which ChEMBL activity JSON field to use as the numeric label.",
    )
    ap.add_argument(
        "--require-value",
        action="store_true",
        default=True,
        help="Drop rows missing the value-field (recommended).",
    )
    ap.add_argument(
        "--dedup",
        type=str,
        default="median",
        choices=["mean", "median", "first"],
        help="How to aggregate duplicate SMILES.",
    )
    ap.add_argument(
        "--limit-per-page",
        type=int,
        default=1000,
        help="Page size for ChEMBL API (<= 1000 is safe).",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="For debugging: stop after N pages.",
    )
    ap.add_argument(
        "--sleep-s",
        type=float,
        default=0.05,
        help="Polite sleep between page fetches.",
    )
    ap.add_argument(
        "--keep-columns",
        type=str,
        nargs="*",
        default=["molecule_chembl_id", "assay_chembl_id", "standard_type", "standard_units"],
        help="Optional metadata columns to keep (if present).",
    )
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    recs = download_activity(
        target_chembl_id=args.target_chembl_id,
        limit_per_page=args.limit_per_page,
        sleep_s=args.sleep_s,
        max_pages=args.max_pages,
    )
    if not recs:
        raise RuntimeError(f"No activity records returned for {args.target_chembl_id}")

    df = pd.DataFrame(recs)
    if "canonical_smiles" not in df.columns:
        raise RuntimeError("ChEMBL activity JSON missing canonical_smiles field")

    keep = [c for c in args.keep_columns if c in df.columns]
    out = pd.DataFrame(
        {
            "SMILES": df["canonical_smiles"].map(_canon_smiles),
            args.label_col: pd.to_numeric(df.get(args.value_field), errors="coerce"),
            **{c: df[c] for c in keep},
        }
    )

    out = out.dropna(subset=["SMILES"]).copy()
    if args.require_value:
        out = out.dropna(subset=[args.label_col]).copy()

    if args.dedup in ("mean", "median"):
        agg = "mean" if args.dedup == "mean" else "median"
        out = out.groupby("SMILES", as_index=False).agg({args.label_col: agg}).copy()
    else:
        out = out.drop_duplicates(subset=["SMILES"]).copy()

    out.to_csv(out_csv, index=False)
    print(f"Downloaded {len(recs)} raw activities")
    print(f"Wrote {len(out)} unique SMILES with labels -> {out_csv}")
    print(f"Label stats for {args.label_col}: n={out[args.label_col].notna().sum()} ")


if __name__ == "__main__":
    main()
