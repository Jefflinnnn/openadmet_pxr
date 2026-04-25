"""Lift a Chemprop splits file from one CSV to another by SMILES matching.

Why
---
Chemprop splits files reference rows by 0-based index. When we move from the
challenge PXR training CSV (4139 rows) to a larger multitask CSV (e.g. 24k rows)
we want to preserve the same *molecule-level* holdouts for PXR while training on
the larger union dataset.

This script:
  - reads a source CSV (used to generate the original splits),
  - reads a destination CSV (e.g. multitask),
  - maps each source row index -> destination row index via canonical SMILES,
  - writes a new splits JSON pointing into the destination CSV.

Assumptions
-----------
- Source and destination SMILES are already canonicalized consistently.
- Destination contains all SMILES from source (if not, the script errors).
- If destination has duplicates, the first occurrence is used.
"""

from __future__ import annotations

import argparse
import json
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


def _build_index(smiles: pd.Series, *, canonicalize: bool) -> dict[str, int]:
    """Map SMILES -> first row index."""
    m: dict[str, int] = {}
    for i, s in enumerate(smiles.astype(str).tolist()):
        if canonicalize:
            s2 = _canon_smiles(s)
            if s2 is None:
                continue
            s = s2
        if s not in m:
            m[s] = int(i)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-csv", type=Path, required=True)
    ap.add_argument("--dst-csv", type=Path, required=True)
    ap.add_argument("--src-smiles-col", type=str, default="SMILES")
    ap.add_argument("--dst-smiles-col", type=str, default="SMILES")
    ap.add_argument("--src-splits", type=Path, required=True)
    ap.add_argument("--out-splits", type=Path, required=True)
    ap.add_argument(
        "--canonicalize",
        action="store_true",
        default=True,
        help="Canonicalize SMILES in both CSVs with RDKit before matching (recommended).",
    )
    args = ap.parse_args()

    src = pd.read_csv(args.src_csv)
    dst = pd.read_csv(args.dst_csv)
    if args.src_smiles_col not in src.columns:
        raise SystemExit(f"--src-csv missing {args.src_smiles_col!r}")
    if args.dst_smiles_col not in dst.columns:
        raise SystemExit(f"--dst-csv missing {args.dst_smiles_col!r}")

    dst_map = _build_index(dst[args.dst_smiles_col], canonicalize=bool(args.canonicalize))
    src_smiles = src[args.src_smiles_col].astype(str).tolist()

    splits = json.loads(args.src_splits.read_text())
    if not isinstance(splits, list) or not splits:
        raise SystemExit("--src-splits must be a non-empty JSON list")

    lifted: list[dict[str, list[int]]] = []
    missing: set[str] = set()
    for split in splits:
        out_split: dict[str, list[int]] = {}
        for key in ("train", "val", "test"):
            idxs = split.get(key, [])
            new_idxs: list[int] = []
            for i in idxs:
                s = src_smiles[int(i)]
                if args.canonicalize:
                    s2 = _canon_smiles(s)
                    if s2 is None:
                        missing.add(s)
                        continue
                    s = s2
                j = dst_map.get(s)
                if j is None:
                    missing.add(s)
                else:
                    new_idxs.append(int(j))
            out_split[key] = new_idxs
        lifted.append(out_split)

    if missing:
        ex = sorted(list(missing))[:10]
        raise SystemExit(
            f"Destination CSV missing {len(missing)} SMILES from source splits. Example: {ex}"
        )

    args.out_splits.parent.mkdir(parents=True, exist_ok=True)
    args.out_splits.write_text(json.dumps(lifted))
    print(f"Wrote lifted splits -> {args.out_splits} (n_splits={len(lifted)})")


if __name__ == "__main__":
    main()
