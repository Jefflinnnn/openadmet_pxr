#!/usr/bin/env python
"""CLI: download all datasets needed for the PXR pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.data.download import download_pxr, download_chembl_target, download_adme, CHEMBL_TARGETS


def main():
    parser = argparse.ArgumentParser(description="Download PXR pipeline datasets")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(CHEMBL_TARGETS.keys()),
        default=list(CHEMBL_TARGETS.keys()),
        help="ChEMBL targets to download (default: all)",
    )
    parser.add_argument("--no-pxr", action="store_true", help="Skip PXR train/test download")
    parser.add_argument("--no-adme", action="store_true", help="Skip ADME public set download")
    args = parser.parse_args()

    if not args.no_pxr:
        download_pxr()

    for name in args.targets:
        download_chembl_target(CHEMBL_TARGETS[name])

    if not args.no_adme:
        download_adme()


if __name__ == "__main__":
    main()
