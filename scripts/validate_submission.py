#!/usr/bin/env python
"""CLI: validate a submission CSV before uploading."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.submission.validate import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Validate submission CSV")
    parser.add_argument("submission", type=str, help="Path to submission CSV")
    args = parser.parse_args()

    try:
        validate_submission(args.submission)
        sys.exit(0)
    except ValueError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
