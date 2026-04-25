"""Append leaderboard results to leaderboard/experiments.csv.

This is intentionally lightweight: it does not talk to any external service.
You paste the leaderboard metrics after submitting.

Example:
  python scripts/log_leaderboard.py \
    --submission-csv submissions/chemeleon_smoke.csv \
    --source runs/chemeleon_smoke \
    --model-family chemeleon \
    --targets pEC50 \
    --lb-mae 0.55 --lb-r2 0.42 --lb-spearman 0.70 --lb-kendall 0.50 \
    --notes "best LB so far"
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


FIELDS = [
    "timestamp",
    "submission_csv",
    "source",
    "model_family",
    "targets",
    "task_weights",
    "featurizers",
    "ensemble_desc",
    "local_cv_mae",
    "local_cv_r2",
    "lb_mae",
    "lb_r2",
    "lb_spearman",
    "lb_kendall",
    "notes",
]


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-csv", type=Path, default=Path("leaderboard/experiments.csv"))
    ap.add_argument("--submission-csv", type=Path, required=True)
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--model-family", type=str, required=True)
    ap.add_argument("--targets", type=str, required=True, help="Comma-separated target list")

    ap.add_argument("--task-weights", type=str, default="")
    ap.add_argument("--featurizers", type=str, default="")
    ap.add_argument("--ensemble-desc", type=str, default="")
    ap.add_argument("--local-cv-mae", type=str, default="")
    ap.add_argument("--local-cv-r2", type=str, default="")

    ap.add_argument("--lb-mae", type=str, default="")
    ap.add_argument("--lb-r2", type=str, default="")
    ap.add_argument("--lb-spearman", type=str, default="")
    ap.add_argument("--lb-kendall", type=str, default="")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    args.log_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": _iso_now(),
        "submission_csv": str(args.submission_csv),
        "source": args.source,
        "model_family": args.model_family,
        "targets": args.targets,
        "task_weights": args.task_weights,
        "featurizers": args.featurizers,
        "ensemble_desc": args.ensemble_desc,
        "local_cv_mae": args.local_cv_mae,
        "local_cv_r2": args.local_cv_r2,
        "lb_mae": args.lb_mae,
        "lb_r2": args.lb_r2,
        "lb_spearman": args.lb_spearman,
        "lb_kendall": args.lb_kendall,
        "notes": args.notes,
    }

    # Create file w/ header if it doesn't exist.
    exists = args.log_csv.exists()
    with args.log_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended to {args.log_csv}: {args.submission_csv}")


if __name__ == "__main__":
    main()
