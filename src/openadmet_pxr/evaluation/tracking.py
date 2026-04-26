"""Weights & Biases experiment tracking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

WANDB_ENTITY = "jefflinnnn-personal"
WANDB_PROJECT = "OpenAdmetPXR"
WANDB_API_KEY = "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"


def setup() -> None:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def log_cv_run(
    run_name: str,
    params: dict[str, Any],
    cv_summary: dict[str, dict[str, float]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    model_object: Any = None,
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Log a completed CV run summary to W&B."""
    setup()
    flat_metrics = {}
    for metric, stats in cv_summary.items():
        flat_metrics[f"{metric}_mean"] = stats["mean"]
        flat_metrics[f"{metric}_std"] = stats["std"]

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        config={**params, **(tags or {})},
        reinit=True,
    )
    run.log(flat_metrics)
    if artifacts:
        for path in artifacts:
            if Path(path).exists():
                run.log_artifact(str(path))
    run.finish()
