"""Weights & Biases experiment tracking."""

from __future__ import annotations

import os

WANDB_ENTITY = "jefflinnnn-personal"
WANDB_PROJECT = "OpenAdmetPXR"
WANDB_API_KEY = "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"


def setup() -> None:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
