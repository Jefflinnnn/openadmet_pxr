"""MLflow experiment tracking using MLflow 3.x LoggedModel pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = "sqlite:///" + str(Path(__file__).parents[3] / "mlruns.db")
EXPERIMENT_NAME = "pxr-activity-prediction"


def setup() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp.experiment_id


def log_cv_run(
    run_name: str,
    params: dict[str, Any],
    cv_summary: dict[str, dict[str, float]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    model_object: Any = None,
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a CV run with MLflow 3.x LoggedModel + Dataset linkage."""
    setup()

    flat_metrics = {}
    for metric, stats in cv_summary.items():
        flat_metrics[f"{metric}_mean"] = stats["mean"]
        flat_metrics[f"{metric}_std"] = stats["std"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        train_dataset = mlflow.data.from_pandas(train_df, name="pxr_train", targets="pEC50")
        mlflow.log_input(train_dataset, context="training")

        if val_df is not None:
            val_dataset = mlflow.data.from_pandas(val_df, name="pxr_val", targets="pEC50")
            mlflow.log_input(val_dataset, context="validation")

        if model_object is not None:
            model_info = mlflow.sklearn.log_model(
                sk_model=model_object,
                name=params.get("model", "model"),
                params={k: str(v) for k, v in params.items()},
            )
            logged_model = mlflow.get_logged_model(model_info.model_id)
            mlflow.log_metrics(
                metrics=flat_metrics,
                model_id=logged_model.model_id,
                dataset=train_dataset,
            )
        else:
            mlflow.log_metrics(flat_metrics)

        if artifacts:
            for path in artifacts:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))

    return run.info.run_id
