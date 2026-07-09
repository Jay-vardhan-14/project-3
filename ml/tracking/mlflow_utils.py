"""MLflow setup and logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

LOGGER = logging.getLogger(__name__)

ModelFlavor = Literal["sklearn", "pytorch", "pyfunc"]


def setup_mlflow(experiment_name: str) -> str | None:
    """Configure MLflow tracking and create or select an experiment."""

    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow is not installed; experiment tracking is disabled.")
        return None

    try:
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else None
        LOGGER.info("MLflow experiment ready: %s (%s).", experiment_name, experiment_id)
        return experiment_id
    except Exception:
        LOGGER.exception("Failed to set up MLflow experiment '%s'.", experiment_name)
        return None


def log_training_run(
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, str | Path] | None,
    model: Any,
    model_name: str,
    flavor: ModelFlavor = "sklearn",
    experiment_name: str | None = None,
) -> str | None:
    """Log parameters, metrics, artifacts, and model to MLflow."""

    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow is not installed; skipping run logging.")
        return None

    try:
        if experiment_name:
            setup_mlflow(experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(_flatten_for_mlflow(params))
            for metric_name, metric_value in _numeric_metrics(metrics).items():
                mlflow.log_metric(metric_name, metric_value)
            for artifact_name, artifact_path in (artifacts or {}).items():
                path = Path(artifact_path)
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=artifact_name)
            if model is not None:
                _log_model(model, model_name, flavor)
            return run.info.run_id
    except Exception:
        LOGGER.exception("Failed to log MLflow training run for '%s'.", model_name)
        return None


def log_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
    filename: str | Path,
) -> Path:
    """Generate and save a minimal black-and-white confusion matrix PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Greys", interpolation="nearest")
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    LOGGER.info("Saved confusion matrix to %s.", output_path)
    return output_path


def write_classification_report(report: dict[str, Any], filename: str | Path) -> Path:
    """Write a classification report artifact to JSON."""

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def create_artifact_dir(prefix: str) -> Path:
    """Create a temporary artifact directory for a training run."""

    return Path(tempfile.mkdtemp(prefix=f"{prefix}-"))


def _log_model(model: Any, model_name: str, flavor: ModelFlavor) -> None:
    import mlflow

    if flavor == "sklearn":
        mlflow.sklearn.log_model(model, artifact_path=model_name)
    elif flavor == "pytorch":
        mlflow.pytorch.log_model(model, artifact_path=model_name)
    else:
        mlflow.pyfunc.log_model(artifact_path=model_name, python_model=model)


def _flatten_for_mlflow(values: dict[str, Any]) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value, default=str)
    return flattened


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric[key] = float(value)
    return numeric
