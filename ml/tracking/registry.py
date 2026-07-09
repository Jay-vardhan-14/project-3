"""MLflow Model Registry operations."""

from __future__ import annotations

import logging
from typing import Any

from ml.config import DEFAULT_CONFIG
from ml.models.evaluation import compare_models

LOGGER = logging.getLogger(__name__)


def register_model(run_id: str, model_name: str, metrics: dict[str, Any]) -> Any:
    """Register a logged MLflow model artifact as a model registry version."""

    import mlflow

    model_uri = f"runs:/{run_id}/{model_name}"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    _set_version_metadata(model_name, str(model_version.version), metrics)
    LOGGER.info("Registered model %s version %s from run %s.", model_name, model_version.version, run_id)
    return model_version


def promote_model(model_name: str, version: str | int, stage: str = "Production") -> Any:
    """Promote a model version to a registry stage."""

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    archived_existing = stage.lower() == "production"
    result = client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archived_existing,
    )
    LOGGER.info("Promoted model %s version %s to %s.", model_name, version, stage)
    return result


def get_production_model(model_name: str) -> Any:
    """Load the current Production model from the MLflow registry."""

    import mlflow

    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


def compare_and_promote(
    model_name: str,
    new_run_id: str,
    new_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Register a model and promote it if it beats current Production F1."""

    current_version = _get_current_production_version(model_name)
    current_metrics = _get_version_metrics(model_name, current_version) if current_version else None
    new_f1 = float(new_metrics.get("f1_macro", 0.0))
    old_f1 = float((current_metrics or {}).get("f1_macro", 0.0))

    model_version = register_model(new_run_id, model_name, new_metrics)
    version = str(model_version.version)

    if compare_models(current_metrics, new_metrics):
        promote_model(model_name, version, stage="Production")
        decision = {
            "promoted": True,
            "reason": "New model beat current Production macro F1.",
            "old_f1": old_f1,
            "new_f1": new_f1,
            "version": version,
            "stage": "Production",
        }
    else:
        promote_model(model_name, version, stage="Staging")
        decision = {
            "promoted": False,
            "reason": "New model did not beat current Production macro F1.",
            "old_f1": old_f1,
            "new_f1": new_f1,
            "version": version,
            "stage": "Staging",
        }

    LOGGER.info("Model promotion decision: %s", decision)
    return decision


def _set_version_metadata(model_name: str, version: str, metrics: dict[str, Any]) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    for key, value in metrics.items():
        if key in {"f1_macro", "accuracy", "precision_macro", "recall_macro"}:
            client.set_model_version_tag(model_name, version, key, str(value))


def _get_current_production_version(model_name: str) -> Any | None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    return versions[0] if versions else None


def _get_version_metrics(model_name: str, model_version: Any | None) -> dict[str, float] | None:
    if model_version is None:
        return None
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    details = client.get_model_version(model_name, str(model_version.version))
    metrics: dict[str, float] = {}
    for key in ["f1_macro", "accuracy", "precision_macro", "recall_macro"]:
        if key in details.tags:
            metrics[key] = float(details.tags[key])
    if metrics:
        return metrics

    run_id = getattr(model_version, "run_id", None)
    if run_id:
        run = client.get_run(run_id)
        return {key: float(value) for key, value in run.data.metrics.items()}
    return None


def compare_and_promote_default(new_run_id: str, new_metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare and promote using the configured registry model name."""

    return compare_and_promote(DEFAULT_CONFIG.model_registry_name, new_run_id, new_metrics)
