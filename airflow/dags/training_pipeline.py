"""Airflow DAG for SentinelML sentiment training and deployment.

The DAG runs the full training lifecycle:
ingest -> validate -> preprocess -> train baseline and transformer -> compare
the two runs -> register/promote the best candidate -> notify serving to reload.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
from typing import Any

from airflow.decorators import dag, task

LOGGER = logging.getLogger(__name__)


DEFAULT_ARGS = {
    "owner": "sentinelml",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


def _config_from_context() -> Any:
    from ml.config import MLConfig

    return MLConfig(
        dataset_name=os.getenv("SENTINELML_DATASET_NAME", "imdb"),
        max_samples=int(os.getenv("SENTINELML_MAX_SAMPLES", "25000")),
        num_epochs=int(os.getenv("SENTINELML_NUM_EPOCHS", "3")),
        batch_size=int(os.getenv("SENTINELML_BATCH_SIZE", "32")),
        model_registry_name=os.getenv("MODEL_NAME", "sentiment-model"),
    )


@dag(
    dag_id="sentiment_training_pipeline",
    description="End-to-end SentinelML sentiment model training and promotion pipeline.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["sentinelml", "training", "mlflow"],
    doc_md=__doc__,
)
def sentiment_training_pipeline() -> None:
    @task
    def ingest_data() -> dict[str, Any]:
        """Download dataset, save to data/raw/, and return summary metadata."""

        from ml.data.ingestion import download_dataset

        config = _config_from_context()
        df = download_dataset(config)
        result = {
            "rows": int(len(df)),
            "raw_path": str(config.raw_data_dir / "latest.parquet"),
            "dataset_name": config.dataset_name,
        }
        LOGGER.info("Ingestion complete: %s", result)
        return result

    @task
    def validate_data(ingestion_result: dict[str, Any]) -> dict[str, Any]:
        """Run validation checks, fail DAG on critical issues."""

        import pandas as pd
        from ml.data.validation import validate_data as run_validation

        raw_path = ingestion_result["raw_path"]
        df = pd.read_parquet(raw_path)
        report = run_validation(df)
        result = {
            "passed": report.passed,
            "rows": report.row_count,
            "warnings": report.warnings,
            "raw_path": raw_path,
        }
        LOGGER.info("Validation complete: %s", result)
        return result

    @task
    def preprocess_data(validation_result: dict[str, Any]) -> dict[str, Any]:
        """Clean text, create splits, and save to data/splits/."""

        import pandas as pd
        from ml.data.preprocessing import preprocess_dataset

        config = _config_from_context()
        df = pd.read_parquet(validation_result["raw_path"])
        splits = preprocess_dataset(df, config)
        result = {
            "train_rows": int(len(splits["train"])),
            "val_rows": int(len(splits["val"])),
            "test_rows": int(len(splits["test"])),
            "splits_dir": str(config.splits_dir),
        }
        LOGGER.info("Preprocessing complete: %s", result)
        return result

    @task
    def train_baseline(preprocess_result: dict[str, Any]) -> dict[str, Any]:
        """Train TF-IDF + Logistic Regression and log to MLflow."""

        from ml.models.baseline import train_baseline as run_baseline

        config = _config_from_context()
        _, metrics = run_baseline(config)
        result = {
            "model_type": "baseline-logreg",
            "run_id": metrics.get("mlflow_run_id"),
            "metrics": _xcom_safe(metrics),
            "splits_dir": preprocess_result["splits_dir"],
        }
        LOGGER.info("Baseline training complete: %s", result)
        return result

    @task
    def train_transformer(preprocess_result: dict[str, Any]) -> dict[str, Any]:
        """Train DistilBERT and log to MLflow."""

        from ml.models.transformer import train_transformer as run_transformer

        config = _config_from_context()
        _, metrics = run_transformer(config)
        result = {
            "model_type": "distilbert",
            "run_id": metrics.get("mlflow_run_id"),
            "metrics": _xcom_safe(metrics),
            "splits_dir": preprocess_result["splits_dir"],
        }
        LOGGER.info("Transformer training complete: %s", result)
        return result

    @task
    def evaluate_and_compare(
        baseline_result: dict[str, Any],
        transformer_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare both models and select winner based on macro F1."""

        candidates = [baseline_result, transformer_result]
        winner = max(candidates, key=lambda item: float(item["metrics"].get("f1_macro", 0.0)))
        result = {
            "winner_model_type": winner["model_type"],
            "winner_run_id": winner["run_id"],
            "winner_metrics": winner["metrics"],
            "candidates": [
                {
                    "model_type": candidate["model_type"],
                    "run_id": candidate["run_id"],
                    "f1_macro": candidate["metrics"].get("f1_macro"),
                    "accuracy": candidate["metrics"].get("accuracy"),
                }
                for candidate in candidates
            ],
        }
        LOGGER.info("Model comparison complete: %s", result)
        return result

    @task
    def register_best_model(winner_result: dict[str, Any]) -> dict[str, Any]:
        """Register winner in MLflow and promote if better than Production."""

        from ml.tracking.registry import compare_and_promote

        config = _config_from_context()
        if not winner_result.get("winner_run_id"):
            raise ValueError("Winner result has no MLflow run id.")
        decision = compare_and_promote(
            model_name=config.model_registry_name,
            new_run_id=winner_result["winner_run_id"],
            new_metrics=winner_result["winner_metrics"],
        )
        result = {
            **winner_result,
            "registration": _xcom_safe(decision),
        }
        LOGGER.info("Registration complete: %s", result)
        return result

    @task
    def deploy_model(registration_result: dict[str, Any]) -> dict[str, Any]:
        """Notify serving API to reload the Production model."""

        import requests

        serving_url = os.getenv("SENTINELML_SERVING_URL", "http://serving:8000")
        reload_url = f"{serving_url.rstrip('/')}/api/v1/model/reload"
        try:
            response = requests.post(reload_url, timeout=30)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            status = "reloaded"
        except Exception as exc:
            LOGGER.warning("Serving reload notification failed: %s", exc)
            payload = {"error": str(exc)}
            status = "reload_failed"

        result = {
            "status": status,
            "reload_response": payload,
            "registration": registration_result["registration"],
        }
        LOGGER.info("Deployment notification result: %s", result)
        return result

    ingested = ingest_data()
    validated = validate_data(ingested)
    preprocessed = preprocess_data(validated)
    baseline = train_baseline(preprocessed)
    transformer = train_transformer(preprocessed)
    winner = evaluate_and_compare(baseline, transformer)
    registered = register_best_model(winner)
    deploy_model(registered)


def _xcom_safe(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = value
        elif isinstance(value, dict):
            safe[key] = _xcom_safe(value)
        else:
            safe[key] = str(value)
    return safe


sentiment_training_pipeline()
