"""TF-IDF + Logistic Regression baseline training."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import MLConfig, DEFAULT_CONFIG
from ml.models.evaluation import (
    compute_inference_latency,
    estimate_model_size_mb,
    evaluate_model,
    generate_classification_report,
)
from ml.tracking.mlflow_utils import (
    create_artifact_dir,
    log_confusion_matrix,
    log_training_run,
    setup_mlflow,
    write_classification_report,
)
from ml.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)


def train_baseline(config: MLConfig = DEFAULT_CONFIG) -> tuple[Pipeline, dict[str, Any]]:
    """Train a TF-IDF + LogisticRegression pipeline and log it to MLflow."""

    set_seed(config.random_seed, include_torch=False)
    train_df, val_df, test_df = load_preprocessed_splits(config)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                _build_vectorizer(config),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.logreg_C,
                    max_iter=config.logreg_max_iter,
                    random_state=config.random_seed,
                    n_jobs=None,
                ),
            ),
        ]
    )

    start_time = time.perf_counter()
    pipeline.fit(train_df["text"], train_df["label"])
    training_time = time.perf_counter() - start_time

    metrics = evaluate_model(pipeline, test_df["text"], test_df["label"])
    y_pred = pipeline.predict(test_df["text"])
    report = generate_classification_report(test_df["label"], y_pred)
    metrics.update(
        {
            "training_time": float(training_time),
            "model_size_mb": float(estimate_model_size_mb(pipeline)),
            "inference_latency_ms": compute_inference_latency(
                pipeline,
                test_df["text"].head(min(8, len(test_df))),
                n_runs=10,
            ),
            "token_or_feature_count": int(
                len(pipeline.named_steps["tfidf"].vocabulary_)
            ),
            "validation_rows": int(len(val_df)),
        }
    )

    artifact_dir = create_artifact_dir("baseline")
    confusion_path = log_confusion_matrix(
        test_df["label"].to_numpy(),
        y_pred,
        labels=["negative", "positive"],
        filename=artifact_dir / "confusion_matrix.png",
    )
    report_path = write_classification_report(report, artifact_dir / "classification_report.json")

    params = {
        "model_type": "baseline-logreg",
        "dataset_name": config.dataset_name,
        "max_samples": config.max_samples,
        "random_seed": config.random_seed,
        "tfidf_max_features": config.tfidf_max_features,
        "tfidf_ngram_range": config.tfidf_ngram_range,
        "logreg_C": config.logreg_C,
        "logreg_max_iter": config.logreg_max_iter,
    }
    setup_mlflow("baseline-logreg")
    run_id = log_training_run(
        params=params,
        metrics=metrics,
        artifacts={
            "confusion_matrix": confusion_path,
            "classification_report": report_path,
        },
        model=pipeline,
        model_name=config.model_registry_name,
        flavor="sklearn",
        experiment_name="baseline-logreg",
    )
    metrics["mlflow_run_id"] = run_id
    LOGGER.info("Finished baseline training with run_id=%s.", run_id)
    return pipeline, metrics


def load_preprocessed_splits(config: MLConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test parquet splits from disk."""

    train_path = config.splits_dir / "train.parquet"
    val_path = config.splits_dir / "val.parquet"
    test_path = config.splits_dir / "test.parquet"
    missing = [str(path) for path in [train_path, val_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed split files: {missing}")
    return (
        pd.read_parquet(train_path),
        pd.read_parquet(val_path),
        pd.read_parquet(test_path),
    )


def _build_vectorizer(config: MLConfig) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=config.tfidf_ngram_range,
    )
