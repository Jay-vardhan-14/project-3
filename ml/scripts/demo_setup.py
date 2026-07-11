"""One-shot demo seed for SentinelML.

Trains the baseline model, registers/promotes it to Production, and populates the
database with sample predictions, drift reports, an alert, and a pipeline run so
the dashboard shows real data immediately after `docker-compose up`.

Run inside the airflow container (it has the training + psycopg2 dependencies):

    docker-compose exec airflow python /opt/airflow/ml/scripts/demo_setup.py

Environment: MLFLOW_TRACKING_URI and SENTINELML_DATABASE_URL are read from the
container env (defaults target the compose service names).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ml.config import MLConfig
from ml.data.ingestion import download_dataset
from ml.data.preprocessing import clean_text, preprocess_dataset
from ml.models.baseline import train_baseline
from ml.monitoring.pipeline_tracking import record_run_finish
from ml.tracking.registry import compare_and_promote
from ml.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("demo_setup")

SAMPLE_REVIEWS = [
    "An absolute masterpiece, I loved every minute of it.",
    "Terrible, boring, and a complete waste of time.",
    "Great acting and a gripping story from start to finish.",
    "The plot made no sense and the pacing dragged.",
    "Heartwarming and beautifully shot — highly recommend.",
    "Dreadful dialogue and wooden performances throughout.",
]


def _db_url() -> str:
    return os.getenv("SENTINELML_DATABASE_URL", "postgresql://postgres:postgres@db:5432/sentinelml")


def _connect() -> Any:
    import psycopg2

    return psycopg2.connect(_db_url())


def _ensure_tables(conn: Any) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                input_text_hash VARCHAR(64) NOT NULL, input_length INTEGER NOT NULL,
                predicted_sentiment VARCHAR(10) NOT NULL, confidence DECIMAL(5,4) NOT NULL,
                model_version VARCHAR(100) NOT NULL, model_name VARCHAR(100) NOT NULL,
                latency_ms INTEGER NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_reports (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), report_date DATE NOT NULL,
                dataset_drift_detected BOOLEAN NOT NULL, drift_score DECIMAL(5,4) NOT NULL,
                features_drifted INTEGER NOT NULL DEFAULT 0, total_features INTEGER NOT NULL,
                prediction_drift_detected BOOLEAN NOT NULL DEFAULT false,
                reference_size INTEGER NOT NULL, current_size INTEGER NOT NULL,
                report_path VARCHAR(500), details JSONB, created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                alert_type VARCHAR(30) NOT NULL, severity VARCHAR(10) NOT NULL,
                message TEXT NOT NULL, is_resolved BOOLEAN DEFAULT false,
                resolved_at TIMESTAMPTZ, metadata JSONB, created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.commit()
    finally:
        cursor.close()


def seed_predictions(pipeline: Any, config: MLConfig, model_version: str, count: int) -> None:
    """Run the trained model over real review texts and log the predictions."""

    test_path = config.splits_dir / "test.parquet"
    texts: list[str] = []
    if test_path.exists():
        texts = pd.read_parquet(test_path)["text"].astype(str).head(count).tolist()
    while len(texts) < count:
        texts.append(random.choice(SAMPLE_REVIEWS))

    now = datetime.now(timezone.utc)
    conn = _connect()
    try:
        _ensure_tables(conn)
        cursor = conn.cursor()
        try:
            for text in texts[:count]:
                proba = pipeline.predict_proba([clean_text(text)])[0]
                idx = int(proba.argmax())
                sentiment = "positive" if int(pipeline.classes_[idx]) == 1 else "negative"
                created = now - timedelta(hours=random.randint(0, 24 * 6), minutes=random.randint(0, 59))
                cursor.execute(
                    "INSERT INTO predictions (input_text_hash, input_length, predicted_sentiment, "
                    "confidence, model_version, model_name, latency_ms, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        len(text), sentiment, round(float(proba[idx]), 4),
                        model_version, config.model_registry_name, random.randint(1, 60), created,
                    ),
                )
            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()
    LOGGER.info("Seeded %d predictions.", count)


def seed_drift_and_alert(config: MLConfig) -> None:
    """Insert a short drift-score history and one warning alert."""

    conn = _connect()
    try:
        _ensure_tables(conn)
        cursor = conn.cursor()
        try:
            for days_ago, score in ((3, 0.06), (2, 0.09), (1, 0.13)):
                cursor.execute(
                    "INSERT INTO drift_reports (report_date, dataset_drift_detected, drift_score, "
                    "features_drifted, total_features, prediction_drift_detected, reference_size, "
                    "current_size, report_path, details) VALUES "
                    "(CURRENT_DATE - %s, %s, %s, %s, 2, false, 2000, 100, NULL, %s)",
                    (days_ago, score > config.drift_threshold, score, 1 if score > config.drift_threshold else 0,
                     json.dumps({"note": "demo drift report"})),
                )
            cursor.execute(
                "INSERT INTO alerts (alert_type, severity, message, metadata) VALUES "
                "('drift_warning', 'warning', 'Drift warning: score=0.1300', %s)",
                (json.dumps({"drift_score": 0.13, "features_drifted": 1, "total_features": 2}),),
            )
            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()
    LOGGER.info("Seeded drift reports and one alert.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed SentinelML with demo data.")
    parser.add_argument("--samples", type=int, default=2000, help="dataset rows to train on")
    parser.add_argument("--predictions", type=int, default=100, help="sample predictions to log")
    args = parser.parse_args()

    set_seed(42, include_torch=False)
    config = MLConfig(max_samples=args.samples)

    LOGGER.info("Downloading dataset (%d samples)...", args.samples)
    frame = download_dataset(config)
    preprocess_dataset(frame, config)

    LOGGER.info("Training baseline model...")
    pipeline, metrics = train_baseline(config)
    run_id = metrics.get("mlflow_run_id")

    decision = {"version": "1"}
    if run_id:
        decision = compare_and_promote(config.model_registry_name, run_id, metrics)
    LOGGER.info("Registry decision: %s", decision)

    version = str(decision.get("version", "1"))
    seed_predictions(pipeline, config, version, args.predictions)
    seed_drift_and_alert(config)
    record_run_finish(
        _db_url(), "sentiment_training_pipeline", "demo_setup",
        "success", {"winner": "baseline-logreg", "baseline_f1": metrics.get("f1_macro"), "promoted": decision.get("promoted", True)},
    )

    # Notify the serving API to load the freshly promoted model (non-fatal).
    try:
        import requests

        serving_url = os.getenv("SENTINELML_SERVING_URL", "http://serving:8000")
        requests.post(f"{serving_url.rstrip('/')}/api/v1/model/reload", timeout=30)
        LOGGER.info("Notified serving to reload the Production model.")
    except Exception as exc:
        LOGGER.warning("Serving reload notification skipped: %s", exc)

    LOGGER.info("Demo setup complete. Open the dashboard at http://localhost:3000")


if __name__ == "__main__":
    main()
