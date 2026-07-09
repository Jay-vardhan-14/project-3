"""Airflow DAG for SentinelML drift detection.

The DAG compares recent prediction logs against the training/reference
distribution, generates a drift report, creates alerts when thresholds are
exceeded, and stores the report metadata for dashboard consumption.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
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
        drift_threshold=float(os.getenv("SENTINELML_DRIFT_THRESHOLD", "0.15")),
        drift_window_days=int(os.getenv("SENTINELML_DRIFT_WINDOW_DAYS", "7")),
    )


def _database_url() -> str:
    return os.getenv(
        "SENTINELML_DATABASE_URL",
        "postgresql://postgres:postgres@db:5432/sentinelml",
    )


@dag(
    dag_id="drift_detection",
    description="Daily SentinelML drift detection and alerting pipeline.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["sentinelml", "drift", "monitoring"],
    doc_md=__doc__,
)
def drift_detection() -> None:
    @task
    def collect_recent_predictions() -> dict[str, Any]:
        """Query recent predictions from DB as current dataset."""

        import pandas as pd

        config = _config_from_context()
        query = f"""
            SELECT predicted_sentiment, confidence, input_length, latency_ms, created_at
            FROM predictions
            WHERE created_at >= NOW() - INTERVAL '{int(config.drift_window_days)} days'
            ORDER BY created_at DESC
        """
        with _connect_db() as connection:
            _ensure_monitoring_tables(connection)
            df = pd.read_sql(query, connection)

        current_path = config.processed_data_dir / "drift_current.parquet"
        current_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(current_path, index=False)
        result = {"current_path": str(current_path), "rows": int(len(df))}
        LOGGER.info("Collected recent predictions: %s", result)
        return result

    @task
    def load_reference_data() -> dict[str, Any]:
        """Load training data distribution as reference dataset."""

        import pandas as pd

        config = _config_from_context()
        train_path = config.splits_dir / "train.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Reference training split not found: {train_path}")

        train_df = pd.read_parquet(train_path)
        reference_df = pd.DataFrame(
            {
                "predicted_sentiment": train_df["label"].map({0: "negative", 1: "positive"}),
                "input_length": train_df["text"].astype(str).str.len(),
            }
        )
        reference_path = config.processed_data_dir / "drift_reference.parquet"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_df.to_parquet(reference_path, index=False)
        result = {"reference_path": str(reference_path), "rows": int(len(reference_df))}
        LOGGER.info("Loaded reference data: %s", result)
        return result

    @task
    def run_drift_detection(
        current_data: dict[str, Any],
        reference_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run drift detection and save HTML report."""

        import pandas as pd
        from ml.monitoring.drift_detector import detect_drift

        if int(current_data["rows"]) == 0:
            raise ValueError("No recent predictions available for drift detection.")

        config = _config_from_context()
        current_df = pd.read_parquet(current_data["current_path"])
        reference_df = pd.read_parquet(reference_data["reference_path"])
        drift_report = detect_drift(reference_df, current_df, config)
        result = {
            "dataset_drift_detected": drift_report.dataset_drift_detected,
            "drift_score": drift_report.drift_score,
            "features_drifted": drift_report.features_drifted,
            "total_features": drift_report.total_features,
            "prediction_drift_detected": drift_report.prediction_drift_detected,
            "reference_size": drift_report.reference_size,
            "current_size": drift_report.current_size,
            "report_path": drift_report.report_path,
            "details": drift_report.details,
        }
        LOGGER.info("Drift detection complete: %s", result)
        return result

    @task
    def evaluate_drift(drift_result: dict[str, Any]) -> dict[str, Any]:
        """Check drift score against threshold and create alert if exceeded."""

        from ml.monitoring.alerts import check_and_create_alert
        from ml.monitoring.drift_detector import DriftReport

        config = _config_from_context()
        report = DriftReport(**drift_result)
        with _connect_db() as connection:
            _ensure_monitoring_tables(connection)
            alert_result = check_and_create_alert(report, config, connection)
        LOGGER.info("Alert evaluation result: %s", alert_result)
        return alert_result

    @task
    def store_drift_report(
        drift_result: dict[str, Any],
        alert_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Store drift report metadata to DB."""

        with _connect_db() as connection:
            _ensure_monitoring_tables(connection)
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO drift_reports (
                        report_date,
                        dataset_drift_detected,
                        drift_score,
                        features_drifted,
                        total_features,
                        prediction_drift_detected,
                        reference_size,
                        current_size,
                        report_path,
                        details
                    )
                    VALUES (
                        CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING id
                    """,
                    (
                        drift_result["dataset_drift_detected"],
                        drift_result["drift_score"],
                        drift_result["features_drifted"],
                        drift_result["total_features"],
                        drift_result["prediction_drift_detected"],
                        drift_result["reference_size"],
                        drift_result["current_size"],
                        drift_result["report_path"],
                        json.dumps(drift_result.get("details", {})),
                    ),
                )
                report_id = str(cursor.fetchone()[0])
                connection.commit()
            finally:
                cursor.close()

        result = {
            "report_id": report_id,
            "alert_created": bool(alert_result.get("created")),
            "report_path": drift_result["report_path"],
        }
        LOGGER.info("Stored drift report: %s", result)
        return result

    current = collect_recent_predictions()
    reference = load_reference_data()
    drift = run_drift_detection(current, reference)
    alert = evaluate_drift(drift)
    store_drift_report(drift, alert)


def _connect_db() -> Any:
    import psycopg2

    return psycopg2.connect(_database_url())


def _ensure_monitoring_tables(connection: Any) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                input_text_hash VARCHAR(64) NOT NULL,
                input_length INTEGER NOT NULL,
                predicted_sentiment VARCHAR(10) NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                latency_ms INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_reports (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                report_date DATE NOT NULL,
                dataset_drift_detected BOOLEAN NOT NULL,
                drift_score DECIMAL(5, 4) NOT NULL,
                features_drifted INTEGER NOT NULL DEFAULT 0,
                total_features INTEGER NOT NULL,
                prediction_drift_detected BOOLEAN NOT NULL DEFAULT false,
                reference_size INTEGER NOT NULL,
                current_size INTEGER NOT NULL,
                report_path VARCHAR(500),
                details JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                alert_type VARCHAR(30) NOT NULL
                    CHECK (alert_type IN ('drift_warning', 'drift_critical', 'model_degradation',
                                          'latency_spike', 'pipeline_failure')),
                severity VARCHAR(10) NOT NULL
                    CHECK (severity IN ('info', 'warning', 'critical')),
                message TEXT NOT NULL,
                is_resolved BOOLEAN DEFAULT false,
                resolved_at TIMESTAMPTZ,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        connection.commit()
    finally:
        cursor.close()


drift_detection()
