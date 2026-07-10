"""Aggregation queries for the dashboard metrics endpoints."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SETTINGS

LOGGER = logging.getLogger(__name__)


def _window_clause(days: int | None, hours: int | None, column: str = "created_at") -> str:
    if hours is not None:
        return f"{column} >= NOW() - INTERVAL '{int(hours)} hours'"
    return f"{column} >= NOW() - INTERVAL '{int(days or 7)} days'"


async def _rows(session: AsyncSession, query: str) -> list[dict[str, Any]]:
    result = await session.execute(text(query))
    return [dict(row) for row in result.mappings().all()]


async def summary(session: AsyncSession, model_version: str | None, model_f1: float | None) -> dict[str, Any]:
    today = await _rows(
        session,
        "SELECT COUNT(*) AS n, COALESCE(AVG(latency_ms), 0) AS avg_latency "
        "FROM predictions WHERE created_at::date = CURRENT_DATE",
    )
    drift = await _rows(
        session,
        "SELECT dataset_drift_detected, drift_score FROM drift_reports "
        "ORDER BY report_date DESC, created_at DESC LIMIT 1",
    )
    drift_status = "healthy"
    if drift:
        score = float(drift[0]["drift_score"])
        threshold = _drift_threshold()
        if score > threshold * 2:
            drift_status = "critical"
        elif score > threshold:
            drift_status = "warning"
    return {
        "predictions_today": int(today[0]["n"]),
        "avg_latency_ms": round(float(today[0]["avg_latency"]), 2),
        "model_version": model_version,
        "model_f1": model_f1,
        "drift_status": drift_status,
    }


async def prediction_volume(session: AsyncSession, days: int | None, hours: int | None) -> list[dict[str, Any]]:
    bucket = "hour" if hours is not None else "day"
    return await _rows(
        session,
        f"SELECT date_trunc('{bucket}', created_at) AS bucket, COUNT(*) AS count "
        f"FROM predictions WHERE {_window_clause(days, hours)} "
        "GROUP BY bucket ORDER BY bucket",
    )


async def latency_percentiles(session: AsyncSession, days: int | None, hours: int | None) -> dict[str, Any]:
    rows = await _rows(
        session,
        "SELECT "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95, "
        "percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99, "
        "COUNT(*) AS n "
        f"FROM predictions WHERE {_window_clause(days, hours)}",
    )
    row = rows[0]
    return {
        "p50": float(row["p50"] or 0),
        "p95": float(row["p95"] or 0),
        "p99": float(row["p99"] or 0),
        "count": int(row["n"]),
    }


async def drift_history(session: AsyncSession, days: int | None) -> list[dict[str, Any]]:
    return await _rows(
        session,
        "SELECT report_date, drift_score, dataset_drift_detected, features_drifted, "
        "total_features, prediction_drift_detected, report_path "
        f"FROM drift_reports WHERE {_window_clause(days, None, 'report_date')} "
        "ORDER BY report_date",
    )


async def sentiment_distribution(session: AsyncSession, days: int | None, hours: int | None) -> list[dict[str, Any]]:
    return await _rows(
        session,
        "SELECT predicted_sentiment, COUNT(*) AS count "
        f"FROM predictions WHERE {_window_clause(days, hours)} "
        "GROUP BY predicted_sentiment",
    )


async def alerts(session: AsyncSession) -> list[dict[str, Any]]:
    return await _rows(
        session,
        "SELECT id, alert_type, severity, message, is_resolved, metadata, created_at "
        "FROM alerts ORDER BY created_at DESC LIMIT 50",
    )


async def pipeline_runs(session: AsyncSession) -> list[dict[str, Any]]:
    return await _rows(
        session,
        "SELECT id, dag_id, run_id, status, started_at, completed_at, duration_seconds, metrics "
        "FROM pipeline_runs ORDER BY started_at DESC LIMIT 50",
    )


def experiments() -> list[dict[str, Any]]:
    """MLflow experiment runs summarized for the dashboard."""

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
    client = MlflowClient()
    runs: list[dict[str, Any]] = []
    for experiment in client.search_experiments():
        for run in client.search_runs([experiment.experiment_id], max_results=50):
            runs.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                    "experiment": experiment.name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": dict(run.data.metrics),
                }
            )
    return runs


def registry_models() -> list[dict[str, Any]]:
    """Registered model versions with stages."""

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
    client = MlflowClient()
    entries: list[dict[str, Any]] = []
    for registered in client.search_registered_models():
        for version in client.search_model_versions(f"name='{registered.name}'"):
            entries.append(
                {
                    "name": version.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "f1_macro": (version.tags or {}).get("f1_macro"),
                    "accuracy": (version.tags or {}).get("accuracy"),
                }
            )
    return entries


def _drift_threshold() -> float:
    from ml.config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.drift_threshold
