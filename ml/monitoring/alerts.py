"""Alert creation and retrieval helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

from ml.config import DEFAULT_CONFIG, MLConfig
from ml.monitoring.drift_detector import DriftReport

LOGGER = logging.getLogger(__name__)


def check_and_create_alert(
    drift_report: DriftReport,
    config: MLConfig = DEFAULT_CONFIG,
    db_session: Any | None = None,
) -> dict[str, Any]:
    """Create an alert record when drift exceeds configured thresholds."""

    alert: dict[str, Any] | None = None
    if drift_report.drift_score > config.drift_threshold * 2:
        alert = {
            "alert_type": "drift_critical",
            "severity": "critical",
            "message": f"Critical drift detected: score={drift_report.drift_score:.4f}",
        }
    elif drift_report.drift_score > config.drift_threshold:
        alert = {
            "alert_type": "drift_warning",
            "severity": "warning",
            "message": f"Drift warning: score={drift_report.drift_score:.4f}",
        }
    elif drift_report.prediction_drift_detected:
        alert = {
            "alert_type": "model_degradation",
            "severity": "warning",
            "message": "Prediction distribution shift detected.",
        }

    if alert is None:
        result = {"created": False, "alert": None}
        LOGGER.info("No alert created for drift score %.4f.", drift_report.drift_score)
        return result

    alert["metadata"] = {
        "drift_score": drift_report.drift_score,
        "features_drifted": drift_report.features_drifted,
        "total_features": drift_report.total_features,
        "report_path": drift_report.report_path,
    }
    alert["created_at"] = datetime.now(timezone.utc).isoformat()

    if db_session is not None:
        _insert_alert(db_session, alert)
    LOGGER.warning("Created alert: %s", alert)
    return {"created": True, "alert": alert}


def resolve_alert(alert_id: str, db_session: Any) -> None:
    """Mark an alert as resolved."""

    _execute(
        db_session,
        """
        UPDATE alerts
        SET is_resolved = true, resolved_at = NOW()
        WHERE id = %s
        """,
        (alert_id,),
    )


def get_active_alerts(db_session: Any) -> list[dict[str, Any]]:
    """Return unresolved alert records."""

    rows = _fetchall(
        db_session,
        """
        SELECT id, alert_type, severity, message, metadata, created_at
        FROM alerts
        WHERE is_resolved = false
        ORDER BY created_at DESC
        """,
    )
    return [dict(row) for row in rows]


def _insert_alert(db_session: Any, alert: dict[str, Any]) -> None:
    _execute(
        db_session,
        """
        INSERT INTO alerts (alert_type, severity, message, metadata)
        VALUES (%s, %s, %s, %s)
        """,
        (
            alert["alert_type"],
            alert["severity"],
            alert["message"],
            json.dumps(alert["metadata"]),  # JSONB column: psycopg2 can't adapt a raw dict
        ),
    )


def _execute(db_session: Any, query: str, params: tuple[Any, ...]) -> None:
    cursor = db_session.cursor()
    try:
        cursor.execute(query, params)
        db_session.commit()
    finally:
        cursor.close()


def _fetchall(db_session: Any, query: str) -> list[Any]:
    cursor = db_session.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
