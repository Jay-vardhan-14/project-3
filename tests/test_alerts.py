from __future__ import annotations

from ml.config import MLConfig
from ml.monitoring.alerts import check_and_create_alert
from ml.monitoring.drift_detector import DriftReport


def test_check_and_create_alert_returns_warning_for_threshold_breach():
    report = DriftReport(
        dataset_drift_detected=True,
        drift_score=0.2,
        features_drifted=1,
        total_features=3,
        prediction_drift_detected=False,
        reference_size=100,
        current_size=20,
        report_path="report.html",
    )

    result = check_and_create_alert(report, MLConfig(drift_threshold=0.15))

    assert result["created"] is True
    assert result["alert"]["alert_type"] == "drift_warning"


def test_check_and_create_alert_returns_critical_for_double_threshold_breach():
    report = DriftReport(
        dataset_drift_detected=True,
        drift_score=0.4,
        features_drifted=2,
        total_features=3,
        prediction_drift_detected=False,
        reference_size=100,
        current_size=20,
        report_path="report.html",
    )

    result = check_and_create_alert(report, MLConfig(drift_threshold=0.15))

    assert result["created"] is True
    assert result["alert"]["alert_type"] == "drift_critical"
