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


class _FakeCursor:
    def __init__(self) -> None:
        self.params: tuple = ()

    def execute(self, _query: str, params: tuple) -> None:
        self.params = params

    def close(self) -> None:
        pass


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        pass


def test_alert_metadata_serialized_as_json_for_jsonb_column():
    # Regression: psycopg2 can't adapt a raw dict into a JSONB column, so the
    # metadata must be json.dumps()'d before insert.
    import json

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
    conn = _FakeConn()

    check_and_create_alert(report, MLConfig(drift_threshold=0.15), conn)

    metadata_param = conn.cursor_obj.params[-1]
    assert isinstance(metadata_param, str)
    assert json.loads(metadata_param)["drift_score"] == 0.2
