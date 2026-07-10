"""Metrics service tests: time windowing and drift status mapping."""

from __future__ import annotations

import pytest

from app.services import metrics_service


def test_window_clause_hours_and_days():
    assert "24 hours" in metrics_service._window_clause(None, 24)
    assert "7 days" in metrics_service._window_clause(7, None)
    # hours takes precedence when both provided
    assert "3 hours" in metrics_service._window_clause(9, 3)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Returns queued result sets in call order."""

    def __init__(self, queued):
        self._queued = list(queued)

    async def execute(self, _query):
        return _FakeResult(self._queued.pop(0))


@pytest.mark.asyncio
async def test_summary_drift_status_healthy():
    session = _FakeSession([
        [{"n": 5, "avg_latency": 42.0}],
        [{"dataset_drift_detected": False, "drift_score": 0.05}],
    ])
    result = await metrics_service.summary(session, "3", 0.85)
    assert result["predictions_today"] == 5
    assert result["avg_latency_ms"] == 42.0
    assert result["drift_status"] == "healthy"
    assert result["model_f1"] == 0.85


@pytest.mark.asyncio
async def test_summary_drift_status_warning():
    session = _FakeSession([
        [{"n": 0, "avg_latency": 0}],
        [{"dataset_drift_detected": True, "drift_score": 0.20}],  # > 0.15 threshold
    ])
    result = await metrics_service.summary(session, "3", None)
    assert result["drift_status"] == "warning"


@pytest.mark.asyncio
async def test_summary_drift_status_critical():
    session = _FakeSession([
        [{"n": 1, "avg_latency": 10}],
        [{"dataset_drift_detected": True, "drift_score": 0.40}],  # > 2x threshold
    ])
    result = await metrics_service.summary(session, "3", None)
    assert result["drift_status"] == "critical"
