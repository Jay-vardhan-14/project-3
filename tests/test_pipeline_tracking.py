from __future__ import annotations

import json
from datetime import datetime, timezone

import ml.monitoring.pipeline_tracking as tracking


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rowcount = 1

    def execute(self, query: str, params: tuple = ()) -> None:
        self.calls.append((query, params))

    def close(self) -> None:
        pass


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        pass


def _patch(monkeypatch) -> _FakeConn:
    conn = _FakeConn()
    monkeypatch.setattr(tracking, "_connect", lambda _url: conn)
    return conn


def test_record_run_start_inserts_running_row(monkeypatch):
    conn = _patch(monkeypatch)
    tracking.record_run_start("db", "sentiment_training_pipeline", "run-1", datetime.now(timezone.utc))

    queries = " ".join(q for q, _ in conn.cursor_obj.calls)
    assert "INSERT INTO pipeline_runs" in queries
    assert "'running'" in queries
    assert conn.committed


def test_record_run_finish_serializes_metrics_and_updates(monkeypatch):
    conn = _patch(monkeypatch)
    tracking.record_run_finish("db", "sentiment_training_pipeline", "run-1", "success", {"winner": "baseline-logreg", "promoted": False})

    update_call = next((c for c in conn.cursor_obj.calls if c[0].strip().startswith("UPDATE")), None)
    assert update_call is not None, "expected an UPDATE on the running row"
    params = update_call[1]
    # metrics param must be a JSON string (JSONB can't adapt a raw dict).
    metrics_param = next(p for p in params if isinstance(p, str) and p.startswith("{"))
    assert json.loads(metrics_param)["winner"] == "baseline-logreg"


def test_record_run_finish_inserts_when_no_running_row(monkeypatch):
    conn = _patch(monkeypatch)
    conn.cursor_obj.rowcount = 0  # UPDATE matched nothing
    tracking.record_run_finish("db", "drift_detection", "run-2", "failed", {"error": "boom"})

    queries = [q.strip() for q, _ in conn.cursor_obj.calls]
    assert any(q.startswith("UPDATE") for q in queries)
    assert any(q.startswith("INSERT INTO pipeline_runs") for q in queries)
