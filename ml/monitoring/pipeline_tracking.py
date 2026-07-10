"""Record DAG run lifecycle into the pipeline_runs table for dashboard observability.

Used by the Airflow DAGs: insert a 'running' row at start, update to
'success'/'failed' at the end. Kept dependency-light (psycopg2 imported lazily)
so it loads inside the Airflow workers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


def _connect(db_url: str) -> Any:
    import psycopg2

    return psycopg2.connect(db_url)


def _ensure_table(conn: Any) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dag_id VARCHAR(100) NOT NULL,
                run_id VARCHAR(100) NOT NULL,
                status VARCHAR(20) NOT NULL
                    CHECK (status IN ('running', 'success', 'failed')),
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                duration_seconds INTEGER,
                metrics JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.commit()
    finally:
        cursor.close()


def record_run_start(db_url: str, dag_id: str, run_id: str, started_at: datetime) -> None:
    """Insert a 'running' row for a DAG run (idempotent per run_id)."""

    conn = _connect(db_url)
    try:
        _ensure_table(conn)
        cursor = conn.cursor()
        try:
            # Clear any stale running row for this run (e.g. a task retry).
            cursor.execute(
                "DELETE FROM pipeline_runs WHERE dag_id = %s AND run_id = %s AND status = 'running'",
                (dag_id, run_id),
            )
            cursor.execute(
                "INSERT INTO pipeline_runs (dag_id, run_id, status, started_at) VALUES (%s, %s, 'running', %s)",
                (dag_id, run_id, started_at),
            )
            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()
    LOGGER.info("Recorded pipeline start: %s %s", dag_id, run_id)


def record_run_finish(
    db_url: str,
    dag_id: str,
    run_id: str,
    status: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Finalize a DAG run row: set terminal status, completion time, duration, metrics."""

    conn = _connect(db_url)
    try:
        _ensure_table(conn)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE pipeline_runs
                SET status = %s,
                    completed_at = NOW(),
                    duration_seconds = GREATEST(0, EXTRACT(EPOCH FROM (NOW() - started_at))::int),
                    metrics = %s
                WHERE dag_id = %s AND run_id = %s AND status = 'running'
                """,
                (status, json.dumps(metrics or {}), dag_id, run_id),  # JSONB: dict must be json.dumps'd
            )
            if cursor.rowcount == 0:
                # No running row to update (start not recorded) — insert a terminal row.
                cursor.execute(
                    "INSERT INTO pipeline_runs (dag_id, run_id, status, started_at, completed_at, duration_seconds, metrics) "
                    "VALUES (%s, %s, %s, NOW(), NOW(), 0, %s)",
                    (dag_id, run_id, status, json.dumps(metrics or {})),
                )
            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()
    LOGGER.info("Recorded pipeline finish: %s %s -> %s", dag_id, run_id, status)
