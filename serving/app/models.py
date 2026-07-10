"""SQLAlchemy models mirroring the PRD schema (section 7.3).

Column types and names match the DDL the drift DAG creates with
``CREATE TABLE IF NOT EXISTS`` so both writers share the same tables.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


_UUID_PK = dict(
    primary_key=True,
    server_default=text("gen_random_uuid()"),
)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), **_UUID_PK)
    input_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_length: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_sentiment: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), **_UUID_PK)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    dataset_drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drift_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    features_drifted: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_features: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_drift_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    reference_size: Mapped[int] = mapped_column(Integer, nullable=False)
    current_size: Mapped[int] = mapped_column(Integer, nullable=False)
    report_path: Mapped[str | None] = mapped_column(String(500))
    details: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), **_UUID_PK)
    alert_type: Mapped[str] = mapped_column(String(30), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_resolved: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    alert_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "alert_type IN ('drift_warning', 'drift_critical', 'model_degradation', "
            "'latency_spike', 'pipeline_failure')",
            name="alerts_alert_type_check",
        ),
        CheckConstraint(
            "severity IN ('info', 'warning', 'critical')",
            name="alerts_severity_check",
        ),
    )


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), **_UUID_PK)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String(50), nullable=False)
    f1_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    accuracy: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    replaced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), **_UUID_PK)
    dag_id: Mapped[str] = mapped_column(String(100), nullable=False)
    run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'success', 'failed')",
            name="pipeline_runs_status_check",
        ),
    )
