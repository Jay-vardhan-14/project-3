"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import SETTINGS


class ModelRef(BaseModel):
    name: str
    version: str


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=SETTINGS.max_text_length)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=SETTINGS.max_batch_size)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model: ModelRef
    latency_ms: int


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    name: str
    version: str | None
    stage: str
    loaded: bool
    flavor: str | None = None
    f1_score: float | None = None
    accuracy: float | None = None
    load_time: str | None = None


class ReloadResponse(BaseModel):
    reloaded: bool
    name: str
    version: str | None
    previous_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    model_load_time: str | None = None
    db_connected: bool
    redis_connected: bool


class MetricsSummary(BaseModel):
    predictions_today: int
    avg_latency_ms: float
    model_version: str | None
    model_f1: float | None
    drift_status: str


class GenericPayload(BaseModel):
    """Loose wrapper for list/aggregate metrics endpoints."""

    data: Any
