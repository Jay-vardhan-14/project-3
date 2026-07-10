"""Dashboard metrics endpoints.

Time windowing: pass ?hours=24 for hourly granularity or ?days=7 (default).
DB-backed aggregates are cached in Redis for a short TTL.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SETTINGS
from app.database import get_db
from app.schemas import MetricsSummary
from app.services import metrics_service
from app.services.predictor import predictor

LOGGER = logging.getLogger(__name__)

router = APIRouter()


async def _cached(request: Request, key: str, ttl: int, producer):
    client = getattr(request.app.state, "redis", None)
    if client is None:
        return await producer()
    try:
        hit = await client.get(key)
        if hit is not None:
            return json.loads(hit)
    except Exception:
        LOGGER.exception("Cache read failed for %s.", key)
    value = await producer()
    try:
        await client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception:
        LOGGER.exception("Cache write failed for %s.", key)
    return value


@router.get("/metrics/summary", response_model=MetricsSummary)
async def summary(db: AsyncSession = Depends(get_db)) -> MetricsSummary:
    info = predictor.info()
    data = await metrics_service.summary(db, info["version"], info["f1_score"])
    return MetricsSummary(**data)


@router.get("/metrics/predictions")
async def predictions(
    request: Request,
    days: int | None = Query(default=None),
    hours: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    key = f"metrics:predictions:{days}:{hours}"
    return await _cached(
        request, key, SETTINGS.metrics_cache_ttl_seconds,
        lambda: metrics_service.prediction_volume(db, days, hours),
    )


@router.get("/metrics/latency")
async def latency(
    days: int | None = Query(default=None),
    hours: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    return await metrics_service.latency_percentiles(db, days, hours)


@router.get("/metrics/drift")
async def drift(days: int | None = Query(default=30), db: AsyncSession = Depends(get_db)):
    return await metrics_service.drift_history(db, days)


@router.get("/metrics/distribution")
async def distribution(
    days: int | None = Query(default=None),
    hours: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    return await metrics_service.sentiment_distribution(db, days, hours)


@router.get("/metrics/distribution/confidence")
async def confidence(
    days: int | None = Query(default=None),
    hours: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    return await metrics_service.confidence_histogram(db, days, hours)


@router.get("/metrics/recent-predictions")
async def recent_predictions(limit: int = Query(default=20, le=200), db: AsyncSession = Depends(get_db)):
    return await metrics_service.recent_predictions(db, limit)


@router.get("/metrics/alerts")
async def alerts(db: AsyncSession = Depends(get_db)):
    return await metrics_service.alerts(db)


@router.get("/metrics/pipeline-runs")
async def pipeline_runs(db: AsyncSession = Depends(get_db)):
    return await metrics_service.pipeline_runs(db)


@router.get("/metrics/experiments")
async def experiments(request: Request):
    return await _cached(
        request, "metrics:experiments", SETTINGS.metrics_cache_ttl_seconds,
        lambda: _sync(metrics_service.experiments),
    )


@router.get("/metrics/models")
async def models(request: Request):
    return await _cached(
        request, "metrics:models", SETTINGS.metrics_cache_ttl_seconds,
        lambda: _sync(metrics_service.registry_models),
    )


async def _sync(func):
    """Run a blocking MLflow call off the event loop."""

    import anyio

    return await anyio.to_thread.run_sync(func)
