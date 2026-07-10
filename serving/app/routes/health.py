"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.database import engine
from app.schemas import HealthResponse
from app.services.predictor import predictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    db_connected = True
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        db_connected = False

    redis_connected = False
    client = getattr(request.app.state, "redis", None)
    if client is not None:
        try:
            redis_connected = bool(await client.ping())
        except Exception:
            redis_connected = False

    return HealthResponse(
        status="ok" if (db_connected and predictor.loaded) else "degraded",
        model_loaded=predictor.loaded,
        model_version=predictor.version,
        model_load_time=predictor.load_time_iso,
        db_connected=db_connected,
        redis_connected=redis_connected,
    )
