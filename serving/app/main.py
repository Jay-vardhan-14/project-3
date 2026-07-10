"""SentinelML model serving API (Phase 4).

Lifespan wiring: ensure DB tables, connect Redis, and load the Production model
from the MLflow registry. Model-load failure is non-fatal — the service starts
in a degraded state and can be recovered via POST /api/v1/model/reload once a
model exists in the registry.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SETTINGS
from app.database import engine, init_models
from app.routes import health, metrics, model, predict
from app.services.predictor import predictor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_models()
    except Exception:
        LOGGER.exception("Table initialization failed at startup.")

    try:
        import redis.asyncio as redis

        app.state.redis = redis.from_url(SETTINGS.redis_url, decode_responses=True)
        await app.state.redis.ping()
        LOGGER.info("Connected to Redis.")
    except Exception:
        LOGGER.exception("Redis connection failed; caching and rate limiting disabled.")
        app.state.redis = None

    try:
        predictor.load_model()
    except Exception:
        LOGGER.exception("Model load failed at startup; serving is degraded until reload.")

    yield

    if getattr(app.state, "redis", None) is not None:
        await app.state.redis.aclose()
    await engine.dispose()


app = FastAPI(title="SentinelML Serving API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (health.router, predict.router, model.router, metrics.router):
    app.include_router(router, prefix="/api/v1")
