"""Serving API settings, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the serving API."""

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@db:5432/sentinelml",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name: str = os.getenv("MODEL_NAME", "sentiment-model")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")

    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    metrics_cache_ttl_seconds: int = int(os.getenv("METRICS_CACHE_TTL_SECONDS", "60"))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))


SETTINGS = Settings()
