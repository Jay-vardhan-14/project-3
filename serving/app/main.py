"""Minimal FastAPI app for Phase 1 infrastructure health checks."""

from __future__ import annotations

import logging

from fastapi import FastAPI

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="SentinelML Serving API")


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    """Return service health for Docker Compose orchestration."""

    LOGGER.debug("Health check requested.")
    return {"status": "ok"}
