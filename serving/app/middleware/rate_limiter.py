"""Redis-backed fixed-window rate limiting for prediction endpoints."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from app.config import SETTINGS

LOGGER = logging.getLogger(__name__)


async def rate_limit(request: Request) -> None:
    """Reject requests exceeding the per-IP per-minute limit.

    Fails open: if Redis is unavailable the request is allowed through.
    ponytail: fixed 60s window; swap for a sliding window only if bursts at the
    window edge become a real problem.
    """

    client = getattr(request.app.state, "redis", None)
    if client is None:
        return

    ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:{ip}"
    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, 60)
    except Exception:
        LOGGER.exception("Rate limiter unavailable; allowing request.")
        return

    if count > SETTINGS.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
        )
