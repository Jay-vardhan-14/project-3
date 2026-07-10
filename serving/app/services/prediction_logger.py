"""Fire-and-forget prediction logging to PostgreSQL."""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Prediction
from app.services.predictor import PredictionResult

LOGGER = logging.getLogger(__name__)


async def log_prediction(
    session: AsyncSession,
    text: str,
    result: PredictionResult,
    model_name: str,
    model_version: str | None,
) -> None:
    """Persist a prediction. Errors are logged, never raised (don't slow inference)."""

    try:
        session.add(
            Prediction(
                input_text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                input_length=len(text),
                predicted_sentiment=result.sentiment,
                confidence=result.confidence,
                model_version=model_version or "unknown",
                model_name=model_name,
                latency_ms=result.latency_ms,
            )
        )
        await session.commit()
    except Exception:
        LOGGER.exception("Failed to log prediction.")
        await session.rollback()
