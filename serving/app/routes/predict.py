"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SETTINGS
from app.database import get_db
from app.middleware.rate_limiter import rate_limit
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelRef,
    PredictRequest,
    PredictResponse,
)
from app.services.predictor import ModelNotLoadedError, PredictionResult, predictor
from app.services.prediction_logger import log_prediction

router = APIRouter()


def _to_response(result: PredictionResult) -> PredictResponse:
    return PredictResponse(
        sentiment=result.sentiment,
        confidence=result.confidence,
        model=ModelRef(name=SETTINGS.model_name, version=predictor.version or "unknown"),
        latency_ms=result.latency_ms,
    )


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(rate_limit)])
async def predict(payload: PredictRequest, db: AsyncSession = Depends(get_db)) -> PredictResponse:
    try:
        result = predictor.predict(payload.text)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    await log_prediction(db, payload.text, result, SETTINGS.model_name, predictor.version)
    return _to_response(result)


@router.post("/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(rate_limit)])
async def predict_batch(
    payload: BatchPredictRequest, db: AsyncSession = Depends(get_db)
) -> BatchPredictResponse:
    for item in payload.texts:
        if not item or len(item) > SETTINGS.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Each text must be 1..{SETTINGS.max_text_length} characters.",
            )
    try:
        results = predictor.predict_batch(payload.texts)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    for text_item, result in zip(payload.texts, results):
        await log_prediction(db, text_item, result, SETTINGS.model_name, predictor.version)
    return BatchPredictResponse(results=[_to_response(result) for result in results])
