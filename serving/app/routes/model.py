"""Model info and hot-reload endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.config import SETTINGS
from app.schemas import ModelInfoResponse, ReloadResponse
from app.services.predictor import ModelNotLoadedError, predictor

LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**predictor.info())


@router.post("/model/reload", response_model=ReloadResponse)
async def model_reload() -> ReloadResponse:
    """Reload the Production model from the registry (called by the training DAG)."""

    try:
        outcome = predictor.reload_model()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Model reload failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return ReloadResponse(
        reloaded=True,
        name=SETTINGS.model_name,
        version=outcome["version"],
        previous_version=outcome["previous_version"],
    )
