"""Test fixtures for the serving API.

Adds the serving dir to sys.path so the ``app`` package imports the same way it
does inside the container (WORKDIR /app), and provides an app whose ``get_db``
dependency is stubbed with an in-memory dummy session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SERVING_DIR = Path(__file__).resolve().parent.parent
if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))

# Point the app at addresses that refuse instantly so the TestClient lifespan
# (DB/Redis/MLflow startup) fails fast instead of stalling on DNS/HTTP retries
# to container hostnames when the Docker stack is running. Must be set before
# any ``app`` module (and its frozen SETTINGS) is imported.
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:1"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@127.0.0.1:1/none"


class FakePipeline:
    """sklearn-like pipeline returning a fixed positive-leaning probability."""

    classes_ = [0, 1]

    def predict_proba(self, texts):
        import numpy as np

        return np.array([[0.12, 0.88] for _ in texts])


@pytest.fixture
def loaded_predictor():
    """Put the global predictor into a loaded sklearn state and restore after."""

    from datetime import datetime, timezone

    from app.services.predictor import predictor

    saved = predictor.__dict__.copy()
    predictor._model = FakePipeline()
    predictor._tokenizer = None
    predictor._flavor = "sklearn"
    predictor._version = "3"
    predictor._tags = {"f1_macro": "0.8496", "accuracy": "0.85"}
    predictor._load_time = datetime.now(timezone.utc)
    yield predictor
    predictor.__dict__.clear()
    predictor.__dict__.update(saved)


class DummySession:
    """Minimal stand-in for AsyncSession used by prediction logging."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:  # noqa: D401
        return None

    async def rollback(self) -> None:
        return None


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app
    from app.services.predictor import predictor

    async def _get_db():
        yield DummySession()

    # Stub the lifespan model-load: without this, startup calls the global
    # predictor.load_model() which retries MLflow (unreachable in tests) with
    # exponential backoff, once per TestClient. Endpoint tests set model state
    # via the loaded_predictor fixture instead.
    original_load = predictor.load_model
    predictor.load_model = lambda: None
    app.dependency_overrides[get_db] = _get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        predictor.load_model = original_load
