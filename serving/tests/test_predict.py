"""Prediction endpoint tests: validation, batch, and rate limiting."""

from __future__ import annotations

from app.config import SETTINGS


def test_predict_returns_sentiment(client, loaded_predictor):
    response = client.post("/api/v1/predict", json={"text": "a wonderful film"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model"]["version"] == "3"
    assert isinstance(body["latency_ms"], int)


def test_predict_rejects_empty_text(client, loaded_predictor):
    assert client.post("/api/v1/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_too_long_text(client, loaded_predictor):
    long_text = "x" * (SETTINGS.max_text_length + 1)
    assert client.post("/api/v1/predict", json={"text": long_text}).status_code == 422


def test_predict_batch(client, loaded_predictor):
    response = client.post("/api/v1/predict/batch", json={"texts": ["good", "bad"]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_predict_batch_rejects_oversize(client, loaded_predictor):
    texts = ["x"] * (SETTINGS.max_batch_size + 1)
    assert client.post("/api/v1/predict/batch", json={"texts": texts}).status_code == 422


def test_predict_returns_503_when_model_missing(client):
    # No loaded_predictor fixture -> model is None.
    from app.services.predictor import predictor

    predictor._model = None
    assert client.post("/api/v1/predict", json={"text": "hi"}).status_code == 503


def test_rate_limit_returns_429(client, loaded_predictor):
    class OverLimitRedis:
        async def incr(self, key):
            return SETTINGS.rate_limit_per_minute + 1

        async def expire(self, key, ttl):
            return True

    client.app.state.redis = OverLimitRedis()
    try:
        response = client.post("/api/v1/predict", json={"text": "hello"})
        assert response.status_code == 429
    finally:
        client.app.state.redis = None
