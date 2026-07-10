"""Predictor loading, inference, and reload tests."""

from __future__ import annotations

import sys
import types

import pytest

from app.services.predictor import ModelNotLoadedError, ModelPredictor
from tests.conftest import FakePipeline


def test_predict_before_load_raises():
    predictor = ModelPredictor()
    with pytest.raises(ModelNotLoadedError):
        predictor.predict("anything")


def test_sklearn_inference(loaded_predictor):
    result = loaded_predictor.predict("a great movie")
    assert result.sentiment == "positive"
    assert result.confidence == 0.88
    assert result.latency_ms >= 0


def _patch_mlflow(monkeypatch, flavors, version="1", run_id="run-abc"):
    """Stub the mlflow surface ModelPredictor.load_model touches."""

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_mlflow.sklearn = types.SimpleNamespace(load_model=lambda uri: FakePipeline())
    fake_mlflow.pytorch = types.SimpleNamespace(load_model=lambda uri: None)

    fake_models = types.ModuleType("mlflow.models")
    fake_models.Model = types.SimpleNamespace(
        load=lambda uri: types.SimpleNamespace(flavors=flavors)
    )

    class FakeVersion:
        def __init__(self):
            self.version = version
            self.run_id = run_id
            self.tags = {"f1_macro": "0.85"}

    class FakeClient:
        def get_latest_versions(self, name, stages):
            return [FakeVersion()]

    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = FakeClient

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)


def test_load_model_detects_sklearn(monkeypatch):
    _patch_mlflow(monkeypatch, flavors={"sklearn": {}, "python_function": {}})
    predictor = ModelPredictor()
    predictor.load_model()
    assert predictor.loaded
    assert predictor.version == "1"
    assert predictor.info()["flavor"] == "sklearn"


def test_reload_reports_previous_version(monkeypatch):
    _patch_mlflow(monkeypatch, flavors={"sklearn": {}}, version="1")
    predictor = ModelPredictor()
    predictor.load_model()
    _patch_mlflow(monkeypatch, flavors={"sklearn": {}}, version="2")
    outcome = predictor.reload_model()
    assert outcome["previous_version"] == "1"
    assert outcome["version"] == "2"


def test_load_model_raises_when_no_version(monkeypatch):
    fake_tracking = types.ModuleType("mlflow.tracking")

    class EmptyClient:
        def get_latest_versions(self, name, stages):
            return []

    fake_tracking.MlflowClient = EmptyClient
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_models = types.ModuleType("mlflow.models")
    fake_models.Model = types.SimpleNamespace(load=lambda uri: None)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)

    with pytest.raises(ModelNotLoadedError):
        ModelPredictor().load_model()
