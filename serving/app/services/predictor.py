"""Model loading from the MLflow registry and inference.

Auto-detects the registered flavor: sklearn Pipeline (baseline) uses
``predict_proba``; a pytorch/transformer model uses the tokenizer + a softmax
over logits. Reuses ``ml.data.preprocessing.clean_text`` so serving cleaning
matches training exactly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import SETTINGS
from ml.data.preprocessing import clean_text

LOGGER = logging.getLogger(__name__)

LABELS = {0: "negative", 1: "positive"}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before a model is available."""


@dataclass
class PredictionResult:
    sentiment: str
    confidence: float
    latency_ms: int


class ModelPredictor:
    """Loads the Production model from MLflow and serves predictions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # ponytail: global lock; inference is CPU-bound and short
        self._model: Any = None
        self._tokenizer: Any = None
        self._flavor: str | None = None
        self._version: str | None = None
        self._run_id: str | None = None
        self._tags: dict[str, str] = {}
        self._load_time: datetime | None = None

    # -- loading -------------------------------------------------------------

    def load_model(self) -> None:
        """Load the current registry model for the configured stage."""

        import mlflow
        from mlflow.models import Model
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(SETTINGS.model_name, stages=[SETTINGS.model_stage])
        if not versions:
            raise ModelNotLoadedError(
                f"No {SETTINGS.model_stage} version for model '{SETTINGS.model_name}'."
            )
        version = versions[0]
        model_uri = f"models:/{SETTINGS.model_name}/{SETTINGS.model_stage}"
        flavors = Model.load(model_uri).flavors

        with self._lock:
            if "sklearn" in flavors:
                self._model = mlflow.sklearn.load_model(model_uri)
                self._tokenizer = None
                self._flavor = "sklearn"
            elif "pytorch" in flavors:
                self._model, self._tokenizer = _load_transformer(model_uri)
                self._flavor = "pytorch"
            else:
                raise ModelNotLoadedError(f"Unsupported model flavors: {sorted(flavors)}")
            self._version = str(version.version)
            self._run_id = version.run_id
            self._tags = dict(version.tags or {})
            self._load_time = datetime.now(timezone.utc)

        LOGGER.info(
            "Loaded model %s v%s (flavor=%s).", SETTINGS.model_name, self._version, self._flavor
        )

    def reload_model(self) -> dict[str, str | None]:
        """Reload the latest Production model, returning old/new version info."""

        previous = self._version
        self.load_model()
        return {"previous_version": previous, "version": self._version}

    # -- inference -----------------------------------------------------------

    def predict(self, text: str) -> PredictionResult:
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded.")
        start = time.perf_counter()
        index, confidence = self._infer_one(text)
        latency_ms = int(round((time.perf_counter() - start) * 1000))
        return PredictionResult(
            sentiment=LABELS.get(int(index), str(index)),
            confidence=round(float(confidence), 4),
            latency_ms=latency_ms,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]

    def _infer_one(self, text: str) -> tuple[int, float]:
        cleaned = clean_text(text)
        with self._lock:
            if self._flavor == "sklearn":
                proba = self._model.predict_proba([cleaned])[0]
                classes = list(self._model.classes_)
                best = int(proba.argmax())
                return int(classes[best]), float(proba[best])
            return self._infer_transformer(cleaned)

    def _infer_transformer(self, cleaned: str) -> tuple[int, float]:
        import torch

        inputs = self._tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        best = int(torch.argmax(probs).item())
        return best, float(probs[best].item())

    # -- metadata ------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def load_time_iso(self) -> str | None:
        return self._load_time.isoformat() if self._load_time else None

    def _tag_float(self, key: str) -> float | None:
        value = self._tags.get(key)
        return float(value) if value is not None else None

    def info(self) -> dict[str, Any]:
        return {
            "name": SETTINGS.model_name,
            "version": self._version,
            "stage": SETTINGS.model_stage,
            "loaded": self.loaded,
            "flavor": self._flavor,
            "f1_score": self._tag_float("f1_macro"),
            "accuracy": self._tag_float("accuracy"),
            "load_time": self.load_time_iso,
        }


def _load_transformer(model_uri: str) -> tuple[Any, Any]:
    # ponytail: serving image ships without torch/transformers; add them to the
    # `serving` extra if a transformer is ever promoted to Production.
    import mlflow
    from transformers import AutoTokenizer

    from ml.config import DEFAULT_CONFIG

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model_name)
    return model, tokenizer


predictor = ModelPredictor()
