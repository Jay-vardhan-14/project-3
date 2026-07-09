"""Model evaluation helpers."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

LOGGER = logging.getLogger(__name__)


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Compute classification metrics for a fitted model."""

    y_pred = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "support_negative": int(support[0]),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "support_positive": int(support[1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }
    LOGGER.info("Evaluation metrics: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


def compute_inference_latency(model: Any, sample_inputs: Any, n_runs: int = 100) -> float:
    """Compute average inference latency in milliseconds."""

    if n_runs <= 0:
        raise ValueError("n_runs must be greater than zero.")

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample_inputs)
    elapsed = time.perf_counter() - start
    return float((elapsed / n_runs) * 1000)


def compare_models(current_metrics: dict[str, Any] | None, new_metrics: dict[str, Any]) -> bool:
    """Return True when the new model beats the current model on macro F1."""

    if not current_metrics:
        return True
    return float(new_metrics.get("f1_macro", 0.0)) > float(current_metrics.get("f1_macro", 0.0))


def generate_classification_report(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return a structured sklearn classification report."""

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    return _to_builtin(report)


def estimate_model_size_mb(model: Any) -> float:
    """Estimate serialized model size in megabytes."""

    import pickle

    return len(pickle.dumps(model)) / (1024 * 1024)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
