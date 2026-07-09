from __future__ import annotations

from sklearn.dummy import DummyClassifier

from ml.models.evaluation import compare_models, compute_inference_latency, evaluate_model


def test_evaluate_model_computes_expected_metric_keys():
    model = DummyClassifier(strategy="most_frequent")
    x_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, [[0], [1]], [0, 1])

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics


def test_compute_inference_latency_returns_milliseconds():
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [0, 1])

    latency = compute_inference_latency(model, [[0]], n_runs=2)

    assert latency >= 0


def test_compare_models_uses_macro_f1_gate():
    assert compare_models({"f1_macro": 0.8}, {"f1_macro": 0.81}) is True
    assert compare_models({"f1_macro": 0.8}, {"f1_macro": 0.8}) is False
    assert compare_models(None, {"f1_macro": 0.1}) is True
