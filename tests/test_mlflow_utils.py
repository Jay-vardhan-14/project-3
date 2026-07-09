from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier

from ml.tracking import mlflow_utils


def test_log_training_run_creates_expected_run(tmp_path, monkeypatch):
    mlflow = pytest_importorskip("mlflow")

    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    experiment_name = "unit-mlflow"

    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame({"x": [0, 1]}), [0, 1])
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")

    run_id = mlflow_utils.log_training_run(
        params={"a": 1},
        metrics={"accuracy": 0.5, "nested": {"ignored": True}},
        artifacts={"reports": artifact},
        model=model,
        model_name="dummy-model",
        flavor="sklearn",
        experiment_name=experiment_name,
    )

    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert run.data.params["a"] == "1"
    assert run.data.metrics["accuracy"] == 0.5


def pytest_importorskip(module_name: str):
    import pytest

    return pytest.importorskip(module_name)
