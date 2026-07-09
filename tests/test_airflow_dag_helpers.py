from __future__ import annotations

import importlib.util
import sys
from types import ModuleType


def test_training_dag_xcom_safe_converts_non_json_values(monkeypatch):
    _install_fake_airflow(monkeypatch)
    module = _load_module("airflow/dags/training_pipeline.py", "training_pipeline_test")

    result = module._xcom_safe({"ok": 1, "nested": {"value": object()}})

    assert result["ok"] == 1
    assert isinstance(result["nested"]["value"], str)


def _install_fake_airflow(monkeypatch):
    airflow = ModuleType("airflow")
    decorators = ModuleType("airflow.decorators")

    def dag(*args, **kwargs):
        def wrapper(fn):
            return fn

        return wrapper

    def task(fn):
        def wrapper(*args, **kwargs):
            return {"task_id": fn.__name__}

        return wrapper

    decorators.dag = dag
    decorators.task = task
    airflow.decorators = decorators
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.decorators", decorators)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
