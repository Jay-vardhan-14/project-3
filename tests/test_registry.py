from __future__ import annotations

from types import SimpleNamespace

from ml.tracking import registry


def test_compare_and_promote_promotes_better_model(monkeypatch):
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        registry,
        "_get_current_production_version",
        lambda model_name: SimpleNamespace(version="1", run_id="old-run"),
    )
    monkeypatch.setattr(registry, "_get_version_metrics", lambda model_name, version: {"f1_macro": 0.7})
    monkeypatch.setattr(
        registry,
        "register_model",
        lambda run_id, model_name, metrics: SimpleNamespace(version="2"),
    )
    monkeypatch.setattr(registry, "promote_model", lambda model_name, version, stage: calls.append((version, stage)))

    decision = registry.compare_and_promote("sentiment-model", "new-run", {"f1_macro": 0.8})

    assert decision["promoted"] is True
    assert decision["stage"] == "Production"
    assert calls == [("2", "Production")]


def test_compare_and_promote_keeps_worse_model_in_staging(monkeypatch):
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        registry,
        "_get_current_production_version",
        lambda model_name: SimpleNamespace(version="1", run_id="old-run"),
    )
    monkeypatch.setattr(registry, "_get_version_metrics", lambda model_name, version: {"f1_macro": 0.9})
    monkeypatch.setattr(
        registry,
        "register_model",
        lambda run_id, model_name, metrics: SimpleNamespace(version="2"),
    )
    monkeypatch.setattr(registry, "promote_model", lambda model_name, version, stage: calls.append((version, stage)))

    decision = registry.compare_and_promote("sentiment-model", "new-run", {"f1_macro": 0.8})

    assert decision["promoted"] is False
    assert decision["stage"] == "Staging"
    assert calls == [("2", "Staging")]


def test_compare_and_promote_promotes_first_registered_model(monkeypatch):
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(registry, "_get_current_production_version", lambda model_name: None)
    monkeypatch.setattr(registry, "_get_version_metrics", lambda model_name, version: None)
    monkeypatch.setattr(
        registry,
        "register_model",
        lambda run_id, model_name, metrics: SimpleNamespace(version="1"),
    )
    monkeypatch.setattr(registry, "promote_model", lambda model_name, version, stage: calls.append((version, stage)))

    decision = registry.compare_and_promote("sentiment-model", "new-run", {"f1_macro": 0.8})

    assert decision["promoted"] is True
    assert decision["old_f1"] == 0.0
    assert decision["stage"] == "Production"
    assert calls == [("1", "Production")]
