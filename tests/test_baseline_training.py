from __future__ import annotations

import pandas as pd

from ml.config import MLConfig
from ml.models import baseline


def test_baseline_training_produces_valid_metrics(monkeypatch, tmp_path):
    train = pd.DataFrame(
        {
            "text": [
                "excellent great good",
                "wonderful good nice",
                "bad awful terrible",
                "poor bad awful",
                "great excellent nice",
                "terrible poor bad",
            ],
            "label": [1, 1, 0, 0, 1, 0],
        }
    )
    val = pd.DataFrame({"text": ["great good", "bad awful"], "label": [1, 0]})
    test = pd.DataFrame({"text": ["excellent good", "terrible bad"], "label": [1, 0]})
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    monkeypatch.setattr(baseline, "setup_mlflow", lambda experiment_name: "1")
    monkeypatch.setattr(baseline, "log_training_run", lambda **kwargs: "run-123")

    config = MLConfig(
        splits_dir=splits_dir,
        tfidf_max_features=100,
        tfidf_ngram_range=(1, 1),
        logreg_max_iter=200,
    )
    model, metrics = baseline.train_baseline(config)

    assert model.predict(["excellent nice"])[0] == 1
    assert metrics["accuracy"] >= 0.7
    assert metrics["mlflow_run_id"] == "run-123"
    assert metrics["token_or_feature_count"] > 0
