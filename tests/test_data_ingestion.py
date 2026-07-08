from __future__ import annotations

import pandas as pd

from ml.config import MLConfig
from ml.data import ingestion


class FakeSplit:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


def test_ingestion_downloads_valid_dataframe(monkeypatch, tmp_path):
    train = pd.DataFrame(
        {
            "text": ["Great movie", "Bad movie", "Loved it", "Awful"],
            "label": [1, 0, 1, 0],
        }
    )
    test = pd.DataFrame({"text": ["Nice", "Poor"], "label": [1, 0]})

    def fake_load_dataset(name):
        assert name == "imdb"
        return {"train": FakeSplit(train), "test": FakeSplit(test)}

    monkeypatch.setattr(ingestion, "load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "datasets", type("Datasets", (), {"load_dataset": fake_load_dataset}))

    config = MLConfig(max_samples=4, raw_data_dir=tmp_path / "raw")
    df = ingestion.download_dataset(config)

    assert set(df.columns) == {"text", "label"}
    assert len(df) == 4
    assert (tmp_path / "raw" / "latest.parquet").exists()
