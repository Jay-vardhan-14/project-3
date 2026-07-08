from __future__ import annotations

import pandas as pd

from ml.config import MLConfig
from ml.data.preprocessing import clean_text, create_tfidf_features, preprocess_dataset


def test_clean_text_removes_html_urls_and_lowercases():
    text = "LOVE <b>This</b>!!! Visit https://example.com now 🤖"

    assert clean_text(text) == "love this!!! visit now"


def test_preprocess_dataset_stratified_split_maintains_class_ratio(tmp_path):
    df = pd.DataFrame(
        {
            "text": [f"positive sample {i}" for i in range(50)]
            + [f"negative sample {i}" for i in range(50)],
            "label": [1] * 50 + [0] * 50,
        }
    )
    config = MLConfig(splits_dir=tmp_path / "splits", test_size=0.1, val_size=0.1)

    splits = preprocess_dataset(df, config)

    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    for split_df in splits.values():
        assert split_df["label"].mean() == 0.5
    assert (tmp_path / "splits" / "train.parquet").exists()


def test_tfidf_creates_expected_shape_matrices():
    config = MLConfig(tfidf_max_features=10, tfidf_ngram_range=(1, 1))

    features = create_tfidf_features(
        ["good movie", "bad movie", "great acting", "bad acting"],
        ["good acting"],
        ["bad plot"],
        config,
    )

    assert features["train"].shape[0] == 4
    assert features["val"].shape[0] == 1
    assert features["test"].shape[0] == 1
    assert features["train"].shape[1] <= 10
