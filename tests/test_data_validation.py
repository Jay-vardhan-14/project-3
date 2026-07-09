from __future__ import annotations

import pandas as pd
import pytest

from ml.data.validation import DataValidationError, validate_data


def test_validation_passes_balanced_dataset():
    df = pd.DataFrame(
        {"text": ["good", "bad", "great", "awful"], "label": [1, 0, 1, 0]}
    )

    report = validate_data(df)

    assert report.passed is True
    assert report.class_distribution == {0: 2, 1: 2}
    assert report.text_length["median"] > 0


def test_validation_catches_nulls():
    df = pd.DataFrame({"text": ["good", None], "label": [1, 0]})

    with pytest.raises(DataValidationError, match="null"):
        validate_data(df)


def test_validation_catches_empty_strings():
    df = pd.DataFrame({"text": ["good", " "], "label": [1, 0]})

    with pytest.raises(DataValidationError, match="empty text"):
        validate_data(df)


def test_validation_warns_on_imbalanced_data():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e"],
            "label": [1, 1, 1, 1, 0],
        }
    )

    report = validate_data(df)

    assert "Class balance is more skewed than 60/40." in report.warnings


def test_validation_warns_on_duplicate_rows():
    df = pd.DataFrame(
        {
            "text": ["good", "good", "bad", "bad"],
            "label": [1, 1, 0, 0],
        }
    )

    report = validate_data(df)

    assert "Dataset contains 2 duplicated rows." in report.warnings
