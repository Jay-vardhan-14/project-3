from __future__ import annotations

import pandas as pd

from ml.config import MLConfig
from ml.monitoring.drift_detector import detect_drift


def test_detect_drift_fallback_returns_report(monkeypatch, tmp_path):
    reference = pd.DataFrame(
        {
            "predicted_sentiment": ["positive", "positive", "negative", "negative"],
            "input_length": [10, 12, 9, 11],
        }
    )
    current = pd.DataFrame(
        {
            "predicted_sentiment": ["positive", "positive", "positive", "positive"],
            "input_length": [10, 10, 10, 10],
        }
    )

    def fail_evidently(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr("ml.monitoring.drift_detector._run_evidently", fail_evidently)
    config = MLConfig(processed_data_dir=tmp_path, drift_threshold=0.1)

    report = detect_drift(reference, current, config)

    assert report.current_size == 4
    assert report.reference_size == 4
    assert report.features_drifted >= 1
    assert report.report_path is not None
    assert (tmp_path / "drift_reports").exists()
