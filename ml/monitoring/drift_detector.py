"""Data drift detection for SentinelML."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import DEFAULT_CONFIG, MLConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftReport:
    """Structured drift detection result."""

    dataset_drift_detected: bool
    drift_score: float
    features_drifted: int
    total_features: int
    prediction_drift_detected: bool
    reference_size: int
    current_size: int
    report_path: str | None
    details: dict[str, Any] = field(default_factory=dict)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: MLConfig = DEFAULT_CONFIG,
) -> DriftReport:
    """Detect drift between reference training data and current prediction data."""

    if reference_df.empty:
        raise ValueError("Reference dataset is empty.")
    if current_df.empty:
        raise ValueError("Current dataset is empty.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = config.processed_data_dir / "drift_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"drift_report_{timestamp}.html"

    try:
        report = _run_evidently(reference_df, current_df, report_path)
        summary = create_drift_summary(report)
        LOGGER.info("Evidently drift detection complete: %s", summary)
        return DriftReport(
            dataset_drift_detected=bool(summary["dataset_drift_detected"]),
            drift_score=float(summary["drift_score"]),
            features_drifted=int(summary["features_drifted"]),
            total_features=int(summary["total_features"]),
            prediction_drift_detected=bool(summary["prediction_drift_detected"]),
            reference_size=int(len(reference_df)),
            current_size=int(len(current_df)),
            report_path=str(report_path),
            details=summary.get("details", {}),
        )
    except Exception as exc:
        LOGGER.warning("Evidently drift detection failed, using fallback detector: %s", exc)
        return _fallback_drift_detection(reference_df, current_df, config, report_path)


def create_drift_summary(report: Any) -> dict[str, Any]:
    """Extract key drift metrics from an Evidently report object or dict."""

    if isinstance(report, dict):
        return report

    data = report.as_dict()
    metrics = data.get("metrics", [])
    features_drifted = 0
    total_features = 0
    prediction_drift_detected = False
    drift_score = 0.0
    details: dict[str, Any] = {"raw_metrics": metrics}

    for metric in metrics:
        result = metric.get("result", {})
        if "number_of_drifted_columns" in result:
            features_drifted = int(result.get("number_of_drifted_columns", 0))
            total_features = int(result.get("number_of_columns", 0))
            drift_share = result.get("share_of_drifted_columns")
            if drift_share is not None:
                drift_score = float(drift_share)
        if metric.get("metric") == "ColumnDriftMetric":
            column_name = result.get("column_name")
            if column_name in {"predicted_sentiment", "label"}:
                prediction_drift_detected = bool(result.get("drift_detected", False))

    return {
        "dataset_drift_detected": drift_score > 0,
        "drift_score": drift_score,
        "features_drifted": features_drifted,
        "total_features": total_features,
        "prediction_drift_detected": prediction_drift_detected,
        "details": details,
    }


def _run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame, report_path: Path) -> Any:
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric
    from evidently.report import Report

    metrics: list[Any] = [DataDriftPreset()]
    if "predicted_sentiment" in current_df.columns and "predicted_sentiment" in reference_df.columns:
        metrics.append(ColumnDriftMetric(column_name="predicted_sentiment"))

    report = Report(metrics=metrics)
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(report_path))
    return report


def _fallback_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: MLConfig,
    report_path: Path,
) -> DriftReport:
    comparable_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns and column not in {"created_at", "timestamp"}
    ]
    details: dict[str, Any] = {}
    drifted = 0

    for column in comparable_columns:
        reference_distribution = reference_df[column].astype(str).value_counts(normalize=True)
        current_distribution = current_df[column].astype(str).value_counts(normalize=True)
        categories = sorted(set(reference_distribution.index).union(current_distribution.index))
        distance = sum(
            abs(float(reference_distribution.get(category, 0.0)) - float(current_distribution.get(category, 0.0)))
            for category in categories
        ) / 2
        is_drifted = distance > config.drift_threshold
        drifted += int(is_drifted)
        details[column] = {
            "distance": distance,
            "drift_detected": is_drifted,
        }

    total_features = max(1, len(comparable_columns))
    drift_score = drifted / total_features
    prediction_drift = bool(details.get("predicted_sentiment", {}).get("drift_detected", False))
    report_payload = {
        "dataset_drift_detected": drift_score > config.drift_threshold,
        "drift_score": drift_score,
        "features_drifted": drifted,
        "total_features": total_features,
        "prediction_drift_detected": prediction_drift,
        "details": details,
    }
    report_path.write_text(
        "<html><body><pre>"
        + json.dumps(report_payload, indent=2, sort_keys=True)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    return DriftReport(
        dataset_drift_detected=bool(report_payload["dataset_drift_detected"]),
        drift_score=float(drift_score),
        features_drifted=int(drifted),
        total_features=int(total_features),
        prediction_drift_detected=prediction_drift,
        reference_size=int(len(reference_df)),
        current_size=int(len(current_df)),
        report_path=str(report_path),
        details=details,
    )
