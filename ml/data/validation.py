"""Data quality checks for sentiment datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"text", "label"}


class DataValidationError(ValueError):
    """Raised when a dataset fails critical validation checks."""


@dataclass(frozen=True)
class ValidationReport:
    """Structured data validation result."""

    passed: bool
    row_count: int
    null_count: int
    duplicate_count: int
    empty_text_count: int
    class_distribution: dict[int, int]
    class_ratio: dict[int, float]
    text_length: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def validate_data(df: pd.DataFrame) -> ValidationReport:
    """Validate required schema, data quality, and class balance."""

    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        errors.append("Dataset is empty.")

    if errors:
        report = _build_report(df, warnings, errors)
        LOGGER.error("Critical data validation failure: %s", errors)
        raise DataValidationError("; ".join(errors))

    null_count = int(df[["text", "label"]].isna().sum().sum())
    if null_count:
        errors.append(f"Dataset contains {null_count} null values.")

    empty_text_count = int(df["text"].astype(str).str.strip().eq("").sum())
    if empty_text_count:
        errors.append(f"Dataset contains {empty_text_count} empty text values.")

    label_values = set(df["label"].dropna().unique().tolist())
    if not label_values.issubset({0, 1}):
        errors.append(f"Labels must be binary 0/1; found {sorted(label_values)}.")

    class_ratio = _class_ratio(df)
    if class_ratio and max(class_ratio.values()) > 0.6:
        warnings.append("Class balance is more skewed than 60/40.")

    duplicate_count = int(df.duplicated(subset=["text", "label"]).sum())
    if duplicate_count:
        warnings.append(f"Dataset contains {duplicate_count} duplicated rows.")

    report = _build_report(df, warnings, errors)
    if errors:
        LOGGER.error("Data validation failed: %s", errors)
        raise DataValidationError("; ".join(errors))

    LOGGER.info(
        "Data validation passed for %s rows. Class distribution: %s",
        report.row_count,
        report.class_distribution,
    )
    return report


def _build_report(
    df: pd.DataFrame, warnings: list[str], errors: list[str]
) -> ValidationReport:
    text_series = df["text"].astype(str) if "text" in df.columns else pd.Series(dtype=str)
    text_lengths = text_series.str.len()

    return ValidationReport(
        passed=not errors,
        row_count=int(len(df)),
        null_count=int(df.isna().sum().sum()),
        duplicate_count=int(df.duplicated().sum()) if not df.empty else 0,
        empty_text_count=int(text_series.str.strip().eq("").sum()) if not text_series.empty else 0,
        class_distribution=_class_distribution(df),
        class_ratio=_class_ratio(df),
        text_length={
            "min": float(text_lengths.min()) if not text_lengths.empty else 0.0,
            "max": float(text_lengths.max()) if not text_lengths.empty else 0.0,
            "mean": float(text_lengths.mean()) if not text_lengths.empty else 0.0,
            "median": float(text_lengths.median()) if not text_lengths.empty else 0.0,
        },
        warnings=warnings,
        errors=errors,
    )


def _class_distribution(df: pd.DataFrame) -> dict[int, int]:
    if "label" not in df.columns:
        return {}
    return {
        int(label): int(count)
        for label, count in df["label"].value_counts().sort_index().items()
        if pd.notna(label)
    }


def _class_ratio(df: pd.DataFrame) -> dict[int, float]:
    distribution = _class_distribution(df)
    total = sum(distribution.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in distribution.items()}
