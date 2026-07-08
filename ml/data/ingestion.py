"""Dataset download and raw storage for SentinelML."""

from __future__ import annotations

from datetime import datetime, timezone
import logging

import pandas as pd

from ml.config import MLConfig
from ml.data.validation import validate_data

LOGGER = logging.getLogger(__name__)


def download_dataset(config: MLConfig) -> pd.DataFrame:
    """Download the configured binary sentiment dataset and persist it as parquet."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for dataset ingestion."
        ) from exc

    LOGGER.info("Downloading dataset '%s'.", config.dataset_name)
    dataset = load_dataset(config.dataset_name)

    if config.dataset_name.lower() == "imdb":
        splits = [dataset["train"].to_pandas(), dataset["test"].to_pandas()]
        df = pd.concat(splits, ignore_index=True)
    elif config.dataset_name.lower() in {"sst2", "glue"}:
        glue_dataset = load_dataset("glue", "sst2")
        df = pd.concat(
            [
                glue_dataset["train"].to_pandas(),
                glue_dataset["validation"].to_pandas(),
            ],
            ignore_index=True,
        ).rename(columns={"sentence": "text"})
    else:
        first_split = next(iter(dataset.keys()))
        df = dataset[first_split].to_pandas()

    df = _normalize_columns(df)
    if config.max_samples and len(df) > config.max_samples:
        sampled_parts = []
        per_class_limit = max(1, config.max_samples // max(1, df["label"].nunique()))
        for _, group in df.groupby("label"):
            sampled_parts.append(
                group.sample(
                    min(len(group), per_class_limit),
                    random_state=config.random_seed,
                )
            )
        df = (
            pd.concat(sampled_parts, ignore_index=True)
            .sample(frac=1.0, random_state=config.random_seed)
            .head(config.max_samples)
            .reset_index(drop=True)
        )

    report = validate_data(df)
    LOGGER.info("Validated raw data: %s", report)

    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = config.raw_data_dir / f"{config.dataset_name}_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)
    latest_path = config.raw_data_dir / "latest.parquet"
    df.to_parquet(latest_path, index=False)
    LOGGER.info("Saved raw dataset to %s and %s.", output_path, latest_path)

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "sentence" in normalized.columns and "text" not in normalized.columns:
        normalized = normalized.rename(columns={"sentence": "text"})
    if "label" not in normalized.columns and "sentiment" in normalized.columns:
        normalized = normalized.rename(columns={"sentiment": "label"})
    available_columns = [column for column in ["text", "label"] if column in normalized.columns]
    return normalized[available_columns].copy()
