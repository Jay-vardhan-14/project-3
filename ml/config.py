"""Configuration objects for SentinelML training and monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class MLConfig:
    """Central configuration for data, training, drift, and serving."""

    dataset_name: str = "imdb"
    max_samples: int = 25000
    test_size: float = 0.1
    val_size: float = 0.1
    random_seed: int = 42

    tfidf_max_features: int = 50000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    logreg_C: float = 1.0
    logreg_max_iter: int = 1000

    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500

    drift_threshold: float = 0.15
    drift_window_days: int = 7

    model_registry_name: str = "sentiment-model"

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/splits")


DEFAULT_CONFIG = MLConfig()
