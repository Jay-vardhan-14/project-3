"""Text cleaning, splitting, and feature creation."""

from __future__ import annotations

import html
import logging
import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from ml.config import MLConfig
from ml.data.validation import validate_data

LOGGER = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHARS_RE = re.compile(r"[^a-z0-9\s.,!?;:'\"()-]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize raw review text while preserving basic punctuation."""

    cleaned = html.unescape(str(text)).lower()
    cleaned = HTML_TAG_RE.sub("", cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = SPECIAL_CHARS_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_dataset(df: pd.DataFrame, config: MLConfig) -> dict[str, pd.DataFrame]:
    """Clean text, create stratified train/validation/test splits, and save them."""

    validate_data(df)
    processed = df[["text", "label"]].copy()
    processed["text"] = processed["text"].map(clean_text)
    validate_data(processed)

    train_val_df, test_df = train_test_split(
        processed,
        test_size=config.test_size,
        stratify=processed["label"],
        random_state=config.random_seed,
    )
    relative_val_size = config.val_size / (1.0 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=config.random_seed,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        output_path = config.splits_dir / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s split with %s rows to %s.", split_name, len(split_df), output_path)

    return splits


def create_tfidf_features(
    train_texts: list[str] | pd.Series,
    val_texts: list[str] | pd.Series,
    test_texts: list[str] | pd.Series,
    config: MLConfig,
) -> dict[str, Any]:
    """Fit a TF-IDF vectorizer on train text and transform all splits."""

    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=config.tfidf_ngram_range,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)
    LOGGER.info("Created TF-IDF matrices with %s features.", len(vectorizer.vocabulary_))
    return {
        "vectorizer": vectorizer,
        "train": x_train,
        "val": x_val,
        "test": x_test,
    }


def create_transformer_dataset(
    texts: list[str] | pd.Series,
    labels: list[int] | pd.Series,
    config: MLConfig,
) -> Any:
    """Create a tokenized HuggingFace Dataset for transformer training."""

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' and 'transformers' packages are required for transformer data."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = Dataset.from_dict({"text": list(texts), "labels": list(labels)})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized.remove_columns(["text"])
