"""DistilBERT sequence classification training."""

from __future__ import annotations

import logging
import inspect
import time
from typing import Any

import numpy as np

from ml.config import DEFAULT_CONFIG, MLConfig
from ml.models.evaluation import generate_classification_report
from ml.tracking.mlflow_utils import (
    create_artifact_dir,
    log_confusion_matrix,
    log_training_run,
    setup_mlflow,
    write_classification_report,
)
from ml.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)


def train_transformer(config: MLConfig = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fine-tune DistilBERT for binary sentiment classification."""

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Transformer training requires torch, datasets, and transformers."
        ) from exc

    set_seed(config.random_seed)
    train_df, val_df, test_df = _load_preprocessed_splits(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, val_df, test_df, effective_epochs = _apply_runtime_limits(
        train_df, val_df, test_df, config, device
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )

    train_dataset = _tokenize_dataset(Dataset.from_pandas(train_df), tokenizer, config)
    val_dataset = _tokenize_dataset(Dataset.from_pandas(val_df), tokenizer, config)
    test_dataset = _tokenize_dataset(Dataset.from_pandas(test_df), tokenizer, config)

    artifact_dir = create_artifact_dir("distilbert")
    training_args_kwargs = {
        "output_dir": str(artifact_dir / "checkpoints"),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": effective_epochs,
        "warmup_steps": config.warmup_steps,
        "seed": config.random_seed,
        "report_to": [],
        "logging_steps": 10,
    }
    strategy_parameter = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_args_kwargs[strategy_parameter] = "epoch"
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_transformer_metrics,
    )

    setup_mlflow("distilbert-sentiment")
    start_time = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - start_time

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    metrics = _compute_transformer_metrics((predictions.predictions, y_true))
    metrics.update(
        {
            "training_time": float(training_time),
            "model_size_mb": _estimate_transformer_size_mb(model),
            "inference_latency_ms": _compute_transformer_latency(trainer, test_dataset),
            "token_or_feature_count": int(tokenizer.vocab_size),
            "device": device,
            "num_epochs": effective_epochs,
        }
    )

    report = generate_classification_report(y_true, y_pred)
    confusion_path = log_confusion_matrix(
        y_true,
        y_pred,
        labels=["negative", "positive"],
        filename=artifact_dir / "confusion_matrix.png",
    )
    report_path = write_classification_report(report, artifact_dir / "classification_report.json")

    params = {
        "model_type": "distilbert",
        "dataset_name": config.dataset_name,
        "max_samples": len(train_df) + len(val_df) + len(test_df),
        "random_seed": config.random_seed,
        "model_name": config.model_name,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": effective_epochs,
        "warmup_steps": config.warmup_steps,
        "device": device,
    }
    run_id = log_training_run(
        params=params,
        metrics=metrics,
        artifacts={
            "confusion_matrix": confusion_path,
            "classification_report": report_path,
        },
        model=model,
        model_name=config.model_registry_name,
        flavor="pytorch",
        experiment_name="distilbert-sentiment",
    )
    metrics["mlflow_run_id"] = run_id
    LOGGER.info("Finished transformer training with run_id=%s.", run_id)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}, metrics


def _tokenize_dataset(dataset: Any, tokenizer: Any, config: MLConfig) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    columns_to_remove = [column for column in ["text", "__index_level_0__"] if column in tokenized.column_names]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def _load_preprocessed_splits(config: MLConfig) -> tuple[Any, Any, Any]:
    import pandas as pd

    train_path = config.splits_dir / "train.parquet"
    val_path = config.splits_dir / "val.parquet"
    test_path = config.splits_dir / "test.parquet"
    missing = [str(path) for path in [train_path, val_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed split files: {missing}")
    return (
        pd.read_parquet(train_path),
        pd.read_parquet(val_path),
        pd.read_parquet(test_path),
    )


def _compute_transformer_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def _apply_runtime_limits(train_df: Any, val_df: Any, test_df: Any, config: MLConfig, device: str) -> tuple[Any, Any, Any, int]:
    if device == "cuda":
        return train_df, val_df, test_df, config.num_epochs

    max_total = min(config.max_samples, 5000)
    total_rows = len(train_df) + len(val_df) + len(test_df)
    if total_rows <= max_total:
        return train_df, val_df, test_df, min(config.num_epochs, 2)

    train_fraction = len(train_df) / total_rows
    val_fraction = len(val_df) / total_rows
    train_limit = max(2, int(max_total * train_fraction))
    val_limit = max(2, int(max_total * val_fraction))
    test_limit = max(2, max_total - train_limit - val_limit)
    return (
        train_df.sample(n=min(train_limit, len(train_df)), random_state=config.random_seed),
        val_df.sample(n=min(val_limit, len(val_df)), random_state=config.random_seed),
        test_df.sample(n=min(test_limit, len(test_df)), random_state=config.random_seed),
        min(config.num_epochs, 2),
    )


def _estimate_transformer_size_mb(model: Any) -> float:
    parameters = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffers = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return float((parameters + buffers) / (1024 * 1024))


def _compute_transformer_latency(trainer: Any, test_dataset: Any) -> float:
    sample_size = min(8, len(test_dataset))
    if sample_size == 0:
        return 0.0
    sample = test_dataset.select(range(sample_size))
    start = time.perf_counter()
    trainer.predict(sample)
    elapsed = time.perf_counter() - start
    return float((elapsed / sample_size) * 1000)
