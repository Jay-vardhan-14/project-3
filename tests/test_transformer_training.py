from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from ml.config import MLConfig
from ml.models import transformer


def test_transformer_training_path_completes_with_mocks(monkeypatch, tmp_path):
    train = pd.DataFrame({"text": ["good", "bad", "great", "awful"], "label": [1, 0, 1, 0]})
    val = pd.DataFrame({"text": ["nice", "poor"], "label": [1, 0]})
    test = pd.DataFrame({"text": ["excellent", "terrible"], "label": [1, 0]})
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    _install_fake_transformer_modules(monkeypatch)
    monkeypatch.setattr(transformer, "setup_mlflow", lambda experiment_name: "1")
    monkeypatch.setattr(transformer, "log_training_run", lambda **kwargs: "run-transformer")

    config = MLConfig(splits_dir=splits_dir, max_samples=8, num_epochs=1, batch_size=2)
    result, metrics = transformer.train_transformer(config)

    assert "model" in result
    assert metrics["mlflow_run_id"] == "run-transformer"
    assert metrics["f1_macro"] >= 0


def _install_fake_transformer_modules(monkeypatch):
    torch_module = ModuleType("torch")
    torch_module.manual_seed = lambda seed: None
    torch_module.cuda = SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch_module.backends = SimpleNamespace(
        cudnn=SimpleNamespace(deterministic=False, benchmark=True)
    )

    class FakeParameter:
        def numel(self):
            return 1

        def element_size(self):
            return 4

    class FakeModel:
        def parameters(self):
            return [FakeParameter()]

        def buffers(self):
            return []

    class FakeTokenizer:
        vocab_size = 10

        def __call__(self, texts, padding, truncation, max_length):
            return {
                "input_ids": [[1, 2]] * len(texts),
                "attention_mask": [[1, 1]] * len(texts),
            }

    class FakeDataset:
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)
            self.column_names = list(self.frame.columns)

        @classmethod
        def from_pandas(cls, frame):
            return cls(frame)

        def map(self, fn, batched):
            values = fn({"text": self.frame["text"].tolist()})
            mapped = self.frame.copy()
            for key, value in values.items():
                mapped[key] = value
            return FakeDataset(mapped)

        def remove_columns(self, columns):
            self.frame = self.frame.drop(columns=columns)
            self.column_names = list(self.frame.columns)
            return self

        def rename_column(self, old, new):
            self.frame = self.frame.rename(columns={old: new})
            self.column_names = list(self.frame.columns)
            return self

        def select(self, indices):
            return FakeDataset(self.frame.iloc[list(indices)].reset_index(drop=True))

        def __len__(self):
            return len(self.frame)

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.model = model
            self.compute_metrics = compute_metrics

        def train(self):
            return SimpleNamespace()

        def predict(self, dataset):
            labels = dataset.frame["labels"].to_numpy()
            logits = np.array([[2.0, 0.1] if label == 0 else [0.1, 2.0] for label in labels])
            return SimpleNamespace(predictions=logits, label_ids=labels)

    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = FakeDataset

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: FakeTokenizer())
    transformers_module.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda name, num_labels: FakeModel()
    )
    transformers_module.Trainer = FakeTrainer
    transformers_module.TrainingArguments = lambda **kwargs: SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
