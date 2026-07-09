# Phase 2 Audit

## Scope

Phase 2 implements Model Training & MLflow from build prompt steps 9-14:

- MLflow experiment setup and run logging
- Confusion matrix and classification report artifacts
- Model evaluation and comparison helpers
- Baseline TF-IDF + Logistic Regression training
- DistilBERT transformer training path
- MLflow Model Registry registration and promotion logic
- Phase 2 unit tests, including the missing duplicate-detection test from Phase 1
- Live MLflow registry promotion verification against `localhost:5000`
- Real HuggingFace transformer sanity run on a tiny CPU subset

## Files Added Or Updated

| File | Purpose | Key functions/exports |
|---|---|---|
| `ml/tracking/mlflow_utils.py` | MLflow setup, run logging, and artifact helpers | `setup_mlflow(experiment_name)`, `log_training_run(...)`, `log_confusion_matrix(...)`, `write_classification_report(...)`, `create_artifact_dir(...)` |
| `ml/models/evaluation.py` | Shared model evaluation utilities | `evaluate_model(...)`, `compute_inference_latency(...)`, `compare_models(...)`, `generate_classification_report(...)`, `estimate_model_size_mb(...)` |
| `ml/models/baseline.py` | Baseline TF-IDF + Logistic Regression trainer | `train_baseline(config)`, `load_preprocessed_splits(config)` |
| `ml/models/transformer.py` | DistilBERT trainer | `train_transformer(config)` plus internal tokenization, metrics, CPU/runtime-limit, size, and latency helpers |
| `ml/tracking/registry.py` | MLflow Model Registry operations | `register_model(...)`, `promote_model(...)`, `get_production_model(...)`, `compare_and_promote(...)`, `compare_and_promote_default(...)` |
| `tests/test_data_validation.py` | Phase 1 gap fix | Added `test_validation_warns_on_duplicate_rows` |
| `tests/test_baseline_training.py` | Baseline training tests | Verifies baseline training returns usable model, valid metrics, mocked MLflow run id, and feature count |
| `tests/test_evaluation.py` | Evaluation helper tests | Verifies metrics shape, latency calculation, and F1 comparison gate |
| `tests/test_mlflow_utils.py` | MLflow logging tests | Uses SQLite MLflow backend and verifies params/metrics are logged |
| `tests/test_registry.py` | Registry promotion tests | Verifies better model promotes to Production and worse model stays in Staging |
| `tests/test_transformer_training.py` | Transformer training path test | Uses mocked Torch/Datasets/Transformers modules to verify orchestration without downloading DistilBERT |
| `pyproject.toml` | Dependency metadata | Aligns Python/dependency targets for MLflow 2.14 and HuggingFace training |
| `uv.lock` | Dependency lockfile | Updated after dependency metadata change |

Dependency alignment completed before Phase 3:

- Python target is `>=3.11,<3.12`, matching the PRD and serving Docker target.
- `mlflow==2.14.3` is pinned in the `training` extra to match the live MLflow server image.
- `pyarrow>=15,<16` is used for MLflow 2.14 compatibility.
- `setuptools>=68,<81` is pinned because MLflow 2.14 imports `pkg_resources`.
- `accelerate>=1.1.0` is included because HuggingFace `Trainer` requires it for PyTorch training.
- `matplotlib` is a base dependency because confusion matrix artifacts are core training behavior.

## Baseline Training

`ml/models/baseline.py` implements a real sklearn pipeline:

```text
TfidfVectorizer -> LogisticRegression
```

Behavior:

- Loads `train.parquet`, `val.parquet`, and `test.parquet` from `config.splits_dir`
- Calls `set_seed(config.random_seed)`
- Trains Logistic Regression using configured `C`, `max_iter`, TF-IDF max features, and n-gram range
- Evaluates on the test split
- Computes:
  - accuracy
  - macro precision, recall, F1
  - per-class precision, recall, F1, support
  - confusion matrix
  - training time
  - model size
  - inference latency
  - TF-IDF feature count
- Generates:
  - `confusion_matrix.png`
  - `classification_report.json`
- Logs params, metrics, artifacts, and sklearn model to MLflow

## Transformer Training

`ml/models/transformer.py` implements the DistilBERT fine-tuning path using HuggingFace:

- Loads split parquet files
- Calls `set_seed(config.random_seed)`
- Detects device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- Uses:
  - `AutoTokenizer.from_pretrained(config.model_name)`
  - `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`
  - HuggingFace `Trainer`
  - `TrainingArguments`
- Tokenizes datasets with `max_length`, truncation, and max-length padding
- Uses validation F1 as the best-model metric
- Evaluates on the test set
- Logs metrics/artifacts/model to MLflow

CPU safeguard:

- If running on CPU, training is capped at `min(config.max_samples, 5000)` and `min(config.num_epochs, 2)` for reasonable runtime.

## MLflow Utilities

`ml/tracking/mlflow_utils.py`:

- Uses Python logging, not `print()`
- Wraps MLflow setup/logging in `try/except`
- MLflow logging failures are logged and return `None`, so training does not crash if tracking is unavailable
- Confusion matrix generation forces matplotlib `Agg` backend for headless Docker/CI compatibility
- Supports sklearn, PyTorch, and pyfunc model logging flavors

## Evaluation

`ml/models/evaluation.py` provides:

- `evaluate_model(model, X_test, y_test) -> dict`
- `compute_inference_latency(model, sample_inputs, n_runs=100) -> float`
- `compare_models(current_metrics, new_metrics) -> bool`
- `generate_classification_report(y_true, y_pred) -> dict`
- `estimate_model_size_mb(model) -> float`

Promotion gate:

```python
new_metrics["f1_macro"] > current_metrics["f1_macro"]
```

Equal F1 does not promote.

## Registry Logic

`ml/tracking/registry.py` implements:

- Register model from `runs:/{run_id}/{model_name}`
- Save key metrics as model version tags
- Promote to `Production`
- Promote weaker models to `Staging`
- Load current Production model from `models:/{model_name}/Production`
- Archive existing Production versions when promoting a new Production model

Decision shape returned by `compare_and_promote(...)`:

```python
{
    "promoted": bool,
    "reason": str,
    "old_f1": float,
    "new_f1": float,
    "version": str,
    "stage": "Production" | "Staging",
}
```

## Tests

Current test command:

```text
uv run --extra training pytest tests -q
```

Output:

```text
.................                                                        [100%]
```

One warning is emitted by dependency internals:

```text
DeprecationWarning: The distutils package is deprecated and slated for removal in Python 3.12.
```

This warning comes from dependency internals, not project code.

## Test Coverage Notes

| Requirement | Covered? | Notes |
|---|---:|---|
| Duplicate detection test added | Yes | `test_validation_warns_on_duplicate_rows` |
| Baseline training produces valid metrics | Yes | Synthetic split data, mocked MLflow logging |
| Baseline model accuracy > 0.7 on small sample | Yes | Test asserts `metrics["accuracy"] >= 0.7` |
| Transformer training path completes | Yes | Mocked Torch/Datasets/Transformers to avoid network/model download |
| MLflow run logging creates expected params/metrics | Yes | SQLite MLflow backend |
| Model comparison logic promotes better F1 | Yes | Unit test |
| Model comparison logic rejects equal/worse F1 | Yes | Unit test |
| Registry operations tested without live registry | Yes | Unit tests are mocked, and a live E2E registry promotion check was also run against `localhost:5000`. |
| Real HuggingFace transformer path | Yes | Tiny CPU sanity run completed with real Torch/Transformers/Datasets/Trainer and logged to MLflow. |

## Live Registry Promotion Verification

Command target:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
```

The script registered a dummy sklearn model version `1`, promoted it to Production, registered a second dummy model with better `f1_macro`, then called project code:

```python
compare_and_promote(model_name, run_2, {"f1_macro": 0.82, "accuracy": 0.82})
```

Before:

```json
[
  {
    "f1_macro_tag": "0.7",
    "run_id": "4a6e8d88eaa04d5a996aec1e0dabecbd",
    "stage": "Production",
    "version": "1"
  }
]
```

After:

```json
[
  {
    "f1_macro_tag": "0.7",
    "run_id": "4a6e8d88eaa04d5a996aec1e0dabecbd",
    "stage": "Archived",
    "version": "1"
  },
  {
    "f1_macro_tag": "0.82",
    "run_id": "3936c092a52745559a95e2ebdce1a563",
    "stage": "Production",
    "version": "2"
  }
]
```

Decision:

```json
{
  "new_f1": 0.82,
  "old_f1": 0.7,
  "promoted": true,
  "reason": "New model beat current Production macro F1.",
  "stage": "Production",
  "version": "2"
}
```

Result:

```text
LIVE_REGISTRY_E2E_OK model_name=sentinelml-registry-e2e-fdbef49c
```

## Real Transformer Sanity Run

The mocked transformer unit test remains fast and CI-friendly. Separately, a real non-mocked sanity run was executed using:

- 200 synthetic samples
- 160 train / 20 validation / 20 test
- CPU
- 1 epoch
- `hf-internal-testing/tiny-random-distilbert`
- MLflow tracking at `http://localhost:5000`

Result:

```text
TRANSFORMER_SANITY_OK
{'run_id': 'fb25539218b54233ae85440a41b3b42d', 'accuracy': 0.55, 'f1_macro': 0.3548387096774194, 'device': 'cpu', 'num_epochs': 1}
```

This validates the real HuggingFace tokenizer/model/Trainer path and MLflow logging path before Phase 3 wraps training in Airflow.

## Live Infrastructure Status

Docker was verified after Docker Desktop started. The exact default host ports `5432` and `6379` were already occupied by another stack, so a temporary override was used:

```text
db:    15432 -> 5432
redis: 16379 -> 6379
mlflow: 5000 -> 5000
```

Current verified status:

```text
sentinelml-db       Up (healthy)   15432->5432
sentinelml-redis    Up (healthy)   16379->6379
sentinelml-mlflow   Up (healthy)   5000->5000
```

MLflow UI:

```text
StatusCode=200
```

PostgreSQL backend:

```text
current_database | current_user
------------------+--------------
mlflow           | postgres
```

MLflow logs:

- Database migrations completed
- Gunicorn started on `0.0.0.0:5000`
- No connection-error lines found in the last checked log output

## Quality Checks

No `print()` calls found in project Python/TS source scanned with:

```text
rg "print\(" ml tests serving dashboard -g "*.py" -g "*.tsx"
```

## Docker Override Decision

`docker-compose.verify.yml` is kept as a permanent local-dev convenience file, not a throwaway workaround. It is documented in `docs/setup-guide.md`.

Use it when another local stack already owns host ports `5432` or `6379`:

```bash
docker-compose -f docker-compose.yml -f docker-compose.verify.yml up -d db redis mlflow
```
