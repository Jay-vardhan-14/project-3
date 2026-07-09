# Phase 3 Audit

## Scope

Phase 3 implements Airflow Orchestration from build prompt steps 15-18:

- `sentiment_training_pipeline` DAG using Airflow TaskFlow API
- `drift_detection` DAG using Airflow TaskFlow API
- Drift detector module with Evidently integration and fallback detector
- Alert creation/retrieval helpers
- Airflow runtime requirements for ML/data/training/orchestration dependencies
- Phase 3 tests for drift detection, alerts, and DAG helper behavior

Phase 2 approved commit:

```text
e6b0f6c24e74b8c9cbc1152c3bd4fc59355aeed2
```

## Files Added Or Updated

| File | Purpose | Key functions/exports |
|---|---|---|
| `airflow/dags/training_pipeline.py` | Main training DAG | `sentiment_training_pipeline()` DAG with TaskFlow tasks: `ingest_data`, `validate_data`, `preprocess_data`, `train_baseline`, `train_transformer`, `evaluate_and_compare`, `register_best_model`, `deploy_model` |
| `airflow/dags/drift_detection.py` | Daily drift DAG | `drift_detection()` DAG with TaskFlow tasks: `collect_recent_predictions`, `load_reference_data`, `run_drift_detection`, `evaluate_drift`, `store_drift_report` |
| `airflow/requirements-airflow.txt` | Runtime dependencies installed in Airflow container | Includes pandas, sklearn, MLflow 2.14.3, Evidently, psycopg2, Torch, Transformers, Accelerate, and related dependencies |
| `ml/monitoring/drift_detector.py` | Drift detection | `DriftReport`, `detect_drift(...)`, `create_drift_summary(...)` |
| `ml/monitoring/alerts.py` | Alert helper logic | `check_and_create_alert(...)`, `resolve_alert(...)`, `get_active_alerts(...)` |
| `docker-compose.yml` | Airflow runtime wiring | Airflow installs `airflow/requirements-airflow.txt` on startup and mounts the requirements file |
| `pyproject.toml` | Dependency extras | Adds `orchestration` extra with Airflow, Evidently, MLflow, psycopg2, requests |
| `uv.lock` | Dependency lockfile | Updated for orchestration dependencies |
| `tests/test_drift_detector.py` | Drift detector test | Forces fallback detector and verifies structured drift report output |
| `tests/test_alerts.py` | Alert tests | Verifies warning and critical alert threshold behavior |
| `tests/test_airflow_dag_helpers.py` | DAG helper test | Parses training DAG with fake Airflow decorators and tests `_xcom_safe` |

## Training Pipeline DAG

DAG id:

```text
sentiment_training_pipeline
```

Schedule:

```text
manual / schedule_interval=None
```

Task order:

```text
ingest_data
  -> validate_data
  -> preprocess_data
  -> train_baseline
  -> train_transformer
  -> evaluate_and_compare
  -> register_best_model
  -> deploy_model
```

Behavior:

- `ingest_data`: downloads the configured dataset and saves raw parquet.
- `validate_data`: validates schema, nulls, empty text, labels, imbalance warnings, and duplicates.
- `preprocess_data`: cleans text, creates stratified train/validation/test parquet splits.
- `train_baseline`: calls the Phase 2 TF-IDF + Logistic Regression trainer and returns MLflow run metadata.
- `train_transformer`: calls the Phase 2 DistilBERT trainer and returns MLflow run metadata.
- `evaluate_and_compare`: compares baseline and transformer by `f1_macro`.
- `register_best_model`: calls `compare_and_promote(...)` to register and promote only if the winner beats current Production.
- `deploy_model`: attempts to call `POST /api/v1/model/reload` on the serving API; logs a warning and returns `reload_failed` if Phase 4 serving is not available yet.

Retry policy:

```text
retries=2
retry_delay=5 minutes
retry_exponential_backoff=True
```

## Drift Detection DAG

DAG id:

```text
drift_detection
```

Schedule:

```text
@daily
```

Task order:

```text
collect_recent_predictions
  -> load_reference_data
  -> run_drift_detection
  -> evaluate_drift
  -> store_drift_report
```

Behavior:

- `collect_recent_predictions`: queries recent prediction logs from PostgreSQL.
- `load_reference_data`: loads the training split and converts labels into a reference sentiment distribution.
- `run_drift_detection`: runs Evidently drift detection and saves an HTML report.
- `evaluate_drift`: creates warning/critical/model degradation alerts when thresholds are exceeded.
- `store_drift_report`: stores drift report metadata in PostgreSQL.

The drift DAG creates the following tables if missing:

- `predictions`
- `drift_reports`
- `alerts`

## Drift Detector

`ml/monitoring/drift_detector.py` first attempts Evidently:

- `DataDriftPreset`
- `ColumnDriftMetric` for `predicted_sentiment` when available
- HTML report saved under `data/processed/drift_reports`

If Evidently’s API fails or changes, the detector falls back to a deterministic distribution-distance implementation so the DAG still produces structured drift output.

`DriftReport` fields:

- `dataset_drift_detected`
- `drift_score`
- `features_drifted`
- `total_features`
- `prediction_drift_detected`
- `reference_size`
- `current_size`
- `report_path`
- `details`

## Alerts

`check_and_create_alert(...)` behavior:

- `drift_score > 2 * drift_threshold`: `drift_critical`
- `drift_score > drift_threshold`: `drift_warning`
- `prediction_drift_detected`: `model_degradation`
- Otherwise no alert is created

Alert helpers can operate with a psycopg2-style DB session or return structured alert dictionaries without DB persistence.

## DistilBERT First-Run Note

The DAG uses the configured default model:

```text
distilbert-base-uncased
```

The first real DAG run will download the tokenizer/model weights unless they are already cached in the Airflow container. That first run can take noticeably longer.

The Phase 2 `train_transformer` safeguards remain in force:

- CPU uses `min(config.max_samples, 5000)`
- CPU uses `min(config.num_epochs, 2)`
- GPU runs the configured sample/epoch values

These caps are intended to keep first CPU DAG runs bounded while still exercising the real transformer training path.

## Verification

Full test suite with training and orchestration extras:

```text
uv run --extra training --extra orchestration pytest tests -q
.....................                                                    [100%]
```

Default test suite:

```text
uv run pytest tests -q
.....................                                                    [100%]
```

Compose config:

```text
docker-compose -f docker-compose.yml -f docker-compose.verify.yml config --quiet
# passed
```

No `print()` calls found in project Python/TS source:

```text
rg "print\(" ml tests airflow serving dashboard -g "*.py" -g "*.tsx"
# no matches
```

## Airflow Runtime Check

A direct local Python import of Airflow DAGs is not reliable on native Windows because Airflow imports POSIX-only modules such as `fcntl`. The test suite covers DAG helper behavior with fake Airflow decorators.

Attempting to start the Airflow container after the code changes failed because Docker Desktop/Linux engine was no longer reachable:

```text
unable to get image 'ghcr.io/mlflow/mlflow:v2.14.3': failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

Before Docker stopped, Phase 2 had already verified the live `db`, `redis`, and `mlflow` services with `docker-compose.verify.yml`. A full Airflow container startup/DAG list should be run once Docker is available again.

Recommended command:

```bash
docker-compose -f docker-compose.yml -f docker-compose.verify.yml up -d airflow
docker-compose -f docker-compose.yml -f docker-compose.verify.yml logs airflow --tail=200
```

## Known Phase Boundary

`deploy_model` calls the Phase 4 serving reload endpoint:

```text
POST /api/v1/model/reload
```

That endpoint is not implemented until Phase 4. For now, the task logs a warning and returns `reload_failed` instead of failing the DAG.
