# Phase 3 Audit

## Scope

Phase 3 implements Airflow Orchestration from build prompt steps 15-18:

- `sentiment_training_pipeline` DAG using Airflow TaskFlow API
- `drift_detection` DAG using Airflow TaskFlow API
- Drift detector module with Evidently integration and deterministic fallback
- Alert creation/retrieval helpers
- Airflow runtime image and Compose wiring
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
| `airflow/Dockerfile` | Airflow runtime image | Builds `sentinelml-airflow:latest` from `apache/airflow:2.9.3-python3.11` and installs ML/orchestration dependencies at image build time |
| `airflow/requirements-airflow.txt` | Airflow image dependencies | Pins Airflow-side datasets, MLflow, Evidently, sklearn, psycopg2, CPU Torch, Transformers, and Accelerate |
| `ml/monitoring/drift_detector.py` | Drift detection | `DriftReport`, `detect_drift(...)`, `create_drift_summary(...)` |
| `ml/monitoring/alerts.py` | Alert helper logic | `check_and_create_alert(...)`, `resolve_alert(...)`, `get_active_alerts(...)` |
| `ml/tracking/registry.py` | Registry promotion logic | Handles first-time model registration when no Production model exists yet |
| `docker-compose.yml` | Runtime wiring | Builds Airflow image, starts scheduler/webserver, sets `PYTHONPATH`, caps Airflow training defaults, and has MLflow repair artifact permissions |
| `pyproject.toml` | Dependency extras | Adds orchestration dependencies |
| `uv.lock` | Dependency lockfile | Updated for orchestration dependencies |
| `tests/test_drift_detector.py` | Drift detector tests | Verifies structured fallback drift report output |
| `tests/test_alerts.py` | Alert tests | Verifies warning and critical alert threshold behavior |
| `tests/test_airflow_dag_helpers.py` | DAG helper tests | Parses training DAG with fake Airflow decorators and tests `_xcom_safe` |

## Training Pipeline DAG

DAG id: `sentiment_training_pipeline`

Schedule: manual only, `schedule_interval=None`

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

Retry policy:

```text
retries=2
retry_delay=5 minutes
retry_exponential_backoff=True
```

## Drift Detection DAG

DAG id: `drift_detection`

Schedule: `@daily`

Task order:

```text
collect_recent_predictions
  -> load_reference_data
  -> run_drift_detection
  -> evaluate_drift
  -> store_drift_report
```

The drift DAG creates missing `predictions`, `drift_reports`, and `alerts` tables.

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

## Real Airflow Runtime Check

Airflow was started through Docker Compose with the verify override:

```text
docker-compose -f docker-compose.yml -f docker-compose.verify.yml up -d --force-recreate mlflow airflow
```

Container status:

```text
NAME                 IMAGE                           COMMAND                  SERVICE   CREATED          STATUS                    PORTS
sentinelml-airflow   sentinelml-airflow:latest       "/bin/bash -c 'airfl..." airflow   21 minutes ago   Up 20 minutes (healthy)   0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp
sentinelml-db        postgres:16                     "docker-entrypoint.s..." db        25 hours ago     Up 2 hours (healthy)      0.0.0.0:15432->5432/tcp, [::]:15432->5432/tcp
sentinelml-mlflow    ghcr.io/mlflow/mlflow:v2.14.3   "/bin/sh -c 'mkdir -..." mlflow    21 minutes ago   Up 20 minutes (healthy)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

Airflow DAG list:

```text
dag_id                      | fileloc                                | owners     | is_paused
============================+========================================+============+==========
drift_detection             | /opt/airflow/dags/drift_detection.py   | sentinelml | True
sentiment_training_pipeline | /opt/airflow/dags/training_pipeline.py | sentinelml | False
```

Airflow import errors:

```text
No data found
```

Airflow startup logs confirmed scheduler and webserver:

```text
Starting the scheduler
Listening at: http://0.0.0.0:8080
```

## Real DAG Run

Triggered run:

```text
airflow dags trigger sentiment_training_pipeline --run-id phase3_airflow_verify_004
```

Final task states:

```text
dag_id                      | execution_date            | task_id              | state
============================+===========================+======================+========
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | ingest_data          | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | validate_data        | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | preprocess_data      | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | train_baseline       | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | train_transformer    | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | evaluate_and_compare | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | register_best_model  | success
sentiment_training_pipeline | 2026-07-09T20:10:48+00:00 | deploy_model         | success
```

Training results from the Airflow run:

```text
Baseline run_id: 3858df0a83c940b989076d19d0231ea7
Transformer run_id: 25a6b9c258c84ca7b726a32b1b89a2ef
Baseline f1_macro: 0.849624060150376
Transformer f1_macro: 0.4949494949494949
Winner: baseline-logreg
```

Registry promotion log:

```text
No registered model exists yet for sentiment-model.
Successfully registered model 'sentiment-model'.
Created version '1' of model 'sentiment-model'.
Registered model sentiment-model version 1 from run 3858df0a83c940b989076d19d0231ea7.
Promoted model sentiment-model version 1 to Production.
Model promotion decision: {'promoted': True, 'reason': 'New model beat current Production macro F1.', 'old_f1': 0.0, 'new_f1': 0.849624060150376, 'version': '1', 'stage': 'Production'}
```

Deployment task result:

```text
Serving reload notification failed: HTTPConnectionPool(host='serving', port=8000): Max retries exceeded with url: /api/v1/model/reload
Deployment notification result: {'status': 'reload_failed', ...}
Marking task as SUCCESS. dag_id=sentiment_training_pipeline, task_id=deploy_model
```

This is expected in Phase 3 because Phase 4 serving is not running yet. The DAG intentionally treats reload notification failure as non-fatal.

## Runtime Fixes From Live Airflow Verification

Real Airflow exposed issues the mocked tests did not catch:

- Airflow workers could not import `ml.*`; fixed with `PYTHONPATH=/opt/airflow`.
- Airflow could not write MLflow artifacts into the shared artifact volume; fixed by having MLflow create and chmod its artifact root before server startup.
- Registry promotion failed when no registered model existed yet; fixed `_get_current_production_version(...)` to return `None` for MLflow `RESOURCE_DOES_NOT_EXIST` and re-raise other MLflow exceptions.

## Phase Boundary

`deploy_model` calls the Phase 4 serving reload endpoint:

```text
POST /api/v1/model/reload
```

That endpoint is not implemented until Phase 4. For now, the task logs a warning and returns `reload_failed` instead of failing the DAG.
