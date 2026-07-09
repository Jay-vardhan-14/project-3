# Phase 1 Audit

## File Breakdown

| File | Purpose | Key functions/exports |
|---|---|---|
| `docker-compose.yml` | Defines 6-service SentinelML stack | `db`: PostgreSQL 16 with init SQL and healthcheck. `redis`: Redis 7 Alpine with `redis-cli ping`. `mlflow`: MLflow 2.14.3 server on `:5000`, PostgreSQL backend, local artifact volume. `airflow`: Airflow 2.9.3 webserver+scheduler, LocalExecutor, PostgreSQL backend. `serving`: FastAPI service on `:8000`, built with root `uv.lock`. `dashboard`: React/nginx placeholder on `:3000`. |
| `.env.example` | Environment template | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `MLFLOW_TRACKING_URI`, `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACT_ROOT`, `AIRFLOW__CORE__SQL_ALCHEMY_CONN`, `AIRFLOW__CORE__EXECUTOR`, `AIRFLOW_ADMIN_USER`, `AIRFLOW_ADMIN_PASSWORD`, `REDIS_URL`, `MODEL_NAME` |
| `ml/config.py` | Central dataclass config | `MLConfig`, `DEFAULT_CONFIG`. Fields: `dataset_name`, `max_samples`, `test_size`, `val_size`, `random_seed`, `tfidf_max_features`, `tfidf_ngram_range`, `logreg_C`, `logreg_max_iter`, `model_name`, `max_length`, `batch_size`, `learning_rate`, `num_epochs`, `warmup_steps`, `drift_threshold`, `drift_window_days`, `model_registry_name`, `data_dir`, `raw_data_dir`, `processed_data_dir`, `splits_dir` |
| `ml/utils/reproducibility.py` | Seed control | `set_seed(seed)`: sets `PYTHONHASHSEED`, `random.seed`, `numpy.random.seed`, and if torch is installed: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False` |
| `ml/data/ingestion.py` | Dataset download and raw parquet storage | `download_dataset(config: MLConfig) -> pd.DataFrame`: loads IMDB/SST2/other HuggingFace dataset, normalizes to `text,label`, optionally samples, validates, writes timestamped parquet and `latest.parquet`. `_normalize_columns(df) -> pd.DataFrame`: maps `sentence`/`sentiment` to expected schema where possible. |
| `ml/data/validation.py` | Schema and quality checks | `DataValidationError`, `ValidationReport`. `validate_data(df) -> ValidationReport`: rejects missing columns, empty dataset, nulls, empty text, non-binary labels; warns on imbalance and duplicates. `_build_report(...)`, `_class_distribution(df)`, `_class_ratio(df)` are internal helpers. |
| `ml/data/preprocessing.py` | Cleaning, splits, feature creation | `clean_text(text: str) -> str`: lowercase, HTML removal, URL removal, special char cleanup, whitespace normalization. `preprocess_dataset(df, config) -> dict[str, pd.DataFrame]`: validates, cleans, stratified train/val/test split, saves parquet. `create_tfidf_features(...) -> dict[str, Any]`: fits TF-IDF on train and transforms all splits. `create_transformer_dataset(...) -> Any`: builds tokenized HuggingFace Dataset using AutoTokenizer. |
| `serving/app/main.py` | Phase 1 serving placeholder | Functional health-only FastAPI stub: `GET /api/v1/health -> {"status":"ok"}`. Full serving API is Phase 4. |
| `serving/Dockerfile` | Phase 1 serving image | Uses `uv sync --frozen --extra serving` from root `pyproject.toml`/`uv.lock`, then runs `uvicorn`. |
| `dashboard/*` | Phase 1 dashboard placeholder | Functional Vite/React/nginx stub displaying `SentinelML`. Real dashboard is Phase 5. Uses Node/npm, not uv, because dashboard is not Python. |

## Tests

| Test file | Test function | Verifies |
|---|---|---|
| `tests/test_data_ingestion.py` | `test_ingestion_downloads_valid_dataframe` | Mocked HuggingFace dataset returns valid `text,label` DataFrame, respects sample size, writes `latest.parquet`. |
| `tests/test_data_validation.py` | `test_validation_passes_balanced_dataset` | Balanced valid data passes and reports distribution/text length. |
| `tests/test_data_validation.py` | `test_validation_catches_nulls` | Null values raise `DataValidationError`. |
| `tests/test_data_validation.py` | `test_validation_catches_empty_strings` | Blank text raises `DataValidationError`. |
| `tests/test_data_validation.py` | `test_validation_warns_on_imbalanced_data` | Imbalance over 60/40 is detected as warning. |
| `tests/test_data_preprocessing.py` | `test_clean_text_removes_html_urls_and_lowercases` | HTML removal, URL removal, lowercasing, special char cleanup. |
| `tests/test_data_preprocessing.py` | `test_preprocess_dataset_stratified_split_maintains_class_ratio` | 80/10/10 split with stratified class balance and parquet output. |
| `tests/test_data_preprocessing.py` | `test_tfidf_creates_expected_shape_matrices` | TF-IDF matrices have expected row counts and feature cap. |

## Live Infrastructure Verification

Requested command output:

```text
PS> docker-compose up -d db redis mlflow
unable to get image 'redis:7-alpine': failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Docker diagnostic:

```text
PS> docker info
Client:
 Version:    29.5.3
 Context:    desktop-linux
...
Server:
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Container health check could not run:

```text
PS> docker-compose ps
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

MLflow UI check:

```text
PS> Invoke-WebRequest http://localhost:5000
Unable to connect to the remote server
```

MLflow logs check:

```text
PS> docker-compose logs mlflow --tail=80
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Initial conclusion: Compose syntax was valid, but live infrastructure could not be verified until Docker Desktop/Linux engine was started.

Follow-up verification after Docker was running:

```text
sentinelml-db       Up (healthy)   15432->5432
sentinelml-redis    Up (healthy)   16379->6379
sentinelml-mlflow   Up (healthy)   5000->5000
```

The alternate host ports came from `docker-compose.verify.yml` because another local stack already occupied `5432` and `6379`.

MLflow UI returned HTTP `200` at `http://localhost:5000`, and the PostgreSQL backend connection was verified with:

```text
current_database | current_user
------------------+--------------
mlflow           | postgres
```

## Test Coverage Confirmation

```text
PS> uv run pytest tests -q
........                                                                 [100%]
```

| Requirement | Covered? | Notes |
|---|---:|---|
| Ingestion produces valid DataFrame with `text`, `label` | Yes | Mocked HuggingFace dataset, validates columns and parquet output. |
| Validation rejects null values | Yes | Raises `DataValidationError`. |
| Validation rejects empty strings | Yes | Raises `DataValidationError`. |
| Validation catches class imbalance | Yes | Warns, does not reject. This matches build prompt wording: warn if >60/40. |
| Validation catches duplicates | Partial gap | Implementation warns on duplicates, but no test currently asserts it. |
| Text cleaning removes HTML, URLs, lowercases | Yes | Direct unit test. |
| Stratified split preserves class ratios | Yes | 50/50 synthetic data remains balanced in all splits. |
| TF-IDF creates correctly-shaped matrices | Yes | Row counts and max feature cap asserted. |

## Test Isolation

`test_ingestion_downloads_valid_dataframe` does not make a live network call. It monkeypatches the `datasets` module with a fake `load_dataset()` returning small synthetic train/test DataFrames.

## Dependency Manager Consistency

Python local dev uses `uv` and the committed `uv.lock`. The serving Dockerfile also uses `uv sync --frozen --extra serving` from the same lockfile.

Dashboard is Node/React, so `uv` does not apply there. Current placeholder uses `npm install`; in Phase 5 it should use a committed lockfile plus `npm ci` for container/local consistency.

## Git

Git initialized and committed.

```text
PS> git log -1 --oneline
0a11f36 Phase 1: Infrastructure & data pipeline
```

Full commit hash:

```text
0a11f36938febab54c9376c4a5a43d04f302694c
```

Working tree was clean after the Phase 1 commit.
