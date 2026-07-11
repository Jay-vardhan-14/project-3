# SentinelML — End-to-End ML Pipeline with MLOps

[![CI](https://github.com/Jay-vardhan-14/project-3/actions/workflows/ci.yml/badge.svg)](https://github.com/Jay-vardhan-14/project-3/actions/workflows/ci.yml)

SentinelML is a production-style machine-learning pipeline for binary sentiment analysis. It ingests text, trains and compares two models (a TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT), tracks every experiment in MLflow, promotes the best model through a registry gate, serves it via a FastAPI inference API with per-prediction logging, detects data drift with a daily Airflow job, and surfaces everything on a monochrome monitoring dashboard — the whole lifecycle orchestrated by Apache Airflow DAGs and deployed as a 6-service Docker Compose stack.

> The numbers in this README are **measured from real runs in this repository**, not aspirational targets. Where a result is constrained (e.g. CPU-capped transformer training), it is called out explicitly.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   Apache Airflow  ──►   │ ingest → validate → preprocess → train        │
   (LocalExecutor)       │ (baseline + DistilBERT) → compare → register  │
                         │ → deploy (notify serving to reload)           │
                         │                                               │
                         │ drift_detection DAG (@daily): collect preds → │
                         │ reference → Evidently drift → alert → store    │
                         └───────────────┬───────────────────────────────┘
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
      ┌────────────┐             ┌────────────────┐          ┌────────────────┐
      │  MLflow    │             │  Serving API   │          │  Dashboard     │
      │ tracking + │◄────────────│  (FastAPI)     │─────────►│  (React/Vite)  │
      │ registry   │  load Prod  │  /predict      │  /metrics│  Overview      │
      └─────┬──────┘   model     │  /model/reload │          │  Experiments   │
            │                    │  /metrics/*    │          │  Drift         │
            ▼                    └───────┬────────┘          │  Predictions   │
      artifact store                     │                   │  Pipeline      │
                                         ▼                   └────────────────┘
                            ┌──────────────────────────┐
                            │ PostgreSQL 16 │ Redis 7   │
                            │ predictions / drift /     │
                            │ alerts / pipeline_runs    │
                            └──────────────────────────┘
```

## Tech stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Apache Airflow 2.9 | Training + drift DAGs, scheduling, retries |
| Experiment tracking | MLflow 2.14 | Params/metrics/artifacts + model registry with stages |
| Baseline model | scikit-learn | TF-IDF + Logistic Regression pipeline |
| Advanced model | HuggingFace Transformers + PyTorch | DistilBERT fine-tuning |
| Serving | FastAPI + SQLAlchemy (async) + asyncpg | Prediction API, prediction logging, metrics |
| Drift detection | Evidently AI | Feature + prediction drift reports |
| Database | PostgreSQL 16 | Predictions, drift reports, alerts, pipeline runs |
| Cache / rate limit | Redis 7 | Per-IP rate limiting, metrics caching |
| Dashboard | React 18 + Vite + Tailwind + Recharts | Monitoring UI |
| Containerization | Docker + Docker Compose | 6-service stack |
| CI/CD | GitHub Actions | ml-tests, serving-tests, dashboard-build, docker-build |
| Testing | pytest, pytest-asyncio, Vitest | 51 tests total |

## Features

- ✅ Airflow-orchestrated training pipeline (8 stages, end-to-end, not stubs)
- ✅ Two models trained and compared per run; best model auto-promoted
- ✅ MLflow experiment tracking + Model Registry (None → Staging → Production)
- ✅ F1-gated promotion — a new model reaches Production only if it beats the current one
- ✅ FastAPI serving with SHA-256-hashed prediction logging (no raw text stored)
- ✅ sklearn/PyTorch flavor auto-detection at load time
- ✅ Redis rate limiting (60 req/min/IP) + metrics caching
- ✅ Daily Evidently drift detection with threshold alerts
- ✅ Monochrome monitoring dashboard (5 pages, 30s auto-refresh)
- ✅ Non-root hardened images + GitHub Actions CI

## Prerequisites

- Docker + Docker Compose
- ~8 GB RAM (DistilBERT training/inference)
- Optional GPU (CPU works; training is auto-capped on CPU — see benchmarks)

## Quick start

```bash
# 1. Start the stack (db, redis, mlflow, airflow, serving, dashboard)
docker-compose up --build -d

# If host ports 5432/6379 are already in use, add the local override:
docker-compose -f docker-compose.yml -f docker-compose.verify.yml up --build -d

# 2. Seed demo data (trains a baseline, registers it Production, logs sample
#    predictions + a drift history + an alert) so the dashboard has data:
docker-compose exec airflow python /opt/airflow/ml/scripts/demo_setup.py
```

Then open:

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| Serving API docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 (admin / admin) |

To run the full training pipeline in Airflow, trigger the `sentiment_training_pipeline` DAG from the UI (or `docker-compose exec airflow airflow dags trigger sentiment_training_pipeline`).

## API

Base path `/api/v1`:

```
POST /predict                       {"text": "..."} -> sentiment, confidence, model, latency_ms
POST /predict/batch                 {"texts": [...]} (max 100)
GET  /model/info                    current model name/version/stage/F1/load time
POST /model/reload                  hot-reload Production model from the registry
GET  /health                        model loaded, DB, Redis status
GET  /metrics/summary               KPIs (predictions today, avg latency, drift status)
GET  /metrics/predictions           volume over time (?days / ?hours)
GET  /metrics/latency               P50 / P95 / P99
GET  /metrics/drift                 drift score history
GET  /metrics/distribution          sentiment distribution
GET  /metrics/distribution/confidence  confidence histogram
GET  /metrics/recent-predictions    recent prediction rows
GET  /metrics/experiments           MLflow runs
GET  /metrics/models                registry entries with stages
GET  /metrics/alerts                active + recent alerts
GET  /metrics/pipeline-runs         DAG run history
```

## Pipeline

**`sentiment_training_pipeline`** (manual trigger): `ingest → validate → preprocess → train_baseline → train_transformer → evaluate_and_compare → register_best_model → deploy_model`. Retries 2× with exponential backoff; `deploy_model` calls the serving `/model/reload` endpoint. Each run writes a `pipeline_runs` row (running → success/failed) with a metrics summary.

**`drift_detection`** (`@daily`): `collect_recent_predictions → load_reference_data → run_drift_detection → evaluate_drift → store_drift_report`. Compares the recent prediction distribution against the training reference via Evidently; if the drift score exceeds the threshold it creates an alert.

## Model comparison & promotion

Promotion is automated and gated on macro F1. `compare_and_promote()` registers the new model, then transitions it to **Production only if its F1 beats the current Production model** — otherwise it stays in **Staging**. Equal F1 does **not** promote.

Verified example (live MLflow registry E2E, Phase 2):

| Step | Model | F1 | Result |
|------|-------|----|--------|
| Register v1 | dummy | 0.70 | → Production |
| Register v2 | dummy | 0.82 | **0.82 > 0.70 → promoted to Production, v1 archived** |

Verified negative case (Phase 5): a retrained baseline with F1 **0.85 == 0.85** (equal to the current Production model) was held in **Staging**; Production stayed on the incumbent — proving the gate blocks non-improvements.

## Performance benchmarks

All measured in this repository on CPU with the demo config (`SENTINELML_MAX_SAMPLES=200`), seed = 42 (deterministic).

**Model quality** (macro metrics on the held-out test split):

| Model | F1 | Accuracy | Precision | Recall |
|-------|----|----------|-----------|--------|
| TF-IDF + LogReg (baseline) | **0.850** | 0.850 | 0.854 | 0.850 |
| DistilBERT (fine-tuned) | 0.495 | 0.500 | 0.500 | 0.500 |

> ⚠️ The DistilBERT F1 (0.495) is **intentionally low**: transformer training is CPU-capped to ≤5000 samples / ≤2 epochs so the DAG completes in ~90s. The PRD's >0.90 target requires full-scale GPU training (25K samples, 3 epochs), which was not run here. Under these constraints the pipeline correctly auto-selects the **baseline** as the better model.

**Inference latency** (baseline, single `/predict`, observed): P50 **1 ms**, P95 **8.1 ms**, P99 **11.2 ms**.

**Pipeline execution** (real, from the `pipeline_runs` table):

| DAG | Duration |
|-----|----------|
| `sentiment_training_pipeline` (8 stages incl. both models) | **91 s** |
| `drift_detection` | **7 s** |

## Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | postgres / postgres / sentinelml | db |
| `MLFLOW_TRACKING_URI` | http://mlflow:5000 | training, serving |
| `MLFLOW_BACKEND_STORE_URI` | postgresql://…@db:5432/mlflow | mlflow |
| `AIRFLOW__CORE__SQL_ALCHEMY_CONN` | postgresql+psycopg2://…@db:5432/airflow | airflow |
| `DATABASE_URL` | postgresql+asyncpg://…@db:5432/sentinelml | serving |
| `REDIS_URL` | redis://redis:6379/0 | serving |
| `MODEL_NAME` | sentiment-model | serving, registry |
| `SENTINELML_MAX_SAMPLES` / `SENTINELML_NUM_EPOCHS` / `SENTINELML_BATCH_SIZE` | 200 / 1 / 8 | training DAG (CPU-friendly demo defaults) |

## Testing

51 tests total. ml and serving tests run in **separate environments** because `apache-airflow` pins `sqlalchemy<2.0` while the serving async stack requires `sqlalchemy>=2.0` — a genuine, non-overlapping version conflict (declared as conflicting `uv` extras).

```bash
# ml + orchestration (26 tests)
uv run --extra training --extra orchestration pytest tests

# serving API (16 tests)
uv run --extra serving pytest serving/tests

# dashboard (9 tests)
cd dashboard && npm ci && npm test
```

CI (`.github/workflows/ci.yml`) runs all four jobs — `ml-tests`, `serving-tests`, `dashboard-build`, `docker-build` — on every push and PR to `main`.

## Project structure

```
airflow/dags/      training_pipeline.py, drift_detection.py
ml/                config, data/, models/, tracking/, monitoring/, utils/, scripts/
serving/app/       main, config, database, models, schemas, routes/, services/, middleware/
dashboard/src/     api/, hooks/, lib/, components/, pages/
docker-compose.yml + docker-compose.verify.yml (local port override)
```

## Limitations (honest notes)

- Transformer accuracy is CPU-capped for runnable demos (see benchmarks); use a GPU + full dataset for the PRD's >0.90 target.
- The demo/verification runs use small sample sizes, so drift scores can be high simply because the logged prediction traffic is skewed relative to the balanced training reference.

## License

MIT
