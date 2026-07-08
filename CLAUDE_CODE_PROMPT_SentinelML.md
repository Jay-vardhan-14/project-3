# Claude Code Prompt — SentinelML
# Copy everything below this line and paste it into Claude Code

---

## PROJECT OVERVIEW

Build "SentinelML" — a production-grade, end-to-end ML pipeline for sentiment analysis with full MLOps tooling. The system ingests text data, trains two models (TF-IDF baseline + DistilBERT), tracks experiments in MLflow, registers the best model, serves it via FastAPI, detects data drift with Evidently, and displays everything on a monitoring dashboard. The entire pipeline is orchestrated by Apache Airflow DAGs.

This is a resume portfolio project. Every component must demonstrate production ML engineering — not notebook-grade code.

## CRITICAL CONSTRAINTS

- Use the exact tech stack specified. Do not substitute tools.
- Every script must have proper logging (Python `logging` module, not `print()`).
- All ML experiments must be fully reproducible: set random seeds everywhere (numpy, torch, sklearn, python hashseed).
- The Airflow DAGs must actually work end-to-end, not be stubs.
- MLflow model comparison and promotion must be automated — no manual steps.
- All predictions must be logged to the database for drift analysis.
- Test coverage > 80% on the serving API.

## CRITICAL: UI DESIGN RULES (Dashboard)

The monitoring dashboard must NOT look AI-generated. Follow these rules:

### DO:
- System font stack: `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
- Light mode default: white (#FFFFFF) and zinc grays
- One accent color: `#2563EB` (blue-600) for interactive elements only
- Typography creates hierarchy: weight (400/500) and size (12px/13px/14px/20px)
- Subtle 1px borders: `border-zinc-200`. No box-shadows on cards
- Dense data display: 12px–13px text in tables, compact padding
- Monochrome charts: `#18181b` (zinc-900) for bars/lines, `#e4e4e7` (zinc-200) for grid
- Use color ONLY for alerts: red for critical drift, amber for warnings, green for healthy
- Status indicators: small 8px dots (green/amber/red), not large colored badges

### DO NOT:
- ❌ NO gradients, glassmorphism, glow effects, purple-blue schemes
- ❌ NO colored card backgrounds or colored left borders on cards
- ❌ NO oversized icons (max 18px), no decorative icons
- ❌ NO border-radius > 8px, no pill shapes
- ❌ NO dark theme with neon accents
- ❌ NO bouncy animations, shimmer effects, floating elements

### Color palette (Tailwind):
```
Page bg:          bg-white
Sidebar bg:       bg-zinc-50 border-r border-zinc-200
Card:             bg-white border border-zinc-200 rounded-md
Primary text:     text-zinc-900
Secondary text:   text-zinc-500
Borders:          border-zinc-200
Hover:            hover:bg-zinc-50

Status healthy:   text-emerald-700 bg-emerald-50 border-emerald-200
Status warning:   text-amber-700 bg-amber-50 border-amber-200
Status critical:  text-red-700 bg-red-50 border-red-200

Chart bars:       #18181b (zinc-900)
Chart grid:       #e4e4e7 (zinc-200)
Chart axis text:  #71717a (zinc-500) at 12px
```

Reference: Grafana (clean mode), Vercel Analytics, Datadog (light mode)

## TECH STACK (do not change)

**Orchestration:** Apache Airflow 2.9
**Experiment Tracking:** MLflow 2.x
**ML (Baseline):** scikit-learn (TF-IDF + LogisticRegression)
**ML (Advanced):** HuggingFace Transformers + PyTorch (DistilBERT)
**Model Serving:** Python 3.11, FastAPI, SQLAlchemy (async), asyncpg
**Drift Detection:** Evidently AI
**Database:** PostgreSQL 16
**Cache:** Redis 7
**Dashboard:** TypeScript, React 18, Vite, Tailwind CSS 3, recharts, lucide-react
**Testing:** pytest, pytest-asyncio, httpx
**Containerization:** Docker + Docker Compose
**CI/CD:** GitHub Actions

## PROJECT STRUCTURE

Create the exact structure from the PRD section 8.

## BUILD ORDER

### PHASE 1: Infrastructure & Data Pipeline

1. Create `docker-compose.yml` with 6 services:
   ```yaml
   services:
     db:          # PostgreSQL 16
     redis:       # Redis 7 Alpine
     mlflow:      # MLflow tracking server (backed by PostgreSQL)
     airflow:     # Airflow webserver + scheduler (all-in-one for simplicity)
     serving:     # FastAPI model serving
     dashboard:   # React monitoring dashboard
   ```
   - MLflow uses PostgreSQL as backend store, local filesystem for artifacts (`/mlflow/artifacts` volume)
   - Airflow uses LocalExecutor with PostgreSQL backend
   - Shared volumes for ML artifacts and data
   - Health checks on all services
   - Dependency ordering: db → mlflow → airflow → serving → dashboard

2. Create `.env.example`:
   ```
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=sentinelml
   MLFLOW_TRACKING_URI=http://mlflow:5000
   MLFLOW_BACKEND_STORE_URI=postgresql://postgres:postgres@db:5432/mlflow
   MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
   AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://postgres:postgres@db:5432/airflow
   AIRFLOW__CORE__EXECUTOR=LocalExecutor
   AIRFLOW_ADMIN_USER=admin
   AIRFLOW_ADMIN_PASSWORD=admin
   REDIS_URL=redis://redis:6379/0
   MODEL_NAME=sentiment-model
   ```

3. Create `ml/config.py`:
   - Dataclass-based configuration with all hyperparameters:
     ```python
     @dataclass
     class MLConfig:
         # Data
         dataset_name: str = "imdb"         # or "sst2"
         max_samples: int = 25000           # limit for faster training
         test_size: float = 0.1
         val_size: float = 0.1
         random_seed: int = 42

         # Baseline (TF-IDF + LogReg)
         tfidf_max_features: int = 50000
         tfidf_ngram_range: tuple = (1, 2)
         logreg_C: float = 1.0
         logreg_max_iter: int = 1000

         # Transformer (DistilBERT)
         model_name: str = "distilbert-base-uncased"
         max_length: int = 256
         batch_size: int = 32
         learning_rate: float = 2e-5
         num_epochs: int = 3
         warmup_steps: int = 500

         # Drift
         drift_threshold: float = 0.15    # Evidently drift score threshold
         drift_window_days: int = 7

         # Serving
         model_registry_name: str = "sentiment-model"
     ```

4. Create `ml/utils/reproducibility.py`:
   - `set_seed(seed)`: sets random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all, PYTHONHASHSEED, torch.backends.cudnn.deterministic = True
   - Called at the start of every training script

5. Create `ml/data/ingestion.py`:
   - `download_dataset(config) -> pd.DataFrame`: download IMDB dataset from HuggingFace `datasets` library
   - Validate: check columns exist ('text', 'label'), check no nulls, log class distribution
   - Save raw data to `data/raw/` as parquet
   - Return DataFrame

6. Create `ml/data/validation.py`:
   - `validate_data(df) -> ValidationReport`:
     - Check for nulls, duplicates
     - Check class balance (warn if > 60/40 split)
     - Check text length distribution (min, max, mean, median)
     - Check for empty strings
   - Return structured report with pass/fail and details
   - Raise `DataValidationError` on critical failures

7. Create `ml/data/preprocessing.py`:
   - `clean_text(text: str) -> str`: lowercase, remove HTML tags, remove URLs, remove special characters (keep alphanumeric + basic punctuation), strip extra whitespace
   - `preprocess_dataset(df, config) -> dict`: clean text, stratified train/val/test split, save splits to `data/splits/` as parquet
   - `create_tfidf_features(train_texts, val_texts, test_texts, config) -> dict`: fit TF-IDF on train, transform all splits, return sparse matrices + fitted vectorizer
   - `create_transformer_dataset(texts, labels, config)`: return HuggingFace Dataset with tokenized inputs (input_ids, attention_mask, labels)

8. Write tests for data pipeline:
   - Test ingestion downloads and returns valid DataFrame
   - Test validation catches nulls, empty strings, imbalanced data
   - Test preprocessing cleans text correctly (HTML removal, URL removal, lowercasing)
   - Test stratified split maintains class ratios
   - Test TF-IDF creates correct-shaped matrices

### PHASE 2: Model Training & MLflow

9. Create `ml/tracking/mlflow_utils.py`:
   - `setup_mlflow(experiment_name: str)`: set tracking URI, create/get experiment
   - `log_training_run(params, metrics, artifacts, model, model_name)`:
     - Log all params (hyperparameters)
     - Log all metrics (accuracy, precision, recall, f1, training_time, model_size_mb, inference_latency_ms)
     - Log artifacts (confusion matrix PNG, classification report JSON)
     - Log model with MLflow's model logging (sklearn or pytorch flavor)
   - `log_confusion_matrix(y_true, y_pred, labels, filename)`: generate and save confusion matrix as PNG using matplotlib (minimal styling — no color maps, just black/white with numbers)

10. Create `ml/models/evaluation.py`:
    - `evaluate_model(model, X_test, y_test) -> dict`: compute accuracy, precision, recall, F1 (macro), per-class metrics, confusion matrix
    - `compute_inference_latency(model, sample_inputs, n_runs=100) -> float`: average inference time in ms
    - `compare_models(current_metrics, new_metrics) -> bool`: return True if new model F1 > current model F1 (the promotion gate)
    - `generate_classification_report(y_true, y_pred) -> dict`: structured report with per-class metrics

11. Create `ml/models/baseline.py`:
    - `train_baseline(config) -> tuple[model, metrics]`:
      1. Load preprocessed data splits
      2. Create TF-IDF features
      3. Train LogisticRegression with configured hyperparameters
      4. Evaluate on test set
      5. Log everything to MLflow (params, metrics, confusion matrix, model)
      6. Return trained pipeline (TfidfVectorizer + LogReg) and metrics dict
    - Package as sklearn Pipeline for single-object serialization

12. Create `ml/models/transformer.py`:
    - `train_transformer(config) -> tuple[model, metrics]`:
      1. Load preprocessed data splits
      2. Load DistilBERT tokenizer + model from HuggingFace
      3. Tokenize datasets
      4. Train with HuggingFace Trainer (or manual training loop):
         - Log metrics per epoch to MLflow
         - Save best checkpoint based on validation F1
      5. Evaluate on test set
      6. Log everything to MLflow
      7. Return model + tokenizer + metrics
    - Handle GPU/CPU automatically: `device = "cuda" if torch.cuda.is_available() else "cpu"`
    - For CPU training: reduce max_samples to 5000 and epochs to 2 for reasonable training time

13. Create `ml/tracking/registry.py`:
    - `register_model(run_id, model_name, metrics)`: register model from MLflow run to Model Registry
    - `promote_model(model_name, version, stage="Production")`: transition model to Production stage
    - `get_production_model(model_name)`: load the current Production model from registry
    - `compare_and_promote(model_name, new_run_id, new_metrics)`:
      1. Get current Production model's metrics (from MLflow tags or logged metrics)
      2. Compare F1 scores
      3. If new F1 > current F1: register new model, promote to Production, archive old
      4. If new F1 <= current F1: register as Staging only, log reason for not promoting
      5. Return decision dict: { promoted: bool, reason: str, old_f1: float, new_f1: float }

14. Write tests:
    - Test baseline training produces valid metrics (accuracy > 0.7 on a small sample)
    - Test transformer training completes without errors (use tiny subset, 1 epoch)
    - Test MLflow logging creates experiment and run with expected params/metrics
    - Test model comparison logic (new > old → promote, new <= old → don't)
    - Test model registry operations

### PHASE 3: Airflow Orchestration

15. Create `airflow/dags/training_pipeline.py`:
    ```python
    # DAG: sentiment_training_pipeline
    # Schedule: None (manual trigger) or weekly
    # Tasks (in order):

    @task
    def ingest_data():
        """Download dataset, save to data/raw/"""

    @task
    def validate_data():
        """Run validation checks, fail DAG on critical issues"""

    @task
    def preprocess_data():
        """Clean text, create splits, save to data/splits/"""

    @task
    def train_baseline():
        """Train TF-IDF + LogReg, log to MLflow, return run_id + metrics"""

    @task
    def train_transformer():
        """Train DistilBERT, log to MLflow, return run_id + metrics"""

    @task
    def evaluate_and_compare(baseline_result, transformer_result):
        """Compare both models, select winner based on F1"""

    @task
    def register_best_model(winner_result):
        """Register winner in MLflow, promote if better than current Production"""

    @task
    def deploy_model(registration_result):
        """Notify serving API to reload model (HTTP call to /model/reload)"""
    ```
    - Use Airflow TaskFlow API (decorated tasks with `@task`)
    - Pass results between tasks via XCom (return values)
    - Retry policy: 2 retries with 5-minute delay
    - Email/log on failure (log-based for this project)
    - DAG documentation string explaining the full pipeline

16. Create `airflow/dags/drift_detection.py`:
    ```python
    # DAG: drift_detection
    # Schedule: daily
    # Tasks:

    @task
    def collect_recent_predictions():
        """Query last N predictions from DB as 'current' dataset"""

    @task
    def load_reference_data():
        """Load training data distribution as 'reference'"""

    @task
    def run_drift_detection(current_data, reference_data):
        """Run Evidently drift report, save HTML report, return metrics"""

    @task
    def evaluate_drift(drift_result):
        """Check drift score against threshold, create alert if exceeded"""

    @task
    def store_drift_report(drift_result, alert_result):
        """Save drift report to DB + artifacts"""
    ```

17. Create `ml/monitoring/drift_detector.py`:
    - `detect_drift(reference_df, current_df, config) -> DriftReport`:
      - Use Evidently's `DataDriftPreset` for feature-level drift detection
      - Use `ColumnDriftMetric` for prediction distribution drift
      - Generate HTML report (save to artifacts directory)
      - Return structured results: overall drift detected (bool), drift score, per-feature drift flags, prediction drift
    - `create_drift_summary(report) -> dict`: extract key metrics for DB storage

18. Create `ml/monitoring/alerts.py`:
    - `check_and_create_alert(drift_report, config, db_session)`:
      - If drift_score > threshold: create `drift_warning` alert
      - If drift_score > 2x threshold: create `drift_critical` alert
      - If prediction distribution shift > 20%: create `model_degradation` alert
    - `resolve_alert(alert_id, db_session)`: mark alert as resolved
    - `get_active_alerts(db_session)`: return unresolved alerts

### PHASE 4: Model Serving API

19. Create `serving/app/main.py`:
    - FastAPI app with CORS, lifespan events
    - On startup: connect to DB, connect to Redis, load model from MLflow registry
    - Mount all route files under `/api/v1/`
    - Global exception handler

20. Create `serving/app/database.py`:
    - Async SQLAlchemy engine (asyncpg)
    - Session factory + `get_db` dependency
    - Initialize tables on startup

21. Create `serving/app/models.py`:
    - SQLAlchemy models for: Prediction, DriftReport, Alert, ModelDeployment, PipelineRun
    - Matching the PRD schema exactly

22. Create `serving/app/services/predictor.py`:
    - `ModelPredictor` class:
      - `load_model(model_name, stage="Production")`: load from MLflow registry, detect model type (sklearn vs pytorch), set up appropriate inference pipeline
      - `predict(text: str) -> PredictionResult`: preprocess → inference → return sentiment + confidence + latency_ms
      - `predict_batch(texts: list[str]) -> list[PredictionResult]`: batch inference
      - `reload_model()`: reload latest Production model from registry
    - Auto-detect model type: if sklearn Pipeline → use `.predict_proba()`, if transformer → use tokenizer + model forward pass
    - Track: model version, model name, load timestamp

23. Create `serving/app/services/prediction_logger.py`:
    - `log_prediction(input_text, result, model_info, db)`:
      - Hash input text with SHA-256 (don't store raw text for privacy)
      - Store: input_hash, input_length, predicted_sentiment, confidence, model_version, model_name, latency_ms, timestamp
      - Fire-and-forget (errors logged, not raised)

24. Create prediction routes `serving/app/routes/predict.py`:
    - `POST /predict`: accepts `{"text": "..."}` (min 1 char, max 5000 chars), returns:
      ```json
      {
        "sentiment": "positive",
        "confidence": 0.9523,
        "model": {
          "name": "sentiment-model",
          "version": "3"
        },
        "latency_ms": 45
      }
      ```
    - `POST /predict/batch`: accepts `{"texts": ["...", "..."]}` (max 100 items), returns list
    - Rate limiting: 60 requests/minute per IP

25. Create model routes `serving/app/routes/model.py`:
    - `GET /model/info`: current model metadata
    - `POST /model/reload`: trigger model reload from registry (returns new version info)

26. Create metrics routes `serving/app/routes/metrics.py`:
    - `GET /metrics/summary`: KPI data for dashboard (predictions today, avg latency, drift status, current model, F1)
    - `GET /metrics/predictions`: prediction volume grouped by hour/day (configurable window)
    - `GET /metrics/latency`: latency percentiles (P50, P95, P99) over configurable window
    - `GET /metrics/drift`: drift score history from drift_reports table
    - `GET /metrics/distribution`: sentiment positive/negative ratio over time windows
    - `GET /metrics/experiments`: MLflow experiment runs (query MLflow API)
    - `GET /metrics/models`: model registry entries with stages
    - `GET /metrics/alerts`: active + recent alerts
    - `GET /metrics/pipeline-runs`: pipeline execution history
    - All endpoints support `?days=7` or `?hours=24` query param for time windowing
    - Cache frequently-accessed aggregations in Redis (60s TTL)

27. Create health route `serving/app/routes/health.py`:
    - `GET /health`: model loaded (bool), model version, model load time, DB connected, Redis connected

28. Write serving tests:
    - `test_predict.py`: valid prediction, empty text (422), too long text (422), batch prediction, rate limiting
    - `test_model_loading.py`: model loads from registry, reload endpoint works, graceful error on missing model
    - `test_metrics.py`: summary returns correct structure, predictions endpoint filters by time window
    - `test_drift.py`: drift detection produces valid report, alert creation on threshold breach

### PHASE 5: Monitoring Dashboard

29. Set up React dashboard:
    - Vite + React 18 + TypeScript + Tailwind CSS + recharts + lucide-react
    - `vite.config.ts`: proxy `/api` to serving backend in dev mode

30. Create `dashboard/src/index.css`:
    ```css
    @tailwind base;
    @tailwind components;
    @tailwind utilities;

    :root {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      font-size: 14px;
      line-height: 1.5;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d4d4d8; border-radius: 3px; }
    ```

31. Create API client `dashboard/src/api/client.ts`:
    - Axios instance with baseURL `/api/v1`
    - Typed functions for each metrics endpoint
    - Auto-refresh: poll every 30 seconds for real-time feel

32. Create dashboard layout:
    - `layout/AppLayout.tsx`: sidebar (200px) + main content
    - `layout/Sidebar.tsx`:
      - Logo/title: "SentinelML" in text (font-medium, text-zinc-900, 15px). No logo icon
      - Navigation: Overview, Experiments, Drift, Predictions, Pipeline
      - Active item: `bg-zinc-100 text-zinc-900`. Inactive: `text-zinc-500 hover:text-zinc-700`
      - Style: `bg-zinc-50 border-r border-zinc-200`. NO shadow. Compact text (13px)

33. Create Overview page (`pages/OverviewPage.tsx`):
    - **Alert banner** (if active alerts): full-width, top of page, `bg-red-50 border border-red-200 text-red-800 text-sm p-3 rounded-md`. Text: "Drift detected: input distribution has shifted significantly. Last checked: 2h ago"
    - **KPI row** (4 cards in a grid):
      - Predictions today: big number (text-2xl font-medium), label below (text-xs text-zinc-500)
      - Avg latency: big number + "ms" suffix
      - Model F1: big number as percentage
      - Drift status: "Healthy" (green dot) / "Warning" (amber dot) / "Alert" (red dot)
      - Card style: `bg-white border border-zinc-200 rounded-md p-4`. NO shadow, NO colored background, NO icon in the card
    - **Prediction volume chart** (recharts AreaChart):
      - Zinc-900 fill with 10% opacity, zinc-900 stroke
      - X axis: dates, Y axis: count
      - Minimal grid, small axis labels (12px zinc-500)
    - **Recent alerts table** (last 5):
      - Columns: Type, Severity (dot), Message, Time
      - Compact rows (py-2), 13px text
      - Style: simple `border-b border-zinc-100` row dividers

34. Create Experiments page (`pages/ExperimentsPage.tsx`):
    - **Runs table**: sortable by any metric column
      - Columns: Run Name, Model Type, F1, Accuracy, Precision, Recall, Latency (ms), Duration, Date
      - F1 and Accuracy formatted as percentages
      - Best F1 row highlighted with `bg-blue-50`
      - Sortable column headers (click to sort, small arrow indicator)
      - Style: compact table, 13px text, `border-b border-zinc-100` dividers
    - **Metrics comparison** (optional, P1): select 2 runs, show side-by-side metrics

35. Create Drift page (`pages/DriftPage.tsx`):
    - **Drift score trend chart** (recharts LineChart):
      - Line: zinc-900
      - Threshold line: dashed, `#dc2626` (red-600)
      - Area above threshold filled with red-50 (subtle)
      - X axis: dates, Y axis: drift score (0–1)
    - **Drift reports table**: date, drift detected (yes/no dot), drift score, features drifted, prediction drift, link to HTML report
    - **Current status card**: last check date, current drift score, status (healthy/warning/critical)

36. Create Predictions page (`pages/PredictionsPage.tsx`):
    - **Sentiment distribution chart** (recharts PieChart or horizontal stacked bar):
      - Positive: zinc-700, Negative: zinc-300 — keep it monochrome
      - Show percentage labels
    - **Confidence distribution histogram** (recharts BarChart):
      - Bucket confidences (0.5–0.6, 0.6–0.7, ..., 0.9–1.0)
      - Zinc-900 bars
    - **Latency distribution**: P50, P95, P99 as inline stats (not a chart)
    - **Recent predictions table**: time, sentiment, confidence, latency, model version

37. Create Pipeline page (`pages/PipelinePage.tsx`):
    - **Pipeline runs table**: DAG name, status (dot: green/red/gray), started, duration, metrics summary
    - Status dot: green for success, red for failed, gray for running
    - Compact, dense table layout

38. Wire up routing in `App.tsx`:
    - / → Overview
    - /experiments → Experiments
    - /drift → Drift
    - /predictions → Predictions
    - /pipeline → Pipeline
    - AppLayout wraps all routes

### PHASE 6: DevOps & Documentation

39. Create Dockerfiles:
    - `serving/Dockerfile`: Python 3.11, multi-stage, non-root user, health check
    - `dashboard/Dockerfile`: Node 20 → Vite build → nginx Alpine, SPA routing + API proxy

40. Create seed/demo script `ml/scripts/demo_setup.py`:
    - Download dataset (small subset: 5000 samples for quick demo)
    - Train baseline model only (fast, no GPU needed)
    - Register in MLflow as Production
    - Generate 100 sample predictions (stored in DB)
    - Generate 1 sample drift report
    - Create 1 sample alert
    - This allows the dashboard to have data immediately on `docker-compose up`

41. Create `.github/workflows/ci.yml`:
    - `ml-tests`: pytest on ml/ directory (mock MLflow, mock data downloads)
    - `serving-tests`: pytest on serving/ with PostgreSQL + Redis services
    - `dashboard-build`: npm ci + npm run build
    - `docker-build`: smoke test building all images

42. Write comprehensive `README.md`:
    - One-paragraph project description
    - Architecture diagram (ASCII)
    - Tech stack table mapping tool → purpose
    - Features list with checkmarks
    - Prerequisites: Docker, 8GB+ RAM (for DistilBERT), optional GPU
    - Quick start: `docker-compose up --build` + `python ml/scripts/demo_setup.py`
    - MLflow UI screenshot placeholder (localhost:5000)
    - Airflow UI screenshot placeholder (localhost:8080)
    - Dashboard screenshot placeholder
    - API documentation (link to /docs)
    - Pipeline DAG explanation with diagram
    - Drift detection explanation
    - Model comparison strategy explanation
    - Environment variables table
    - Testing instructions
    - Deployment guide
    - Performance benchmarks table
    - License (MIT)

## QUALITY REQUIREMENTS

### ML Code:
- Every function has type hints
- Every script uses `logging` module (not print)
- Seeds set at the start of every training function
- No hardcoded paths — all from config
- MLflow tracking is wrapped in try/except (tracking failures shouldn't crash training)
- Models saved with all preprocessing artifacts (vectorizer + model for baseline, tokenizer + model for transformer)
- Confusion matrices saved as clean, minimal PNGs (black text on white, no color heatmap)

### API Code:
- Pydantic schemas for all request/response models
- Proper HTTP status codes
- Async everywhere (async def on all route handlers)
- Structured error responses
- Rate limiting on prediction endpoints
- Fire-and-forget prediction logging (don't slow down inference)

### Dashboard Code:
- TypeScript strict mode
- All API responses typed
- Auto-refresh every 30 seconds (configurable)
- Loading skeletons (zinc-200 pulse blocks), not spinners
- Empty states with helpful messages
- No `any` types

### Design (enforce on every component):
- Run the "would this look at home on Vercel Analytics?" test
- If any element has a gradient or glow → remove it
- If any icon is > 18px → shrink it
- Charts: zinc-900 data, zinc-200 grid, zinc-500 labels. Color ONLY for alert indicators

## FINAL CHECKLIST

- [ ] `docker-compose up --build` starts all 6 services
- [ ] `python ml/scripts/demo_setup.py` populates demo data
- [ ] MLflow UI at localhost:5000 shows experiments and runs
- [ ] Airflow UI at localhost:8080 shows both DAGs
- [ ] Training DAG runs end-to-end (at least with baseline model on small data)
- [ ] Drift detection DAG runs without errors
- [ ] POST /api/v1/predict returns valid prediction
- [ ] POST /api/v1/predict/batch handles multiple texts
- [ ] GET /api/v1/health shows model loaded with correct version
- [ ] Dashboard Overview shows KPIs and charts with real data
- [ ] Dashboard Experiments page shows MLflow runs
- [ ] Dashboard Drift page shows drift history
- [ ] Dashboard Predictions page shows prediction distribution
- [ ] Model comparison works: better model gets promoted, worse model stays in Staging
- [ ] All tests pass with 80%+ coverage on serving API
- [ ] CI pipeline passes
- [ ] README has architecture diagram, quick start, and screenshot placeholders
- [ ] UI has NO gradients, NO glows, NO purple-blue, NO oversized icons
- [ ] No print() statements — all logging module
- [ ] No hardcoded paths or secrets
