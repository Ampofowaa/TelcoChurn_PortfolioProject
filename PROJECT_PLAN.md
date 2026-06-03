# Telco Churn ML Project — Build Plan

## What This Project Is

An end-to-end machine learning system that predicts which telecom customers will churn and quantifies the revenue impact of early intervention. The project is built to production-grade standards: modular Python package, automated tests, reproducible data pipeline, model registry, REST API, continuous training, and cloud deployment.

The intended audience for the deployed system is a business stakeholder asking: *"Which customers should we call this week, and what is the expected return?"*
The intended audience for this codebase is a technical hiring manager asking: *"Can this person build and operate an ML system, not just train a model?"*

Full modelling rationale and business results are documented in `README.md` and `ANALYSIS.md`. This document is the engineering build plan.

---

## Background

A PhD researcher in Operations Research/Management Science built a thorough single-notebook churn model on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The notebook (`notebooks/_archive/EDA-original.ipynb`, 148 cells) already contains rigorous science: EDA, statistical testing, sklearn `Pipeline` + `ColumnTransformer` preprocessing, baseline model selection, Optuna hyperparameter tuning, sigmoid calibration, cost-sensitive threshold derivation (three scenarios), SHAP error analysis, MLflow logging, and a sealed test-set evaluation with 1,000-iteration bootstrap confidence intervals.

**What was missing was the engineering and operations wrapper around that science** — the part that turns a notebook into a system a team can own, operate, and extend:

- Modular, testable Python package
- Dependency lock and reproducible environment
- Data validation in code (Pandera)
- Data version control (DVC)
- Configuration management (Hydra)
- Unit and integration tests (≥ 80 % coverage)
- Pre-commit hooks, linting, formatting, type-checking
- CI/CD (GitHub Actions)
- SQL-driven feature pipeline (Postgres in Docker)
- Workflow orchestration and continuous training (Prefect)
- Real-time serving (FastAPI) and demo UI (Streamlit)
- Containerisation (Docker / Docker Compose)
- Cloud deployment (AWS)
- Drift and performance monitoring (Evidently + Prometheus + Grafana)

This plan describes how to build that wrapper, phase by phase, without re-doing the science.

---

## Architecture Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Cloud | AWS | Strongest resume signal; free tier covers all required services |
| Serving | FastAPI | Industry-standard Python API framework; native async |
| UI | Streamlit | Clean separation from the API; widely used in industry DS demos |
| Orchestration | Prefect 3 | Best learning-curve-to-signal ratio; handles retraining and drift DAGs |
| SQL layer | Postgres in Docker | Mirrors warehouse-driven DS workflows; shows SQL fluency |
| Experiment tracking | MLflow | Already wired in the notebook; promote to Postgres + S3 backend |

---

## Tech Stack

| Concern | Tool |
|---|---|
| Environment / dependencies | `uv` + `pyproject.toml` + `uv.lock` |
| Lint / format | `ruff` + `black` |
| Type checking | `mypy` (strict, `src/` only) |
| Pre-commit | `pre-commit` framework |
| Data validation | `pandera` (schema + 5 quality gates) |
| Data version control | `dvc` (local cache → S3 in Phase 12) |
| Configuration | `hydra` (YAML under `configs/`) |
| SQL store | Postgres 16 in Docker; SQLAlchemy from Python |
| Experiment tracking | MLflow (local `mlruns/` → RDS + S3 in Phase 12) |
| Model registry | MLflow Model Registry (`champion` / `challenger` aliases) |
| Hyperparameter tuning | Optuna (TPE sampler, 50 trials) |
| Modelling | LightGBM + scikit-learn Pipelines |
| Explainability | SHAP |
| Testing | `pytest` + `hypothesis` (property-based) |
| Serving | FastAPI + Pydantic v2 + uvicorn |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| Orchestration | Prefect 3 (self-hosted; UI included) |
| Drift detection | Evidently AI |
| Metrics / alerting | Prometheus + Grafana |
| Logging | `structlog` (JSON structured logs) |
| CI/CD | GitHub Actions (free tier) |
| Container registry | Amazon ECR |
| Deployment | AWS App Runner |
| Secrets | AWS Secrets Manager (`.env` + `python-dotenv` locally) |

---

## Repository Layout (target)

```
TelcoChurn_PortfolioProject/
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint, type-check, unit tests on every push/PR
│       ├── integration.yml     # integration tests (Docker) on PRs to main
│       ├── cd.yml              # build + push image + deploy on merge to main
│       └── data-quality.yml    # weekly DVC pull + Pandera validation
├── .pre-commit-config.yaml
├── pyproject.toml              # deps + ruff/black/mypy/pytest config
├── uv.lock
├── Makefile
├── Dockerfile                  # FastAPI serving image
├── Dockerfile.ui               # Streamlit image
├── docker-compose.yml          # local stack: postgres + mlflow + prefect + api + ui + prometheus + grafana
├── .env.example
├── .gitignore
├── dvc.yaml                    # pipeline: ingest → validate → features → train → evaluate
│
├── configs/                    # Hydra config tree
│   ├── config.yaml             # top-level (paths, MLflow URI, random seed, validation thresholds)
│   ├── data/telco.yaml
│   ├── model/lightgbm.yaml
│   ├── training/optuna.yaml
│   └── serving/api.yaml
│
├── sql/
│   ├── schema/001_create_raw.sql
│   └── features/
│       ├── tenure_buckets.sql
│       ├── charge_per_service.sql
│       └── customer_features.sql   # final feature view
│
├── src/telco_churn/            # importable package
│   ├── __init__.py             # exposes __version__
│   ├── data/
│   │   ├── ingest.py           # CSV → Postgres loader (idempotent)
│   │   ├── schema.py           # Pandera RawSchema + CleanedSchema
│   │   ├── checks.py           # CheckResult types + 5 gate functions
│   │   └── validate.py         # orchestrates gates; structured logging; report writer
│   ├── features/
│   │   ├── sql_features.py     # runs sql/features/*.sql via SQLAlchemy
│   │   └── build.py            # ColumnTransformer definition + any residual pandas transforms
│   ├── models/
│   │   ├── train.py            # Optuna + LightGBM + MLflow
│   │   ├── calibrate.py        # CalibratedClassifierCV
│   │   ├── threshold.py        # cost-sensitive threshold derivation
│   │   ├── evaluate.py         # test-set metrics + bootstrap CIs
│   │   └── register.py         # MLflow Model Registry promotion logic
│   ├── serving/
│   │   ├── app.py              # FastAPI app
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── predict.py          # champion model loader + predict_proba wrapper
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── monitoring/
│   │   ├── drift.py            # Evidently report generator
│   │   └── metrics.py          # Prometheus metric definitions
│   └── utils/
│       ├── logging.py          # structlog JSON config
│       ├── db.py               # SQLAlchemy engine factory
│       └── io.py
│
├── tests/
│   ├── conftest.py             # root: --run-integration flag; integration skip guard
│   ├── unit/
│   │   ├── conftest.py         # shared DataFrame fixtures
│   │   ├── test_checks.py
│   │   ├── test_validate.py
│   │   ├── test_features.py
│   │   ├── test_threshold.py
│   │   └── test_serving_schemas.py
│   └── integration/
│       ├── test_ingest_postgres.py
│       ├── test_sql_features.py
│       ├── test_train_pipeline.py
│       └── test_api.py
│
├── pipelines/                  # Prefect flows
│   ├── retrain.py
│   ├── drift_check.py
│   └── batch_predict.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/
│       ├── api_metrics.json
│       └── drift.json
│
├── notebooks/
│   ├── _archive/EDA-original.ipynb  # original 148-cell monolith; frozen reference
│   ├── 00-data-quality.ipynb        # Phase 2 — 5 gates on live Postgres data
│   ├── 01-eda.ipynb                 # Phase 3 — statistical tests, distributions
│   ├── 02-feature-experiments.ipynb # Phase 4 — SQL view + ColumnTransformer output
│   ├── 03-model-selection.ipynb     # Phase 5 — Optuna study + baseline comparison
│   ├── 04-calibration-and-threshold.ipynb  # Phase 6 — reliability diagrams + cost curves
│   └── 05-error-analysis.ipynb      # Phase 7 — SHAP + FN/FP analysis
│
├── datasets/                   # tracked by DVC, not Git
│   ├── raw/
│   ├── interim/
│   └── processed/
│
└── docs/
    ├── architecture.md
    ├── runbook.md
    └── monitoring.md
```

---

## Phase Checklist

| Phase | Goal | Key Deliverable | Status |
|---|---|---|---|
| 0 | Project foundation | `pyproject.toml`, pre-commit, skeleton dirs, Hydra root, structlog | ✅ Done |
| 1 | Data ingestion (CSV → Postgres) | `docker-compose.yml`, `sql/schema/`, `data/ingest.py` | ✅ Done |
| 2 | Data validation (Pandera + 5 gates) | `data/schema.py`, `data/checks.py`, `data/validate.py`, 40 tests | ✅ Done |
| 3 | EDA notebook | `notebooks/01-eda.ipynb` importing from `src/` | Not started |
| 4 | Feature engineering (SQL + Python) | `sql/features/*.sql`, `features/sql_features.py`, `features/build.py` | Not started |
| 5 | Model training (LightGBM + Optuna + MLflow) | `models/train.py`, `configs/model/`, `configs/training/` | Not started |
| 6 | Calibration + cost-sensitive threshold | `models/calibrate.py`, `models/threshold.py` | Not started |
| 7 | Evaluation + error analysis + registry promotion | `models/evaluate.py`, `models/register.py`, `notebooks/05-error-analysis.ipynb` | Not started |
| 8 | DVC pipeline wrap | `dvc.yaml` with 5 stages; reproducible end-to-end | Not started |
| 9 | Serving + UI | FastAPI (`/predict`, `/health`, `/metrics`) + Streamlit + Dockerfiles | Not started |
| 10 | Orchestration | Prefect retrain flow (weekly) + drift check flow (daily) | Not started |
| 11 | CI/CD | GitHub Actions `ci.yml`, `integration.yml`, `cd.yml`, `data-quality.yml` | Not started |
| 12 | AWS deployment | ECR + App Runner + RDS + S3 | Not started |
| 13 | Monitoring | Prometheus + Grafana + Evidently drift | Not started |
| 14 | Documentation polish | README, runbook, architecture diagram | Not started |

---

## Phased Execution Plan

Phases follow the data scientist's natural workflow: environment → ingest → validate → explore → features → train → evaluate → reproducibility → serving → orchestration → monitoring. Tests are written alongside each module — there is no separate testing phase. Each phase ends in a working, demonstrable state.

---

### Phase 0 — Project Foundation *(1–2 days)*

**What this achieves:** Any engineer can clone the repo, install dependencies with a single command, and run the test suite — with linting, type-checking, and secret detection enforced automatically on every commit.

**Deliverables:**
- `uv init` → `pyproject.toml` + `uv.lock` (Python 3.11+)
- `ruff`, `black`, `mypy` (strict on `src/`), `pytest` configured in `pyproject.toml`
- `.pre-commit-config.yaml`: ruff, black, mypy, end-of-file-fixer, check-yaml, detect-secrets
- Skeleton directories: `src/telco_churn/`, `tests/{unit,integration}/`, `configs/`, `sql/`, `pipelines/`, `monitoring/`, `docs/`
- `configs/config.yaml` — Hydra root (data paths, MLflow URI, random seed, validation thresholds)
- `src/telco_churn/utils/logging.py` — structlog JSON logging
- `.gitignore` extended for `mlruns/`, `datasets/`, `.dvc/cache/`, `*.log`, `.env`; `.env.example` added
- `docs/architecture.md` — Mermaid system diagram stub
- `Makefile` shortcuts: `lint`, `format`, `test`, `train`

**Verification:** `uv run pre-commit run --all-files` passes; `uv run pytest` runs cleanly on an empty suite.

---

### Phase 1 — Data Ingestion (CSV → Postgres) *(1–2 days)*

**What this achieves:** Raw data lives in a real database from day one, not a locally-read CSV. Every downstream step reads from Postgres, mirroring warehouse-driven workflows that interviewers expect to see.

**Deliverables:**
- `docker-compose.yml` — `postgres:16` service with persistent volume and env-var credentials
- `sql/schema/001_create_raw.sql` — `customers_raw` table with explicit types (`totalcharges NUMERIC NULL` — 11 zero-tenure customers have no first bill in the source CSV; the column must be nullable)
- `src/telco_churn/data/ingest.py` — idempotent CSV → Postgres loader; accepts `--csv-path` via `argparse`
- `src/telco_churn/utils/db.py` — SQLAlchemy engine factory reading `POSTGRES_URL` from the environment
- `tests/unit/test_ingest.py` — unit tests for parsing helpers
- `tests/integration/test_ingest_postgres.py` — `testcontainers` Postgres; asserts row count and schema after load

> **Data acquisition note (Phase 14 docs):** `make data` calls `uv run kaggle datasets download -d blastchar/telco-customer-churn`. The Kaggle CLI accepts either a `KGAT_` token (`~/.kaggle/access_token`) for new accounts or a legacy `kaggle.json` for old ones. The Kaggle API is deliberately not wired into the ingestion path — the dataset is static and DVC (Phase 8) already provides content-hashed reproducibility.

**Verification:** `docker compose up postgres -d && uv run python -m telco_churn.data.ingest` loads 7,043 rows; `SELECT COUNT(*) FROM customers_raw` returns 7043.

---

### Phase 2 — Data Validation (Pandera + the 5 Gates) *(1–2 days)*

**What this achieves:** Data quality problems are caught automatically, with a clear severity model — blocking errors stop the pipeline immediately; warnings are logged and allow it to continue. Any schema drift (unexpected column or type change) requires a conscious code change, not just a silent load.

**Deliverables:**
- `src/telco_churn/data/schema.py` — `RawSchema` (21 columns, `totalcharges` nullable) and `CleanedSchema` (inherits; overrides `totalcharges` non-nullable); `Config.strict = True` so unexpected columns are a blocking error
- `src/telco_churn/data/checks.py` — `Severity` (ERROR / WARNING), `CheckResult` (frozen dataclass; `failure_severity` describes what happens *if* the check fails, not the data state; `detail` carries row-level evidence), `ValidationResult` (computed `errors` / `warnings` / `passed`); five gate functions:
  1. `check_schema` — Pandera schema validation
  2. `check_duplicate_ids` — no customer appears twice
  3. `check_churn_labels` — target is strictly binary {0, 1}
  4. `check_totalcharges_nulls` — null rate within the known 11-row tolerance
  5. `check_distribution_sanity` — row count ≥ `validation.min_rows`; per-column null rate ≤ `validation.max_null_rate` (both configurable in `configs/config.yaml`)
- `src/telco_churn/data/validate.py` — `validate_raw` / `validate_clean` (both accept `strict` and `reports_dir`; `strict=True` for the DVC stage entry point, `strict=False` for the Prefect flow which inspects the full `ValidationResult`); `save_validation_report` (timestamped `summary.csv` + per-check `_failures.csv`); `clean_dataframe` (Phase 2 placeholder — median-imputes NULL `totalcharges`; **removed in Phase 8** when the features stage takes over imputation); `__main__` CLI exits 0 on pass, 1 on failure
- `tests/unit/conftest.py` — shared fixtures: `valid_raw_df`, `zero_tenure_df`, `empty_telco_df`, `large_valid_df`
- `tests/unit/test_checks.py` — 26 tests: `ValidationResult` properties; Gates 1–5 pass/fail/severity/detail assertions
- `tests/unit/test_validate.py` — 14 tests: `validate_raw` strict/non-strict; `validate_clean`; `clean_dataframe` imputation; `save_validation_report` artifacts; `hypothesis` property tests for NaN propagation and column isolation
- `notebooks/00-data-quality.ipynb` — runs the 5 gates against the live Postgres table; renders `summary.csv` and `schema_failures.csv` for inspection

**Phase 8 cleanup required:**
- Remove `clean_dataframe()` once the Phase 4 `features` stage owns imputation via a fitted `sklearn SimpleImputer` that stores the training-set median
- In the DVC stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event listing which gates failed, then call `sys.exit(1)` — do not let Python's default traceback handler produce the exit, which is noisy and unsearchable in pipeline logs

**Verification:** `uv run python -m telco_churn.data.validate` against the loaded Postgres data exits 0; an injected violation exits 1 and writes a timestamped report to `reports/validation/`.

---

### Phase 3 — Exploratory Data Analysis Notebook *(1 day)*

**What this achieves:** The original 148-cell monolith is archived and replaced by a clean, importable-function-backed notebook. A reviewer can read it as a narrative — not as a research scratch-pad — and verify that the modelling decisions are grounded in the data.

**Deliverables:**
- `notebooks/01-eda.ipynb` — slim notebook that imports from `src/` and renders outputs; covers:
  - Churn rate by contract type, tenure cohort, and internet service type
  - Distribution plots for `MonthlyCharges` and `TotalCharges` (churned vs. retained)
  - Chi-squared tests for categorical features vs. churn
  - Correlation heatmap and VIF check for multicollinearity
  - Class imbalance summary (73.5 % No / 26.5 % Yes)
- Any helper functions promoted to `src/telco_churn/data/` or `src/telco_churn/features/` as needed; no duplicated logic between notebook and `src/`

> The original `notebooks/_archive/EDA-original.ipynb` remains frozen. It is the authoritative reference for all modelling decisions; it is not retroactively split or edited.

**Verification:** Notebook executes end-to-end without errors (`jupyter nbconvert --execute notebooks/01-eda.ipynb`).

---

### Phase 4 — Feature Engineering (SQL + Python) *(2 days)*

**What this achieves:** Features are built in two layers — SQL views in Postgres (tenure bucketing, charge-per-service ratios) and a scikit-learn `ColumnTransformer` (scaling, encoding, imputation). This mirrors how features are built in a real warehouse-backed ML system. The ColumnTransformer definition is lifted verbatim from the original notebook to preserve the science exactly.

**Deliverables:**
- `sql/features/tenure_buckets.sql` — `CASE`-based tenure cohorts (e.g., 0–12, 13–24, 25–48, 49+ months)
- `sql/features/charge_per_service.sql` — `MonthlyCharges` divided by the count of active services; NULL-safe
- `sql/features/customer_features.sql` — final feature view joining `customers_raw` with both SQL-derived features; this is the table the Python feature builder reads
- `src/telco_churn/features/sql_features.py` — `build_sql_features(engine)` runs the three SQL files in dependency order via SQLAlchemy; idempotent (`CREATE OR REPLACE VIEW`)
- `src/telco_churn/features/build.py` — `build_feature_matrix(df)` applying the `ColumnTransformer` lifted from `notebooks/_archive/EDA-original.ipynb`:
  - Numeric pipeline: `SimpleImputer(strategy="median")` → `StandardScaler`
  - Binary pipeline: passthrough (already 0/1)
  - Categorical pipeline: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="first", handle_unknown="ignore")`
  - Saves fitted transformer to `datasets/processed/preprocessing.pkl`
- `tests/unit/test_features.py` — dtype invariants (`hypothesis`), no-NaN propagation, column count stability
- `tests/integration/test_sql_features.py` — `testcontainers` Postgres; assert view row count matches `customers_raw`; assert no NULL in derived columns where not expected
- `notebooks/02-feature-experiments.ipynb` — SQL view output vs. `ColumnTransformer` output side by side; engineered-feature distributions

**Verification:** `uv run python -m telco_churn.features.build` reads from the SQL feature view, applies the ColumnTransformer, and writes `datasets/processed/telco_churn_processed.csv` (7,043 rows, expected feature count).

---

### Phase 5 — Model Training (LightGBM + Optuna + MLflow) *(2–3 days)*

**What this achieves:** A reproducible, experiment-tracked training run that searches hyperparameters with Bayesian optimisation and logs every trial to MLflow. The best model is registered in the MLflow Model Registry as `challenger`, ready for evaluation and promotion.

**Deliverables:**
- `configs/model/lightgbm.yaml` — LightGBM parameter ranges (lifted from the original notebook's Optuna study as warm-start defaults): `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- `configs/training/optuna.yaml` — Optuna config: `n_trials: 50`, `sampler: tpe`, `direction: maximize`, `metric: roc_auc`, `cv_folds: 5`, `random_state: 42`
- `src/telco_churn/models/train.py`:
  - Reads config via Hydra
  - Loads the feature matrix from `datasets/processed/`
  - Stratified 5-fold cross-validation; optimisation metric is ROC-AUC (validation set)
  - Class imbalance: `scale_pos_weight` set to negative/positive ratio (~2.77)
  - Each Optuna trial is a nested MLflow child run; the study itself is the parent run
  - Best model logged as `pyfunc` artifact with `feature_columns.txt` and `preprocessing.pkl`
  - Best model registered in MLflow Model Registry as `telco-churn-pipeline` with alias `challenger`
- `tests/unit/test_train.py` — config loading, metric logging contract (mock MLflow client)
- `notebooks/03-model-selection.ipynb` — loads the Optuna study from MLflow; renders parallel-coordinates plot, hyperparameter importance, and validation AUC distribution across trials

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and produces an MLflow run whose cross-validation AUC falls within the bootstrap CI reported in `README.md`.

---

### Phase 6 — Calibration + Cost-Sensitive Threshold *(1 day)*

**What this achieves:** The model outputs calibrated probabilities (not raw scores), and the decision threshold is set to reflect the actual business cost of a missed churner versus a wasted retention call — not the default 0.5. Three cost scenarios are evaluated so the business can choose the level of risk they are comfortable with.

**Deliverables:**
- `src/telco_churn/models/calibrate.py` — `CalibratedClassifierCV` wrapping the best LightGBM model; tests both sigmoid and isotonic methods; keeps whichever achieves the lower Brier score; logs calibrated model as a new MLflow artifact
- `src/telco_churn/models/threshold.py` — cost-sensitive threshold search over the OOF probability distribution; three scenarios from `notebooks/_archive/EDA-original.ipynb`:
  - **Conservative** (high cost of a missed churner): threshold ~0.22
  - **Base** (balanced): threshold ~0.30
  - **Optimistic** (high cost of a wasted call): threshold ~0.38
- `tests/unit/test_threshold.py` — synthetic cost matrix → expected threshold; degenerate cases (zero cost, single class)
- Both modules log artifacts to the same MLflow run as Phase 5
- `notebooks/04-calibration-and-threshold.ipynb` — reliability diagrams before and after calibration; cost curve annotated with each scenario threshold

**Verification:** Calibrated Brier score ≤ uncalibrated Brier score; base-scenario threshold matches the value documented in `README.md`.

---

### Phase 7 — Evaluation + Error Analysis + Registry Promotion *(2 days)*

**What this achieves:** A sealed test-set evaluation (the test set has never been touched until this point) produces bootstrap-confidence-interval-bounded metrics that are honest estimates of production performance. A structured promotion decision replaces the `challenger` alias with `champion` only when the new model improves on both recall and calibration.

**Deliverables:**
- `src/telco_churn/models/evaluate.py`:
  - Sealed test-set metrics: ROC-AUC, PR-AUC, recall, precision, F1, Brier score
  - 1,000-iteration bootstrap 95 % CIs (routine lifted verbatim from the original notebook)
  - Writes `reports/metrics.json`
  - Logs all metrics and the report to the MLflow run
- `src/telco_churn/models/register.py` — promotes `challenger` → `champion` if and only if it beats the current `champion` on both recall (≥ 0) and Brier score (lower is better); no promotion otherwise; logs the decision with structured event `model_promoted` or `model_rejected`
- `tests/unit/test_evaluate.py` — bootstrap CI math verified on a synthetic dataset with known population AUC
- `notebooks/05-error-analysis.ipynb` — SHAP global feature importance, SHAP local explanations for representative FN/FP cases, confusion matrix at each cost scenario threshold

**Verification:** `uv run python -m telco_churn.models.evaluate` produces `reports/metrics.json` whose AUC CI overlaps the CI reported in `README.md`.

---

### Phase 8 — DVC Pipeline Wrap *(1 day)*

**What this achieves:** The five-stage pipeline (ingest → validate → features → train → evaluate) becomes a content-hashed DAG. Changing a hyperparameter re-runs only the training and evaluation stages — not the full pipeline. This is the reproducibility guarantee that separates a real MLOps workflow from ad-hoc notebooks.

**Deliverables:**
- `dvc init`; `dvc add datasets/raw/Telco-Customer-Churn.csv`
- `dvc.yaml` — five stages with explicit `deps` (code + configs) and `outs` (data artifacts, models, metrics):

  | Stage | Deps | Outs |
  |---|---|---|
  | `ingest` | `data/ingest.py`, raw CSV | `customers_raw` table hash |
  | `validate` | `data/validate.py`, `data/schema.py` | `reports/validation/` |
  | `features` | `features/build.py`, `sql/features/` | `datasets/processed/`, `preprocessing.pkl` |
  | `train` | `models/train.py`, `configs/` | MLflow run ID, `feature_columns.txt` |
  | `evaluate` | `models/evaluate.py` | `reports/metrics.json` |

- DVC local remote configured for now; swapped to S3 in Phase 12
- **Phase 2 cleanup:** Remove `clean_dataframe()` from `validate.py` — imputation now belongs to the `features` stage's fitted `SimpleImputer`. Update `validate_clean()` to expect the features stage output directly. Remove the associated tests.
- **Phase 2 cleanup:** In the DVC `validate` stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event, and call `sys.exit(1)`.

**Verification:** Change a hyperparameter in `configs/model/lightgbm.yaml`; `dvc repro` re-runs `train` and `evaluate` only, not `ingest`, `validate`, or `features`.

---

### Phase 9 — FastAPI Serving + Streamlit UI *(3 days)*

**What this achieves:** The champion model is accessible via a REST API with a health endpoint, batch prediction support, and built-in Prometheus metrics. A Streamlit UI lets a non-technical user submit a customer profile and see the churn probability alongside the top reasons — using the same API.

**Deliverables:**
- `src/telco_churn/serving/schemas.py` — Pydantic v2 request/response models; field constraints aligned with the Pandera schema (single source of truth)
- `src/telco_churn/serving/predict.py` — loads the `champion` model and preprocessor from MLflow at startup; exposes `predict_single` and `predict_batch`
- `src/telco_churn/serving/app.py` — FastAPI app:
  - `POST /predict` — single customer prediction
  - `POST /predict/batch` — batch scoring (array of customers)
  - `GET /health` — liveness probe
  - `GET /ready` — readiness probe (model loaded)
  - `GET /metrics` — Prometheus metrics via `prometheus_fastapi_instrumentator`
  - Structured log per prediction: request ID, features, probability, threshold, decision
- `src/telco_churn/ui/streamlit_app.py` — 19-feature form → `POST /predict` → probability gauge + top-5 SHAP contributions
- `Dockerfile` (multi-stage, FastAPI) + `Dockerfile.ui` (Streamlit); both added to `docker-compose.yml`
- `tests/integration/test_api.py` — FastAPI test client: `/predict` returns valid schema; `/health` returns 200; batch endpoint accepts arrays

**Verification:** `docker compose up && curl -X POST http://localhost:8000/predict -d @example_payload.json` returns a churn probability; Streamlit at `:8501` displays a prediction with SHAP contributions.

---

### Phase 10 — Prefect Orchestration (Continuous Training) *(2–3 days)*

**What this achieves:** The model retrains automatically every week and checks for data drift every day — without manual intervention. The Prefect UI provides a full audit trail of every run, including failures.

**Deliverables:**
- Prefect 3 server added to `docker-compose.yml` (UI at `:4200`)
- `pipelines/retrain.py` — weekly Sunday 02:00 schedule; runs `ingest → validate → features → train → evaluate → register`; promotes `challenger` to `champion` on recall AND Brier improvement
- `pipelines/drift_check.py` — daily 06:00; pulls the last 24 hours of predictions from the API structured logs; runs an Evidently data drift report; alerts (Prefect notification) when PSI > 0.2 on any top-5 feature
- `pipelines/batch_predict.py` (optional) — nightly scoring of all customers; writes results to a `predictions` table in Postgres

**Verification:** Trigger each flow from the Prefect UI; inject synthetic rows with shifted `MonthlyCharges`; confirm the drift alert fires within the next scheduled drift check run.

---

### Phase 11 — CI/CD with GitHub Actions *(2 days)*

**What this achieves:** Every pull request is gated — no broken code reaches `main`. Every merge to `main` automatically builds, pushes, and deploys the container. A weekly data-quality check catches upstream data problems before the next retrain.

**Deliverables:**
- `.github/workflows/ci.yml` — on every push and PR: `uv sync --frozen`, pre-commit, mypy, `pytest --cov=src --cov-fail-under=80` (unit tests only; no Docker required), Docker build (no push)
- `.github/workflows/integration.yml` — on PRs to `main` only: `docker compose --profile infra up -d`, wait for Postgres healthcheck, `pytest tests/integration/ --run-integration`, `docker compose down`
- `.github/workflows/cd.yml` — on merge to `main`: all of CI, then build + tag image with commit SHA, push to ECR, trigger App Runner redeploy
- `.github/workflows/data-quality.yml` — weekly cron: `dvc pull` from S3, run Pandera validation, alert on failure
- GitHub repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_TRACKING_URI`  # pragma: allowlist secret

**Verification:** Open a PR with a deliberate lint error → CI fails. Fix and merge → CD ships to AWS within ~10 minutes.

---

### Phase 12 — AWS Deployment *(2–3 days)*

**What this achieves:** The API and UI are publicly accessible via HTTPS URLs. All data and model artifacts are stored durably in S3. The MLflow tracking server runs against a managed Postgres database. Infrastructure is least-privilege throughout.

**Deliverables:**
- S3 buckets: `telco-churn-data` (DVC remote), `telco-churn-mlflow` (artifact store)
- RDS Postgres (`db.t3.micro`, free tier) — MLflow backend store + production application database
- ECR repositories for `api` and `ui` images
- App Runner services for both (auto-scales; scales to zero on idle)
- CloudWatch log groups; AWS Secrets Manager for database credentials; minimum-privilege IAM roles
- DVC remote swapped from local to S3 (`dvc remote modify ...`)
- `docs/architecture.md` updated with the deployed-system diagram

**Verification:** `curl https://<app-runner-url>/predict` with a valid payload returns a churn probability; the Streamlit URL loads and submits successfully.

---

### Phase 13 — Monitoring Stack *(2 days)*

**What this achieves:** Latency, error rates, and prediction drift are visible in real time. Alerts fire before users notice a problem. The monitoring setup mirrors what is expected in a production ML team.

**Deliverables:**
- Prometheus + Grafana added to `docker-compose.yml` (local); Grafana Cloud free tier (10K series) used in AWS
- FastAPI `/metrics` endpoint exposes: request count by endpoint, request latency (p50/p95/p99), prediction probability histogram, model version label
- Grafana dashboards:
  - **API health** — RPS, latency percentiles, error rate
  - **Prediction distribution** — rolling histogram vs. training reference
  - **Feature drift** — PSI per top-5 feature over time (sourced from Evidently reports)
- Alert rules: p95 latency > 500 ms; error rate > 1 %; PSI > 0.2 sustained over 24 hours
- `src/telco_churn/utils/logging.py` updated so `log_level` is read from `LOG_LEVEL` environment variable (fallback `"INFO"`), enabling temporary debug logging in production without a redeploy

**Verification:** Grafana dashboards populate within minutes of the API receiving traffic; simulated drift raises a PSI alert; load test shows p95 latency panel updating live.

---

### Phase 14 — Documentation Polish *(1 day)*

**What this achieves:** A recruiter or hiring manager can understand the project's scope, results, and architecture within 90 seconds of landing on the repo.

**Deliverables:**
- `README.md` — top of file: 1-paragraph elevator pitch, architecture diagram, "Quick demo" GIF, tech-stack table with phase links, headline metrics table
- `ANALYSIS.md` — full modelling narrative (already written; verify it references `src/` functions, not notebook code)
- `docs/runbook.md` — how to retrain, roll back to the previous champion, and debug a drift alert
- `docs/architecture.md` — complete system diagram (Mermaid), data lineage from raw CSV to API response, decision log for major architecture choices
- README "Project status & lessons learned" section: 3–4 bullet points on non-obvious choices and their trade-offs

---

## Execution Order

Phase 0 is the only hard prerequisite. Phases 1–8 are sequential — each phase's output is the next phase's input. After Phase 8, the engineering wrapper is largely independent:

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
                                    ↓
                              9 → 10 → 11 → 12 → 13 → 14
```

Conservative estimate: **5–7 weeks** part-time; **~3 weeks** full-time.

---

## Testing Discipline

Tests are a per-module habit, not a milestone:

- Every phase that adds code to `src/` adds matching unit tests in the same PR.
- Integration tests appear as their dependencies come online: Postgres tests from Phase 1, SQL feature tests from Phase 4, train-pipeline smoke from Phase 5, API tests from Phase 9.
- Property-based tests (`hypothesis`) cover data transforms — no NaN propagation, dtype invariants.
- **Coverage target: ≥ 80 % on `src/`**, enforced in CI from Phase 11 (`pytest --cov=src --cov-fail-under=80`) and locally via `[tool.coverage.report] fail_under = 80` in `pyproject.toml`.
- When writing data, schema, or validation tests, cover: normal case, missing values, wrong dtypes, and empty DataFrame.
- Run `pytest` before marking any task complete; if a phase has no tests yet, note it explicitly rather than skipping.

---

## Notebook Discipline

Notebooks are for exploration and narrative, not for owning logic:

1. Exploration starts in a phase-scoped notebook — new feature ideas, model comparisons, threshold scenarios.
2. Stable logic is promoted to `src/` once the experiment converges. The notebook becomes a thin demo: `from telco_churn... import X; render(X(...))`.
3. No duplicated logic between notebooks and `src/`. If a function lives in both places, the notebook copy is wrong by construction.
4. Notebooks must run end-to-end. CI (Phase 11) executes every notebook via `jupyter nbconvert --execute`; a broken notebook fails the build.
5. `notebooks/_archive/EDA-original.ipynb` is frozen. It is the authoritative reference for all modelling decisions; it is not retroactively split or edited.

| Notebook | Phase | What It Demonstrates |
|---|---|---|
| `00-data-quality.ipynb` | 2 | 5 Pandera gates on the live Postgres table; example violations |
| `01-eda.ipynb` | 3 | Statistical tests, distributions, churn-rate breakdowns |
| `02-feature-experiments.ipynb` | 4 | SQL view vs. ColumnTransformer output; engineered-feature distributions |
| `03-model-selection.ipynb` | 5 | Optuna study summary + baseline comparison loaded from MLflow |
| `04-calibration-and-threshold.ipynb` | 6 | Reliability diagrams + 3-scenario cost curves |
| `05-error-analysis.ipynb` | 7 | SHAP global/local plots + FN/FP analysis |

---

## Critical Files and Their Roles

| File | Role |
|---|---|
| `notebooks/_archive/EDA-original.ipynb` | **Source of truth for the modelling science.** Every function in `src/` migrates from here. Do not alter it. |
| `README.md` | Master narrative and project landing page; updated each phase |
| `ANALYSIS.md` | Full modelling rationale, hyperparameters, error analysis, and business impact |
| `pyproject.toml` | Dependencies + all tool configuration (single source of truth) |
| `dvc.yaml` | Reproducible pipeline graph |
| `docker-compose.yml` | One-command local stack for development |
| `configs/config.yaml` | Hydra root; controls data paths, thresholds, MLflow URI, random seed |
| `src/telco_churn/data/schema.py` | Pandera schema — referenced by validation, training, and FastAPI request models |
| `src/telco_churn/models/train.py` | Optuna + LightGBM + MLflow; called by both CLI and Prefect |
| `src/telco_churn/serving/app.py` | FastAPI app; loads `champion` model from MLflow Registry at startup |
| `pipelines/retrain.py` | Continuous-training DAG; ties together every `src/` module |
| `.github/workflows/cd.yml` | Deploy automation; ties code changes to AWS |

---

## Reuse from the Original Notebook

The science is done — do not redo it. These outputs from `notebooks/_archive/EDA-original.ipynb` are authoritative and must be carried through unchanged in logic:

| Notebook artifact | Destination in `src/` |
|---|---|
| 5 data quality gates | `data/schema.py` + `data/checks.py` |
| `ColumnTransformer` definition | `features/build.py` (lifted verbatim) |
| Optuna best hyperparameters | Default values in `configs/training/optuna.yaml` (still searchable; warm-start from these) |
| Cost-sensitive threshold logic (3 scenarios) | `models/threshold.py` |
| Bootstrap CI evaluation routine | `models/evaluate.py` |
| 8 documented limitations | README "known limitations" section; surfaced as Grafana alert thresholds where relevant |

---

## End-to-End Verification (post Phase 14)

1. **Reproducibility:** Fresh clone → `uv sync` → `docker compose up` → `dvc repro` → metrics within bootstrap CI of `README.md`.
2. **API correctness:** `pytest tests/integration/test_api.py` passes; manual `curl` to the deployed App Runner URL returns a churn probability.
3. **UI:** Streamlit form submission shows a probability with SHAP contributions; response matches the direct API call.
4. **Continuous training:** Trigger `retrain.py` from the Prefect UI → new MLflow run appears → if metrics beat the champion, the registry alias flips → next API container deploy picks up the new model.
5. **Drift detection:** Run `drift_check.py` after injecting synthetic drift → Evidently report shows the shift; Grafana PSI dashboard reflects the jump; Prefect notification fires.
6. **CI/CD:** Open a PR that breaks a unit test → CI blocks the merge. Fix and merge → CD ships to AWS within ~10 minutes.
7. **Monitoring:** Load-test the API; Grafana latency and error-rate panels update live.

---

## What This Plan Deliberately Does Not Include

These are reasonable "future work" items for the README, not omissions:

- **A/B testing infrastructure** — the README notes uplift modelling as a next step; not part of MVP
- **Online learning / streaming** — batch retraining is appropriate for a monthly-churn problem
- **Multi-model serving** — a single champion model is sufficient
- **API authentication** — fine for a portfolio demo; a production system would add OAuth/JWT
- **Multi-region / HA deployment** — single App Runner region is sufficient
- **Cost dashboards** — AWS Cost Explorer's free view covers the free-tier project

---

## Pre-Phase 3 To-Do

Items identified in a post-QA audit (June 2026). All must be resolved before Phase 3 begins. Ordered by priority.

| Priority | Status | Item | Detail |
|---|---|---|---|
| High | [x] | Fix 4 stale references in `CLAUDE.md` | (1) Source of Truth: `README.md` → `ANALYSIS.md` for modelling rationale; (2) Testing section example: `test_schema.py` → `test_checks.py`; (3) Phase 2 deliverable: `test_schema.py` → `test_checks.py` + `test_validate.py`; (4) Phase 7 notebook: `03-error-analysis.ipynb` → `05-error-analysis.ipynb` |
| High | [x] | Bump pre-commit mypy to match project requirement | `.pre-commit-config.yaml` pins `mirrors-mypy` at `v1.13.0`; `pyproject.toml` requires `mypy>=2.1.0`. Hook and project are running different mypy versions — they can diverge on what they catch. Update `rev` to the latest `mirrors-mypy` release that corresponds to mypy v2.x |
| Medium | [x] | Add direct Pandera schema constraint tests | Added `test_invalid_contract_type_fails_schema_check` and `test_unexpected_column_fails_schema_check` to `tests/unit/test_checks.py`. Other cases (`gender`, `churn`, NULL `totalcharges`) were already covered — no duplicate test file needed |

---

## Completed: QA & Standards Hardening (June 2026)

All items below were identified in a review against industry DS standards and completed before the repo was shared publicly. Full details are in `CHANGELOG.md`.

| Priority | Item | Resolution |
|---|---|---|
| Critical | GitHub Actions CI pipeline | Added `.github/workflows/ci.yml` (lint, type-check, unit tests + coverage) |
| Critical | README recruiter-facing rewrite | Slimmed to elevator pitch + results + pipeline overview + tech stack |
| Critical | Coverage threshold enforced locally | `fail_under = 80` in `[tool.coverage.report]` |
| High | PEP 561 typed-package marker | Added `src/telco_churn/py.typed` |
| High | `customerid` index in SQL schema | Covered by `PRIMARY KEY` constraint (implicit B-tree index) |
| High | `CHANGELOG.md` | Generated from Conventional Commits history |
| High | `clean_dataframe()` placeholder documented | Docstring explains Phase 8 removal and why |
| Medium | Gate 5 thresholds config-driven | `validation.min_rows` and `validation.max_null_rate` in `configs/config.yaml` |
| Medium | `__version__` exposed | `importlib.metadata.version()` in `src/telco_churn/__init__.py` |
| Medium | Integration test skip guard | `--run-integration` flag + `pytest.mark.integration` in root `conftest.py` |
| Low | `argparse` in `ingest.py` CLI | `--csv-path` argument replaces hardcoded path |
| Low | `CONTRIBUTING.md` | Setup, make commands, pre-commit hooks, PR conventions |
