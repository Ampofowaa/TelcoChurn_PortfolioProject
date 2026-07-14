# Telco Customer Churn — End-to-End ML Portfolio Project

[![CI](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml/badge.svg)](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml)

Predicts which telecom customers are likely to churn and quantifies the revenue impact of early intervention. Built as a production-grade system covering the full MLOps lifecycle: validation → data ingestion → feature engineering → model training → calibration → cost-sensitive threshold optimisation → serving → monitoring.

Full modelling rationale, hyperparameter search, error analysis, SHAP explainability, and business impact → **[ANALYSIS.md](ANALYSIS.md)**
Engineering build plan (15 phases, current status) → **[PROJECT_PLAN.md](PROJECT_PLAN.md)**

---

## Results (development-set diagnostics — sealed test-set evaluation is a Phase 7 deliverable)

*Figures below are out-of-fold (OOF) diagnostics on the development set, not sealed test-set results — `models/evaluate.py` (Phase 7) has not yet run. See [ANALYSIS.md §7](ANALYSIS.md#7-final-test-set-results) for why the sealed test set stays untouched until then.*

| Metric | Value |
|---|---|
| Model family | LightGBM — **+0.007 PR-AUC** over `LogisticRegressionCV` (95 % CI [+0.002, +0.012], excludes zero) |
| CV PR-AUC (tuned, 1-SE rule) | **0.6690** |
| Calibration method | Sigmoid (`CalibratedClassifierCV`) — selected over isotonic via a PR-AUC-preservation gate |
| Pooled Brier (calibrated) | **0.1345** — 16.5 % better than uncalibrated |
| Brier Skill Score | 0.3098 |
| Production threshold (base scenario) | **0.3941** — closed-form `t* = c / (r × LTV)`, no leakage |
| Implied contact rate | 30.8 % of the development set |

---

## Dataset

| Property | Detail |
|---|---|
| Source | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (public) |
| Rows | 7,043 customers |
| Features | 20 (demographics, account info, 9 service flags) |
| Target | `Churn` — left within the last month |
| Class split | 73.5 % No / 26.5 % Yes (2.8:1 imbalance) |
| Missing data | 11 `TotalCharges` NaN — zero-tenure customers, imputed with median |

---

## Pipeline

```
Raw CSV
  └─ Pandera validation (5 quality gates) → ingest to Postgres (`customers_raw`)
       └─ Feature engineering (SQL views on Postgres; 9-lap OOF discovery; charge_per_service adopted)
            └─ LightGBM baseline → Optuna tuning (50 trials, TPE)
                 └─ Sigmoid calibration (CalibratedClassifierCV, cv=5) → MLflow registration (challenger alias)
                      └─ OOF cost-optimised threshold (no leakage)
                           └─ Sealed-test evaluation + full-data refit → champion (Phase 7)
                                └─ FastAPI serving + Streamlit UI
```

**Data splits** — stratified by `customerid`, sealed once before feature discovery (`data/split.py`):

| Split | n | Churners | Role |
|---|---|---|---|
| Dev | 5,634 | 1,495 (26.5 %) | Cross-validation (`RepeatedStratifiedKFold`, 10×10) for every modelling decision — family selection, feature selection, hyperparameter tuning, calibration |
| Test | 1,409 | 374 (26.5 %) | Final evaluation — sealed until all decisions finalised (Phase 7) |

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Project foundation (tooling, pre-commit, configs) | ✅ Done |
| 1 | Data ingestion — CSV → Postgres | ✅ Done |
| 2 | Data validation — Pandera + 5 quality gates | ✅ Done |
| 3 | EDA notebook | ✅ Done |
| 4a | Feature discovery — 9-lap OOF search → adoption gate | ✅ Done |
| 4b | Feature engineering — SQL views + Pandera-validated feature interface | ✅ Done |
| 5 | Model training — LightGBM + Optuna tuning + MLflow logging | ✅ Done |
| 6 | Calibration + cost-sensitive threshold | ✅ Done |
| 7–14 | Evaluation → serving → cloud → monitoring | Planned |

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full phase-by-phase roadmap.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Package management | [uv](https://docs.astral.sh/uv/) |
| Data ingestion | SQLAlchemy → Postgres (`customers_raw`), idempotent upsert + row-count assertion |
| Data validation | Pandera (5 quality gates) |
| Feature engineering | SQL views on Postgres (`customer_features`) |
| Experiment tracking | MLflow |
| Modelling | LightGBM (tuned challenger); `LogisticRegressionCV` baseline compared; `DummyClassifier(strategy='prior')` as leakage safeguard + BSS reference |
| Hyperparameter tuning | Optuna (TPE sampler, 50 trials) |
| Calibration | `CalibratedClassifierCV` (sigmoid) |
| Cost-sensitive threshold | Closed-form `t* = c / (r × LTV)`, 3-scenario cost model |
| Explainability | SHAP (`TreeExplainer`) |
| Pipeline versioning (Phase 8) | DVC |
| Serving (Phase 9) | FastAPI + Streamlit |
| Orchestration (Phase 10) | Prefect |
| Infrastructure | Docker, Postgres |
| CI/CD | GitHub Actions |
| Cloud (Phase 12) | AWS ECR + App Runner + RDS + S3 |

---

## Quick Start

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), and a [Kaggle API token](https://www.kaggle.com/settings/api).

**1 — Clone and install**

```bash
git clone <repo-url>
cd TelcoChurn_PortfolioProject
uv sync
pre-commit install
```

**2 — Download the dataset**

```bash
make data
```

Places the raw CSV at `datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

**3 — Start Postgres and load the data**

```bash
cp .env.example .env   # defaults work with Docker out of the box
make db-up             # start postgres:16 in Docker
make ingest            # load 7,043 rows → customers_raw
```

**4 — Run validation and tests**

```bash
make validate          # run the 5 Pandera quality gates
uv run pytest          # unit tests (no Docker required)
make test-integration  # integration tests (requires Docker)
```

**5 — Build features**

```bash
make features          # build SQL views → write datasets/processed/telco_churn_processed.csv
```

**6 — Train, calibrate, and derive the threshold**

```bash
make train                                     # LightGBM + Optuna tuning; logs to MLflow (note the printed run_id)
make calibrate RUN_ID=<run_id from above>      # sigmoid calibration; registers the challenger model version
make threshold MODEL_VERSION=<version above>   # closed-form cost-sensitive threshold, ships configs/policy/threshold.yaml
```

Each step's console output (structlog JSON) prints the `run_id` / `model_version` the next command needs.

**7 — Browse experiment runs**

`make db-up` already started the MLflow tracking server alongside Postgres —
open [http://localhost:5000](http://localhost:5000) to explore the logged runs, the registered `telco-churn-pipeline` model, and the `challenger` alias.

---

## Project Structure

```
src/telco_churn/
  data/                  # ingest, Pandera validation/schema, dev/test split
  features/              # SQL feature views, 9-lap discovery, permutation-importance selection
  models/
    train/                # candidate comparison, feature freeze, Optuna tuning, model logging
    calibrate.py           # CalibratedClassifierCV method selection + registration (challenger)
    threshold.py            # closed-form cost-sensitive threshold derivation
    plots.py, diagnostics.py
  utils/                 # paths, logging, db, mlflow, stats helpers
configs/                 # Hydra YAML — training/, tuning/, calibration/, threshold/, costs.yaml, policy/
sql/                     # Postgres schema + feature SQL views
tests/unit/              # pytest unit tests (≥80 % coverage target)
tests/integration/       # Postgres-backed tests (ingest, split, sql_features, validate) + subprocess CLI tests (train, calibrate, threshold)
pipelines/               # Prefect flows (retrain, drift check, batch predict) — Phase 10
monitoring/              # Prometheus config + Grafana dashboard JSON — Phase 13
datasets/                # gitignored; tracked by DVC (Phase 8)
mlruns/                  # gitignored; MLflow local tracking store
notebooks/               # 00–04: ingestion → EDA → feature discovery/engineering →
                         #        model selection/feature selection/hyperparameter tuning →
                         #        calibration & threshold (05-error-analysis is Phase 7)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, make commands, and PR conventions.
