# Telco Customer Churn — End-to-End ML Portfolio Project

[![CI](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml/badge.svg)](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml)

Predicts which telecom customers are likely to churn and quantifies the revenue impact of early intervention. Built as a production-grade system covering the full MLOps lifecycle: data ingestion → validation → feature engineering → model training → calibration → cost-sensitive threshold optimisation → serving → monitoring.

Full modelling rationale, hyperparameter search, error analysis, SHAP explainability, and business impact → **[ANALYSIS.md](ANALYSIS.md)**
Engineering build plan (15 phases, current status) → **[PROJECT_PLAN.md](PROJECT_PLAN.md)**

---

## Results (sealed test set, n = 1,409)

| Metric | Value |
|---|---|
| ROC-AUC | **0.8413** |
| Recall | **0.786** — 294 of 374 churners caught |
| Precision | 0.524 |
| F1 | 0.629 |
| Production threshold | 0.2956 (OOF cost-optimised, no val/test leakage) |
| Budget required | 39.8 % of customer base |
| Churners caught vs random at same budget | **+97 %** |
| Annualised P&L uplift — base scenario | **$753,806** |

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
  └─ Pandera validation (5 quality gates)
       └─ Feature engineering (9-lap OOF discovery; charge_per_service adopted)
            └─ LightGBM baseline → Optuna tuning (50 trials, TPE)
                 └─ Sigmoid calibration (CalibratedClassifierCV, cv=5)
                      └─ OOF cost-optimised threshold (no leakage)
                           └─ MLflow registration (champion alias)
                                └─ FastAPI serving + Streamlit UI
```

**Data splits** — stratified, fixed before any modelling:

| Split | n | Churners | Role |
|---|---|---|---|
| Train | 5,070 | ~26.5 % | Fitting + CV folds |
| Val | 564 | 150 (26.6 %) | Diagnostics — never used for model selection |
| Test | 1,409 | 374 (26.5 %) | Final evaluation — sealed until all decisions finalised |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Package management | [uv](https://docs.astral.sh/uv/) |
| Data validation | Pandera (5 quality gates) |
| Experiment tracking | MLflow |
| Modelling | LightGBM (tuned champion); XGBoost + Logistic Regression (baselines compared) |
| Hyperparameter tuning | Optuna (TPE sampler, 50 trials) |
| Calibration | `CalibratedClassifierCV` (sigmoid) |
| Explainability | SHAP (`TreeExplainer`) |
| Pipeline versioning | DVC |
| Serving | FastAPI + Streamlit |
| Orchestration | Prefect |
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

**6 — Browse experiment runs (Phase 5+)**

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open [http://localhost:5000](http://localhost:5000) to explore logged experiments.

---

## Project Structure

```
src/telco_churn/        # importable Python package
configs/                # Hydra YAML configs (model params, paths, thresholds)
sql/                    # Postgres schema + feature SQL views
tests/unit/             # pytest unit tests (≥80 % coverage target)
tests/integration/      # integration tests (Postgres, API, pipeline smoke)
pipelines/              # Prefect flows (retrain, drift check, batch predict)
monitoring/             # Prometheus config + Grafana dashboard JSON
datasets/               # gitignored; tracked by DVC (Phase 8)
mlruns/                 # gitignored; MLflow local tracking store
notebooks/              # EDA and error-analysis notebooks
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, make commands, and PR conventions.

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
| 5–14 | Model training → serving → cloud → monitoring | Planned |

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full phase-by-phase roadmap.
