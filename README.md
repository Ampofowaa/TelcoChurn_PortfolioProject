# Telco Customer Churn — End-to-End ML Portfolio Project

[![CI](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml/badge.svg)](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-835%20passing-brightgreen)](CONTRIBUTING.md#testing)
[![Coverage](https://img.shields.io/badge/coverage-95.7%25-brightgreen)](CONTRIBUTING.md#testing)

Predicts which telecom customers are likely to churn and quantifies the revenue impact of early intervention. Built as a production-grade system covering the full MLOps lifecycle: data ingestion → validation → feature engineering → model training → calibration → cost-sensitive threshold optimisation → serving → monitoring.

**0.670 PR-AUC · +$15,061 expected value per test cohort (95% CI [$11,215, $18,604]) · promotion gate: pass** — a one-time evaluation on the test set, held out since the original split and never touched until now. Full breakdown below.

Full modelling rationale, hyperparameter search, error analysis, SHAP explainability, and business impact → **[ANALYSIS.md](ANALYSIS.md)**
Rendered notebooks — every figure, every phase, outputs included → **[notebooks/](notebooks/)**
Engineering build plan (15 phases, current status) → **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
System architecture, ML workflow, data flow & MLflow layout diagrams → **[docs/architecture.md](docs/architecture.md)**
Release history → **[CHANGELOG.md](CHANGELOG.md)**
Known limitations & open questions → **[ANALYSIS.md §9](ANALYSIS.md#9-known-limitations)**

### Contents

- [Results](#results-sealed-test-set-evaluation)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Results (sealed test-set evaluation)

*Figures below are the one-time sealed test-set evaluation (n = 1,409 customers, untouched since the original split). Full breakdown, error analysis, and SHAP explainability → [ANALYSIS.md §7](ANALYSIS.md#7-final-test-set-results).*

| Metric | Value |
|---|---|
| Model family | LightGBM — **+0.007 PR-AUC** over `LogisticRegressionCV` (95 % CI [+0.002, +0.012], excludes zero) |
| **Test-set PR-AUC** | **0.670** (95 % CI [0.619, 0.714]) vs. a 0.265 dummy-prior floor |
| **Test-set recall** (shipped threshold) | **0.698** (95 % CI [0.651, 0.743]) — 261 of 374 churners caught |
| Brier Skill Score | **0.301** — Brier 0.136 vs. 0.195 dummy-prior floor |
| Calibration slope | **0.992** (95 % CI [0.891, 1.100]) — within the [0.80, 1.25] guardrail band |
| Production threshold (base scenario) | **0.3941** — closed-form `t* = c / (r × LTV)`, no leakage |
| Contact rate (base scenario) | 31.2 % of the test set |
| Expected value (base scenario) | **+$15,061** (95 % CI [$11,215, $18,604]) vs. both `treat-all` and `treat-none` baselines |
| **Promotion gate** | **Pass** (cold start) — human review **approved** |

**What drives churn, and what it's worth:**

- **Contract type and tenure dominate** — together they account for over a third of the model's total signal (SHAP); short-tenure, month-to-month customers are the highest-risk group by a wide margin.
- **The campaign pays for itself** — expected value beats both calling everyone and calling no one, under all three cost scenarios tested (conservative, base, optimistic — [full breakdown](ANALYSIS.md#7-final-test-set-results)).
- **Known blind spot: long-tenure, annual-contract customers — and the cause runs deeper than one segment.** The model rarely flags them, and the root cause is a signal missing from nearly every error type: nothing in this dataset measures customer loyalty or satisfaction directly. Two cheap fixes close part of the gap without new data (a manual outreach rule today, an engineered interaction feature next cycle); closing it fully needs new data collection ([business takeaways](ANALYSIS.md#business-takeaways)).
- **Retention effort is worth front-loading** — a customer's first few months move the predicted risk score more than any other window, so early outreach buys the most risk reduction per dollar.
- **Fairness reviewed, not just measured** — the disparities the model does show across protected groups track real underlying differences in churn risk rather than proxy discrimination, and are carried into ongoing monitoring rather than treated as a blocker.

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
  └─ Ingest to Postgres (`customers_raw`) → Pandera validation (5 quality gates)
       └─ Feature engineering (SQL views on Postgres; 9-lap OOF discovery; charge_per_service adopted)
            └─ LightGBM baseline → Optuna tuning (50 trials, TPE)
                 └─ Sigmoid calibration (CalibratedClassifierCV, cv=5) → MLflow registration (challenger alias)
                      └─ OOF cost-optimised threshold (no leakage)
                           └─ Sealed-test evaluation + error analysis + human review — gate: pass (Phase 7)
                                └─ Champion registration + drift baseline + model card (Phase 7, done)
                                     └─ FastAPI serving + Streamlit UI
```

**Data splits** — stratified by `customerid`, sealed once before feature discovery (`data/split.py`):

| Split | n | Churners | Role |
|---|---|---|---|
| Dev | 5,634 | 1,495 (26.5 %) | Cross-validation (`RepeatedStratifiedKFold`, 10×10) for every modelling decision — family selection, feature selection, hyperparameter tuning, calibration |
| Test | 1,409 | 374 (26.5 %) | Final evaluation — opened once, in Phase 7 (`evaluate.py`) |

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
| 7a | Sealed test-set evaluation, error analysis, human review — gate: pass | ✅ Done |
| 7b | Registry promotion, drift baseline, model card (`register.py`, `drift_reference.py`) | ✅ Done |
| 8 | DVC pipeline wrap | 🔜 Next |
| 9 | Serving + UI — FastAPI + Streamlit | 🔜 Next |
| 10 | Orchestration — Prefect retrain/drift flows | ⏸ Scoped, deferred until after 12 |
| 11 | CI/CD — GitHub Actions | 🔜 Next |
| 12 | AWS deployment | 🔜 Next |
| 13 | Monitoring — Prometheus/Grafana/Evidently | ⏸ Scoped, deferred until after 12 |
| 14 | Documentation polish | Planned |

**Current focus:** 8 → 9 → 11 → 12 — DVC, serving, CI/CD, and cloud deployment, in that order, since each is a dependency of the next and together they form one working, deployed system. Orchestration (10) and monitoring (13) are fully scoped but deliberately sequenced *after* deployment — both are more meaningful run against a live system with real traffic than built ahead of one.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full phase-by-phase roadmap and the execution-order rationale.

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

**5 — Split the data and build features**

```bash
make split             # canonical dev/test partition, sealed before feature discovery
make features          # build SQL views → write datasets/processed/telco_churn_processed.csv
```

**6 — Train, calibrate, and derive the threshold**

```bash
make train                                     # LightGBM + Optuna tuning; logs to MLflow (note the printed run_id)
make calibrate RUN_ID=<run_id from above>      # sigmoid calibration; registers the challenger model version
make threshold MODEL_VERSION=<version above>   # closed-form cost-sensitive threshold, ships configs/policy/threshold.yaml
```

Each step's console output (structlog JSON) prints the `run_id` / `model_version` the next command needs.

**7 — Evaluate on the sealed test set and run error analysis**

```bash
make evaluate MODEL_VERSION=<version above>        # one-time sealed-test scoring; runs the promotion gate
make error-analysis MODEL_VERSION=<version above>  # SHAP explainability + error diagnosis
```

Opens `notebooks/05-evaluation-and-error-analysis.ipynb` to review the gate result and error diagnostics.

**8 — Record human sign-off, then promote the champion**

```bash
uv run python -m telco_churn.models.review review.verdict=approved review.approver="Your Name" review.notes="..."  # verdict, approver, and a non-empty reason are all required; prints where to find this cycle's MLflow diagnostics first
make register MODEL_VERSION=<version above>  # acts on the gate verdict + review: flips the champion alias on a pass, tags rejected on a fail
```

**9 — Browse experiment runs**

`make db-up` already started the MLflow tracking server alongside Postgres —
open [http://localhost:5000](http://localhost:5000) to explore the logged runs, the registered `telco-churn-pipeline` model, and the `challenger` alias.

---

## Project Structure

```
src/telco_churn/
  data/                          # ingest, Pandera validation/schema, dev/test split
  features/                      # SQL feature views, 9-lap discovery, permutation-importance selection
  models/
    train/                       # candidate comparison, feature freeze, Optuna tuning, model logging
    calibrate.py                 # CalibratedClassifierCV method selection + registration (challenger)
    threshold.py                 # closed-form cost-sensitive threshold derivation
    evaluate.py                  # one-time sealed-test scoring + promotion gate (Phase 7)
    gate.py                      # decide_promotion — pure gate function, PR-AUC selection + 3 veto guardrails
    economics.py                 # expected-value scenarios, sensitivity, break-even analysis
    explain.py, error_analysis.py # SHAP explainability + error diagnosis (Phase 7)
    drift_reference.py           # champion drift-monitoring baseline builder (Phase 7)
    register.py                  # registry alias flip, smoke check, rollback, model card (Phase 7)
    plots.py, diagnostics.py
  utils/                         # paths, logging, db, mlflow, stats helpers
configs/                 # Hydra YAML — training/, tuning/, calibration/, threshold/, evaluate/, error_analysis/, register/, costs.yaml, policy/, model_promotion.yaml
sql/                     # Postgres schema + feature SQL views
tests/unit/              # pytest unit tests (≥80 % coverage target)
tests/integration/       # Postgres-backed tests (ingest, split, sql_features, validate) + subprocess CLI tests (train, calibrate, threshold, evaluate, error_analysis)
pipelines/               # Prefect flows (retrain, drift check, batch predict) — Phase 10
monitoring/              # Prometheus config + Grafana dashboard JSON — Phase 13
datasets/                # gitignored; tracked by DVC (Phase 8)
mlruns/                  # gitignored; MLflow local tracking store
notebooks/               # 00–05: ingestion → EDA → feature discovery/engineering →
                         #        model selection/feature selection/hyperparameter tuning →
                         #        calibration & threshold → evaluation & error analysis (Phase 7)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, make commands, and PR conventions.

---

## Author

**Richlove Frimpong** — [LinkedIn](https://www.linkedin.com/in/richlove-frimpong) · [GitHub](https://github.com/Ampofowaa)
