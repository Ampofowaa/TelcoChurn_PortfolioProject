# Telco Customer Churn — End-to-End ML Portfolio Project

[![CI](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml/badge.svg)](https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1085%20passing-brightgreen)](CONTRIBUTING.md#testing)
[![Coverage](https://img.shields.io/badge/coverage-96.8%25-brightgreen)](CONTRIBUTING.md#testing)

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
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Modelling](#modelling)
- [Pipeline Versioning (DVC)](#pipeline-versioning-dvc)
- [Serving](#serving)
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

## Pipeline

```
Raw CSV
  └─ Ingest → Postgres (`customers_raw`)
       └─ Validation (Pandera, 5 quality gates)
            └─ Feature engineering (SQL views; 9-lap out-of-fold discovery search over candidate features)
                 └─ Model training (LightGBM + Optuna tuning) → calibration → register challenger
                      └─ Threshold → sealed-test evaluation → error analysis → human review
                           └─ Champion registration
                                └─ Serving (FastAPI + Streamlit)
```

Mechanism and rationale for each stage below → [Modelling](#modelling), [Pipeline Versioning](#pipeline-versioning-dvc), [Serving](#serving). This is the simplified view — full tool-level architecture, feedback loops, and MLflow run layout → [docs/architecture.md](docs/architecture.md).

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
| Modelling | LightGBM, `LogisticRegressionCV`, `DummyClassifier` (see [Modelling](#modelling)) |
| Hyperparameter tuning | Optuna |
| Calibration | `CalibratedClassifierCV` |
| Explainability | SHAP |
| Pipeline versioning | DVC (see [Pipeline Versioning](#pipeline-versioning-dvc)) |
| Serving | FastAPI + Streamlit (see [Serving](#serving)) |
| Monitoring instrumentation | `prometheus_client` |
| Orchestration (Phase 10) | Prefect |
| Infrastructure | Docker, Postgres |
| CI/CD | GitHub Actions |
| Cloud (Phase 12) | AWS ECR + App Runner + RDS + S3 |

---

## Modelling

Mechanism only — the reasoning behind each choice (why LightGBM over `LogisticRegressionCV`, why sigmoid calibration, why these guardrails) lives in `ANALYSIS.md`, linked per row rather than repeated here.

| Stage | Mechanism |
|---|---|
| **Family selection** | Three candidates cross-validated on the dev partition — `DummyClassifier(strategy='prior')` (leakage safeguard + BSS floor), `LogisticRegressionCV` baseline, LightGBM challenger. PR-AUC is the sole selection criterion. → [ANALYSIS.md §4a](ANALYSIS.md#4a-model-selection) |
| **Feature selection** | Permutation-importance ablation over 100 CV folds, stability-voted rather than decided from a single all-dev fit. → [ANALYSIS.md §4b](ANALYSIS.md#4b-feature-selection-concluded-ablation--importance-diagnostic) |
| **Hyperparameter tuning** | Optuna, TPE sampler, 50 trials; final pick is the 1-SE-rule value, not the raw-best trial. → [ANALYSIS.md §4c](ANALYSIS.md#4c-hyperparameter-tuning-optuna) |
| **Calibration** | `CalibratedClassifierCV` (sigmoid) — fit and logged only; `calibrate.py` never registers a model on its own. → [ANALYSIS.md §5](ANALYSIS.md#5-probability-calibration) |
| **Threshold** | Closed-form `t* = c / (r × LTV)` from `configs/costs.yaml`, derived out-of-fold with no leakage into the sealed test set. → [ANALYSIS.md §6](ANALYSIS.md#6-business-impact--threshold-selection) |
| **Promotion gate** | PR-AUC drives selection; three veto-only guardrails (recall at `t*`, Brier, calibration slope) can reject a candidate but never promote one. → [ANALYSIS.md §0](ANALYSIS.md#success-criterion--the-promotion-gate) |
| **Registration** | `register.py` is the sole registry-write entry point — mints the challenger, tags it `pending`, and flips (or rejects) the `champion` alias only after both the automated gate and a human review pass. → [ANALYSIS.md §8](ANALYSIS.md#8-model-registration--promotion) |

Full modelling narrative, every number, every deviation from the archived exploratory pass → [ANALYSIS.md](ANALYSIS.md).

---

## Pipeline Versioning (DVC)

`dvc.yaml` wraps the file-producing half of the pipeline above as a reproducible, content-hashed DAG — the registry-mutating half (minting/tagging/flipping a model version) structurally cannot live in it, since a registry alias isn't a file DVC can hash.

| Component | What it does |
|---|---|
| **9-stage DAG** (`dvc.yaml`) | `ingest → validate → split → features → train → calibrate → threshold → evaluate → error_analysis`, each stage's `deps`/`params`/`outs` explicitly declared. `dvc repro <stage>` only re-runs that stage and its upstream chain when something it actually depends on changed — never the whole DAG unconditionally. |
| **Registry writes stay outside the DAG** | `register.py` (mint challenger, tag, flip/reject champion) runs as its own step, never a `dvc.yaml` stage. A full training cycle is therefore two `dvc repro` calls bracketing one registration step, not one — [Quick Start steps 5–7](#quick-start) run through it. |
| **Receipts bridge the two worlds** | Six small JSON receipts (`reports/*_receipt.json`) stand in for side effects DVC can't hash directly — a Postgres load, an MLflow run — giving `validate`/`train`/`calibrate`/`eval`/`error_analysis` a real dependency edge on "did the step before me actually succeed," not just "did its script exit 0." |
| **Local cache only** | No DVC remote yet — `.dvc/cache` is local-only through Phase 11; Phase 12 adds an S3 remote for shared caching, not a migration off the current setup. |

```bash
dvc repro <stage>   # re-run one stage and its upstream chain
dvc dag             # print the pipeline DAG
dvc metrics show    # diff metrics.json / params across commits
```

Run it yourself → [Quick Start step 5](#quick-start). Full architecture → [docs/architecture.md](docs/architecture.md).

---

## Serving

The champion (`telco-churn-pipeline@champion` in the MLflow registry) is served behind a FastAPI app, with a Streamlit UI as a thin client — both run as their own Docker images and come up together with one `docker compose up`.

| Component | What it does |
|---|---|
| **FastAPI** (`serving/app.py`, `serving/predict.py`) | Loads the champion at startup and hot-reloads it on a TTL poll — a promotion is just an MLflow alias flip, so a new champion goes live without a redeploy. Exposes `POST /predict`, `POST /predict/batch`, `GET /customer/{id}`, `GET /health`/`GET /ready` (liveness vs. readiness), and `GET /metrics` (Prometheus). Runs on `uvicorn` inside `docker/api/Dockerfile`, a multi-stage build from the project's own `uv.lock`. |
| **Contact policy** | `POST /predict/batch` doesn't just score every row — it ranks them by expected value and caps who gets contacted at a configurable `contact_capacity`/`campaign_budget`, so the response reflects an operationally realistic campaign, not an unlimited-budget fantasy. |
| **Shadow/canary rollout** | Config-gated, off by default: whenever a `challenger` model exists in the registry, it can be dual-scored against every request for comparison (**shadow**, zero routing risk) or actually serve a consistent-hash slice of live traffic (**canary**). Mechanism, Prometheus metrics, and a real evidence log → [docs/architecture.md § Shadow/Canary Serving](docs/architecture.md#shadowcanary-serving--servingpredictpy-phase-9). |
| **Streamlit UI** (`ui/streamlit_app.py`) | **Score a Customer** tab: look up a real customer from `customers_crm` (a seeded "current state" derivation of the training data, never the frozen snapshot itself) or manually enter one for a what-if scenario. Mechanism, rationale, and what's/isn't durably recorded → [docs/architecture.md § Live Customer Lookup](docs/architecture.md#live-customer-lookup--customers_crm-phase-9). **Batch Prediction** tab: score a CSV upload — [`examples/sample_batch_predictions.csv`](examples/sample_batch_predictions.csv) demonstrates all three `/predict/batch` item shapes in one upload (ID-only, full inline, ID-plus-override). **Model Info** tab: read the champion's model card. All three drive the same FastAPI endpoints a real client would call. Runs from `docker/ui/Dockerfile`, same build pattern as the API image. |
| **Docker Compose** | `docker-compose.yml` wires four services — `postgres`, `mlflow`, `fastapi`, `streamlit` — so `docker compose up -d --build` brings up the entire stack in one command, with `fastapi`/`streamlit` waiting on `postgres`/`mlflow` via `depends_on`. |

Run it yourself → [Quick Start step 9](#quick-start).

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
| 8 | DVC pipeline wrap — 9-stage reproducible DAG (`dvc.yaml`), `ingest` through `error_analysis` | ✅ Done |
| 9 | Serving + UI — FastAPI + Streamlit, shadow/canary rollout mechanism | ✅ Done |
| 10 | Orchestration — Prefect retrain/drift flows | 🔜 Next |
| 11 | CI/CD — GitHub Actions | ⏳ Queued after 10 |
| 12 | AWS deployment | ⏳ Queued after 11 |
| 13 | Monitoring — Prometheus/Grafana/Evidently | ⏸ Scoped, deferred until after 12 |
| 14 | Documentation polish | Planned |

**Current focus:** 10 → 11 → 12 — orchestration, then CI/CD, then cloud deployment. Monitoring (13) stays deferred until after 12 — it needs a live system to alert on. Orchestration doesn't have that same dependency: this dataset never has live traffic for the retrain/drift flows to route through either way (`ANALYSIS.md` §9 item 13), so there's no reason to wait on deployment before building the mechanism.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full phase-by-phase roadmap and the execution-order rationale.

---

## Quick Start

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), and a [Kaggle API token](https://www.kaggle.com/settings/api).

**1 — Clone and install**

```bash
git clone <repo-url>
cd TelcoChurn_PortfolioProject
uv sync --all-extras
pre-commit install
```

**2 — Download the dataset**

```bash
make data
```

Places the raw CSV at `datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

**3 — Run tests**

```bash
uv run pytest          # unit tests — needs nothing but the install above, no Docker or pipeline run required
make test-integration  # integration tests — spins up its own ephemeral Postgres, requires only a running Docker daemon
```

A sanity check that the install and dataset download worked, not a pipeline step — neither command depends on anything below, which is why it's not sitting in the middle of the pipeline run.

**4 — Start Postgres**

```bash
cp .env.example .env   # defaults work with Docker out of the box
make db-up             # start postgres:16 in Docker
```

**5 — Run the pipeline through calibration**

```bash
make repro STAGE=calibrate   # ingest -> validate -> split -> features -> train -> calibrate
```

Exits 0 without touching `threshold` — that stage needs a registered model version, minted next. See [Pipeline Versioning](#pipeline-versioning-dvc) for why this is deliberate, not a partial failure.

**6 — Mint the challenger, then finish the pipeline**

```bash
make register-challenger        # mints calibrate's output as a new registry version, tagged pending
make repro STAGE=error_analysis # threshold -> evaluate -> error_analysis
```

Review the gate result in MLflow (`mlflow ui` → this cycle's `evaluation` run) rather than the notebook, which only reflects this cycle if someone re-executes it.

**7 — Record human sign-off, then promote the champion**

```bash
make review VERDICT=approved APPROVER="Your Name" NOTES="..."
make register MODEL_VERSION=<version>  # flips champion on a pass, tags rejected on a fail
```

`MODEL_VERSION` (printed by step 6, or `mlflow ui`) has no default — deliberately, since this is the step that can put a model into production.

<details>
<summary>Re-running one stage against a specific historical run instead</summary>

Every command above works with no arguments because it always resolves "whatever the previous step just produced." `calibrate`/`threshold`/`evaluate`/`error-analysis` also accept an explicit override — `make calibrate RUN_ID=<run_id>` or `make threshold MODEL_VERSION=<version>` — to target a different run/version than the default chain, and `make repro STAGE=<stage>` (e.g. `make repro STAGE=train`) re-runs one DVC stage alone. Neither is needed for a first run.

</details>

**8 — Browse experiment runs**

`make db-up` already started the MLflow tracking server alongside Postgres —
open [http://localhost:5000](http://localhost:5000) to explore the logged runs, the registered `telco-churn-pipeline` model, and the `challenger` alias.

**9 — Serve the champion and try it**

```bash
docker compose up -d --build   # full stack: postgres, mlflow, fastapi, streamlit
```

Once `/ready` returns `200` (`curl http://localhost:8000/ready`), open [http://localhost:8501](http://localhost:8501) for the Streamlit UI, or hit the API directly:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"customerid":"0002-ORFBO"},{"customerid":"0003-MKNFE"}]'
```

Try the Batch Prediction tab with [`examples/sample_batch_predictions.csv`](examples/sample_batch_predictions.csv) (50 real customers, resolved live from `customers_crm`, regenerated deterministically by [`scripts/generate_sample_batch_predictions.py`](scripts/generate_sample_batch_predictions.py)) — see [Serving](#serving) for what it demonstrates. In the Score a Customer tab, hit "Fetch customer" with any of `0094-OIFMO`, `0096-BXERS`, `0096-FCPUF`, `0098-BOWSO`, `0100-DUVFC`. `make smoke-test-serving` runs the same checks as a script; `docker compose down` tears the stack back down.

---

## Project Structure

```
src/telco_churn/
  data/                          # ingest, Pandera validation/schema, dev/test split
  features/                      # SQL feature views, 9-lap discovery, permutation-importance selection
  models/
    train/                       # candidate comparison, feature freeze, Optuna tuning, model logging
    calibrate.py, threshold.py, evaluate.py, gate.py    # calibration, threshold, sealed-test evaluation, promotion gate
    economics.py, explain.py, error_analysis.py, shap_values.py  # business impact, SHAP, error diagnosis
    register.py, review.py       # registry writes, human review verdict
    drift_reference.py, environment_parity.py  # drift baseline, hot-reload dependency diff
    artifacts.py, policy_config.py, dev_features.py, calibration_metrics.py  # shared loaders/helpers
    plots.py, diagnostics.py
  serving/                       # FastAPI app + shadow/canary-aware scoring
    predict.py, app.py           # hot-reloading model bundle, /predict endpoints
    contact_policy.py, customer_lookup.py, schemas.py  # contact ranking, Postgres lookup, request/response models
  ui/                            # Streamlit demo app
    streamlit_app.py             # Score a Customer, Batch Prediction, Model Info tabs
  utils/                         # paths, logging, db, mlflow, stats, hashing helpers
configs/                 # Hydra YAML — training/, tuning/, calibration/, threshold/, evaluate/, error_analysis/, review/, register/, serving/, costs.yaml, model_promotion.yaml
sql/                     # Postgres schema + feature SQL views
tests/unit/              # pytest unit tests (≥80 % coverage target)
tests/integration/       # Postgres-backed tests (ingest, split, sql_features, validate) + subprocess CLI tests (train, calibrate, threshold, evaluate, error_analysis, predict) + FastAPI TestClient suite
tests/streamlit/         # headless Streamlit `AppTest` smoke suite (Phase 9)
pipelines/               # Prefect flows (retrain, drift check, batch predict) — Phase 10
monitoring/              # Prometheus config + Grafana dashboard JSON — Phase 13
docker/api/, docker/ui/  # multi-stage Dockerfiles (FastAPI, Streamlit), built from uv.lock (Phase 9)
examples/                # sample_batch_predictions.csv — real dev-partition customer IDs for /predict/batch and the Batch Prediction tab
                         #   regenerate: uv run python scripts/generate_sample_batch_predictions.py
dvc.yaml, .dvc/, .dvcignore  # DVC pipeline — 9 stages, ingest through error_analysis (Phase 8)
datasets/raw/             # committed to git — the source CSV DVC hashes as a stage dep, never modified
datasets/processed/, datasets/interim/  # gitignored; DVC-cached stage outputs
mlruns/                  # gitignored; MLflow local tracking store
notebooks/               # 00–05: ingestion → EDA → feature discovery/engineering →
                         #        model selection/feature selection/hyperparameter tuning →
                         #        calibration & threshold → evaluation & error analysis (Phase 7)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, make commands, and PR conventions.

---

## Author

**Richlove Frimpong** — [LinkedIn](https://www.linkedin.com/in/richlove-frimpong) · [GitHub](https://github.com/Ampofowaa)
