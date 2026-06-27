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
│   │   ├── build.py            # column group exports (raw IBM columns + charge_per_service)
│   │   ├── preprocessing.py    # shared ColumnTransformer builder (Phase 4a; reused by train.py)
│   │   └── generate.py         # error-driven discovery machinery: OOF profiler, gate, bootstrap CI (Phase 4a)
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
│   ├── 00-data-ingestion.ipynb      # Phase 2 — raw CSV → 5 Pandera gates → Postgres ingest
│   ├── 01-eda.ipynb                 # Phase 3 — statistical tests, distributions
│   ├── 02a-feature-discovery.ipynb  # Phase 4a — structured feature search: domain hypotheses + OOF blind-spot profiling → adoption gate
│   ├── 02b-feature-engineering.ipynb # Phase 4b — distribution/justification view of the adopted set
│   ├── 03-model-selection.ipynb     # Phase 5 — Optuna study + baseline comparison
│   ├── 03b-feature-selection.ipynb  # Phase 5 — null-importance selection experiment
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
| 3 | EDA notebook | `notebooks/01-eda.ipynb` importing from `src/` | ✅ Done |
| 4a | Feature discovery: structured feature search (domain hypotheses + OOF profiling → adoption gate) | `features/generate.py`, `features/preprocessing.py`, `notebooks/02a-feature-discovery.ipynb`, provenance log | ✅ Done |
| 4b | Feature engineering — encode the 4a-adopted set | `sql/features/*.sql`, `features/sql_features.py`, `features/build.py`, `features/schema.py` | ✅ Done |
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
- `notebooks/00-data-ingestion.ipynb` — demonstrates the raw CSV → 5 Pandera gates → Postgres ingest journey; renders `summary.csv` and `schema_failures.csv` for inspection

**Phase 8 cleanup required:**
- Remove `clean_dataframe()` once imputation is owned by the **Phase 5 training `ColumnTransformer`** (`SimpleImputer(strategy='median')` fit on `X_train` only); the Phase 4b `features` stage deliberately preserves the 11 NaNs so no training-set statistic leaks into feature engineering
- In the DVC stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event listing which gates failed, then call `sys.exit(1)` — do not let Python's default traceback handler produce the exit, which is noisy and unsearchable in pipeline logs

**Verification:** `uv run python -m telco_churn.data.validate` against the loaded Postgres data exits 0; an injected violation exits 1 and writes a timestamped report to `reports/validation/`.

---

### Phase 3 — Exploratory Data Analysis Notebook *(1 day)*

> **Why Phase 2 (Validation) comes before Phase 3 (EDA):** This ordering looks backwards — you would normally explore data before writing validation rules. The explanation is that this project migrates a *completed* notebook (`EDA-original.ipynb`), not a greenfield build. The five validation gates were *discovered* during the original EDA session and are already documented in `ANALYSIS.md`. Phase 2 *enforces* those already-known rules as automated Pandera checks; Phase 3 *re-presents* the EDA as a clean, importable-function-backed notebook. In the original science timeline the order was: explore → discover quality issues → define gates. In the migration timeline, enforcement (Phase 2) is built first because downstream phases (features, training) depend on the validation pipeline being in place before they run. These two orderings serve different purposes: discover-vs-enforce.

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

**Verification:** Notebook executes end-to-end without errors (`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-eda.ipynb`).

---

### Phase 4a — Feature Discovery: Structured Feature Search *(2–3 days)*

**What this achieves:** Establishes which engineered features earn a place in the model through a narrated, audited discovery loop. Candidates originate from two sources: EDA-anchored domain hypotheses (e.g., tenure survival-curve segmentation, service-normalised pricing, the fiber × contract interaction) and OOF false-negative profiling that surfaces systematic blind spots the baseline cannot recover. Every candidate — regardless of origin — passes through a four-screen adoption gate: leakage pre-gate → redundancy screen → OOF PR-AUC + subgroup recall → importance vs. noise floor. The loop is human-in-the-loop: the analyst writes each candidate in the notebook; `generate.py` supplies the mechanical scaffolding. Starts from raw IBM columns only. At least one decoy must be introduced and rejected.

**Not a DVC stage:** run-once R&D, seeded (`random_state=42`). Commits its provenance and adopted-set list; the production pipeline ships the frozen result via Phase 4b `build.py`.

**Deliverables:**

*Source:*
- `src/telco_churn/features/preprocessing.py` — `build_preprocessor(binary, multi_cat, numeric)` returning a `ColumnTransformer` (median-impute + scale numerics; OHE categoricals; `FunctionTransformer(astype str)` on the mixed-dtype binary group); reused verbatim by Phase 5 `train.py`
- `src/telco_churn/features/generate.py` — discovery machinery: `oof_predictions`, `profile_false_negatives` (single-feature scan + cross-tab → ranked blind spots), `serving_available` (leakage pre-gate), `redundancy_screen` (|corr| + VIF / Cramér's V), `candidate_importance` (permutation importance vs. decoy noise floor), `adoption_gate` (composes the four screens; PR-AUC is the sole selector, the others are guardrails), `bootstrap_pr_auc_ci`, `backward_elimination` (post-loop pruning pass over the adopted set), `LapRecord` + provenance writer. Pure, typed, testable — no feature logic.

*Tests:*
- `tests/unit/test_generate.py` — planted recoverable blind spot: profiler surfaces it; gate adopts a known-good feature, rejects a decoy, rejects a leaked candidate at the pre-gate (before any metric is computed), flags and rejects a collinear duplicate with no marginal recall gain, rejects a noise-floor-importance candidate despite a fold-noise recall blip; the flat-global-PR-AUC-but-recall-up case is adopted; OOF predictions are leak-free; bootstrap-CI math on a known-population case; empty-frame / all-noise edge cases.

*Notebook:*
- `notebooks/02a-feature-discovery.ipynb` — narrated discovery session: lap-by-lap from the raw base, error profile → analyst hypothesis → redundancy screen → re-measure → gate decision, including ≥1 rejected decoy. Candidate feature logic lives here during the loop, then migrates to `build.py` in Phase 4b.

*Artifact:*
- `reports/feature_discovery/provenance.json` — run-level header (random state, gate-threshold constants) + per-lap record (blind spot, hypothesis, EDA anchor, serving-availability verdict, redundancy stats, ΔPR-AUC, Δsubgroup recall, importance vs. noise floor, decision). The auditable trail proving the feature set was discovered, not asserted.

**Verification:** `notebooks/02a-feature-discovery.ipynb` runs end-to-end from the raw base, discovers candidates lap-by-lap from the error profile (expected to converge near the six features in `build.py`), rejects ≥1 decoy through the gate, and emits the adopted-set list + `reports/feature_discovery/provenance.json`. `uv run pytest tests/unit/test_generate.py` passes.

---

### Phase 4b — Feature Engineering *(1–2 days)*

**What this achieves:** Takes the single Phase 4a adoption — `charge_per_service` — builds it in a SQL view, and folds it into the feature set alongside the 19 raw IBM columns. The training pipeline inherits a clean, Pandera-validated column interface: nothing provisional, no dead feature code.

**Deliverables:**
- `sql/features/charge_per_service.sql` — `monthlycharges ÷ GREATEST(service_count, 1)`; nine binary `CASE WHEN` flags summed in a subquery; `GREATEST` guards divide-by-zero; `internetservice <> 'No'` catches both DSL and Fiber optic
- `sql/features/customer_features.sql` — final feature view; LEFT JOIN over `customers_raw` so any row filtered by a dependent view surfaces as NULL (caught by Pandera) rather than being silently dropped
- `src/telco_churn/features/sql_features.py` — `build_sql_features(engine, sql_dir)`: runs the two SQL files in dependency order inside one `engine.begin()` transaction; idempotent (`CREATE OR REPLACE VIEW`)
- `src/telco_churn/features/build.py` — `BINARY_STR_COLS` (5), `BINARY_INT_COLS` (1), `MULTI_CAT_COLS` (10), `NUMERIC_COLS` (3 raw + `charge_per_service`); `build_feature_df()` is a Pandera-decorated pass-through; `__main__` CLI retained until Phase 8
- `src/telco_churn/features/schema.py` — `CustomerFeaturesSchema` (20 columns; `strict=False` so `customerid`/`churn` pass through; `coerce=True` for Postgres type coercion on read); `FeatureOutputSchema` inherits and overrides `coerce=False`
- `tests/unit/test_build.py` — shape, NaN passthrough, immutability, schema rejection, `hypothesis` property tests, provenance cross-check against `adopted_features.json`
- `tests/unit/test_sql_features.py` — execution count, dependency order, single-transaction contract, error propagation with filename, idempotency, file existence
- `tests/integration/test_sql_features_postgres.py` — `testcontainers` Postgres covering both views and the full SQL → `build_feature_df` pipeline; `__main__` CLI subprocess test
- `notebooks/02b-feature-engineering.ipynb` — loads the feature view, renders the 20-column inventory, confirms output shape; Phase 4a outcome in the opening cell; full narrative in `ANALYSIS.md §3b`

**Verification:** `uv run python -m telco_churn.features.build` exits 0 and writes `datasets/processed/telco_churn_processed.csv` (7,043 rows × 20 feature columns + `customerid` + `churn`). `uv run pytest tests/unit/test_build.py tests/unit/test_sql_features.py` passes. Integration tests require the stale `tenure_buckets` cleanup before running.

---

### Phase 5 — Model Training (LightGBM + Optuna + MLflow) *(3–4 days)*

**What this achieves:** A reproducible, experiment-tracked training run that searches hyperparameters with Bayesian optimisation and logs every trial to MLflow. The best model is registered in the MLflow Model Registry as `challenger`, ready for calibration (Phase 6) and sealed-test evaluation (Phase 7).

**Order of operations (do not reorder):**

1. **Build candidates** — train `DummyClassifier` (no-information floor) and `LogisticRegression` (linear reference) alongside the tree candidates (LightGBM, XGBoost, RandomForest) through the *identical* pipeline and CV splits. Baselines must run through the same pipeline to be a fair measuring instrument.
   > **Expected finding:** Telco churn is near-linear — logistic regression typically reaches PR-AUC within the test-set CI of tuned LightGBM. If so, state in `ANALYSIS.md` that LightGBM is chosen for calibration quality, SHAP interaction structure, and error-analysis-driven features — **not** because it materially out-predicts the linear model.

2. **Compare on PR-AUC** → confirm LightGBM beats the baselines; document the margin. PR-AUC is the sole selection metric — threshold-free and imbalance-appropriate at a ~27 % positive rate. ROC-AUC, recall, precision, and F1 are logged as diagnostics only.
   > **⚠ Flagged deviation:** the notebook selects and tunes on CV recall@0.35. Standardising to PR-AUC separates ranking quality from operating-point selection — the threshold is set in Phase 6 and is a business-owned knob. The notebook's preprocessing, hyperparameter ranges, calibration, threshold logic, and evaluation math are otherwise preserved verbatim. Decision approved June 2026; record in `ANALYSIS.md` when Phase 5 lands.

3. **Select features (freeze the input space)** — run null-importance / target-permutation selection inside CV on train + val only, against a *default-config* LightGBM (not the tuned model). Refit on survivors; confirm the reduced model's CV PR-AUC sits within the bootstrap CI of the full-feature model. The expected, fully acceptable outcome is "keep most/all" — documented honestly either way.
   > **📌 Discussion (revisit at Phase 5 implementation):** sklearn's built-in selectors — `SelectFromModel`, `SelectKBest`, `RFE` — all implement `fit`/`transform` and can slot directly as a pipeline step between the `ColumnTransformer` and the model. The selector is then refitted on each CV fold's training portion automatically, giving the same leak-free guarantee as the custom null-importance wrapper but with less code. Use a built-in if the null-importance experiment shows most features are informative (the expected outcome here); build the custom wrapper only if per-fold stability reporting is a required output.
   > **⚠ Flagged deviation:** the notebook performs no feature selection — VIF / Cramér's V / permutation importance are diagnostics, not a selection gate. This step is a deliberate methodological addition. The reduced set is adopted **only if** its CV PR-AUC is within the full set's bootstrap CI and there is a parsimony reason to prefer it; otherwise the full set stands. Record the keep/drop decision in `ANALYSIS.md`.

4. **Tune only the confirmed family** (LightGBM) with Optuna — *after* the feature set is frozen by step 3. Each trial is a nested MLflow child run; the study is the parent run. Optimisation objective is PR-AUC (`average_precision`).

5. **Register the tuned model** as `challenger`. Calibration, thresholding (Phase 6), and sealed-test evaluation (Phase 7) are deliberately separate, later phases.

> **📌 `customerid` flow through the training pipeline.**
> The industry standard is to keep identifiers alongside the feature matrix to the model boundary and exclude them passively — not to drop them at feature engineering time. `customerid` is available for error analysis (Phase 7), prediction tracing, and serving but must never reach the model.
>
> | Stage | `customerid` treatment |
> |---|---|
> | `build_feature_df` | Present in the input DataFrame; absent from the returned `feature_df` because it is not in `BINARY_STR_COLS`, `BINARY_INT_COLS`, `MULTI_CAT_COLS`, or `NUMERIC_COLS` |
> | `train.py` train/val/test split | Split `customerid` out as a separate `pd.Series` alongside `X_train`, `X_val`, `X_test` — do not drop it |
> | `ColumnTransformer` | `remainder='drop'` passively excludes any column not listed in a transformer — `customerid` never reaches the model even if accidentally present in `X` |
> | Error analysis (Phase 7) | Re-attach `customerid` Series to `y_pred` for joins back to raw customer records |

**Deliverables:**

*Configs:*
- `configs/model/lightgbm.yaml` — parameter ranges warm-started from the notebook's Optuna best: `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- `configs/training/optuna.yaml` — `n_trials: 50`, `sampler: tpe`, `direction: maximize`, `metric: average_precision`, `cv_folds: 5`, `random_state: 42`

*Source:*
- `src/telco_churn/models/train.py` — reads config via Hydra; loads feature DataFrame from `datasets/processed/` via `build_feature_df`; performs the stratified train/val/test split (`random_state=42`, **the only place the split is defined**); fits the `ColumnTransformer` on the training split only and wraps it with the model in a single sklearn `Pipeline` (OHE over native encoding — all categoricals ≤ 4 unique values; `StandardScaler` included for linear baseline fairness); stratified 5-fold CV with `scale_pos_weight ≈ 2.77`; best model logged as `pyfunc` with `feature_space.txt`, `feature_columns.txt`, and `preprocessing.pkl` and registered as `telco-churn-pipeline` / alias `challenger`. **Two-layer artifact logging (feature space vs model input space):** `feature_space.txt` is logged at the start of the MLflow run by reading `BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS` from `build.py` — it records the full *feature space* (every column `build_feature_df` produced, owned by `FeatureSchema`). It is generated here, not in `build.py`, because it is an MLflow artifact — Phase 4 has no MLflow context. `feature_columns.txt` records the *model input space* — the subset that survived `select.py` and entered the `ColumnTransformer`. The diff between these two files is what selection dropped for that specific run; any MLflow run is self-describing without a git lookup. `feature_space.txt` is identical across most runs (it only changes when `build.py` changes); `feature_columns.txt` can differ on every run where selection experiments change. `preprocessing.pkl` is the fitted `ColumnTransformer` encoding the exact transformations applied to the model input space at training time. The test split is importable **only** by `evaluate.py` (Phase 7) — the "test set touched once" invariant is structural, not conventional. **Binary dtype note:** the binary group is split by dtype across two lists — `BINARY_STR_COLS` (`gender`, `has_partner`, `dependents`, `phoneservice`, `paperlessbilling`; `'Yes'`/`'No'` strings) and `BINARY_INT_COLS` (`seniorcitizen`, `is_long_month_to_month`; `0`/`1` integers). The binary branch of the `ColumnTransformer` is fed their union (`BINARY_STR_COLS + BINARY_INT_COLS`) with a `FunctionTransformer(lambda X: X.astype(str))` as its first step, normalising all binary inputs to strings before OHE. **Imputation note:** `totalcharges` and `monthly_to_total_ratio` are NaN for 11 zero-tenure rows (`tenure = 0`, `monthlycharges > 0` — first bill not yet issued); both must be imputed via `SimpleImputer(strategy='median')` inside the numeric branch of the `ColumnTransformer` — fit on `X_train` only, never on the full dataset. This is the only place imputation is applied; `build_feature_df` deliberately preserves the NaN values so no training-set statistics leak into the feature-engineering step. Median imputation is appropriate here for three reasons: (1) all 11 rows are non-churners, so they are not in the minority class the model is tuned to identify; (2) 11 rows is 0.15 % of the dataset — negligible impact on any split; (3) LightGBM splits on individual thresholds, not linear combinations, so inter-column consistency between the imputed `totalcharges` and `monthly_to_total_ratio` values does not affect model behaviour.
> **📌 Discussion (revisit at Phase 5 implementation) — model-specific preprocessors:** The current design uses a single shared `ColumnTransformer` with `StandardScaler` included for linear baseline fairness. However, full OHE without `drop='first'` on multi-categorical features reintroduces perfect multicollinearity for linear models (the dummy variable trap), inflating coefficient variance and making estimates unstable — tree models are immune since each split is evaluated independently. The principled fix is two named preprocessors: `linear_preprocessor` (OHE `drop='first'` on all categoricals + `StandardScaler`) paired with `DummyClassifier` and `LogisticRegression`, and `tree_preprocessor` (OHE `drop='if_binary'` on binary cols, no drop on multi-cat, scaling optional) paired with LightGBM, XGBoost, and RandomForest. This costs two `ColumnTransformer` definitions and one extra line per `Pipeline` but makes the baseline comparison genuinely apples-to-apples — each family's pipeline is tuned to that family's assumptions. Adopt this pattern if the linear baseline is being taken seriously as a potential champion; the single-transformer shortcut is acceptable if trees dominate by a clear margin and the linear model is a floor only.
   >
   > **`tenure_cohort` vs `tenure` for the linear preprocessor (Phase 4a finding):** Phase 4a rejected `tenure_cohort` for LightGBM — continuous `tenure` already encodes all cohort signal and the tree learns the same non-linear boundaries through its own splits. For logistic regression, `tenure_cohort` (OHE’d with `drop='first'`) is the correct substitution: a single linear slope on continuous `tenure` cannot represent the survival-distribution shape (steep drop in the first 12 months, then gradual flattening), whereas OHE’d cohorts give the model a separate coefficient per risk tier. When building the `linear_preprocessor`, move `tenure` from the numeric group and replace it with `tenure_cohort` in the categorical group.

- `src/telco_churn/features/schema.py` (**extended, not created — the file already exists from Phase 4b, holding the Pandera `CustomerFeaturesSchema` / `FeatureOutputSchema` contracts**) — Phase 5 adds a `FeatureSchema` frozen dataclass (or Pydantic `BaseModel` with `model_config = ConfigDict(frozen=True)`) *alongside* those contracts, owning `binary`, `multi_cat`, and `numeric` as `tuple[str, ...]` fields. This replaces the bare `list[str]` column-group constants in `build.py` with a single typed, immutable object (the Pandera schemas validate row *values*; `FeatureSchema` owns the column *grouping* — two complementary roles in one module). `train.py` imports one `FeatureSchema` instance and feeds its fields directly to the `ColumnTransformer` column lists; `build_feature_df` imports the same instance to select columns — one source of truth, no mutation risk. Rationale: `Final[list[str]]` prevents rebinding but not `.append()`; `tuple` prevents mutation but not misuse; a `frozen=True` schema prevents both and adds field-level validation (non-empty, no duplicates) at no extra cost. `build.py`'s bare column-group lists remain until this phase lands; they are removed in the same PR that adds `FeatureSchema` to the existing `schema.py`.
  **Feature space vs model input space — two distinct concepts:** `FeatureSchema` owns the *feature space*: all engineered columns that `build_feature_df` can produce (the full Phase 4 output). The *model input space* is a separate, narrower concept — it is what actually enters the `ColumnTransformer` after `select.py` narrows the feature space down to survivors. `FeatureSchema` does not define the model input space; `select.py`'s output does. These must be kept conceptually separate: the feature space is stable across many model versions (adding a new engineered column updates it once), while the model input space changes every time selection runs a different experiment. **`frozen=True` semantics:** `frozen=True` means the `FeatureSchema` *instance* is immutable at runtime — no `.append()` or mutation by a caller. It does **not** mean the feature set is version-locked. Evolving the feature set (e.g., adding a new engineered column) requires explicit code changes in `build.py`, `FeatureSchema`, and the relevant SQL/Python feature builder — the frozen instance just enforces that those changes are deliberate rather than accidental runtime side-effects.

- `src/telco_churn/features/select.py` — null-importance / target-permutation selector; fits inside CV on train + val only against a default-config LightGBM; returns surviving feature list, importance/null table, and per-fold stability scores; wrapped in an sklearn `Pipeline` so `cross_val_score` re-fits selection on each fold's training portion (leak-free by construction).

*Tests:*
- `tests/unit/test_train.py` — config loading, metric logging contract (mock MLflow client)
- `tests/unit/test_select.py` — synthetic data with planted noise and known-informative columns; assert selector drops noise and keeps signal; assert selection is fit inside the fold; cover empty-dataframe and all-noise edge cases

*Notebooks:*
- `notebooks/03-model-selection.ipynb` — loads the Optuna study from MLflow; renders parallel-coordinates plot, hyperparameter importance, CV PR-AUC distribution across trials, and comparison table with reference baselines
- `notebooks/03b-feature-selection.ipynb` — full selection experiment: full-set CV PR-AUC + bootstrap CI → null-importance ranking → reduced-set refit → overlapping-CI check → documented keep/drop decision; imports from `select.py`

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and produces an MLflow run whose cross-validation PR-AUC falls within the bootstrap CI reported in `README.md`; reference baselines appear as rows in the comparison and LightGBM's PR-AUC is ≥ both. `notebooks/03b-feature-selection.ipynb` runs end to end and records a keep/drop decision in `ANALYSIS.md`.

**Prep checklist (task → deliverable):** the five ordered steps form the spine; two foundation tasks precede them because Step 1 cannot run without the schema and configs.

| # | Task | Deliverable |
|---|---|---|
| 1 | **Foundation:** `FeatureSchema` (frozen, typed; replaces the bare `list[str]` constants in `build.py`) | `src/telco_churn/features/schema.py` + `build.py` edits |
| 2 | **Foundation:** model + training Hydra configs | `configs/model/lightgbm.yaml`, `configs/training/optuna.yaml` |
| 3 | **Step 1** — build candidates (Dummy + LogReg + LightGBM/XGBoost/RandomForest) through the *identical* pipeline + CV; split defined once (`random_state=42`); `customerid` split out as a Series; `ColumnTransformer` fit on train only | `src/telco_churn/models/train.py` (split + `ColumnTransformer`/`Pipeline` + candidates) |
| 4 | **Step 2** — compare on PR-AUC (sole selection metric); confirm LightGBM ≥ baselines; ROC-AUC/recall/precision/F1 as diagnostics only | comparison logic in `train.py` + `ANALYSIS.md` note (recall@0.35→PR-AUC deviation + baseline margin) |
| 5 | **Step 3** — null-importance/target-permutation selection inside CV on train+val vs default LightGBM; refit on survivors; adopt reduced set only if within full-set bootstrap CI; **freezes the input space** | `src/telco_churn/features/select.py` + keep/drop decision in `ANALYSIS.md` |
| 6 | **Step 4** — tune LightGBM with Optuna (50 TPE trials, `average_precision`) *after* the freeze; trials as nested MLflow child runs | Optuna tuning logic in `train.py` |
| 7 | **Step 5** — register tuned model as `telco-churn-pipeline` / `challenger`; two-layer artifact logging | registration + `feature_space.txt`, `feature_columns.txt`, `preprocessing.pkl` in `train.py` |
| 8 | Tests — config loading + metric-logging contract (mock MLflow); split reproducibility | `tests/unit/test_train.py` |
| 9 | Tests — planted-noise synthetic data; selector drops noise/keeps signal; leak-free; edge cases | `tests/unit/test_select.py` |
| 10 | Notebook — Optuna study + baseline comparison loaded from MLflow | `notebooks/03-model-selection.ipynb` |
| 11 | Notebook — full selection experiment (full-set CI → null-importance → reduced refit → overlap check → decision) | `notebooks/03b-feature-selection.ipynb` |
| 12 | Verification — 50-trial run within README CI; LightGBM ≥ baselines; green `pytest` | passing verification + green test suite |

> **Order is load-bearing:** selection (Step 3) freezes the input space *before* tuning (Step 4) — changing features invalidates tuning. The test split is defined once in `train.py` and stays importable only by `evaluate.py` (Phase 7). Calibration/thresholding (Phase 6) and sealed-test evaluation + promotion to `champion` (Phase 7) are out of scope here.

---

### Phase 6 — Calibration + Cost-Sensitive Threshold *(1 day)*

**What this achieves:** The model outputs calibrated probabilities (not raw scores), and the decision threshold is set to reflect the actual business cost of a missed churner versus a wasted retention call — not the default 0.5. Three cost scenarios are evaluated so the business can choose the level of risk they are comfortable with.

**Deliverables:**
- `src/telco_churn/models/calibrate.py` — `CalibratedClassifierCV` wrapping the best LightGBM model; tests both sigmoid and isotonic methods; keeps whichever achieves the lower Brier score; logs calibrated model as a new MLflow artifact
  > **⚠ Flagged deviation from the archived notebook (per CLAUDE.md):** the notebook uses a *fixed* `method='sigmoid'` (Platt) calibrator (§14.1). Selecting sigmoid-vs-isotonic by Brier is a deliberate, small methodological change — not a transcription. Caveat to apply: isotonic can over-fit on a small calibration set (~1,400 val rows), so sigmoid may legitimately win; **document the method actually chosen** and its Brier in `ANALYSIS.md` rather than assuming isotonic. If the result is sigmoid, the outcome matches the notebook and the deviation is moot. The notebook's threshold logic and evaluation math are otherwise preserved verbatim.
- `src/telco_churn/models/threshold.py` — cost-sensitive threshold search over the OOF probability distribution; three scenarios from `notebooks/_archive/EDA-original.ipynb`:
  - **Conservative** (high cost of a missed churner): threshold ~0.22
  - **Base** (balanced): threshold ~0.30
  - **Optimistic** (high cost of a wasted call): threshold ~0.38
- `tests/unit/test_threshold.py` — synthetic cost matrix → expected threshold; degenerate cases (zero cost, single class)
- Both modules log artifacts to the same MLflow run as Phase 5
- `notebooks/04-calibration-and-threshold.ipynb` — reliability diagrams before and after calibration; cost curve annotated with each scenario threshold

**Verification:** Calibrated Brier score ≤ uncalibrated Brier score; base-scenario threshold matches the value documented in `README.md`.

> **📌 Cross-reference with Phase 4a (`notebooks/02a-feature-discovery.ipynb`):** The discovery notebook explicitly names `models/calibrate.py` and `models/threshold.py` as the home of the production decision threshold — telling readers that `DISCOVERY_THRESHOLD` (the prevalence-based reference used for lap-to-lap delta comparisons) is *not* the production threshold. When this phase lands: (1) confirm both modules exist at the paths named in the notebook (`src/telco_churn/models/calibrate.py` and `src/telco_churn/models/threshold.py`); (2) verify the notebook's cross-reference text is still accurate (function names, file paths); (3) update the notebook's threshold documentation block if the calibration/threshold API has changed from what was described.

> **⚠ Layout flag (from Phase 3):** Add `reports/figures/` to the repository layout in the "Repository Layout (target)" section when this phase lands. `reports/` is already used implicitly (Phase 2 writes `reports/validation/`, Phase 7 writes `reports/metrics.json`) but was omitted from the top-level layout. Phase 6 is the first phase to save evaluation charts (reliability diagrams, cost curves) to disk, making the omission visible. Update the layout table and add `reports/figures/` as the destination for saved charts.

---

### Phase 7 — Evaluation + Error Analysis + Registry Promotion *(2 days)*

**What this achieves:** A sealed test-set evaluation (the test set has never been touched until this point) produces bootstrap-confidence-interval-bounded metrics that are honest estimates of production performance. A structured promotion decision replaces the `challenger` alias with `champion` only when the new model improves on both **ranking (PR-AUC)** and **calibration (Brier)**; recall at the operating threshold is reported but does not gate the decision.

**Deliverables:**
- `src/telco_churn/models/evaluate.py`:
  - Sealed test-set metrics: ROC-AUC, PR-AUC, recall, precision, F1, Brier score
  - 1,000-iteration bootstrap 95 % CIs (routine lifted verbatim from the original notebook)
  - Writes `reports/metrics.json`
  - Logs all metrics and the report to the MLflow run
- `src/telco_churn/models/register.py` — promotes `challenger` → `champion` if and only if it beats the current `champion` on both **PR-AUC** (ranking quality; threshold-free) and **Brier score** (calibration; lower is better); no promotion otherwise; logs the decision with structured event `model_promoted` or `model_rejected`. **The operating threshold is shipped as a separate versioned config artifact** alongside the model — *not* folded into the promotion comparison — so "is the new model better at ranking?" and "where do we cut?" stay independent, separately-auditable decisions. (Recall@threshold remains a *reported* metric; it does not gate promotion, because it inherits the fixed-threshold fragility discussed in `summary.md` §4.2.) **Model versioning is automatic:** MLflow auto-increments integer version numbers on every `log_model` call — there is nothing to manage manually. What `register.py` manages is the *alias layer*: flipping the `champion` alias to point at the new version number when the promotion gate passes, and leaving `challenger` on the new version otherwise. The FastAPI service (Phase 9) always loads whichever run holds the `champion` alias at startup — it is decoupled from version numbers entirely. The Phase 10 Prefect retrain flow automates this promotion on every weekly retrain cycle; the only case requiring manual intervention is an emergency rollback, where `champion` is re-pointed at an older version number via `mlflow.MlflowClient().set_registered_model_alias()`.
- `tests/unit/test_evaluate.py` — bootstrap CI math verified on a synthetic dataset with known population AUC
- `notebooks/05-error-analysis.ipynb` — SHAP global feature importance, SHAP local explanations for representative FN/FP cases, confusion matrix at each cost scenario threshold

> **Ordering & test-set discipline:** the error analysis here is *confirmatory* (notebook §12 / §16.4 — SHAP + FN/FP profiling of the final model), distinct from the *generative* error-feature loop already baked into Phase 4. The sealed test set is touched exactly **once**, here — `evaluate.py` is the *only* module permitted to import the test split (the structural isolation set up in Phase 5). Under continuous retraining (Phase 10) do **not** re-use this same sealed test set for every challenger-vs-champion comparison — that erodes it; promote on a rolling/time-based holdout instead. This preserves the "test set touched once" invariant (Lifecycle & Framing Gaps, Group A). Turning the *generative* loop into reproducible production code (so the repo demonstrates the full lifecycle in code, not just the migration) is a **deliberately deferred v2** — see "What This Plan Deliberately Does Not Include" — sequenced after Phase 14 so the production spine ships first.

**Verification:** `uv run python -m telco_churn.models.evaluate` produces `reports/metrics.json` whose PR-AUC and ROC-AUC CIs overlap the CIs reported in `README.md`.

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
  | `features` | `features/build.py`, `sql/features/` | `datasets/processed/telco_churn_features.parquet` (Parquet — static snapshot for downstream stages), `preprocessing.pkl` |
  | `train` | `models/train.py`, `configs/` | MLflow run ID, `feature_space.txt`, `feature_columns.txt` |
  | `evaluate` | `models/evaluate.py` | `reports/metrics.json` |

> **Feature versioning and lineage — what Phase 8 closes:** the `features` stage deps (`features/build.py` + `sql/features/*.sql` + the DVC-tracked raw CSV) are content-hashed by DVC. This is the *provenance* half of feature lineage — it records exactly which feature-engineering code ran on which data version to produce the processed dataset. The *membership* half is already covered by Phase 5 MLflow artifacts (`feature_space.txt` — what was available; `feature_columns.txt` — what was selected). Cross-referencing the DVC `features` stage cache entry with the MLflow run ID (logged as a `train` stage out) gives a complete, reproducible lineage chain: raw data version → feature code version → feature space → selection decision → model. This combination — DVC for provenance, MLflow for membership — is the standard approach for projects without a dedicated feature store.

- DVC local remote configured for now; swapped to S3 in Phase 12

> **Deliberate scope — why the DAG stops at `evaluate`:** the five DVC stages cover the *data-transform* pipeline (raw → reproducible metrics). Calibration + thresholding (Phase 6) and registry promotion (Phase 7) are intentionally **not** DVC stages. They are *decision* steps, not deterministic data transforms: calibration depends on a held-out fold, the threshold encodes a business cost choice (owned outside the pipeline — see `summary.md` §4.5), and promotion compares against the live `champion` in the MLflow registry, which is mutable state DVC cannot content-hash. Folding them in would make `dvc repro` non-deterministic (its output would depend on whatever `champion` currently exists). Instead, those steps are driven by the Phase 10 Prefect `retrain` flow, which calls `train → evaluate` (reproducible, DVC-tracked) and then `calibrate → threshold → register` (decision layer) as explicit flow tasks. If full champion reproducibility is ever required, the fix is to pin the comparison baseline to a specific run ID rather than the `champion` alias — not to add these as DVC stages.

- **SQL view materialisation (required):** The Phase 4 SQL views recompute on every read — acceptable for development but not for a DVC pipeline. The `features` stage entry point must call `build_feature_df(engine)` to execute the SQL graph **once**, then immediately write the result to `datasets/processed/features.parquet` before exiting. The `train` stage lists that Parquet file as its sole data dependency, not Postgres. This gives three guarantees: (1) every training run reads a static, content-hashed snapshot; (2) `dvc repro` never blocks on the DB when the features hash is unchanged; (3) if Postgres is unavailable, all downstream stages still run from the cached Parquet. The DB is only contacted during the `features` stage, which DVC skips if its deps (raw CSV + `build.py` + SQL files) are unchanged.

- **Retire `build.py __main__` block:** The `if __name__ == "__main__"` block in `src/telco_churn/features/build.py` is a Phase 4 development scaffold — it wires together the full feature pipeline (config load → DB connect → SQL views → `build_feature_df` → write CSV) so the pipeline could be verified manually before DVC existed. In Phase 8 the DVC `features` stage entry point takes over that responsibility with two changes: output is Parquet instead of CSV, and DVC manages invocation. The `__main__` block should be removed from `build.py` at this point — it becomes dead code once the stage entry point exists. The core logic (`build_feature_df`, `_add_python_features`, column constants) stays in `build.py` permanently; only the CLI scaffold is retired. Also delete `datasets/processed/telco_churn_processed.csv` from the repo — it is superseded by the DVC-tracked `datasets/processed/features.parquet`. **The new stage entry point must assert `df_out.shape[0] > 0` before writing the Parquet output** — a zero-row result from a broken SQL view should be a hard failure, not a silent empty artifact (eighth-pass QA item 4).

- **No manual retraining flags (replaces the notebook's `RETRAIN_BEST` / `LOG_ARTIFACTS` booleans):** the hand-set flags that decided what to recompute do **not** migrate into `src/`. DVC's content-hashed DAG determines staleness — changing a dep reruns exactly the affected stages and nothing else. This is the engineering replacement for the manual flags (former Group B item).
- **Phase 2 cleanup:** Remove `clean_dataframe()` from `validate.py` — imputation now belongs to the `features` stage's fitted `SimpleImputer`. Update `validate_clean()` to expect the features stage output directly. Remove the associated tests.
- **Phase 2 cleanup:** In the DVC `validate` stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event, and call `sys.exit(1)`.
- **SQL migration strategy:** `CREATE TABLE IF NOT EXISTS` is idempotent for creation but blind to changes — adding a column, renaming one, or tightening a constraint will be silently skipped on re-run. Adopt Alembic (the natural fit for a SQLAlchemy-backed project) before Phase 8 ships so that schema changes are applied reproducibly across local, CI, and AWS environments. Existing DDL in `sql/schema/001_create_raw.sql` becomes the initial migration version.

**Verification:** Change a hyperparameter in `configs/model/lightgbm.yaml`; `dvc repro` re-runs `train` and `evaluate` only, not `ingest`, `validate`, or `features`.

---

### Phase 9 — FastAPI Serving + Streamlit UI *(3 days)*

**What this achieves:** The champion model is accessible via a REST API with a health endpoint, batch prediction support, and built-in Prometheus metrics. A Streamlit UI lets a non-technical user submit a customer profile and see the churn probability alongside the top reasons — using the same API.

**Deliverables:**
- `src/telco_churn/serving/schemas.py` — Pydantic v2 request/response models; field constraints aligned with the Pandera schema (single source of truth)
- `src/telco_churn/serving/predict.py` — loads the `champion` model and preprocessor from MLflow at startup; exposes `predict_single` and `predict_batch`
- **Threshold as policy, not model state:** `/predict` returns the **calibrated `P(churn)`**; the decision rule (operating threshold, any per-segment cuts, EV formula) lives in a **separate versioned config / policy layer** loaded at startup, changeable without redeploying the model artifact. This keeps the business-owned operating point decoupled from the model — see `summary.md` §4.5. The response includes both the probability and the decision so callers can apply their own threshold if they prefer.
- `src/telco_churn/serving/app.py` — FastAPI app:
  - `POST /predict` — single customer prediction
  - `POST /predict/batch` — batch scoring (array of `CustomerFeatures` objects); same model and preprocessor as the single endpoint — batch is a delivery mode, not a model change; the prediction unit (one customer per score) is identical in both modes
  - `GET /health` — liveness probe
  - `GET /ready` — readiness probe (model loaded)
  - `GET /metrics` — Prometheus metrics via `prometheus_fastapi_instrumentator`
  - Structured log per prediction: request ID, features, probability, threshold, decision

> **Note — batch as the operational backbone:** In production, batch is typically the primary delivery mode. A nightly/weekly job scores the entire active customer base, writes results to a `churn_scores` table, and the CRM reads from there. Real-time (`/predict`) is the supplement — used for event-triggered interventions (e.g., a customer calls to cancel). The `pipelines/batch_predict.py` flow in Phase 10 is the scheduled incarnation of this pattern. The prediction unit (`a single customer per score`) is identical in both modes.

- `src/telco_churn/ui/streamlit_app.py` — 19-feature form → `POST /predict` → probability gauge + top-5 SHAP contributions
- `Dockerfile` (multi-stage, FastAPI) + `Dockerfile.ui` (Streamlit); both added to `docker-compose.yml`
- `tests/integration/test_api.py` — FastAPI test client: `/predict` returns valid schema; `/health` returns 200; batch endpoint accepts arrays

**Verification:** `docker compose up && curl -X POST http://localhost:8000/predict -d @example_payload.json` returns a churn probability; Streamlit at `:8501` displays a prediction with SHAP contributions.

---

### Phase 10 — Prefect Orchestration (Continuous Training) *(2–3 days)*

**What this achieves:** The model retrains automatically every week and checks for data drift every day — without manual intervention. The Prefect UI provides a full audit trail of every run, including failures.

**Deliverables:**
- Prefect 3 server added to `docker-compose.yml` (UI at `:4200`)
- `pipelines/retrain.py` — weekly Sunday 02:00 schedule; runs `ingest → validate → features → train → evaluate → register`; promotes `challenger` to `champion` on PR-AUC AND Brier improvement (same gate as `register.py`)
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
0 → 1 → 2 → 3 → 4a → 4b → 5 → 6 → 7 → 8
                                        ↓
                                  9 → 10 → 11 → 12 → 13 → 14
```

Phase 4a (structured feature search) is the authoritative gate that **decides** the feature set; Phase 4b **engineers** the survivors into SQL + `build.py` + schema. 4a sits at the feature ↔ model boundary because it needs a baseline model to profile errors against — it reuses `features/preprocessing.py` (built in 4a, shared with Phase 5 `train.py`).

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
| `00-data-ingestion.ipynb` | 2 | Raw CSV → 5 Pandera gates → Postgres ingest; example violations |
| `01-eda.ipynb` | 3 | Statistical tests, distributions, churn-rate breakdowns |
| `02a-feature-discovery.ipynb` | 4a | Structured feature search: domain hypotheses + OOF blind-spot profiling → adoption gate, incl. a rejected decoy |
| `02b-feature-engineering.ipynb` | 4b | Distribution / justification view of the 4a-adopted feature set |
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
| `ColumnTransformer` definition | `models/train.py` (Phase 5) — fitted on training split only; wrapped with the model in a single sklearn `Pipeline` |
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
- **An *automated / online* generative feature loop (v2)** — the *human-in-the-loop, reproducible* generative loop is now **in scope as Phase 4a**: the analyst profiles errors on an OOF substrate, hypothesises features, and the adoption gate keeps/rejects them (`features/generate.py` + `notebooks/02a-feature-discovery.ipynb`), reproducing the convergence of `EDA-original.ipynb` §10.5 as runnable, audited code. What remains deferred to **v2** is the *automated* version — a drift-triggered loop that re-opens discovery on new data post-deployment without an analyst in the seat (Phases 10 & 13 supply the triggers). The automated engine is deliberately out of MVP scope: unattended feature invention is a governance and reproducibility hazard and is **not** standard practice for classical tabular ML — the human-in-the-loop loop (Phase 4b) is the industry-correct form. See `summary.md` §4.4.
