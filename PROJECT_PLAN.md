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
│   ├── performance_check.py    # realised-performance feedback loop (matured-label PR-AUC)
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
| 11 | CI/CD + staging | GitHub Actions `ci.yml`, `integration.yml`, `cd.yml` (staging → approval → production), `data-quality.yml`; GitHub Environments | Not started |
| 12 | AWS deployment | ECR + App Runner + RDS + S3 | Not started |
| 13 | Monitoring | Prometheus + Grafana + Evidently drift | Not started |
| 14 | Documentation polish | README, runbook, architecture diagram | Not started |

---

## Phased Execution Plan

**Precondition — problem framing is done before Phase 0.** The prediction unit, label definition, cost structure, success criteria, and the single selection metric are decided *first* and documented in `ANALYSIS.md §0`; every phase below presupposes that framing. This is the true step 0 of any ML project — it is a document, not a build phase, because for this migration it was settled in the original notebook. Do not treat "environment" as the start of the lifecycle; treat it as the start of the *engineering wrapper* around an already-framed problem.

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

> **📌 Methodological caveat — split timing (known limitation, documented not silently accepted).** The strict-rigour ideal is to seal the dev/test split *before* any decision that optimises against a metric — including this phase's OOF-PR-AUC **adoption gate**. In this build the split is defined later, in Phase 5 `train.py`, so 4a's adoption gate technically profiles over the full dataset, test rows included. Why this is accepted here rather than reworked: (1) the **only** feature adopted is `charge_per_service`, a *deterministic domain transform* (`monthlycharges ÷ service_count`) that would be adopted on its domain rationale regardless of which rows the gate saw — there is no fitted statistic to leak; (2) the leakage channel is the *adoption decision*, not the feature values, and a deterministic transform carries no target signal across the split; (3) the honest performance number still comes only from the Phase 7 sealed-test read, which no feature decision touched downstream of adoption. **The correct discipline is nonetheless split-first**, and it is enforced where it actually bites: the Phase 10 retrain loop re-runs selection (Phase 5 §17.1 re-selection cadence) on a split established *before* selection, and any *future* candidate that is a fitted/data-derived feature (target/frequency encoding, learned bin edges, anything with a trainable parameter) **must** seal the split before its adoption gate runs — for those, this caveat is not a free pass. Record this limitation in `ANALYSIS.md` so it reads as a considered trade-off, not an oversight. *(A `refactor/canonical-split` task list exists — `docs/canonical-split-refactor-tasks.md` — that resolves this by sealing the split before discovery and re-running 4a on dev; when that lands, this caveat is rewritten to "resolved.")*

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

> **📌 Workflow — `src`-first, notebook renders; `train.py` is the full bake-off, the winner is its *output*.** A common misreading is "experiment in the notebook, then port only the winning model to `train.py`." This project is built the other way (per CLAUDE.md's *notebooks are thin wrappers* rule): **the logic lives in `src/` from the start** — `train.py` and `select.py` run the candidate comparison, feature selection, and Optuna tuning — and the notebooks (`03-model-selection.ipynb`, `03b-feature-selection.ipynb`) **import from `src/` and render/narrate** the results loaded from MLflow; they do not compute them. Two consequences: **(1)** `train.py` contains the *entire* bake-off (`DummyClassifier` + `LogisticRegressionCV` + LightGBM, all candidates), not just the chosen model — the selection is an *output* of running it under the Step 2 decision rule (logged to MLflow, registered as `challenger`), not a manual notebook call hand-coded afterward; **(2)** "only the selected model goes forward" is a *downstream* property (Phases 6→9 consume the committed `tree_preprocessor` + LightGBM; the `linear_preprocessor` is comparison-only and nothing downstream imports it) — *not* a property of `train.py`, which must keep the full comparison to stay reproducible and auditable. Scratch-notebook exploration is still fine; the discipline is the **graduation rule** — once something works, it moves *into* `src/` (the single source of truth), it does not stay in the notebook to be reimplemented later (the same notebook→`src/`→thin-notebook path Phase 3 followed with the archived EDA notebook). The prep checklist encodes this order: `src/` logic (Steps 1–5) is built first; the notebooks (deliverables 10–11) load and render last.

**Order of operations (do not reorder):**

1. **Build candidates** — train three models on the *same data, CV folds, and metric*, each through the preprocessing its model family requires (see the model-specific-preprocessors decision below): `DummyClassifier(strategy='prior')` (no-information floor) and `LogisticRegression` (a genuine linear contender — see Step 2) through the `linear_preprocessor`, and LightGBM (the pre-committed nonlinear model) through the `tree_preprocessor`. Holding the data, folds, and metric fixed — *not* the encoding — is what makes the comparison a fair measuring instrument: identical preprocessing would quietly handicap the linear model (dummy-variable trap) and is therefore not the neutral choice. **Imbalance parity across candidates:** the LogReg candidate is `LogisticRegressionCV(Cs=10, cv=5, scoring='average_precision', penalty='l2', class_weight='balanced', max_iter=1000, random_state=42)`. **Why `LogisticRegressionCV`, not a fixed `C`:** sklearn's `LogisticRegression` is L2-regularized by default at an arbitrary `C=1.0`; comparing that against a tuned LightGBM (Step 4) would violate the plan's own "compare ceilings, not defaults" principle in reverse — handicapping the linear model at its default while the tree reaches its ceiling, and undermining the "genuine contender, not a formality" claim. The inner `Cs` grid tunes the L2 strength `C` on each outer fold's training portion (leak-free by construction, same guarantee as `select.py`), so the linear model is compared *at its own ceiling*. Scope is kept to `C` only (L2 fixed — the conventional linear-baseline ceiling); `l1`/`elasticnet`/solver permutations are a deliberately-avoided rabbit hole. **Why L2 (ridge), not L1 (lasso):** feature selection is owned by a separate, model-agnostic step (Step 3) — using L1 here would let the penalty silently zero out features, putting LogReg on a *different, smaller* input space than LightGBM and confounding the linear-vs-tree comparison (a performance gap would no longer isolate functional form from feature count). L2 keeps every feature in play so the comparison measures only the thing it is meant to. L2 is also the more stable choice given the dataset's correlated predictors (`tenure`/`totalcharges`/`monthlycharges`): L1 picks arbitrarily among a correlated group and the pick can flip across folds, whereas L2 shrinks correlated coefficients together — yielding a more stable, more interpretable standardized-coefficient table. `class_weight='balanced'` handles the ~27 % positive rate at the loss level (the linear analog to LightGBM's `scale_pos_weight ≈ 2.77`, satisfying the "imbalance handling is required" rule for this candidate too), so neither contender is handicapped on imbalance and the bake-off stays fair in *both* directions (the dummy-trap guard protects the encoding; `class_weight` protects the loss). The `DummyClassifier(strategy='prior')` needs no weighting — it returns the base rate by construction. `max_iter=1000` (above the sklearn default of 100) avoids silent `ConvergenceWarning`s inside the CV folds that would otherwise hand back a half-optimized linear model.
   >
   > **📌 Why class weights, not resampling (record the rationale in `ANALYSIS.md`).** Cost-reweighting (`scale_pos_weight` / `class_weight='balanced'`) is the deliberate choice over SMOTE / under- / over-sampling — not a default. Four reasons specific to this build: (1) **~27 % positive is mild imbalance** — resampling earns its keep at extreme ratios (<5 %), not here, where the loss reweighting already sees ample minority signal; (2) **the feature space is mostly categorical/one-hot** — SMOTE interpolates incoherently across OHE columns (SMOTE-NC is awkward and fabricates implausible synthetic customers); (3) **calibration is a first-class Phase 6 deliverable** — every base-rate-altering resampler decalibrates probabilities, muddying the reliability diagram and Brier signal, whereas reweighting perturbs them least and most predictably; (4) **the operating point is set explicitly in Phase 6** — once the threshold is tuned against business cost, resampling (a crude boundary shift) solves the same problem twice and adds no PR-AUC (rank-based selection makes any difference likely within CI). A resampling bake-off is therefore **not** added to `src/`; the written rationale is the deliverable. *(If a visual exhibit is ever wanted, it lives as a single contained, gates-nothing cell in `03-model-selection.ipynb` — class_weight vs in-fold SMOTE vs undersampling on CV PR-AUC **and** Brier, resampling applied inside the fold's training portion only, never on full data.)*
   > **⚠ Flagged deviation:** the notebook trains three tree ensembles (LightGBM, XGBoost, RandomForest) and selects the best. We keep only LightGBM. On tabular data the three trees land within noise of each other; a *default-config* bake-off compares defaults rather than ceilings (an unsound selection mechanism); and each extra model carries a maintenance tail for no decision value. LightGBM is committed up front for the reasons below — if it ever underperforms, adding XGBoost back is a cheap one-off experiment. Record this deviation in `ANALYSIS.md` when Phase 5 lands.
   >
   > **📌 Baselines are measuring instruments, not predictors.** The `DummyClassifier` uses `strategy='prior'` (its `predict_proba` returns the base rate), so it anchors **two** floors at once: PR-AUC (= prevalence ≈ 0.27) and Brier/calibration — `most_frequent` would give degenerate one-hot probabilities and a useless Brier reference. Three uses: (1) **leakage canary** — run through the *identical* pipeline + dev CV it must score ROC-AUC ≈ 0.5 / PR-AUC ≈ prevalence; anything higher means target leakage or a fold peek, not skill; (2) **accuracy-trap exhibit** — a `most_frequent` dummy would score ~73 % accuracy while being worthless, the one-line proof of why accuracy is banned here; (3) **read PR-AUC as lift over the prevalence floor**, never the raw number — the floor moves with churn rate, so it is recomputed each retrain (Phase 10), not hardcoded. Business-policy baselines (*treat all* / *treat none*) live in the Phase 6/7 cost analysis, not here.
   >
   > **📌 `customerid` flow through the training pipeline.**
   > The industry standard is to keep identifiers alongside the feature matrix to the model boundary and exclude them passively — not to drop them at feature engineering time. `customerid` is available for error analysis (Phase 7), prediction tracing, and serving but must never reach the model.
   >
   > | Stage | `customerid` treatment |
   > |---|---|
   > | `build_feature_df` | Present in the input DataFrame; absent from the returned `feature_df` because it is not in `BINARY_STR_COLS`, `BINARY_INT_COLS`, `MULTI_CAT_COLS`, or `NUMERIC_COLS` |
   > | `train.py` dev/test split | Split `customerid` out as a separate `pd.Series` alongside `X_dev`, `X_test` — do not drop it |
   > | `ColumnTransformer` | `remainder='drop'` passively excludes any column not listed in a transformer — `customerid` never reaches the model even if accidentally present in `X` |
   > | Error analysis (Phase 7) | Re-attach `customerid` Series to `y_pred` for joins back to raw customer records |
   >
   > **📌 Decision — model-specific preprocessors (adopted).** Two named preprocessors, *not* a single shared `ColumnTransformer`. Full OHE without `drop='first'` on multi-categorical features reintroduces perfect multicollinearity for linear models (the dummy-variable trap), inflating coefficient variance and destabilising estimates — tree models are immune since each split is evaluated independently. So: `linear_preprocessor` (OHE `drop='first'` on all categoricals + `StandardScaler`) pairs with `DummyClassifier` and `LogisticRegression`; `tree_preprocessor` (OHE `drop='if_binary'` on binary cols, no drop on multi-cat, no scaling) pairs with LightGBM. This costs one extra `ColumnTransformer` definition and lives **only in the Steps 1–2 comparison** — once LightGBM is committed at the end of Step 2, every downstream stage (selection, tuning, calibration, threshold, serving) uses `tree_preprocessor` + LightGBM. The cost is justified because the linear model is a genuine contender (Step 2): a shared preprocessor would handicap it and risk a false "LightGBM out-predicts" reading on the very comparison that decides whether the tree's added complexity is warranted.
   >
   > **Native categorical vs OHE for LightGBM (deliberate, record in `ANALYSIS.md`).** LightGBM supports native categorical splits (Fisher-optimal partitioning), often preferred over OHE for trees — but OHE is the *considered* choice here, not an oversight: every categorical is ≤ 4 unique values (low cardinality, where OHE's column-blowup cost is negligible), OHE keeps the tree pipeline uniform with the linear one, and it yields **cleaner per-level SHAP attribution** for the Phase 9 Streamlit top-5 contributions — the explainability deliverable LightGBM was committed for. A one-liner in `ANALYSIS.md` keeps OHE reading as deliberate rather than as not-knowing-about native categoricals.
   >
   > **`tenure_cohort` vs `tenure` for the linear preprocessor (Phase 4a finding):** Phase 4a rejected `tenure_cohort` for LightGBM — continuous `tenure` already encodes all cohort signal and the tree learns the same non-linear boundaries through its own splits. For logistic regression, a cohort-binned `tenure` (OHE’d with `drop='first'`) is the correct substitution: a single linear slope on continuous `tenure` cannot represent the survival-distribution shape (steep drop in the first 12 months, then gradual flattening), whereas OHE’d cohorts give the model a separate coefficient per risk tier — and the per-tier coefficients are directly readable in the LogReg odds-ratio table (the interpretability exhibit), which a `SplineTransformer` basis (the modern alternative, considered and rejected for this build) would not yield. **`tenure_cohort` is *not* added to `build_feature_df`/`FeatureSchema`** — it was deliberately rejected for the tree (importance 0.0024), so adding it to the shared feature space would force `select.py` to re-drop it every run and pollute the production feature space with a comparison-only column. Instead, `build_linear_preprocessor` **routes `tenure` through an internal cohort-binning branch** (a stateless `FunctionTransformer(pd.cut)` over a module-level `TENURE_COHORT_EDGES` constant — the fixed edges from `ANALYSIS.md` — feeding `OneHotEncoder(drop='first', handle_unknown='ignore')`); `train.py` hands `build_linear_preprocessor` the **same** column groups as the tree path (`tenure` stays in `numeric`), with no caller-side column swapping. **Leak-free by construction:** the cohort edges are *fixed domain constants*, not data-derived quantiles, so the binning needs no `fit` and leaks no train-set statistics — uniquely in this pipeline it does not even require per-fold refitting (quantile binning would; fixed edges are why this stays simple). The cohort encoding has the same comparison-only lifetime as the `linear_preprocessor` itself — nothing downstream imports either.
   >
   > **📌 Why LightGBM is the pre-committed model.** Not because it out-predicts XGBoost/RF (it doesn't on tabular data — they sit within noise), but for build-specific reasons: (1) **fast exact TreeSHAP** — explainability is a first-class deliverable (Phase 7 error analysis, Phase 9 Streamlit top-5 contributions) and the whole error-driven feature loop (Phase 4a) is tree-centric; (2) **training speed** (histogram + leaf-wise growth) keeps Optuna's ~250 fits and Phase 10's weekly retrain cheap; (3) **clean imbalance + calibration** — `scale_pos_weight` for the ~27 % positive rate (Phase 5) and well-behaved probabilities (Phase 6); (4) **operational fit** — sklearn API, MLflow support, small artifact for the registry/serving path (Phases 7–9); (5) **continuity** with the source-of-truth notebook. These build-specific reasons — not predictive superiority — are why the linear model, even if within CI, does not displace it here.
   >
   > **Expected finding:** Telco churn is near-linear — logistic regression typically reaches PR-AUC within the test-set CI of tuned LightGBM. If so, state in `ANALYSIS.md` that LightGBM is chosen for calibration quality, SHAP interaction structure, and error-analysis-driven features — **not** because it materially out-predicts the linear model.

2. **Compare on PR-AUC** → confirm LightGBM clears the `DummyClassifier` floor and document its margin over `LogisticRegression`. **The linear model is a genuine contender, not a formality:** if the **paired PR-AUC difference** (LightGBM − LogReg) is within noise — its bootstrap CI on Δ includes 0 — or real but immaterial (`|Δ|` below the pre-registered materiality threshold `Δ* = 0.01`), say so plainly in `ANALYSIS.md` and acknowledge the simplicity LogReg would buy — easier to ship, calibrate, monitor, and explain. LightGBM is still adopted, for the build-specific explainability and feature-loop reasons in Step 1 — *not* because it out-predicts the linear model — making this a documented, reasoned choice rather than an automatic one. PR-AUC is the sole selection metric — threshold-free and imbalance-appropriate at a ~27 % positive rate. ROC-AUC is logged as a threshold-free diagnostic; the **threshold-dependent** diagnostics (precision, F1) are **not** reported at the default 0.5 — they are reported as a **precision/F1-at-fixed-recall profile** over OOF predictions at recall ∈ {0.70, 0.80, 0.90}, computed identically across all three candidates. **Calibration caveat to record in `ANALYSIS.md`:** `class_weight='balanced'` decalibrates LogReg's raw probabilities (shifts them toward the minority class) but barely moves PR-AUC, because PR-AUC is rank-based and the reweighting is near-monotone — a concrete demonstration that reweighting is an operating-point lever, not a ranking lever. The comparison is therefore unaffected; if LogReg's calibrated probabilities were ever needed, they would be recovered via the Phase 6 calibration path.
   > **⚠ Flagged deviation:** the notebook selects and tunes on CV recall@0.35. Standardising to PR-AUC separates ranking quality from operating-point selection — the threshold is set in Phase 6 and is a business-owned knob. The notebook's preprocessing, hyperparameter ranges, calibration, threshold logic, and evaluation math are otherwise preserved verbatim. Decision approved June 2026; record in `ANALYSIS.md` when Phase 5 lands.
   > **📌 Fixed-recall profile and per-segment robustness/fairness checks** are threshold-planning and model-characterisation tools, not selection tools — they belong in Phase 7 (`05-error-analysis.ipynb`) where the champion is examined in depth. See Phase 7 deliverables for the full specification of each.
   > **📌 Comparison fairness — the LightGBM side is a *conservative* default at this step.** LogReg enters Step 2 tuned to its ceiling (`LogisticRegressionCV` over `C`), but LightGBM is still **default-config** here — its Optuna tuning is Step 4, *after* the selection freeze, and cannot run earlier without violating select-then-tune. The asymmetry is therefore *structural*, and it cuts in the safe direction: default LightGBM is the **floor** of the tree's range, so if it already clears tuned-LogReg the family decision holds *a fortiori* (tuning only widens the gap). **Two-stage reporting (resolves the inconsistency):** the family *decision* is made here at Step 2 against default LightGBM; the LogReg-vs-LightGBM margin *reported* in `ANALYSIS.md` / the model card is **refreshed after Step 4** against the tuned model — the "within the CI of *tuned* LightGBM" phrasing above is a post-tuning number, not a Step-2 one. Recording only the Step-2 default-vs-tuned margin would understate the gap and contradict the narrative.
   > **📌 Comparison as a tracked artifact (not just a notebook cell).** Each candidate (`DummyClassifier`, `LogisticRegressionCV`, default LightGBM) is logged as its **own MLflow run** with CV mean ± std PR-AUC and **wall-clock train/predict time** — the comparison table is then reproducible from the tracking store, and "LogReg is within CI *and* ~N× cheaper to train and serve" becomes a recordable `ANALYSIS.md` sentence, not just a PR-AUC reading. **Paired folds are load-bearing:** the `RepeatedStratifiedKFold` splitter is instantiated **once** (`random_state=42`) and the *identical* fold assignments are passed to every candidate, so the comparison is paired by construction — the precondition for both the McNemar test (notebook §11.1) and the bootstrap-CI margin to be valid (an unpaired comparison would confound model differences with fold-assignment differences).
   > **📌 Selection significance test + pre-registered decision rule.** The "is LightGBM better than LogReg" call is made with a **paired bootstrap on the PR-AUC *difference*** Δ = AP(LightGBM) − AP(LogReg), *not* an overlapping-CI eyeball of the two marginal CIs. Both models are scored on the same OOF rows, so their absolute PR-AUCs are strongly positively correlated (a hard resample drags both down together); that shared resample-difficulty noise dominates the marginal CIs but **cancels in Δ** (`Var(Δ) = Var(A) + Var(B) − 2·Cov(A,B)`, with `Cov ≫ 0` here), so the difference is estimated far more precisely than either absolute score. Overlapping-CI is biased toward declaring a *tie* — exactly the project's expected finding ("churn is near-linear"), so leaning on it would be a confirmation-bias trap. Procedure: per bootstrap draw, resample dev-row indices **once** (stratified, prevalence-preserving) and score *both* models on those **same** indices (pairing relies on the identical-fold-indices guarantee above), record Δ_b; the percentile CI on {Δ_b} is the test. Companion: the **per-fold win rate** across the 15 (5×3) folds. McNemar (notebook §11.1) stays a *hard-prediction* disagreement diagnostic — it does not test the PR-AUC difference and does not drive this decision.
   > **Pre-registered decision rule** (fixed *before* seeing results, to bar post-hoc retrofitting; materiality threshold **`Δ* = 0.01` PR-AUC**):
   > 1. **Floor (non-negotiable):** LightGBM must clear the Dummy floor (AP ≈ prevalence).
   > 2. **Tie / immaterial** — CI on Δ includes 0, *or* excludes 0 but `|Δ| < Δ*`: practical tie → **explainability + feature-loop rationale decides** → adopt LightGBM (the planned outcome); record the within-noise/immaterial margin in `ANALYSIS.md`.
   > 3. **Material LightGBM win** — CI excludes 0 in LightGBM's favour, `Δ ≥ Δ*`: adopt LightGBM on performance *and* explainability.
   > 4. **Material LogReg win** — CI excludes 0 in LogReg's favour, `Δ ≤ −Δ*` — **the kill condition:** explainability does **not** override a real accuracy gap (LogReg's standardized-coefficient table is itself a first-class explainability artifact) → **ship LogReg** and re-examine whether the tree family is warranted. This abort branch is what makes "reasoned, not automatic" airtight.
   > The rule is *applied* at Step 2 against default LightGBM (conservative floor); the Δ and its CI *reported* in `ANALYSIS.md` / the model card are refreshed post-Step-4 against the tuned model (per the two-stage-reporting note above).
   > **📌 Contingency — if rule 4 fires, Step 3 selection swaps to a linear-appropriate method (selection is model-dependent).** The Step 3 selector is null-importance against **tree gain** — a *tree-specific* signal. Selecting LogReg's input space with a tree's notion of importance is incoherent, so a LogReg winner invalidates the planned selector. (This is the seam behind Step 1's "model-agnostic step" claim: the null-importance *framework* is model-agnostic, but its *base signal* is not.) Pick **one** model-matched method **a priori** — not an empirical bake-off of selectors (trying many and keeping the best CV PR-AUC is selection-method overfitting, the same multiple-comparisons / cherry-picking trap the one-metric invariant bars): **(A)** re-run the *same* null-importance framework with the LogReg's standardized-coefficient / permutation-importance signal (apparatus unchanged, only the signal swaps); or **(B, leaning — better for the `tenure`/`totalcharges`/`monthlycharges` correlation)** switch the production model to **elastic-net LogReg** (embedded selection, penalty CV-tuned; L1 alone flips arbitrarily across the correlated group, elastic-net's L2 term groups them stably) — note this drops the comparison-time **L2-only** choice, which Step 1 adopted as a *comparison device*, not a production commitment. Either way the reduced set is **validated and frozen identically** (reduced CV PR-AUC within the full set's bootstrap CI) and **stepwise / p-value selection is explicitly avoided** (inflated p-values, unstable, overfits). Record the chosen method and rationale in `ANALYSIS.md`. *(Low-probability branch — the expected outcome is LightGBM adopted; this note exists so the abort path is not under-specified.)*

**Iterative feature loop — Steps 2c → 2d (runs between Step 2 and Step 3; exit to Step 3 when convergence is reached)**

After Step 2 confirms the model family, enter a diagnostic loop on the **development set only** (sealed test remains untouched) before committing the freeze in Step 3. The loop has two sub-steps run in order; Step 3 is reached only when the loop exits via its convergence criterion.

   **2c. Diagnose bias/variance.** Fit the confirmed model family (default-config LightGBM, `scale_pos_weight ≈ 2.77`) on the *current* feature set via `RepeatedStratifiedKFold` over development and record two quantities: the **train-vs-CV PR-AUC gap** (variance signal) and the **slope of the learning curve at maximum training size** (bias signal — still rising means the model has not saturated the available signal). Decision:
   - Gap < 0.05 and learning curve has flattened → feature set is fit-for-purpose; proceed to 2d to check for segment-level failures.
   - Gap ≥ 0.05 or curve still rising → proceed to 2d to identify *which segments* are driving the failure before deciding whether the fix is regularisation or new features.

   This is a **named decision point**, not notebook decoration. Record t he gap, the curve slope, and the decision in `ANALYSIS.md` and `03-model-selection.ipynb`.

   > **📌 Step 2c reads the same model state as the "full-feature default ①" in Step 3's selection-boundary diagnostic.** The two checks share the same CV fit — no extra run. Step 3's ① is the *entry* state into selection; Step 2c uses it as the *exit gate* from the loop. They are complementary, not duplicated.

   **2d. OOF profiling and error analysis.** Score development rows using **out-of-fold (OOF) predictions** from the same `RepeatedStratifiedKFold` run in 2c — the only unbiased predictions available at this stage; the sealed test is never opened here. Segment OOF errors by: contract type (`month-to-month` vs longer), tenure band (< 12 months / 12–36 / > 36), service count quintile, and `charge_per_service` outliers (> 95th percentile). Decision:
   - **No systematic failure pattern *and* CV PR-AUC gain from the previous loop round (vs baseline for round 1) < 0.005** → convergence reached; exit loop and proceed to Step 3.
   - **Systematic failure identified** → hypothesize the missing signal → engineer the feature in Phase 4 (`build.py` / SQL) → re-enter at Step 2b (re-compare candidates on the updated feature set) before running 2c/2d again.
   - **3 rounds completed without convergence** → exit regardless; document the remaining failure pattern in `ANALYSIS.md` as a known limitation and a candidate for Phase 10 retraining.

   > **📌 The "profiling holdout" is OOF development predictions — never the sealed test.** The Phase 7 error analysis (`05-error-analysis.ipynb`) re-runs the same segment breakdown on the sealed test after champion promotion — that is the *reporting* version (one-shot, the number that stands). This loop is the *generative* version (iterative, drives feature engineering). Conflating the two means the sealed test's signal is consumed while choosing features — it is no longer sealed.

   > **📌 For this dataset the loop is expected to converge in 0–1 rounds.** Phase 4a already ran an error-driven feature search against a baseline model and engineered the candidates now in `build_feature_df`. If Phase 4a closed the loop, Step 2c will show a clean gap and Step 2d will find no systematic patterns — both recorded explicitly so "we checked and it converged" is auditable, not assumed. The loop structure is here so Phase 5 is methodologically sound regardless of how exhaustive Phase 4a was.

3. **Select features (freeze the input space)** — run null-importance / target-permutation selection inside CV **on the development set** (test untouched), against a *default-config* LightGBM (not the tuned model). Refit on survivors; confirm the reduced model's CV PR-AUC sits within the bootstrap CI of the full-feature model. The expected, fully acceptable outcome is "keep most/all" — documented honestly either way.
   > **📌 Discussion (revisit at Phase 5 implementation):** sklearn's built-in selectors — `SelectFromModel`, `SelectKBest`, `RFE` — all implement `fit`/`transform` and can slot directly as a pipeline step between the `ColumnTransformer` and the model. The selector is then refitted on each CV fold's training portion automatically, giving the same leak-free guarantee as the custom null-importance wrapper but with less code. Use a built-in if the null-importance experiment shows most features are informative (the expected outcome here); build the custom wrapper only if per-fold stability reporting is a required output.
   > **⚠ Flagged deviation:** the notebook performs no feature selection — VIF / Cramér's V / permutation importance are diagnostics, not a selection gate. This step is a deliberate methodological addition. The reduced set is adopted **only if** its CV PR-AUC is within the full set's bootstrap CI and there is a parsimony reason to prefer it; otherwise the full set stands. Record the keep/drop decision in `ANALYSIS.md`.
   > **📌 Bias/variance diagnostic across the selection boundary (notebook-only, gates nothing).** Record the train-vs-CV PR-AUC gap at two model states bracketing this step: ① the **full-feature default** model (read *before* selection) and ② the **reduced default** model (after refit on survivors). The ①→② delta isolates *selection's* variance effect — expected small given the "keep most/all" outcome, because in this build the primary overfit lever is **Optuna regularisation (Step 4), not feature selection**. This does **not** influence the keep/drop call — PR-AUC alone governs selection (one-metric invariant); it is a `03-model-selection.ipynb` diagnostic only. ② is the *same* model state Step 4 reads as its before-tuning baseline, so the two checks chain into one trajectory **full → reduced → tuned**; report all three gaps in `ANALYSIS.md`.
   > **📌 Governance gate — runs *before* statistical selection (checks importance cannot see).** Three governance checks applied first, each recorded in `ANALYSIS.md`, *not* emergent outputs of null-importance: (1) **Protected attributes — policy: awareness + measurement (adopted).** All four protected / quasi-protected attributes — `gender` (sex), `seniorcitizen` (age), `has_partner` (marital status), `dependents` (familial status) — **remain model inputs; none is hand-excluded.** Rationale: this is a retention/marketing *benefit* allocation (not a credit/housing/employment *denial* where statutory protection binds), where demographic targeting is normal practice; all carry genuine churn signal; and whether `gender` is usable is left to **multivariate null-importance** — a better judge than the **univariate** EDA association (V 0.0086 is bivariate and blind to interactions). Fairness is enforced by **measuring disparate impact across all four axes** (Step 2 fairness lens), not by exclusion. **Two principles recorded for a future higher-stakes model:** (a) excluding a protected attribute is a **normative** call, *never* delegated to a statistical selector — "let the model decide" must not make *using a protected attribute* contingent on profitability; (b) direct use of a protected attribute is **disparate treatment** (the sharper line in regulated domains) vs proxy-driven **disparate impact** — both are acceptable-but-monitored *here* given the benefit context, but that distinction would govern a denial model. **Proxy caveat:** exclusion ≠ fairness anyway — other features reconstruct protected signal, which is why the measurement lens, not the input toggle, is the load-bearing control. (2) **Serving-time availability** — confirm every surviving feature is computable at inference (trivially true here — all from the customer snapshot — but the check is recorded). (3) **Leakage** — confirm no feature encodes the target (mostly a Phase 4 concern; selection is the natural audit point). Only features clearing this gate enter null-importance.
   > **📌 Imbalance parity in the selection model.** "Default-config" means **untuned hyperparameters**, *not* unweighted — the selection LightGBM still carries `scale_pos_weight ≈ 2.77` (the "imbalance handling is required" rule applies to *every* fit, including selection). Importances from an unweighted model rank features for a *different* model than the one that ships; the selection model must mirror the final model's imbalance handling to be representative.
   > **📌 The freeze is not permanent — re-selection cadence (Phase 10 link).** The frozen input space is fixed for *this* training cycle, not forever. Under continuous retraining (Phase 10 / §17.1) data drifts and a frozen set can go stale, so the selection experiment is re-run as a **periodic audit on the retrain cadence** (not every retrain — selection is not a per-fit step) and the keep/drop is re-confirmed. The frozen list is a versioned artifact that can change across cycles, never a one-time permanent commitment.

> **── Internal boundary (decide → optimize) ──** Steps 1–3 *decide* — the model family and the frozen feature set; Steps 4–5 *optimize* that fixed model. The freeze is a hard barrier (re-running selection invalidates any tuning after it), which is why this is the natural 5a/5b seam — kept as one phase, but split here if the two halves are ever shipped as separate PRs.

4. **Tune only the confirmed family** (LightGBM) with Optuna — *after* the feature set is frozen by step 3. Each trial is a nested MLflow child run; the study is the parent run. Optimisation objective is PR-AUC (`average_precision`).
   > **📌 Optuna is LightGBM-only by design — match the search method to the search space.** Bayesian optimisation (TPE) earns its keep only when the space is high-dimensional, the objective is non-convex with interacting parameters, and evaluations are expensive enough that a sampler must *prune* rather than enumerate — all true for LightGBM's 7–8 interacting hyperparameters. None hold for the linear family, so it is **never** handed to Optuna. The LogReg / elastic-net contender has a **1–2 parameter convex search space** (`C`, and `l1_ratio` for elastic-net), which is tuned at *fit time* via its **regularisation path** (`LogisticRegressionCV`, with `l1_ratios=` for elastic-net) — not as a separate tuning step. The path warm-starts each `C` from the previous coefficients, so the full grid costs ~1 fit, and it is deterministic (no sampler warm-up variance to corrupt the `Δ = AP(LightGBM) − AP(LogReg)` margin). This is why LogReg enters Step 1 already at its ceiling and why the rule-4 contingency (Step 2) reads "penalty CV-tuned," not "Optuna-tuned." **General rule:** *few hyperparameters + convex loss → enumerate (grid / regularisation path); many + non-convex → sample (Randomized / Bayesian).* `GridSearchCV`/`RandomizedSearchCV` are the generic enumerate/sample tools and would be the fallback if LogReg lacked a path estimator — but it has one, and `GridSearchCV` only becomes necessary here if tuning something *outside* the penalty path (e.g. `solver`), which this build does not.
   > **📌 Which parameters enter the search — hyperparameter triage (keep the space small).** Each hyperparameter falls in one of three buckets, and only the first is searched: **(1) tune** — params that move the bias–variance tradeoff, whose optimum is data-dependent (no robust default); **(2) fix by procedure** — params with a *better* resolution than blind search (here `n_estimators`, resolved by early stopping below); **(3) pin to a constant** — determinism/correctness knobs, problem-structure constants, and *modelling decisions* that should be reasoned about, not searched. The discipline is keeping bucket 1 small: every dead dimension dilutes the 50-trial budget and pollutes the hyperparameter-importance plot. **LightGBM (cluster the knobs, tune one or two per cluster — they are substitutes that interact):** *capacity* → `num_leaves` tuned, `max_depth` a guard rail (`num_leaves ≤ 2^max_depth`); *learning dynamics* → `learning_rate` tuned low, `n_estimators` fixed + early-stopped (the canonical decoupling of the most-correlated GBM pair — see below); *subsampling* → `subsample` (+ enabling `subsample_freq=1`), `colsample_bytree`; *leaf regularisation* → `min_child_samples`, `reg_alpha`, `reg_lambda`. Bucket 3 for LightGBM is `deterministic`, `force_row_wise`, `n_jobs`, `verbose` (and `subsample_freq=1`, a *fixed* param that **enables** a tuned one — without it `subsample` is a silent no-op). **LogReg (a linear model has almost no structural knobs — capacity is fixed by the feature set, so only regularisation is left):** bucket 1 is `C` alone (or `C` × `l1_ratio` for elastic-net); `penalty` (L2 chosen a priori for correlated-predictor stability — Step 1), `class_weight='balanced'` (imbalance policy), and `solver`/`max_iter` (convergence knobs that change only *whether* the fit converges, not the fitted model) are all bucket 2/3 — modelling decisions or numerics, never searched. **Industry practice that produces this list:** start from published ranges / prior Optuna bests rather than from scratch (this build warm-starts from the notebook's best); after an initial study, read `plot_param_importances` (fANOVA) and **demote** flatlined params to fixed; decouple known-correlated pairs by procedure; and let the trial budget set the width (small budget → fewer params, wider ranges). Record any demote/keep change in `ANALYSIS.md` when Phase 5 lands.
   > **📌 Early stopping over a tuned `n_estimators` (industry-standard tree-count selection).** `n_estimators` is **not** an Optuna-searched integer. Each trial instead fixes a high ceiling (`n_estimators ≈ 2000`) with a low `learning_rate` and lets `early_stopping_rounds ≈ 50` resolve the tree count per fold, evaluated on `average_precision` — the selection metric, so stopping never optimises a different signal than selection does. This decouples the two most-correlated hyperparameters (`learning_rate` × `n_estimators`), spends the trial budget only on parameters that actually shape the decision boundary, and generalises better than a directly-tuned count. **Nesting cost (accepted):** early stopping needs a held-out portion *inside* each outer CV fold, so each fold's training portion is further split into a fit set and a thin early-stopping validation set (`random_state=42`); the model is then scored on the fold's untouched held-out portion exactly as before. On ~5,600 development rows this inner carve-out is thin but acceptable; the final development-fit model uses the median early-stopped tree count across folds. The early-stopping validation set lives **entirely inside development** — the sealed test is never touched.
   > **📌 Study hygiene — four standard checks around the search (not just "run 50 trials").** A mature Optuna study is not only a sampler + objective; four practices guard the result, each recorded in `ANALYSIS.md` / `03-model-selection.ipynb`: **(1) Pruning** — pair the TPE sampler with a **MedianPruner**, reporting each fold's PR-AUC as an intermediate value (`trial.report(fold_ap, fold_idx)` + `trial.should_prune()`) so a trial clearly behind the running median is killed mid-CV. *Decision recorded either way:* adopt it for the budget saving, or state explicitly that 50 trials × 5 folds is cheap enough to skip pruning — the silence is the only wrong answer. **(2) Boundary-hit check (the most common practical tuning bug)** — after the study, confirm the best trial is **interior** to every searched range; a best value sitting on a range edge (e.g. `reg_lambda` at its max, `num_leaves` at its ceiling) means the range was too narrow and the true optimum is outside it → **widen that range and re-run**. **(3) Best-trial selection rule — `1-SE`, not raw max.** Consistent with this build's overfit-conscious stance: among trials whose CV PR-AUC is within one standard error of the best, pick the **most-regularized (simplest)** config, not the bare argmax — a sliver of CV score traded for a config less likely to be a lucky sampler draw (Breiman / `glmnet` / `caret` convention). Record raw-max vs `1-SE` explicitly. **(4) Convergence check** — plot `plot_optimization_history` (or the EDF) and confirm the study **plateaued** rather than still climbing at trial 50; a still-rising curve means the budget was too small. Gates nothing (PR-AUC selection still governs); it justifies the trial budget honestly, alongside the overfit diagnostic below. **Parallel trials deliberately not used** (`n_jobs>1` / RDB multi-process): they forfeit the bit-reproducibility the LightGBM determinism block guarantees — trial *completion order* becomes a race, so the seeded sampler's "trial → result → next suggestion" sequence is no longer fixed — and they degrade sequential TPE at this small budget (parallel workers sample from a stale surrogate), all to save single-digit-minutes of wall-clock that does not need saving. Parallelism is the right call only at large budgets with expensive per-trial fits, via Optuna's RDB `storage=` path — not in-process `n_jobs` — and neither condition holds here or at the Phase 10 retrain scale.
   > **Overfit diagnostic (notebook §10.6 / §10.8):** the notebook runs a bias/variance check both *before* tuning and *after* tuning (on the Optuna best) — train-vs-CV PR-AUC gap and a learning curve. The *before-tuning* model here is the **reduced default** (② from Step 3's selection-boundary diagnostic), so this check completes the **full → reduced → tuned** trajectory begun there: ②→③ isolates *tuning's* variance effect, the build's primary overfit lever. Carry this through as a **diagnostic in `03-model-selection.ipynb`**, not a new `src/` module: it gates nothing (PR-AUC selection still governs) but it documents that the tuned model did not overfit the train split before it is handed to Phase 6/7. Report all three gaps in `ANALYSIS.md`.

5. **Register the tuned model** as `challenger`. Calibration, thresholding (Phase 6), and sealed-test evaluation (Phase 7) are deliberately separate, later phases.
   > **📌 Registration is a packaging contract, not a one-liner — four standard inclusions.** "Register the challenger" carries an industry-standard packaging checklist; the full **`Pipeline`** (`tree_preprocessor` + LightGBM as one pyfunc — not the bare estimator, which is the real train/serve-skew guard) is logged with: **(1) a model signature + input example** — `mlflow.<flavor>.log_model(..., signature=infer_signature(X, preds), input_example=X.head())` — so the Phase 9 FastAPI service can **enforce input schema** at inference instead of failing deep in the pipeline; logging a pyfunc without a signature is the most common "worked in training, broke in serving" gap. **(2) A log → reload → predict round-trip check** — load the just-logged pyfunc back and assert prediction parity with the in-memory model on a sample, catching serialization / missing-dependency / unpicklable-preprocessing bugs *before* Phase 9, not after (lands in `tests/unit/test_train.py`; the existing 500-row smoke test exercises *training*, not the registered-artifact load path). **(3) `model_card.json` populated at registration** — the audit artifact this phase has the inputs for: data version (DVC hash), git SHA, final hyperparameters, CV PR-AUC + the paired-Δ vs LogReg, the fixed-recall profile, the Step 2 fairness/robustness flag results, intended use, and known limitations (also the Phase 7 promotion-decision input). **(4) Pinned environment logged with the model** — confirm the auto-logged `python_env.yaml` / `requirements.txt` carries the **resolved** versions (from `uv.lock`), not floating ranges, so the Phase 9/12 serving container loads under the same deps it trained on. **Aliases, not stages** — `champion`/`challenger` is the modern MLflow approach (registry stages are deprecated).
   > **📌 Sequencing — the Phase 5 `challenger` is the raw tuned model: uncalibrated and un-thresholded.** It is *not* serving-ready, and nothing should wire serving to it. The registry lineage is: Phase 5 registers the tuned model as `challenger`; **Phase 6** adds calibration + the cost-sensitive threshold (logged to the same run); **Phase 7** evaluates on the sealed test once and promotes `champion`. Serving (Phase 9) loads `champion`, never the Phase-5 `challenger`.

> **📌 Split & resampling strategy — which set feeds what.** The dataset is small (~7k rows), so a static validation holdout would be noisy and wasteful (a ~1,400-row val set is a thin, high-variance place to fit a calibrator). We therefore use a **two-way stratified split** (`random_state=42`, defined once in `train.py`): **development (~80%)** and a **sealed test (~20%)**. Everything the val set used to do is done by **cross-validation over development**, so every dev row serves in each role via rotation — nothing is spent on a static val set.
>
> **CV scheme:** the development CV is `RepeatedStratifiedKFold` (5 folds × 3 repeats, `random_state=42`) for the **model comparison and feature-selection confirmation** — on ~5,600 rows a single 5-fold split is a high-variance point estimate, and the LogReg-vs-LightGBM margin is exactly what that noise corrupts; repeating tightens it at 3× CV cost. **Optuna tuning (Step 4) keeps a single stratified 5-fold** — the TPE sampler is robust to per-trial noise and tripling 50 trials × the early-stopping inner split is not worth the cost there. (The dataset is a customer *snapshot* with no time axis and one row per `customerid`, so a temporal or grouped split does not apply — stratified random splitting is the correct choice.)
>
> | Set | ~Share | Role | Consumed in |
> |---|---|---|---|
> | **development** | ~80% | Model fitting + **CV over development** for selection — `RepeatedStratifiedKFold (5×3)` for model comparison + feature-selection confirmation, single stratified 5-fold for Optuna tuning; **cross-fit calibration** (`CalibratedClassifierCV(cv=5)`); **threshold** search over **OOF** calibrated probabilities; confirmatory **error analysis** on OOF predictions | Steps 1–4, Phase 6, Phase 7 error analysis |
> | **test** | ~20% | **Sealed**; the final reporting metric only | Phase 7 (`evaluate.py`, once) |
>
> **Everything is fit on development (via CV / OOF); nothing is conflated because the final metric is read on the sealed test, which the calibrator and threshold never saw.** The anti-conflation guard is structural: the data the operating point is *fit* on (development) is disjoint from the data it is *reported* on (test) — the rotation inside development keeps every fit step leak-free (a row is either in a fold's training portion or its held-out portion, never both), and the sealed test is the final arbiter. The production refit (Phase 10, §17.1) retrains on development + test (100 %).

**Deliverables:**

*Configs:*
- `configs/model/lightgbm.yaml` — **searched** parameter ranges warm-started from the notebook's Optuna best: `num_leaves`, `learning_rate` (biased low — see early stopping, Step 4), `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, plus a `max_depth` guard rail (capped so `num_leaves ≤ 2^max_depth` holds — stops leaf-wise growth from building deep, overfit branches on ~5,600 rows). **`n_estimators` is not searched** — it is a fixed high ceiling resolved per fold by early stopping (Step 4). **Fixed (non-searched) params:** `subsample_freq=1` — without it `subsample` is a *silent no-op* (the sklearn wrapper disables bagging by default, so a tuned `subsample` would have zero effect and pollute the hyperparameter-importance plot); `deterministic=True`, `force_row_wise=True`, pinned `n_jobs`, and `verbose=-1` for bit-reproducible, quiet fits (LightGBM's histogram build is otherwise non-deterministic across thread counts, which would undercut the `random_state=42` reproducibility guarantee).
- `configs/training/diagnose.yaml` — convergence and segmentation knobs for `diagnose.py`: `gap_threshold: 0.05` (train-vs-CV PR-AUC gap ceiling for Step 2c), `gain_threshold: 0.005` (per-round CV PR-AUC gain floor for Step 2d exit), `max_rounds: 3` (iteration cap), `cv_folds: 5`, `cv_repeats: 3`, `random_state: 42`; tenure band edges (`[0, 12, 36, inf]`), service count quantile boundaries, and `charge_per_service_outlier_pct: 95` for the OOF segment definitions — surfaced as config knobs so the convergence criterion is a recorded, reproducible decision and not a buried constant.
- `configs/training/optuna.yaml` — `n_trials: 50`, `sampler: tpe`, `direction: maximize`, `metric: average_precision`, `cv_folds: 5`, `random_state: 42`, `early_stopping_rounds: 50`, `n_estimators_ceiling: 2000` (the fixed upper bound early stopping resolves against — see Step 4). **Study-hygiene knobs (Step 4 four-checks note):** `sampler_seed: 42` and `n_startup_trials: 10` — TPE needs an explicit seed for reproducibility and a random warm-up before the surrogate engages; at a 50-trial budget the 10-trial warm-up is a deliberate 20 % allocation, not a silent default. `pruner: median` (or `none` if pruning is explicitly waived) and `selection_rule: 1se` (vs `argmax`) — surfaced as recorded knobs so the pruning and best-trial decisions are reproducible, not buried constants.

*Experiment tracking (MLflow) — layout for the whole phase, not just one step:*

MLflow threads through every step (Step 2 candidate runs, Step 3 selection runs, Step 4 nested tuning, Step 5 registration), so the tracking layout is specified once here rather than per step. The governing rule: **runs that must share one sortable PR-AUC leaderboard to drive a decision belong in one experiment** — experiments are split by *which decision on which data*, never by *which activity within one decision*.

- **One experiment for all of Phase 5** — `telco-churn-training`. Every candidate, the selection runs, and the tuning study are comparable on the single selection metric (PR-AUC), and seeing them ranked together *is* the Step 2 bake-off — splitting baselines / selection / tuning into separate experiments would fragment the very comparison Step 2 makes reproducible from the tracking store (`mlflow.search_runs`). One-experiment-per-activity is the anti-pattern; one-experiment-per-decision is the standard.
- **What is a run** = a unit of comparison you would rank: each candidate (`DummyClassifier`, `LogisticRegressionCV`, default LightGBM) is a sibling run; the full-feature vs reduced-feature default LightGBM (Step 3) are runs; the Optuna study is **one parent run** with its 50 trials as **nested child runs** (collapsed in the UI so the sweep is one row, not 50). **Parent/child logging division:** the **parent** carries the search config, the best trial's metric (so it ranks against the sibling candidates on the PR-AUC axis), the study-level plots (optimization history, param importance, parallel coordinates), and **the one refit model** — the `challenger` registered in Step 5 is promoted from the **parent** run. **Children stay lightweight** — one sampled param set + that trial's CV mean ± std, *no per-trial model artifact* (50 serialized models is pure storage waste and no mid-search trial is ever served). **What is *not* a run:** the 5 CV folds inside a trial (aggregated into the trial's mean ± std), and the 50–100 null-importance permutation refits (Step 3) — these are *artifacts* of a run (the importance/null table), not runs. The common failure is exploding every fit into a run (unreadable leaderboard) or logging nothing.
- **Run content — log the data and the *evidence*, not just the metrics (the run is the source of truth).** Two things beyond params/metrics make a run self-describing from the tracking store: **(1) Native dataset tracking** — log the development split as a first-class input via `mlflow.log_input(mlflow.data.from_pandas(dev_df, source=..., name=..., targets="churn"), context="training")`, which records the dataset's **schema, digest, and role** (queryable, shown in the UI's "Datasets used" panel). This is *complementary* to the DVC-hash tag, not a duplicate — the DVC tag says *which version*; `mlflow.data` records the *schema/digest/role inside the run*. The **sealed test is not logged here** — it is a Phase 7 input. **(2) Diagnostic exhibits as artifacts, not only notebook renders** — the comparison table, PR curves, the fixed-recall profile, the bootstrap-Δ distribution, the per-segment robustness/fairness panel, and the study plots are **logged to their runs as artifacts**, so "the comparison is reproducible from the tracking store" (Step 2) covers the *evidence*, not just the numbers. This is the tracking-store half of the `src`-first / notebook-renders workflow: the run holds the exhibit, the notebook re-renders it from MLflow rather than being its only home.
- **Organize within the one experiment with tags, not more experiments** — `stage` (`baseline`/`comparison`/`selection`/`tuning`), `model_family` (`dummy`/`logreg`/`lightgbm`), the **git SHA**, and the **DVC data hash** (pins every run to its data snapshot — load-bearing once Phase 8/10 land). `mlflow.search_runs(filter_string="tags.stage='comparison'")` then reconstructs each table programmatically — the mechanism behind Step 2's "comparison table reproducible from the tracking store." Stable `run_name`s for readability.
- **Autolog policy — off in `train.py` (explicit logging), allowed in notebooks.** `mlflow.autolog()` instruments `.fit()` to auto-log hyperparameters, training curves (LightGBM's `best_iteration`, eval-set metrics), and a model artifact. It is **not** used in `train.py`: it logs what the *framework* sees at `.fit()`, not the *experiment design* — it cannot compute CV PR-AUC mean ± std, the paired-bootstrap Δ, or the fixed-recall profile (all live in the CV loop, not a single fit); it logs *every* fit (per-fold, per-trial), flooding the leaderboard against the run-granularity rule above; it logs a model per fit by default (`log_models=True`), the storage-waste anti-pattern the parent/child division avoids; and it has no concept of the two-layer artifacts, nesting, or `champion`/`challenger` aliasing. The production path therefore logs explicitly. *(Narrow optional exception: `mlflow.lightgbm.autolog(log_models=False)` scoped to the **final refit on the parent** to grab hyperparameters + curves cheaply — everything else stays manual.)* **In exploratory notebook cells, autolog + explicit logging may be mixed**, under one rule: open the run yourself (`with mlflow.start_run():`, so autolog logs into *your* run rather than creating and closing its own around `.fit()`), let autolog own its keys (hyperparameters / model / training curves), and manually log only the gaps in *different* keys (CV/derived metrics, custom plots, tags) — duplicate keys with differing values raise, since params are immutable once set. Most Phase 5 notebooks *load* the study from MLflow rather than train, so this mixing is a general-exploration allowance, not a structured-notebook need.
- **Considered and not used — `mlflow.evaluate()` and MLflow Projects (record so the omissions don't read as oversight).** **`mlflow.evaluate()`** auto-computes a metric battery + diagnostic plots, but it imposes *its own* metric set and a **0.5-threshold confusion matrix** — colliding with the one-metric (PR-AUC) invariant, the deliberate fixed-recall profile (0.5 is explicitly rejected — Step 2), and the sealed-test-touched-once rule (no convenience API casually scoring test). Evaluation stays the bespoke, controlled `evaluate.py` (Phase 7); the same reasoning bars reaching for it in the Step 2 comparison. **MLflow Projects (`MLproject`)** is not used either — reproducible pipeline execution is owned by **DVC (Phase 8)**; an `MLproject` entry-point/env wrapper would duplicate it.
- **Rule-4 consequence (LogReg wins) — the nesting degenerates to a flat run.** The parent/child structure above is LightGBM-specific: it exists only because Optuna produces discrete, rankable trials. If Step 2 rule 4 fires (material LogReg win — see the Step 2 contingency note), there is **no Optuna study**, because LogReg/elastic-net is tuned by its regularization path *inside a single `fit`* (`LogisticRegressionCV`, with `l1_ratios=` for elastic-net) — the per-`C` evaluations are internal computation (same category as CV folds), logged as an **artifact** (the `C`-vs-CV-PR-AUC validation curve, a `C × l1_ratio` heatmap for elastic-net), never as child runs. So the winner is **one flat run** (no parent/child), the Optuna study plots are replaced by that validation curve plus the standardized-coefficient / odds-ratio table (notebook `03`), `preprocessing.pkl` is the `linear_preprocessor`, and the `challenger` is promoted from that flat run. The `optuna.yaml` study-hygiene knobs (`pruner`, `n_startup_trials`, `sampler_seed`) are moot; only the `1se` rule ports — applied to the path (strongest-regularization `C` within 1 SE of the best, post-processed from `LogisticRegressionCV.scores_`, which otherwise returns the raw-best `C`).
- **Second experiment reserved for Phase 10** — `telco-churn-retrain`. Production retrains answer a *different* question (challenger-vs-champion on a rolling holdout, not the sealed test) on a *different* cadence; isolating them keeps hundreds of automated operational runs from burying the human-made Phase 5 selection decisions. This is the *lifecycle-stage* boundary — the only experiment split that earns its keep here.
- **Orthogonal to the Model Registry** — experiments/runs track the *search*; the registry (`telco-churn-pipeline`, aliases `champion`/`challenger`) tracks *deployable versions*. A run is *promoted* into the registry (Step 5 registers `challenger`); the experiment layout above does not touch the alias design in CLAUDE.md. **CI/CD handoff (forward-reference, not Phase 5 work):** this `challenger` registration is the artifact the later CI/CD pillars gate on — the registry alias is the deployment contract (serving loads `champion`), the `challenger → champion` promotion gate lands in **Phase 7** (`register.py`), continuous training in **Phase 10** (Prefect), the GitHub Actions wiring in **Phase 11**, and the RDS+S3 registry backend in **Phase 12**. Phase 5 only needs to keep producing a registered, reproducible, config-driven challenger (it does — determinism block, `model_card.json`, the 500-row ROC-AUC ≥ 0.75 smoke test is the metric-floor CI hook); the gates themselves are deliberately out of scope here.
- **Tool-integration seams (forward-reference, not Phase 5 work) — loose coupling, each layer owns one concern.** MLflow is the experiment/model layer sitting *alongside* the others, joined by thin seams that Phase 5 only has to leave open, not build: **DVC** (Phase 8) — the **data-hash run tag** above is the seam (DVC versions data/pipelines, MLflow versions experiments/models — both, never double-tracked); **Prefect** (Phase 10) — `train.py` is a clean Hydra `__main__`, so a retrain task invokes it as-is and reads the registry to decide promotion (Prefect owns *when* runs happen, MLflow owns *what* they recorded); **AWS** (Phase 12) — MLflow's backend store → **RDS**, artifact store → **S3**, reached by pointing `MLFLOW_TRACKING_URI` at the server, so the local-`mlruns/`→prod migration is a **config/env change, not code**. The forward-compatibility property Phase 5 must preserve is exactly this: tracking URI, experiment name, and entry point stay **config-driven**, so all three layers bolt on later without rework.

*Source:*
- `src/telco_churn/features/preprocessing.py` (**extended from Phase 4a**) — Phase 4a shipped a single `build_preprocessor` (median-impute + scale numerics; OHE categoricals; `FunctionTransformer(astype str)` on the binary group). Phase 5 reuses it as the **`tree_preprocessor`** (scaling is harmless to LightGBM) and **adds `build_linear_preprocessor`** (OHE `drop='first'` + `StandardScaler`, plus an internal stateless cohort-binning branch that routes `tenure` through `FunctionTransformer(pd.cut)` over a module-level `TENURE_COHORT_EDGES` constant → `OneHotEncoder(drop='first')` — per Step 1) for the `DummyClassifier` / `LogisticRegression` baselines. `train.py` hands it the **same** column groups as the tree path (`tenure` stays in `numeric`); the binning is encapsulated inside the preprocessor and `tenure_cohort` is **not** a `build_feature_df`/`FeatureSchema` column. Additive — the existing tree path is unchanged, so Phase 4a is not broken. The `linear_preprocessor` is comparison-only (Steps 1–2); nothing downstream imports it.
- `src/telco_churn/models/train.py` — reads config via Hydra; sets the tracking experiment once at entry via `mlflow.set_experiment("telco-churn-training")` (the single Phase 5 experiment — see the Experiment tracking layout above; the name is config-driven, not hardcoded at the call site, so Phase 10 can point its retrain runs at `telco-churn-retrain` without a code change); loads feature DataFrame from `datasets/processed/` via `build_feature_df`; performs the stratified development/test split (~80/20, `random_state=42`, **the only place the split is defined**); fits each candidate's family preprocessor (`linear_preprocessor` for Dummy/LogReg, `tree_preprocessor` for LightGBM — see Step 1) on each CV fold's training portion (and on all of development for the final model) and wraps it with the model in an sklearn `Pipeline`; the committed production pipeline is `tree_preprocessor` + LightGBM (OHE `drop='if_binary'` on binary, no drop on multi-cat, no scaling — all categoricals ≤ 4 unique values); stratified 5-fold CV **on the development split** (test excluded — see the Split & resampling note) with `scale_pos_weight ≈ 2.77`, `subsample_freq=1`, and the reproducibility block (`deterministic=True`, `force_row_wise=True`, pinned `n_jobs`, `verbose=-1`) for LightGBM — its `n_estimators` resolved per fold by early stopping on `average_precision` (Step 4), not Optuna-searched — and `LogisticRegressionCV(scoring='average_precision', class_weight='balanced', max_iter=1000, random_state=42)` for the LogReg candidate — its inner `Cs` search tunes L2 strength `C` per outer fold so the linear contender is compared at its own ceiling, not at sklearn's arbitrary `C=1.0` (imbalance parity + fair-ceiling tuning — see Step 1); best model logged as `pyfunc` with `feature_space.txt`, `feature_columns.txt`, and `preprocessing.pkl` and registered as `telco-churn-pipeline` / alias `challenger`. **Two-layer artifact logging (feature space vs model input space):** `feature_space.txt` is logged at the start of the MLflow run by reading the `FeatureSchema` instance's `binary + multi_cat + numeric` fields — it records the full *feature space* (every column `build_feature_df` produced, owned by `FeatureSchema`, which replaces the bare `BINARY_STR_COLS`/`BINARY_INT_COLS`/`MULTI_CAT_COLS`/`NUMERIC_COLS` constants). It is generated here, not in `schema.py`/`build.py`, because it is an MLflow artifact — Phase 4 has no MLflow context. `feature_columns.txt` records the *model input space* — the subset that survived `select.py` and entered the `ColumnTransformer`. The diff between these two files is what selection dropped for that specific run; any MLflow run is self-describing without a git lookup. `feature_space.txt` is identical across most runs (it only changes when `build.py` changes); `feature_columns.txt` can differ on every run where selection experiments change. `preprocessing.pkl` is the fitted `ColumnTransformer` encoding the exact transformations applied to the model input space at training time. The test split is importable **only** by `evaluate.py` (Phase 7) — the "test set touched once" invariant is structural, not conventional. **Binary dtype note:** `FeatureSchema.binary` is a single dtype-heterogeneous group — five `'Yes'`/`'No'` string columns (`gender`, `has_partner`, `dependents`, `phoneservice`, `paperlessbilling`) plus one `0`/`1` integer column (`seniorcitizen`). The binary branch of the `ColumnTransformer` is fed the whole `binary` field with a `FunctionTransformer(lambda X: X.astype(str))` as its first step, normalising the mixed string/int inputs to strings before OHE. **Imputation note:** `totalcharges` is NaN for 11 zero-tenure rows (`tenure = 0`, `monthlycharges > 0` — first bill not yet issued); it is the **only** column with missing values — the other engineered numeric `charge_per_service` derives from the always-present `monthlycharges` (`monthlycharges / GREATEST(service_count, 1)`), so it is never NaN. `totalcharges` must be imputed via `SimpleImputer(strategy='median')` inside the numeric branch of the `ColumnTransformer` — fit on the training data only (each CV fold's training portion; all of development for the final fit), never on test. This is the only place imputation is applied; `build_feature_df` deliberately preserves the NaN values so no training-set statistics leak into the feature-engineering step. Median imputation is appropriate here for three reasons: (1) all 11 rows are non-churners, so they are not in the minority class the model is tuned to identify; (2) 11 rows is 0.15 % of the dataset — negligible impact on any split; (3) only `totalcharges` is imputed, and `charge_per_service` is computed upstream in SQL from `monthlycharges` (not from `totalcharges`), so no derived numeric depends on the imputed cell and inter-column consistency is not a concern.
> *Preprocessor design for the candidate pipelines — the two model-specific preprocessors (`linear_preprocessor` / `tree_preprocessor`) and the `tenure_cohort`-for-`tenure` substitution for the linear family — is **decided** under **Step 1** of the Order of operations (model-specific preprocessors adopted).*

- `src/telco_churn/models/diagnose.py` — diagnostic loop for Steps 2c and 2d; runs against the **development set only** (sealed test never opened); fits a default-config LightGBM (`scale_pos_weight ≈ 2.77`) via `RepeatedStratifiedKFold` over development to compute the **train-vs-CV PR-AUC gap and learning curve** (Step 2c) and generate **OOF predictions** for per-segment error profiling (Step 2d: contract type, tenure band, service count quintile, `charge_per_service` outliers > 95th percentile); logs each loop round as an MLflow run under `tags.stage='diagnosis'` in the `telco-churn-training` experiment so the iteration history is auditable alongside the formal candidate runs; exits when the round-on-round CV PR-AUC gain < `gain_threshold` (0.005) and no systematic failure pattern is found, or after `max_rounds` (3); has a `__main__` Hydra entry point (`make diagnose`). Reads convergence knobs and segment definitions from `configs/training/diagnose.yaml`. Deliberately does **not** consume the sealed test — OOF predictions are the only unbiased scores available at this stage; the sealed test stays for Phase 7 (`evaluate.py`). **Execution order:** `make diagnose` is run before `make train` in the iterative loop; once `diagnose.py` exits with a convergence flag, `train.py` runs the full formal pipeline (Steps 1–5). There is no code dependency between the two — `diagnose.py` is a standalone diagnostic that `train.py` neither imports nor calls.

- `src/telco_churn/features/schema.py` (**extended, not created — the file already exists from Phase 4b, holding the Pandera `CustomerFeaturesSchema` / `FeatureOutputSchema` contracts**) — Phase 5 adds a `FeatureSchema` frozen dataclass (or Pydantic `BaseModel` with `model_config = ConfigDict(frozen=True)`) *alongside* those contracts, owning `binary`, `multi_cat`, and `numeric` as `tuple[str, ...]` fields. This replaces the bare `list[str]` column-group constants in `build.py` with a single typed, immutable object (the Pandera schemas validate row *values*; `FeatureSchema` owns the column *grouping* — two complementary roles in one module). `train.py` imports one `FeatureSchema` instance and feeds its fields directly to the `ColumnTransformer` column lists; `build_feature_df` imports the same instance to select columns — one source of truth, no mutation risk. Rationale: `Final[list[str]]` prevents rebinding but not `.append()`; `tuple` prevents mutation but not misuse; a `frozen=True` schema prevents both and adds field-level validation (non-empty, no duplicates) at no extra cost. `build.py`'s bare column-group lists remain until this phase lands; they are removed in the same PR that adds `FeatureSchema` to the existing `schema.py`.
  **Feature space vs model input space — two distinct concepts:** `FeatureSchema` owns the *feature space*: all engineered columns that `build_feature_df` can produce (the full Phase 4 output). The *model input space* is a separate, narrower concept — it is what actually enters the `ColumnTransformer` after `select.py` narrows the feature space down to survivors. `FeatureSchema` does not define the model input space; `select.py`'s output does. These must be kept conceptually separate: the feature space is stable across many model versions (adding a new engineered column updates it once), while the model input space changes every time selection runs a different experiment. **`frozen=True` semantics:** `frozen=True` means the `FeatureSchema` *instance* is immutable at runtime — no `.append()` or mutation by a caller. It does **not** mean the feature set is version-locked. Evolving the feature set (e.g., adding a new engineered column) requires explicit code changes in `build.py`, `FeatureSchema`, and the relevant SQL/Python feature builder — the frozen instance just enforces that those changes are deliberate rather than accidental runtime side-effects.

- `src/telco_churn/features/select.py` — null-importance / target-permutation selector; fits inside CV on the development set against a default-config LightGBM; returns surviving feature list, importance/null table, and per-fold stability scores; wrapped in an sklearn `Pipeline` so `cross_val_score` re-fits selection on each fold's training portion (leak-free by construction).
   > **📌 Base signal — `gain`, used *inside the null*, not as the raw criterion.** The per-feature importance fed to the selector is LightGBM `importance_type='gain'` (total loss reduction from splits on the feature) — *not* `'split'` (mere split count, which ignores split quality). Raw `gain` is a **biased** standalone selection signal: it is computed on the training set (so it credits splits that overfit), inflates for high-cardinality / continuous features (the `tenure`/`monthlycharges`/`totalcharges` group), and allocates credit arbitrarily among correlated features. The selector therefore **never thresholds on raw gain** — gain is only the base signal compared against its own **target-permutation null distribution** (real gain vs gain under permuted targets). That comparison is what cancels the cardinality/overfit bias (Altmann et al. 2010): a feature's bias appears in *both* its real and its null gain, so it differences away, leaving "is this gain beyond chance?" Swapping the base signal to mean|SHAP| (above) improves consistency but does **not** fix correlated-feature credit-splitting — the null wrapper, not the signal choice, is the real safeguard, which makes `gain` an appropriate and defensible base signal here. **Model-dependence caveat:** `gain` is tree-specific, so this selector is valid only while LightGBM is the committed family. If Step 2 rule 4 fires (material LogReg win), selection swaps to a linear-appropriate method per the contingency note under Step 2 — *not* this tree-gain selector.
   > **📌 Correlation-aware selection (the `tenure`/`totalcharges`/`monthlycharges` cluster).** The significance test does **not** resolve correlated-feature credit-splitting: one of a redundant pair can fall below its null *purely because its partner absorbed the gain*, silently dropping a feature that is individually fine. Mitigate explicitly — either a **correlation pre-filter** (collapse a highly-correlated pair to one representative before selection) or **grouped/clustered permutation** (permute the correlated block together so the null reflects the group, not the individual). At minimum, verify the correlated trio is not *both*-dropped by accident; record the kept/dropped choice and reason in `ANALYSIS.md`.
   > **📌 Selector hyperparameters — pin them; the cutoff *is* the selection decision.** Three parameters the spec must fix, surfaced in `configs/` (alongside the Optuna config) so the run is reproducible and the cutoff is an explicit recorded knob, not a buried constant: (1) **permutation rounds** — number of permuted-target refits building each feature's null (≈50–100, enough to estimate the upper percentile stably); (2) **keep/drop cutoff** — a feature survives if its real gain exceeds, e.g., the 95th/99th percentile of its null (or a fitted-distribution p-value with a stated α); (3) **permutation seed** — `random_state=42` pinned so the null draws are reproducible.
   > **📌 Implementation note — two jobs, two leakage boundaries (do not conflate).** "Selection is separate from training" means the *frozen list* is committed before tuning — **not** that selection runs outside CV. The module has two distinct jobs with different leakage rules:
   > 1. **Validate the keep-vs-reduce decision (leak-free → pipeline + CV).** Wrap `[tree_preprocessor → selector → default LightGBM]` in a `Pipeline` and pass it to `cross_val_score`, so the selector's `fit` (null-importance computation) sees only each fold's *training* portion. This yields the unbiased reduced-model CV PR-AUC compared against the full-feature model's bootstrap CI. **The anti-pattern to avoid:** fitting the selector once on all of `X_dev`, freezing that list, *then* cross-validating with it — this leaks the target into every fold's held-out rows and inflates the estimate.
   > 2. **Mint the single committed list.** CV produces per-fold survivors that disagree (the `per-fold stability scores` output); reconcile to one list by either a stability rule (survive in ≥ k of N folds) or a single selector fit on **all of development**. Fitting on all of development to derive the frozen list is **not** leakage — the leakage boundary that matters is development vs the **sealed test**, and the honest read happens in Phase 7 on that test, which the selector never touches.
   > The CV+pipeline keeps the *estimate* honest; the sealed test keeps the *final number* honest — two separate guards. After the list is frozen, the selector is **removed** from the downstream pipeline; tuning/calibration/serving consume the hardcoded list, not a live selector. **Open sub-decision (see `ANALYSIS.md` §4 stub):** whether the selector operates on the 20 named features (pre-OHE) or the ≈40 one-hot columns (post-OHE) — leaning pre-OHE.
   > **📌 Alternative considered — Boruta-SHAP (rejected for this build, recorded as deliberate).** Boruta-SHAP (shadow-feature null + mean|SHAP| as the importance signal) is the modern synthesis of the two rigorous ideas and a reasonable selector for trees. It is **not** adopted here, for build-specific reasons — not unfamiliarity: (1) the `BorutaShap` package does not expose a clean sklearn `fit`/`transform` selector API, so it cannot drop into the per-fold `Pipeline` that the leakage guard above depends on without hand-wrapping; (2) it is a thinly-maintained third-party dependency, a net-negative signal for a build whose headline is production discipline (the chosen path leans only on first-tier `lightgbm`/`shap`/`sklearn`); (3) at ~20 low-cardinality features with an expected "keep most/all", every rigorous method (null-importance, Boruta, Boruta-SHAP) converges on the same set — so it changes the *result* by ~nothing while adding integration and dependency cost, which also runs against this step's deliberate bias toward the simplest selector that works (the `SelectFromModel` note above). **Cheap available upgrade if ever wanted:** keep the target-permutation wrapper and its `Pipeline` integration, and swap only the per-feature signal from gain to **mean|SHAP|** — captures Boruta-SHAP's consistency benefit (~10 lines) without the library or the API problem. Treated as optional polish, not substance, at this feature count. Record this alternative-considered rationale in `ANALYSIS.md` alongside the keep/drop decision.

*Tests:*
- `tests/unit/test_train.py` — config loading, metric logging contract (mock MLflow client)
- `tests/unit/test_diagnose.py` — synthetic 200-row dev set with a planted segment failure; assert gap and learning curve metrics are computed and logged to MLflow; assert OOF segment table covers all rows with no NaN; assert exit fires when gain < `gain_threshold`; assert max-rounds cap exits without error when convergence is not reached
- `tests/unit/test_select.py` — synthetic data with planted noise and known-informative columns; assert selector drops noise and keeps signal; assert selection is fit inside the fold; cover empty-dataframe and all-noise edge cases

*Notebooks:*
- `notebooks/03-model-selection.ipynb` — loads the Optuna study from MLflow; renders parallel-coordinates plot, hyperparameter importance, CV PR-AUC distribution across trials, and comparison table with reference baselines; **precision/F1-at-fixed-recall profile** (recall ∈ {0.70, 0.80, 0.90} on OOF predictions, all candidates) plus the overlaid PR curves — the threshold-dependent diagnostics reported at recall-first operating points, never the 0.5 default (see Step 2); **LogReg standardized-coefficient odds-ratio table** — because the `linear_preprocessor` scales features, `exp(coef_)` yields per-SD odds ratios that are directly rankable as signed, additive feature effects (with `drop='first'`, each categorical coefficient is the log-odds shift vs its reference category); rendered as an interpretability exhibit alongside the LightGBM SHAP summary — notebook-only, gates nothing; **bias/variance diagnostic** (notebook §10.6 / §10.8): train-vs-CV PR-AUC gap + learning curve for the default-config and the tuned LightGBM, confirming no overfit before hand-off; **paired-bootstrap PR-AUC-difference test** (LightGBM − LogReg) — the Δ bootstrap distribution, its percentile CI, and the per-fold win rate; the instrument behind the Step 2 decision rule (materiality `Δ* = 0.01`); **per-segment robustness + fairness panel** — paired Δ_s on `contract_type` / `tenure_cohort` / `internetservice` and per-group PR-AUC parity on all four protected axes (`gender` / `seniorcitizen` / `has_partner` / `dependents`), flag-only (see Step 2), with the two-year `contract_type` tier marked low-support; **McNemar's test** (notebook §11.1) on the best-baseline-vs-tuned prediction disagreement, reported as a hard-prediction significance diagnostic only — it does not test the PR-AUC difference and does not override PR-AUC selection
- `notebooks/03b-feature-selection.ipynb` — full selection experiment: full-set CV PR-AUC + bootstrap CI → null-importance ranking → reduced-set refit → overlapping-CI check → documented keep/drop decision; imports from `select.py`

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and produces an MLflow run whose cross-validation PR-AUC falls within the bootstrap CI reported in `README.md`; reference baselines (`DummyClassifier`, `LogisticRegression`) appear as rows in the comparison; LightGBM clears the Dummy floor and either beats `LogisticRegression` or sits within its CI with the closeness documented in `ANALYSIS.md`. `notebooks/03b-feature-selection.ipynb` runs end to end and records a keep/drop decision in `ANALYSIS.md`.

**Prep checklist (task → deliverable):** the five ordered steps form the spine; two foundation tasks precede them because Step 1 cannot run without the schema and configs.

| # | Task | Deliverable |
|---|---|---|
| 1 | **Foundation:** `FeatureSchema` (frozen, typed; replaces the bare `list[str]` constants in `build.py`) | `src/telco_churn/features/schema.py` + `build.py` edits |
| 2 | **Foundation:** model + training Hydra configs | `configs/model/lightgbm.yaml`, `configs/training/optuna.yaml` |
| 3 | **Step 1** — build candidates (Dummy `strategy='prior'` + `LogisticRegressionCV` over `C` `scoring='average_precision', class_weight='balanced', max_iter=1000` via `linear_preprocessor`; LightGBM via `tree_preprocessor`; XGBoost/RF dropped — deviation logged) on the **same `RepeatedStratifiedKFold (5×3)` dev folds** (splitter instantiated once, identical indices per candidate → paired); split defined once (`random_state=42`); each candidate logged as its own MLflow run with CV mean ± std PR-AUC + train/predict time; `customerid` split out as a Series; preprocessors fit per fold (development only) | `src/telco_churn/models/train.py` + `features/preprocessing.py` (`build_linear_preprocessor`) |
| 4 | **Step 2** — compare on PR-AUC (sole selection metric); confirm LightGBM clears the Dummy floor and document its margin over LogReg (a genuine contender, not a formality) via a **paired bootstrap on the PR-AUC difference** (not overlapping-CI) under the pre-registered decision rule (materiality `Δ* = 0.01`; kill condition → ship LogReg on a material LogReg win); diagnostics = ROC-AUC (threshold-free) + precision/F1-at-fixed-recall profile (recall ∈ {0.70, 0.80, 0.90} on OOF), never the 0.5 default; **disaggregated check** (pre-registered, flag-only): robustness Δ_s on `contract_type` / `tenure_cohort` / `internetservice` + fairness parity on all four protected axes (`gender` / `seniorcitizen` / `has_partner` / `dependents`) — aggregate PR-AUC still selects | comparison logic in `train.py` + `ANALYSIS.md` note (XGB/RF drop + recall@0.35→PR-AUC deviation + LogReg-vs-LightGBM margin **refreshed post-Step-4 against the tuned model** (the Step-2 decision uses default LightGBM as a conservative floor; report the paired-Δ CI and which decision-rule branch fired) + `class_weight` calibration caveat + "why class weights, not resampling" rationale + native-categorical-vs-OHE deliberate choice + train/predict-time-vs-PR-AUC simplicity note + chosen LogReg `C` reported alongside the standardized-coefficient table, since regularization strength sets the coefficient shrinkage; + per-segment Δ_s and fairness-parity results/flags from the disaggregated check) |
| 5 | **Steps 2c/2d — diagnostic loop** (run after Step 2, before Step 3; repeat until convergence): fit default-config LightGBM via `RepeatedStratifiedKFold` on dev; compute train-vs-CV PR-AUC gap + learning curve (2c); OOF segment profiling on contract type / tenure band / service count / `charge_per_service` outliers (2d); log each round to MLflow under `tags.stage='diagnosis'`; exit when gain < 0.005 and no systematic failure, or at 3-round cap | `src/telco_churn/models/diagnose.py` + `configs/training/diagnose.yaml` |
| 6 | **Step 3** — null-importance/target-permutation selection inside CV on the development set vs default LightGBM; refit on survivors; adopt reduced set only if within full-set bootstrap CI; **freezes the input space** | `src/telco_churn/features/select.py` + keep/drop decision in `ANALYSIS.md` |
| 7 | **Step 4** — tune LightGBM with Optuna (50 TPE trials, `average_precision`) *after* the freeze; `n_estimators` resolved per fold by early stopping on `average_precision` (not searched); `subsample_freq=1` + determinism block (`deterministic`, `force_row_wise`, pinned `n_jobs`, `verbose=-1`) fixed; `max_depth` guard rail; trials as nested MLflow child runs | Optuna tuning logic in `train.py` + `configs/model/lightgbm.yaml` |
| 8 | **Step 5** — register tuned model as `telco-churn-pipeline` / `challenger`; two-layer artifact logging | registration + `feature_space.txt`, `feature_columns.txt`, `preprocessing.pkl` in `train.py` |
| 8 | Tests — config loading + metric-logging contract (mock MLflow); split reproducibility | `tests/unit/test_train.py` |
| 9 | Tests — planted-noise synthetic data; selector drops noise/keeps signal; leak-free; edge cases | `tests/unit/test_select.py` |
| 10 | Notebook — Optuna study + baseline comparison loaded from MLflow; bias/variance diagnostic (train-vs-CV gap + learning curve, pre/post tuning) and McNemar significance test (diagnostic only) | `notebooks/03-model-selection.ipynb` |
| 11 | Notebook — full selection experiment (full-set CI → null-importance → reduced refit → overlap check → decision) | `notebooks/03b-feature-selection.ipynb` |
| 12 | Verification — 50-trial run within README CI; LightGBM ≥ baselines; green `pytest` | passing verification + green test suite |

> **Order is load-bearing:** selection (Step 3) freezes the input space *before* tuning (Step 4) — changing features invalidates tuning. The test split is defined once in `train.py` and stays importable only by `evaluate.py` (Phase 7). Calibration/thresholding (Phase 6) and sealed-test evaluation + promotion to `champion` (Phase 7) are out of scope here.

---

### Phase 6 — Calibration + Cost-Sensitive Threshold *(1 day)*

**What this achieves:** The model outputs calibrated probabilities (not raw scores), and the decision threshold is set to reflect the actual business cost of a missed churner versus a wasted retention call — not the default 0.5. Three cost scenarios are evaluated so the business can choose the level of risk they are comfortable with.

**Deliverables:**
- `src/telco_churn/models/calibrate.py` — `CalibratedClassifierCV(cv=5, ensemble=False)` **cross-fit on the development set** — no static val holdout (see the Phase 5 Split & resampling note): the calibrator is trained on **out-of-fold predictions** (each row calibrated by a model that did not train on it) while the base LightGBM is refit on all of development. Uses 100 % of development for both base fitting and calibration via rotation, so nothing is spent on a static val set. Tests both sigmoid and isotonic methods; keeps whichever achieves the lower Brier score; logs calibrated model as a new MLflow artifact
  > **⚠ Flagged deviation from the archived notebook (per CLAUDE.md):** the notebook uses a *fixed* `method='sigmoid'` (Platt) calibrator (§14.1). Selecting sigmoid-vs-isotonic by Brier is a deliberate, small methodological change — not a transcription. Caveat to apply: isotonic can over-fit when the calibration signal is thin — even cross-fit over the ~5,600-row development set, only ~27 % are churners — so sigmoid (Platt) may legitimately win; **document the method actually chosen** and its Brier in `ANALYSIS.md` rather than assuming isotonic. If the result is sigmoid, the outcome matches the notebook and the deviation is moot. The notebook's threshold logic and evaluation math are otherwise preserved verbatim.
- `src/telco_churn/models/threshold.py` — cost-sensitive threshold search over the **calibrated out-of-fold (OOF) probabilities on the development set** — `cross_val_predict(method='predict_proba')` over the calibrated pipeline, so every probability is out-of-sample and the search is leak-free (*not* raw scores; the sealed test stays untouched for Phase 7). **No conflation results:** the operating point is *fit* on development but the final cost/recall is *reported* on the sealed test (Phase 7), which the threshold never saw. The dependency is load-bearing: the threshold must be derived on the same calibrated scores production will serve, so **calibration precedes thresholding** and a re-calibration invalidates a previously-derived threshold. Three scenarios from `notebooks/_archive/EDA-original.ipynb`:
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
  - **Lift, gains & ranking diagnostics** (notebook §15 + §16.2): cumulative gains curve, lift curve, and a per-decile lift table (decile → count, churn rate, cumulative captured churn, lift vs. base rate). These answer the project's actual business question — *"which customers should we call this week?"* — by quantifying how much churn the top-k% capture; threshold-free, so they complement PR-AUC rather than competing with it.
  - **Business-impact summary** (notebook §16.3): the expected-value calculation at the base-scenario threshold — retained-revenue vs. retention-call cost — emitted as a single headline figure for `README.md`. The cost inputs come from the Phase 6 threshold scenarios; this step turns the chosen operating point into a dollar number, it does not re-select the threshold. The EV is reported **relative to two policy baselines** — *treat all* (`DummyClassifier(strategy='constant', constant=1)`: call everyone — maximum recall, maximum spend) and *treat none* (`constant=0`: do nothing — zero spend, full churn loss). The model must beat both on expected value; these are the cost-curve endpoints, bracket the achievable value, and persuade stakeholders more directly than a statistical floor.
  - Writes `reports/metrics.json` (metrics + bootstrap CIs + decile lift table + business-impact figure); saves gains/lift charts to `reports/figures/`
  - Logs all metrics and the report to the MLflow run
- `src/telco_churn/models/register.py` — promotes `challenger` → `champion` if and only if it beats the current `champion` on both **PR-AUC** (ranking quality; threshold-free) and **Brier score** (calibration; lower is better); no promotion otherwise; logs the decision with structured event `model_promoted` or `model_rejected`. **The operating threshold is shipped as a separate versioned config artifact** alongside the model — *not* folded into the promotion comparison — so "is the new model better at ranking?" and "where do we cut?" stay independent, separately-auditable decisions. (Recall@threshold remains a *reported* metric; it does not gate promotion, because it inherits the fixed-threshold fragility discussed in `summary.md` §4.2.) **Model versioning is automatic:** MLflow auto-increments integer version numbers on every `log_model` call — there is nothing to manage manually. What `register.py` manages is the *alias layer*: flipping the `champion` alias to point at the new version number when the promotion gate passes, and leaving `challenger` on the new version otherwise. The FastAPI service (Phase 9) always loads whichever run holds the `champion` alias at startup — it is decoupled from version numbers entirely. The Phase 10 Prefect retrain flow automates this promotion on every weekly retrain cycle; the only case requiring manual intervention is an emergency rollback, where `champion` is re-pointed at an older version number via `mlflow.MlflowClient().set_registered_model_alias()`.
- `tests/unit/test_evaluate.py` — bootstrap CI math verified on a synthetic dataset with known population AUC
- `notebooks/05-error-analysis.ipynb` — SHAP global feature importance, SHAP local explanations for representative FN/FP cases, confusion matrix at each cost scenario threshold; plus three structured characterisation analyses run on the promoted champion:
  - **Fixed-recall profile** (recall ∈ {0.70, 0.80, 0.90}): precision and F1 at each target recall level, derived from the three cost-scenario thresholds (conservative ≈ 0.22 / base ≈ 0.30 / optimistic ≈ 0.38) logged by Phase 6 `threshold.py`. A threshold-planning tool — shows the business the precision trade-off at each campaign coverage level. Values are read from the Phase 6 MLflow artifacts, not hard-coded here.
  - **Per-segment robustness check**: champion PR-AUC within three EDA-anchored axes — `contract_type` (Cramér's V 0.41), `tenure_cohort` (Pearson r −0.48 on continuous tenure; reuses `TENURE_COHORT_EDGES` from `preprocessing.py`), `internetservice` (V 0.32) — reported as per-segment PR-AUC and lift over the segment base rate. Checks whether the aggregate PR-AUC is uniform or concentrated in one cohort. A concentrated-edge flag routes to SHAP interaction-value drill-down on the `fiber-optic × month-to-month` compound cohort (54.6 % churn, the highest-risk documented interaction). Low-support caveat: the two-year `contract_type` tier churns <3 %; read its CI with caution.
  - **Fairness parity check**: per-group PR-AUC across four protected / quasi-protected axes — `gender` (sex; V 0.0086), `seniorcitizen` (age proxy), `has_partner` (marital status), `dependents` (familial status). `gender` is included precisely because its near-zero bivariate signal does not preclude proxy discrimination through correlated features; a clean parity result is the wanted, documentable outcome. Input policy: all four remain model inputs (benefit-allocation context — not a credit/employment denial); hard-excluding a protected attribute is a deliberate normative override, never a statistical-selector outcome. Flag any material gap for sign-off in `ANALYSIS.md`.

> **Within-phase order: `evaluate` → error-analysis review → `register`.** Run `evaluate.py` first (sealed-test metrics + lift/gains + business impact), then review `05-error-analysis.ipynb`, then run `register.py`. The deliverables are listed artifact-first (both `src/` modules together) but execute in this sequence. For the **first champion promotion this is a human review gate, not just confirmation**: subgroup FN/FP breakdown and SHAP sanity can veto a model that passes the aggregate PR-AUC + Brier gate but fails a protected/high-value segment — aggregate metrics hide subgroup collapse. The automated metric-only gate in `register.py` is what the **Phase 10** weekly retrain uses (no human in the loop every week); the initial Phase 7 promotion additionally requires the error-analysis review to pass. This is why error analysis is sequenced *before* `register.py`, not after.

> **Ordering & test-set discipline:** the error analysis here is *confirmatory of the metric decision* (notebook §12 / §16.4 — SHAP + FN/FP profiling of the final model), distinct from the *generative* error-feature loop already baked into Phase 4. The sealed test set is touched exactly **once**, here — `evaluate.py` is the *only* module permitted to import the test split (the structural isolation set up in Phase 5). Under continuous retraining (Phase 10) do **not** re-use this same sealed test set for every challenger-vs-champion comparison — that erodes it; promote on a rolling/time-based holdout instead. This preserves the "test set touched once" invariant (Lifecycle & Framing Gaps, Group A). Turning the *generative* loop into reproducible production code (so the repo demonstrates the full lifecycle in code, not just the migration) is a **deliberately deferred v2** — see "What This Plan Deliberately Does Not Include" — sequenced after Phase 14 so the production spine ships first.

> **The evaluated model is not the deployed artifact (notebook §17.1 — full-data production refit):** the metrics produced here are honest estimates from a model fit, tuned, and calibrated on the development set (all via CV), with the test split held out. Before the model serves traffic it is refit on **100 % of the data** (development + test) using the frozen feature set and tuned hyperparameters — there is no metric to estimate at that point, so withholding data only discards signal. That **full-data production refit does not live in Phase 7** (it must not run before the sealed-test metrics are recorded, or the test set is no longer sealed). It is a step in the **Phase 10 Prefect `retrain` flow**, which refits on all available data after `evaluate` passes and *then* registers/promotes the champion. The Phase 7 `challenger`/`champion` aliases point at the development-set model for the evaluation record; Phase 10 re-points them at the full-data refit for serving.

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
- **Update `train.py` data loader:** `_load_processed` in `src/telco_churn/models/train.py` currently reads `datasets/processed/telco_churn_processed.csv` — a Phase 5 development scaffold. In Phase 8 update it to read the Parquet artifact produced by the `features` stage (`datasets/processed/telco_churn_features.parquet`, Parquet replaces CSV): change `pd.read_csv(path)` to `pd.read_parquet(path)` and update the filename. The `train` stage in `dvc.yaml` declares this file as a dep, so DVC re-runs training automatically whenever the features hash changes.

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
- `pipelines/retrain.py` — weekly Sunday 02:00 schedule; runs `ingest → validate → features → train → evaluate → refit-full → register`; promotes `challenger` to `champion` on PR-AUC AND Brier improvement (same gate as `register.py`). **`refit-full` (notebook §17.1):** after `evaluate` records the sealed-test metrics, refit the frozen-feature, tuned-hyperparameter pipeline on **100 % of the data** (development + test) — the deployed artifact uses all available signal, while the promotion gate is still decided on the held-out metrics from `evaluate`. This step runs **only here**, never in Phase 7, so the test set stays sealed at evaluation time. **Dummy-floor guardrail:** each cycle recomputes the prevalence / `DummyClassifier(strategy='prior')` floor on the current data and asserts the retrained model clears it by a margin — a near-dummy result signals a broken feature/label pipeline (stale join, dropped column), not a modelling regression, and blocks promotion with a structured alarm. The floor is recomputed each cycle, never hardcoded, because churn prevalence drifts.
- `pipelines/drift_check.py` — daily 06:00; pulls the last 24 hours of predictions from the API structured logs; runs an Evidently data drift report; alerts (Prefect notification) when PSI > 0.2 on any top-5 feature
- `pipelines/performance_check.py` — **realized-performance feedback loop (closes the ML loop — `input` drift is only a proxy for what we actually care about).** PSI and prediction-distribution drift detect that *inputs* moved; they do not measure whether the model is still *right*. For churn the ground truth does arrive — observed weeks/months later — so this flow joins **logged predictions → realised outcomes** once labels mature and computes **realised PR-AUC** (plus Brier and the decile-lift table) over the matured cohort, comparing it to the Phase 7 offline estimate. Mechanics: predictions are logged with `customerid` + `request_id` + timestamp (Phase 9); a `prediction_outcomes` table in Postgres records the eventual churn label per `customerid`; the flow runs on a **label-maturity cadence** (e.g. monthly, after the observation window closes), scoring only the cohort whose outcomes are now known. **Delayed-label discipline:** never score a cohort before its outcome window matures (premature labels bias the metric optimistic); the maturity lag is a configured constant. This realised number — not PSI — is the **authoritative health signal** and the primary, *performance-based* retrain trigger (see below).
  > **📌 Retraining is trigger-based, not only scheduled.** The weekly cron is the *floor*; a sustained realised-PR-AUC drop below a configured fraction of the Phase 7 baseline (or a sustained PSI breach) should *trigger* `retrain.py` off-cycle. Schedule-only retraining either wastes compute when nothing moved or reacts too slowly when something did. Connect `performance_check.py` / `drift_check.py` alerts to an event-triggered run of the retrain flow, with the weekly schedule as the backstop.
- `pipelines/batch_predict.py` (optional) — nightly scoring of all customers; writes results to a `predictions` table in Postgres

**Verification:** Trigger each flow from the Prefect UI; inject synthetic rows with shifted `MonthlyCharges`; confirm the drift alert fires within the next scheduled drift check run. Seed `prediction_outcomes` with a matured cohort of known labels and confirm `performance_check.py` computes a realised PR-AUC, compares it to the Phase 7 baseline, and (when forced below the configured fraction) fires the performance-based retrain trigger.

---

### Phase 11 — CI/CD with GitHub Actions *(2–3 days)*

**What this achieves:** Every pull request is gated — no broken code reaches `main`. Every merge to `main` automatically deploys to a staging environment where a smoke test runs before a human approves promotion to production. This is where Continuous Delivery (CD) lives: automated deployment to staging, manual gate to production. A weekly data-quality check catches upstream data problems before the next retrain.

**CI/CD flow (the two-stage deployment pipeline):**

```
PR opened
  → ci.yml (lint + type-check + unit tests + Docker build)
  → integration.yml (Postgres + API integration tests) — PRs to main only

Merge to main
  → cd.yml stage 1: build image → push to ECR → deploy to App Runner (staging)
  → smoke test: GET /health + POST /predict with a valid payload against staging URL
  → [manual approval gate — GitHub Environment protection rule on "production"]
  → cd.yml stage 2: deploy same image SHA to App Runner (production)
```

The key boundary: **staging is automatic, production requires a human.** The same image SHA (built once, tagged with the commit SHA) is deployed to both environments — no rebuild on promotion, so what was tested in staging is exactly what goes to production.

**Why staging matters for this project:**
- Confirms the Docker image and serving stack work end-to-end before real traffic sees it
- The smoke test catches serialisation failures, missing dependencies, or broken feature pipelines that unit tests cannot reach
- Establishes the `env` tag distinction on MLflow runs: `development` (local), `staging` (CI-triggered), `production` (post-approval) — making the MLflow experiment metadata meaningful rather than decorative
- Mirrors real team practice: ML engineers push to staging, a tech lead or SRE approves production

**Deliverables:**
- `.github/workflows/ci.yml` — on every push and PR: `uv sync --frozen`, pre-commit, mypy, `pytest --cov=src --cov-fail-under=80` (unit tests only; no Docker required), Docker build (no push)
- `.github/workflows/integration.yml` — on PRs to `main` only: `docker compose --profile infra up -d`, wait for Postgres healthcheck, `pytest tests/integration/ --run-integration`, `docker compose down`
- `.github/workflows/cd.yml` — two-stage on merge to `main`:
  - **Stage 1 (automatic):** build + tag image with commit SHA, push to ECR, deploy to App Runner staging service, run smoke test (`/health` returns 200; `/predict` with a sample payload returns a valid churn probability)
  - **Stage 2 (gated):** `environment: production` block triggers the GitHub Environment protection rule (manual approval required); on approval, deploy the same image SHA to App Runner production service
- `.github/workflows/data-quality.yml` — weekly cron: `dvc pull` from S3, run Pandera validation, alert on failure
- **GitHub Environments** — two environments configured in the repo settings:
  - `staging` — no protection rules; deployed automatically
  - `production` — required reviewer(s) set; deployment blocked until approval
- **Two App Runner services in AWS (Phase 12):** `telco-churn-api-staging` and `telco-churn-api-production` — same image, different environment variables (staging points at staging RDS/S3 prefixes)
- GitHub repository secrets scoped per environment: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_TRACKING_URI`, `STAGING_APP_RUNNER_ARN`, `PRODUCTION_APP_RUNNER_ARN`  # pragma: allowlist secret

**Verification:** Open a PR with a deliberate lint error → CI fails. Fix and merge → staging deploys automatically and smoke test passes → approval grants production deploy within ~10 minutes of approval.

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
  - **Realised performance** — realised PR-AUC / Brier over each matured cohort vs. the Phase 7 offline baseline (sourced from `performance_check.py`); the panel that distinguishes "inputs moved" from "the model is actually degrading." Updates on the label-maturity cadence, not in real time — annotate it as such so a flat line between updates is not misread as staleness.
- Alert rules: p95 latency > 500 ms; error rate > 1 %; PSI > 0.2 sustained over 24 hours; **realised PR-AUC below a configured fraction of the Phase 7 baseline for one matured cohort** (the performance-based retrain trigger — see Phase 10 `performance_check.py`)
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
| `ColumnTransformer` definition | `features/preprocessing.py` → `models/train.py` (Phase 5) — fitted per CV fold on development; production path is `tree_preprocessor` + LightGBM in an sklearn `Pipeline` (linear baselines use a separate `linear_preprocessor`, Step 1) |
| Optuna best hyperparameters | Default values in `configs/training/optuna.yaml` (still searchable; warm-start from these) |
| Cost-sensitive threshold logic (3 scenarios) | `models/threshold.py` |
| Bootstrap CI evaluation routine | `models/evaluate.py` |
| Lift / gains curves + decile lift table (§15, §16.2) | `models/evaluate.py` (Phase 7) → `reports/metrics.json` + `reports/figures/` |
| Business-impact / EV figure (§16.3) | `models/evaluate.py` (Phase 7) → README headline metric |
| Bias/variance + McNemar diagnostics (§10.6, §10.8, §11.1) | `notebooks/03-model-selection.ipynb` (Phase 5) — notebook-only, gates nothing |
| Full-data production refit (§17.1) | `pipelines/retrain.py` `refit-full` step (Phase 10); never in Phase 7 |
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
