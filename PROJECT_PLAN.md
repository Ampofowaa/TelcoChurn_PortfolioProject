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
│   ├── costs.yaml              # ARPU/LTV/retention/intervention per scenario (Phase 6)
│   ├── policy/threshold.yaml   # shipped operating point + costs_config_hash (written by Phase 6; no model stamp — t* is model-independent)
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
│   │   ├── validate.py         # orchestrates gates; structured logging; report writer
│   │   ├── split.py            # canonical dev/test partition, sealed before Phase 4a
│   │   └── eda.py              # statistical helpers backing notebooks/01-eda.ipynb (Phase 3)
│   ├── features/
│   │   ├── sql_features.py     # runs sql/features/*.sql via SQLAlchemy
│   │   ├── build.py            # column group exports (raw IBM columns + charge_per_service)
│   │   ├── schema.py           # CustomerFeaturesSchema / FeatureOutputSchema (Phase 4b); FeatureSchema dataclass (Phase 5)
│   │   ├── preprocessing.py    # shared ColumnTransformer builder (Phase 4a; reused by models/train/)
│   │   ├── generate.py         # error-driven discovery machinery: OOF profiler, gate, bootstrap CI (Phase 4a)
│   │   ├── select.py           # permutation-importance feature selection vs. a noise decoy (Phase 5)
│   │   └── accessor.py         # load_features() + sha256 content hash; owns path & format (Phase 6)
│   ├── models/
│   │   ├── train/               # Optuna + LightGBM + MLflow: candidates, comparison, feature_freeze, tuning, log_model, common (Phase 5)
│   │   ├── calibrate.py        # CalibratedClassifierCV
│   │   ├── threshold.py        # closed-form t* = c/(r*LTV) + argmax-EV agreement check + pre-seal dev-OOF screen (Phase 6, last step)
│   │   ├── diagnostics.py      # shared V1/V2/V2b slicing helpers: segment collapse, fairness gaps, per-group calibration (Phase 5–7)
│   │   ├── evaluate.py         # test-set metrics + bootstrap CIs + promotion_decision.json
│   │   ├── plots.py            # pure helpers: reliability bins, EV curves, r-sensitivity (Phase 6)
│   │   ├── gate.py             # pure promotion gate (ANALYSIS.md §0); called by evaluate, read by register
│   │   ├── economics.py        # pure campaign maths: EV, budget curve, break-even, sensitivity, tornado
│   │   ├── explain.py          # pure SHAP helpers: global importance, dependence (V3), signed cohort, local
│   │   ├── error_analysis.py   # error concentration/confidence, value-weighted errors → error_analysis.json
│   │   ├── drift_reference.py  # pure builder for the champion's drift baseline (Phase 7)
│   │   └── register.py         # MLflow Model Registry alias flip; acts on the persisted decision
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
│       ├── paths.py            # get_project_root() anchor for all src/ file I/O (Phase 0/4.1)
│       ├── mlflow.py           # tracking-URI resolution, registry/run lookups, experiment metadata (Phase 5+)
│       └── stats.py            # shared bootstrap-CI helpers (Phase 4a+)
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
│   ├── 03a-model-selection.ipynb     # Phase 5 — model-family selection: baseline comparison + paired-bootstrap decision
│   ├── 03b-feature-selection.ipynb  # Phase 5 — permutation-importance selection experiment
│   ├── 03c-hyperparameter-tuning.ipynb # Phase 5 — Optuna study (history, parallel-coords, importance) + model logging
│   ├── 04-calibration-and-threshold.ipynb  # Phase 6 — reliability diagrams + cost curves
│   └── 05-evaluation-and-error-analysis.ipynb  # Phase 7 — renders evaluation + error analysis; hosts the V3 human review and the reported dev-OOF diagnostics (V1/V2/V2b)
│
├── datasets/                   # tracked by DVC, not Git
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── split_manifest.parquet  # canonical customerid → {dev, test} partition
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
| 5 | Model training (LightGBM + Optuna + MLflow) | `models/train/` (candidates, comparison, feature_freeze, tuning, registration), `features/select.py`, `models/diagnostics.py`, `configs/training/`, `configs/tuning/`, 371 tests, 93.25% coverage | ✅ Done |
| 6 | Calibration + cost-sensitive threshold | `models/calibrate.py`, `models/threshold.py`, `models/plots.py`, `features/accessor.py`, `utils/mlflow.py`, `configs/{calibration,costs,policy,threshold}/`, `notebooks/04-calibration-and-threshold.ipynb`, 420 tests, 94.15% coverage | ✅ Done |
| 7 | Evaluation + error analysis + registry promotion | `models/evaluate.py`, `models/gate.py`, `models/economics.py`, `models/explain.py`, `models/error_analysis.py`, `models/drift_reference.py`, `models/register.py`, `notebooks/05-evaluation-and-error-analysis.ipynb` | ✅ Done |
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

> The original `notebooks/_archive/EDA-original.ipynb` remains frozen and is not retroactively split or edited. It is kept as a comparison notebook for the original exploratory pass, not a source of truth — `src/` and `ANALYSIS.md` are authoritative wherever later phases diverge from it.

**Verification:** Notebook executes end-to-end without errors (`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-eda.ipynb`).

---

### Canonical Data Split

**What this achieves:** Seals the dev/test partition as a versioned artifact *before* any
metric-driven decision runs against it — including Phase 4a's OOF-PR-AUC feature-adoption gate.
Without this, 4a's discovery loop would need its own ad-hoc split (or run over the full dataset,
test rows included), and Phase 5 would define a second, independent split — two divergent splits
in the tree, one of which would silently leak test rows into a metric decision.

**Deliverables:**
- `src/telco_churn/data/split.py` — `make_split(ids, labels, test_size, random_state)`, stratified
  by `churn`, depends only on `(customerid, churn)` — no engineered feature — so it can run
  immediately after Phase 2 validation; `write_split`/`load_split`; `dev_ids()`/`test_ids()`/
  `partition(df)` helpers. `__main__` CLI writes `datasets/processed/split_manifest.parquet`.
- `make split` — new Makefile target, between `validate` and `features`.

**Consumers:** Phase 4a discovery (`notebooks/02a-feature-discovery.ipynb`) reads the `dev`
partition via `partition()`; Phase 5 `train.py` reads `dev` the same way (`common.py` imports
`data.split.partition()` directly — no internal split of its own); Phase 7 `evaluate.py` is the
**sole** reader of `test`, via `test_ids()`/`partition()`.

**Verification:** `make split` writes the manifest; re-running with the same seed is
byte-identical; `dev`/`test` are disjoint and cover every `customerid` exactly once. See
`docs/canonical-split-refactor-tasks.md` for the full refactor trail.

---

### Phase 4a — Feature Discovery: Structured Feature Search *(2–3 days)*

**What this achieves:** Establishes which engineered features earn a place in the model through a narrated, audited discovery loop. Candidates originate from two sources: EDA-anchored domain hypotheses (e.g., tenure survival-curve segmentation, service-normalised pricing, the fiber × contract interaction) and OOF false-negative profiling that surfaces systematic blind spots the baseline cannot recover. Every candidate — regardless of origin — passes through a four-screen adoption gate: leakage pre-gate → redundancy screen → OOF PR-AUC + subgroup recall → importance vs. noise floor. The loop is human-in-the-loop: the analyst writes each candidate in the notebook; `generate.py` supplies the mechanical scaffolding. Starts from raw IBM columns only. At least one decoy must be introduced and rejected.

**Not a DVC stage:** run-once R&D, seeded (`random_state=42`). Commits its provenance and adopted-set list; the production pipeline ships the frozen result via Phase 4b `build.py`.

> **📌 Split timing — resolved.** The dev/test split is now sealed *before* any decision that
> optimises against a metric, including this phase's OOF-PR-AUC **adoption gate**: the Canonical
> Data Split step above establishes `datasets/processed/split_manifest.parquet` immediately after
> Phase 2 validation, and 4a discovery loads the `dev` partition only, via `data.split.partition()`.
> The adopted feature (`charge_per_service`) was re-confirmed on this leak-free dev-only rerun —
> unchanged from the earlier full-dataset run, as expected for a deterministic domain transform
> (`monthlycharges ÷ service_count`) with no fitted statistic to leak. The correct discipline —
> split-first — now holds structurally rather than as a documented trade-off: any *future*
> candidate that is a fitted/data-derived feature (target/frequency encoding, learned bin edges,
> anything with a trainable parameter) inherits the same seal automatically, since the split it
> would be evaluated against is already sealed before this notebook runs. Phase 5 `train.py` now
> imports this canonical manifest directly (`common.py`'s `data.split.partition()`) rather than
> deriving its own split. See `docs/canonical-split-refactor-tasks.md` for the full refactor trail.

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

**What this achieves:** A reproducible, experiment-tracked training pipeline that decides the model family, freezes the feature set, tunes hyperparameters, and logs the result as a run artifact. Five ordered steps — candidate comparison → decision → feature selection → Optuna tuning → model logging — each logged as its own MLflow run under experiment `telco-churn-training`. The phase deliberately stops short of the model *registry*: the artifact it produces is uncalibrated and therefore not a valid rollback target (`CLAUDE.md` § *Run artifacts vs. registry versions*); Phase 6 registers the calibrated pipeline. All decision logic lives in `src/`; the `03a`–`03c` notebooks import from it and render results, they do not compute them.

**Order of operations (do not reorder — each step freezes an input to the next):**
1. **Build candidates** — `DummyClassifier(strategy='prior')`, `LogisticRegressionCV` (linear preprocessor: OHE `drop='first'` + `StandardScaler`), and default-config LightGBM (tree preprocessor), scored on one shared `RepeatedStratifiedKFold(5×3)` over the development split so folds are paired across candidates. `class_weight='balanced'` handles the ~27% positive rate for both families; a `DummyClassifier` leakage canary is a hard assertion (ROC-AUC ≈ 0.5, PR-AUC ≈ prevalence), not a soft warning.
2. **Compare on PR-AUC** — a pre-registered paired-bootstrap decision rule (materiality Δ\*=0.005) decides LightGBM vs. LogReg (one-metric invariant: PR-AUC alone selects). Non-gating fixed-recall and per-segment fairness/robustness diagnostics are logged alongside but never decide the family.
3. **Generative diagnostic loop (2c/2d)** — human-in-the-loop bias/variance and OOF segment profiling on the development set only, between Steps 2 and 3; engineer a feature and re-enter at Step 2, or stop (≈3-round cap).
4. **Select features** — permutation-importance selection against a noise-decoy column (the same rule as Phase 4a's Screen 4) inside CV against default LightGBM; freezes the input space via a paired-bootstrap keep-vs-reduce test; a non-gating SHAP audit cross-checks the result.
5. **Tune LightGBM with Optuna** — 50 TPE trials on PR-AUC after the freeze; `n_estimators` resolved per fold by early stopping, not searched (median `best_iteration_` across folds, each trial's early-stopping validation slice carved from that fold's own *training* partition so it never leaks into the CV score); a 1-SE selection rule over raw argmax picks the most-regularized trial within noise of the best.
6. **Log the tuned pipeline to the run — do not register it** — the final fit refits the selected trial on all of development, with **`n_estimators` scaled from the early-stopped median to this fit's larger row count** (`n_estimators_final = median(best_iteration_) × n_final_fit / n_fold_fit`, since the per-fold median was measured on a smaller carved-out training partition than the final fit actually trains on; a two-count CV diagnostic — scoring the tuned spec at both the raw and the scaled count on the same folds — confirms the correction against this project's own data rather than trusting the formula alone, and is logged alongside the full derivation in `training_manifest.json`). The full `[preprocessor → model]` `Pipeline` is then logged as pyfunc with `pyfunc_predict_fn="predict_proba"`, a signature inferred from that same probability output, and a log→reload→`predict_proba` parity check, retrievable at `runs:/<run_id>/model`; `training_manifest.json` records the engineering audit trail (git SHA, DVC hash, hyperparameters, feature space/columns, CV PR-AUC, paired-Δ vs. LogReg, and the returned `ModelInfo`'s **`logged_model_id`** — MLflow 3's `LoggedModel` id, which does not auto-populate on the registry and must be persisted deliberately or Phase 7's evaluation logging has no `model_id` to attach sealed-test metrics to). Uncalibrated and un-thresholded — not serving-ready, therefore **not a registry version**. See `CLAUDE.md` § *Run artifacts vs. registry versions*: the registry holds only valid rollback targets, and an uncalibrated pipeline is a stage of construction, not a deployable model. Phase 6 `calibrate.py` performs the single registration of this training cycle, on the calibrated artifact, and points `challenger` at it.

**Deliverables:**

*Configs:* `configs/training/lightgbm.yaml` (searched ranges + fixed determinism/imbalance knobs), `configs/training/logreg.yaml`, `configs/training/selection.yaml`, `configs/tuning/optuna.yaml` (`n_trials`, sampler, pruner, `selection_rule`).

*Source:*
- `src/telco_churn/features/preprocessing.py` — adds `build_linear_preprocessor` alongside the existing tree-family `build_preprocessor`
- `src/telco_churn/features/schema.py` — adds a frozen `FeatureSchema` dataclass owning the binary/multi_cat/numeric column groups
- `src/telco_churn/data/split.py` — canonical dev/test split (sealed once, order-invariant), imported by every downstream consumer
- `src/telco_churn/models/train/` — one module per step (`candidates.py`, `comparison.py`, `feature_freeze.py`, `tuning.py`, `log_model.py`) plus shared `common.py` (fold-parallel CV scoring, default hyperparameters, dev-split loading, and tracking-URI/git/DVC resolution — `_resolve_tracking_uri` anchors a relative fallback to the project root and defaults a fresh clone or CI run with no `.env`/infra profile to `sqlite:///mlflow.db`, never the bare `mlruns` file store MLflow 3.14 has frozen and which raises on use); Optuna trials nested under a `tuning_study` parent run. Every step-level run (`model_comparison`, `feature_selection`, `tuning_study`), not only the per-candidate ones, logs its input dataset via `common.py::_log_dev_input`, resolving through `features/accessor.py`'s canonical path rather than a hardcoded one, so `mlflow.log_input`'s lineage is legible on every run. `log_model.py`'s `run_model_logging_step` logs the pipeline and manifest onto the `tuning_study` run but passes no `registered_model_name=` and sets no alias — those two lines move to Phase 6 `calibrate.py`
- `src/telco_churn/features/select.py` — the permutation-importance selector, wrapped in a `Pipeline` so it refits leak-free per fold
- `src/telco_churn/models/diagnostics.py` — pure helpers for the 2c/2d loop (`generalization_gap`, `learning_curve_points`, `segment_oof_errors`) and Step 2's fairness/robustness diagnostics
- `docker/mlflow/Dockerfile`, `sql/schema/000_create_mlflow_db.sql` — Postgres-backed MLflow tracking server

*Tests:* `tests/unit/test_train_*.py`, `test_select.py`, `test_diagnostics.py`, `test_split.py` — leakage-canary assertion, run-twice determinism, leak-free-refit proofs, synthetic planted-signal/noise fixtures; `tests/integration/test_train_subprocess.py` — full CLI composition path via Hydra fast-path config overrides.

*Notebooks:* `03a-model-selection.ipynb` (candidate comparison, bias/variance loop, fairness/robustness panel), `03b-feature-selection.ipynb` (selection experiment), `03c-hyperparameter-tuning.ipynb` (Optuna study, full→reduced→tuned progression, model-logging confirmation).

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and logs a tuned pipeline at `runs:/<run_id>/model` with the reload-parity assertion passing and `training_manifest.json` attached; `telco-churn-pipeline` has **no new registry version** at the end of this phase (the registry stays empty until Phase 6 registers the calibrated artifact). LightGBM clears the Dummy floor and beats/ties LogReg under the decision rule; full `pytest` green. See `ANALYSIS.md` §4 for the recorded results and rationale, `CHANGELOG.md` `[0.5.0]`–`[0.5.3]` for delivery history, and `[0.6.2]`–`[0.6.3]` for the tracking-URI, dataset-lineage, and tree-count-scaling fixes applied to this phase's code after Phase 6 shipped.

---

### Phase 6 — Calibration + Cost-Sensitive Threshold *(2 days)*

**What this achieves:** The model outputs calibrated probabilities instead of raw, imbalance-reweighted scores, and the decision threshold reflects the actual business cost of a missed churner versus a wasted retention call — not the default 0.5. This phase performs the training cycle's single MLflow registration; three cost scenarios are shipped so the business can choose its own risk posture without re-deriving anything. **Its last step is a pre-seal screen, not a footnote.** Before the sealed test set is ever touched, this phase re-runs `ANALYSIS.md` §0's V1/V2/V2b dev-OOF diagnostics and the aggregate calibration-slope band check against the model's own dev-OOF probabilities, and hard-stops the cycle if that screen fails. A check that can only reject, never approve, costs nothing to run here and everything to skip — this is the one point in the pipeline a badly-calibrated model can be caught without spending the one-time sealed-test evaluation (Phase 7) on it.

**Order of operations (do not reorder — each step depends on the last):**
1. **Calibrate** — `CalibratedClassifierCV(ensemble=False)` wraps the **unfitted** `[preprocessor → LightGBM]` `Pipeline`, resolved via `sklearn.clone(mlflow.sklearn.load_model(manifest["logged_model_uri"]))` — never `runs:/<run_id>/model`, which becomes ambiguous once this module logs a second model onto the same run. `ensemble=False` cross-fits on the development set (an explicit, seeded outer/inner `StratifiedKFold(5)`) with no static val holdout, and collapses `calibrated_classifiers_` to one entry whose `.estimator` is the Pipeline Phase 7's SHAP step needs. The resulting dev-OOF probability vector — the numbers that select the calibration method, produce BSS, validate `t*`, and (below) power the dev-OOF screen — is logged as a run artifact (`calibration/dev_oof_predictions.parquet`: `customer_id`, `y_true`, `p_hat`), not only summarized into JSON and discarded; the threshold step, Phase 7's drift reference, and `error_analysis.py` all read this vector directly rather than recomputing it (`CLAUDE.md` § *Persist the evidence, not just the conclusion*). A `calibration_spec` block (`method`, `inner_cv_folds`, `random_state`, `ensemble`) is written alongside in `calibration_summary.json`, so the calibration method and its exact configuration are frozen and legible without re-deriving them from `configs/` later.
2. **Select the calibration method** — sigmoid is the incumbent; isotonic must earn the switch. A hard PR-AUC-preservation gate runs first (`training_setup.delta_threshold`, per-fold mean AP, never pooled — pooling would dilute isotonic's ranking damage across five distinct calibration maps). Only a method that clears the gate proceeds to a paired-bootstrap CI on per-fold Brier (block-bootstrapped by outer fold, reusing `models/train/comparison.py`'s paired-bootstrap idiom). Both methods and their Brier/ECE/PR-AUC are logged regardless of which wins, so the loser's numbers make the winner's selection legible. The selected method's dev-OOF **calibration slope** (`calibrate.py::calibration_slope` — regress `y` on `logit(p)`, Cox calibration, with a bootstrap CI) is reported in `calibration_summary.json` beside Brier/BSS/ECE: it is a calibration diagnostic computed where the calibration decision is made, and it is what step 7 below screens against the band, and what §0's promotion guardrail later consumes on the sealed-test side at Phase 7.
3. **Register** — the training cycle's only registration point (`CLAUDE.md` § *Run artifacts vs. registry versions*): Phase 5 logs an uncalibrated pipeline and stops; this step calls `mlflow.sklearn.log_model(..., name="calibrated_model", registered_model_name=...)` onto that same run and points `challenger` at it. `name="calibrated_model"` (not `"model"`) is load-bearing — reusing the name would rebind `runs:/<run_id>/model` away from Phase 5's pipeline. Blocked outright (`RuntimeError`) if `training_manifest.json`'s `tuning_summary.trial_count_below_threshold` is true, overridable only by an explicit config flag (`calibration.override_trial_count_gate`) — a data-quality gate on the tuning result, not a performance comparison. The registered version also carries a **`logged_model_id`** tag, set from the returned `ModelInfo.model_id` — since `ModelVersion.model_id` does not auto-populate in OSS MLflow, this tag is the hop Phase 6's own dev-OOF screen and Phase 7's evaluation logging both resolve through to attach metrics to the right `LoggedModel`. The version's training-scope tag is named **`training_data_scope`** (not `refit_scope` — there is no full-data-refit module in this plan to name it after; that concept has been dropped in favour of a Phase 10 recalibration flow, not yet designed).
   - **Also freezes the golden-parity fixture, at this exact moment.** `select_golden_rows(X_dev, customer_ids, n_rows)` picks the `cfg.calibration.golden_n_rows` (5) dev rows with the **lowest `customerid`** — never a positional `.head()`, so the fixture is stable against an unrelated reordering of the processed feature table upstream — and scores them with the **in-memory** fitted pipeline before `log_model`/pickling touches it at all. Logged as `calibration/golden_predictions.json`: the pinned `customerid`s, the rows themselves (at the model's committed input schema), the reference `p_hat` scores, and a `"purpose": "serving-parity fixture — reproducibility only; scores are in-sample and are not performance evidence"` key (every golden row is a development-partition row this model trained on). This is what makes Phase 7 `register.py`'s serving-parity smoke check a genuine independent round trip rather than a tautology — the reference must come from a different process and a different code path than the assertion that later checks it, and it is captured here, before this run's own model has ever been reloaded from disk.
   - **Tags the newly minted version `promotion_status: pending` at this exact moment** — before `threshold.py`'s dev-OOF screen, `evaluate.py`, `error_analysis.py`, or `register.py` can fail — so a crash anywhere downstream leaves the version in the one state every abort path is guaranteed to have started from, rather than depending on some later step to remember to write it (`CLAUDE.md` § MLflow Model Registry: the fail-safe default must be the one you get for free).
4. **Derive the threshold** — closed form `t* = c / (r × LTV)`, from `q·r·LTV − c > 0` (contact iff expected value exceeds doing nothing). This supersedes the classical `C_FP/(C_FP + C_FN)` cost-matrix rule, which implicitly treats correct decisions as free; here every contact costs `c` regardless of outcome. `c`, `LTV`, and `r` are resolved per scenario from `configs/costs.yaml` (ARPU from churner `MonthlyCharges` quantiles, development set only; `r` — the one parameter this dataset cannot supply — taken from an industry benchmark range, not fit).
5. **Validate, don't select, against the data** — an empirical argmax-EV check (`cross_val_predict` over the calibrated pipeline) confirms `t*` falls inside a 1,000-resample bootstrap CI of where realized expected value actually peaks. `t*` itself has zero sampling variance (a closed-form function of cost parameters alone), so nothing here re-derives it — the CI describes the noisy empirical estimate, not the threshold. A retention-rate sensitivity sweep (holding cost/LTV fixed) is logged as a headline result, not a footnote: `t*` is inversely proportional to `r`, the model's single largest source of uncertainty.
6. **Ship — and split the output along the line the closed form already draws.** Every scalar this phase computes — `dev_brier`, `dev_bss`, `dev_ece`, `dev_per_fold_mean_ap`, `dev_calibration_slope` (step 2), and per scenario `t_star_{scenario}`, `implied_contact_rate_{scenario}`, `dev_ev_at_t_star_{scenario}` — is logged as an MLflow **metric**, not only as a field inside a JSON blob; the EV curve itself is persisted as `threshold/ev_curve.parquet` rather than only rendered. JSON remains the audit record. `threshold.py::expected_value_at_threshold(proba, y, scenario, t)` is the one place `p·r·LTV − c` is computed as a pure function; Phase 7's `economics.py` imports it rather than redefining it. `threshold/threshold.json` (all three scenarios' full diagnostic bundles) is logged to the run, alongside `threshold/figures/`. What it mirrors out to disk splits in two, because the file mixes two kinds of content with two different lifetimes:
   - **`configs/policy/threshold.yaml` — the policy, and *nothing model-dependent*.** All of it is a pure function of `configs/costs.yaml`, because `t* = c/(r × LTV)` **is model-independent by construction**. Provenance here is pinned by a **`costs_config_hash`**, not by `model_run_id`/`model_version` — same mechanism `CLAUDE.md` already mandates for the economics metrics.
   - **`threshold/threshold_validation.json` — the model-dependent half, logged as an artifact on the model's run.** `model_run_id`, `model_version`, `calibration_method`, `argmax_ev_threshold`, `argmax_ev_bootstrap_ci`, `within_ci`, `implied_contact_rate`. **Same rule as `drift_reference.json`: an artifact describing a specific version travels with that version, never sits at a fixed path.**
   - **⚠ Why the split, and not a restamp.** The model stamp is honest *today*. It goes wrong once a future retrain promotes a new challenger over the incumbent: the stamp does not change, the champion does, and `implied_contact_rate` silently becomes the *wrong model's* contact rate. Rewriting the stamp on promotion would be worse — it would convert a *visibly stale* stamp into a *silent misattribution*. Splitting removes the stamp from the file that has no business carrying one.
   - **Retires a third copy of the calibration method** — `calibration_method` now lives only in `calibration_summary.json` (source of truth) and `threshold_validation.json` (a record of what was calibrated), not a third time in `threshold.yaml`.
7. **Screen — the pre-seal dev-OOF check, and this module's own last step.** Immediately after deriving and shipping the threshold above, `run_threshold_step` re-uses the exact aligned `(customerid, y_true, p_hat)` vector it already built for the derivation — never a second fetch of `dev_oof_predictions.parquet` — and joins it to the `dev` partition's raw segment/protected columns (`build_dev_oof_screen_frame`, reading the full feature table via `features/accessor.py::load_features()` + `data.split.partition()`, restricted to `dev` only: this is the one place in `threshold.py` that imports `telco_churn.data.split`, and it never touches `test`).
   - **The aggregate calibration-slope screen.** `calibrate.py`'s already-logged, already-selected slope (from `calibration_summary.json`, never recomputed) is checked against §0's `[0.80, 1.25]` band via `gate.py::slope_passes` — promoted from a private `_slope_passes` to a public function precisely so this module reuses the identical check `gate.py`'s own cold-start/comparative decisions apply, rather than reimplementing the band logic a second time. This is a **screen, not the gate** — measured at n = 5,634 (higher-powered than the eventual sealed-test slope's n = 1,409, but mildly optimistic, since Phase 6 selected the calibration method on these same probabilities), so it can **reject early, never approve**. **On failure, `run_threshold_step` raises `RuntimeError` after logging** — a hard stop: the model never reaches `evaluate.py`, the sealed test set stays unspent, and the version sits at `promotion_status: pending` (never `rejected` — no evaluation verdict was ever reached) until it is re-calibrated and the cycle re-run.
   - **V1/V2/V2b — computed here, not in Phase 7.** `compute_dev_oof_diagnostics` calls `diagnostics.py`'s shared slicing helpers — `sliced_ranking_metrics`/`flag_segment_collapse` (V1, segment collapse), `sliced_decision_rates` + `equal_opportunity_difference_by_axis`/`demographic_parity_difference_by_axis` (V2, fairness disparity at the decision), `sliced_calibration`/`flag_calibration_collapse` (V2b, per-group calibration collapse) — on this dev-OOF frame. These are the same functions Phase 7's `evaluate.py` calls on its own sealed-test frame for its reporting-only slices, so the two surfaces (dev-OOF, screened; sealed-test, reported) are never two implementations of one idea. Reported, never gating (`ANALYSIS.md` §0): a flagged segment or fairness gap changes nothing about the promotion decision — it only feeds the model card and ongoing monitoring.
   - **Writes and logs `reports/dev_oof_predictions.parquet` + `reports/dev_oof_diagnostics.json`** (the latter mirrored under the run's `threshold/` artifact folder), plus `dev_oof_calibration_slope` (+ CI bounds) as metrics against the model's `model_id` and a named `dev_oof_screen` MLflow dataset, and a `dev_oof_screen_result: pass | fail` tag. Every later reader of this diagnostic — Phase 7's `evaluate.py::load_dev_oof_diagnostics`, `error_analysis.py` — fetches it by explicit `run_id` from this run, never from the local `reports/` mirror, which reflects whichever run last executed `threshold.py` on this machine, not necessarily the run being asked about.

**Deliverables:**

*Configs:* `configs/calibration/default.yaml` (`method: sigmoid | isotonic | auto`, fold counts, ECE binning, **`golden_n_rows`** — rows in the golden-parity fixture, 5 by default), `configs/costs.yaml` (the three scenarios as data — conservative/base/optimistic), `configs/threshold/default.yaml` (`random_state` for the argmax-EV bootstrap; **`n_bootstrap`** — a separate 1,000-resample knob for the dev-OOF screen's V1/V2/V2b slicing, distinct from `costs.yaml`'s `argmax_ev_bootstrap_n_samples` since it resamples segment-sliced ranking/calibration metrics, not the argmax-EV curve), `configs/policy/threshold.yaml` (generated output, not hand-authored).

*Source:*
- `src/telco_churn/models/calibrate.py` — method selection, the single registration, `select_golden_rows`, per the steps above.
- `src/telco_churn/models/threshold.py` — closed-form `t*`, the argmax-EV agreement check, the `r`-sensitivity sweep, and — its own last step — the pre-seal dev-OOF screen (`build_dev_oof_screen_frame`, `compute_dev_oof_diagnostics`). No longer leak-free by the *absence* of a `data.split` import (step 7 needs the `dev` partition's segment columns); it remains leak-free by construction in the sense that matters — no `.fit(` call anywhere, and the one `data.split.partition()` call is restricted to `dev`, never `test`. Also now owns `load_policy_thresholds`/`resolve_policy_scenarios`/`resolve_policy_thresholds_by_scenario` (moved from `evaluate.py`, which imports them back) — this module is the one that writes `configs/policy/threshold.yaml` in the first place, so reading it back is now writer/reader-local.
- `src/telco_churn/models/diagnostics.py` — gains the shared V1/V2/V2b surface this step and Phase 7's `evaluate.py` both call: public `ROBUSTNESS_AXES`/`FAIRNESS_AXES` constants and `build_segment_lookup` (promoted out of what was a private, evaluate.py-local helper), plus `sliced_ranking_metrics`/`flag_segment_collapse`, `sliced_decision_rates`/`equal_opportunity_difference_by_axis`/`demographic_parity_difference_by_axis`, and `sliced_calibration`/`flag_calibration_collapse`. Full description under Phase 7's `diagnostics.py` bullet, since `evaluate.py`'s sealed-test reporting slices are this surface's other caller.
- `src/telco_churn/models/gate.py` — `slope_passes` (renamed from private `_slope_passes`) is now public specifically so this phase's dev-OOF screen can reuse the exact band check `gate.py`'s own cold-start/comparative decisions apply, rather than a second copy of the same tolerance-region logic.
- `src/telco_churn/models/plots.py` — pure plotting-data helpers (reliability-diagram bins); no rendering, no matplotlib import. Cost/EV curves and the sensitivity sweep stay in `threshold.py` itself, since it needs them internally.
- `src/telco_churn/features/accessor.py` — `load_features()`, the single accessor owning the processed-features path, format, and `sha256` content hash (anticipates Phase 8's CSV→Parquet swap); consumed by `calibrate.py`, `evaluate.py`, and now `threshold.py`'s dev-OOF screen.
- `src/telco_churn/utils/mlflow.py` — `resolve_tracking_uri()` (as before), plus **`resolve_model_run_id`, `resolve_logged_model_id`, `load_model_promotion_bars`, and `ensure_experiment_metadata`**, all now shared here rather than living in `evaluate.py` (their original, single-caller home): `threshold.py`'s dev-OOF screen needs the same registry lookups `evaluate.py` needs, and importing them from `evaluate.py` would be circular, since `evaluate.py` already imports `CostScenario`/`costs_config_hash`/`load_costs_config` back from `threshold.py`. `ensure_experiment_metadata(cfg)` replaces the bare `mlflow.set_experiment(...)` call across every training-cycle module, setting an experiment-level description and tags once so the MLflow UI is self-documenting regardless of which module runs first.
- `src/telco_churn/utils/stats.py::paired_bootstrap_ci` — extracted from `models/train/comparison.py`, reused by `threshold.py`'s argmax-EV agreement check.

*Tests:* `tests/unit/test_calibrate.py`, `test_threshold.py`, `test_mlflow.py`, `test_accessor.py`; `tests/integration/test_calibrate_subprocess.py`, `test_threshold_subprocess.py` — leak canary (preprocessor refit count under `ensemble=False`), run-twice determinism, an AST/grep-style scan proving the module's leak-free claims structurally rather than by convention (`test_threshold_dev_oof_screen_has_no_refit_or_reslope_machinery` — the dev-OOF screen never re-fits the calibrated pipeline or re-derives a slope `calibrate.py` already computed), an inherited-contamination canary (the argmax-EV check passes on OOF probabilities and fails on in-sample ones), and hermetic dev-features fixtures. For the dev-OOF screen specifically: `test_run_threshold_step_screens_dev_oof_slope_and_writes_reports` (asserts `reports/dev_oof_predictions.parquet` + `reports/dev_oof_diagnostics.json` are written and match what's logged to MLflow) and `test_run_threshold_step_raises_on_bad_slope_read_from_calibration_summary` (a planted out-of-band slope raises `RuntimeError` and leaves `promotion_status` at `pending` — never `rejected`, since no evaluation verdict was ever reached). The integration suite covers both CLIs' exit-0 and exit-1 paths (missing/invalid `configs/costs.yaml`, `r = 0`, `t* ≥ 1`) plus `test_threshold_main_cli_ships_policy_regardless_of_dev_oof_screen_verdict` (the policy YAML and MLflow artifacts are written even on a screen failure — the audit trail records the failing attempt rather than it vanishing) and `test_threshold_main_cli_exits_one_when_model_version_does_not_exist`.

*Notebook:* `notebooks/04-calibration-and-threshold.ipynb` — reliability diagrams before/after calibration, method-selection diagnostics, per-scenario cost breakdowns, threshold-by-scenario and expected-value curves, the retention-rate sensitivity plot, and the dev-OOF V1/V2/V2b screen panels (segment PR-AUC floors, per-group decision rates, per-group calibration slopes) — rendered from `reports/dev_oof_diagnostics.json`, never recomputed.

**Verification:** `uv run python -m telco_churn.models.calibrate calibration.run_id=<id>` registers version 1 (`challenger`, `promotion_status: pending`) and logs `calibration/golden_predictions.json` onto that same run. `uv run python -m telco_churn.models.threshold threshold.model_version=1` writes `threshold/threshold.json` + `configs/policy/threshold.yaml`, with the base scenario's `t*` falling inside its own empirical argmax-EV bootstrap CI, **then runs the dev-OOF screen as its own last step**: confirm the dev-OOF slope sits inside §0's [0.80, 1.25] band (settling `ANALYSIS.md` §9 #4), and that `reports/dev_oof_predictions.parquet` + `reports/dev_oof_diagnostics.json` are written with V1/V2/V2b reported (never gating) alongside them. This is a screen, not the gate — the gate is the *sealed-test* slope, one of Phase 7's four criteria of record; the screen is higher-powered (n = 5,634 vs. 1,409) though mildly optimistic, so it can reject early but never approve. A champion outside the band on the surface biased in its favour must be re-calibrated — far better discovered here, with the seal intact, than after Phase 7 spends the one-time sealed-test evaluation on it. Full `pytest` green. See `ANALYSIS.md` §5–§6 for the recorded numbers and rationale (sigmoid selected via `isotonic_disqualified_pr_auc_gate`, BSS 0.31, `t* = 0.3941`, and §6's dev-OOF screen results: 0 flagged on V1/V2b, V2 flagging `seniorcitizen`/`has_partner`/`dependents` against pre-existing churn-rate gaps in the population rather than proxy discrimination) and `CHANGELOG.md` `[0.6.0]`–`[0.6.3]` for delivery history, including a QA pass that hardened test hermeticity and the `calibration.method='auto'` decision path, the later MLflow-lineage/evidence-persistence/tree-count-scaling fixes, and the dev-OOF screen's relocation from a standalone Phase 7 module into this phase's last step.

---

### Phase 7 — Evaluation + Error Analysis + Registry Promotion *(2–3 days)*

**What this achieves:** A sealed test-set evaluation (touched here, once) produces bootstrap-CI-bounded metrics that honestly estimate production performance. A structured promotion gate — specified in `ANALYSIS.md` §0, not restated here — decides on those metrics for the first time, in its cold-start regime (no incumbent `champion`, so the absolute bars apply). A pass flips `champion` directly onto the registered version the metrics were recorded against — nothing is refit.

**This phase begins where Phase 6 leaves off, not from a clean slate.** `threshold.py`'s pre-seal dev-OOF screen (Phase 6's last step) has already run and passed — there is no standalone `calibration_screen.py` module here (an earlier draft of this plan put the screen in this phase; it moved to Phase 6, since its entire value is rejecting a model *before* the seal breaks). `evaluate.py` reads Phase 6's V1/V2/V2b diagnostics rather than recomputing them.

**Deliverables:**

*Configs:* `configs/model_promotion.yaml` — the five pre-registered `GateBars` (see `ANALYSIS.md` §0), loaded by path, never through Hydra composition. `configs/register/default.yaml` — `model_version` (required, never inferred from the `challenger` alias), `require_review` (`true` here; Phase 10's weekly retrain sets it `false`), `alias` (parameterized so Phase 10's shadow/canary staging reuses `promote_to_alias` unchanged), `golden_atol`, `environment_packages`, `drift_reference_n_bins`.

*Source:*
- `src/telco_churn/models/gate.py` — `decide_promotion(candidate, incumbent, cfg) -> Decision`, a pure function implementing §0 (no I/O, no MLflow). `evaluate.py` calls it once; `register.py` only re-reads its persisted verdict — two independent evaluations of one rule is how it becomes two rules. `GateInputs` carries interval-valued (Δ + CI) statistics, not bare scalars, since the comparative regime is a paired-bootstrap rule: a `(challenger_metric, champion_metric)` signature would silently collapse into `if new > old` and must never be "simplified" to one. `record_review(...)` stamps the human verdict onto the decision the notebook re-saves.
- `src/telco_churn/models/evaluate.py` — the only module permitted to touch the sealed test set (`CLAUDE.md`'s modelling invariant); resolves the model by explicit version, never alias. One scoring pass computes: sealed-test PR-AUC/ROC-AUC against the Dummy floor; a purpose-built classification summary (not `sklearn.classification_report`, whose negative-class/weighted averages mislead at ~27% prevalence) at all three shipped thresholds with confusion counts; the fixed-recall profile; calibration (BSS, ECE, Murphy's decomposition, and the sealed-test **calibration slope — the one `gate.py` reads**, distinct from Phase 6's dev-OOF screen slope); decile lift/gains; business-impact EV per scenario with bootstrap CIs *and* the three-scenario parameter-uncertainty bracket kept as a separate, un-conflated error bar (`ANALYSIS.md` §0); sensitivity sweeps and a tornado on the base scenario; and sliced fairness/robustness test metrics, reported only — V1/V2/V2b were already screened in Phase 6 and are read here (`load_dev_oof_diagnostics`), never recomputed. Writes `test_predictions.parquet`, `metrics.json`, `economics.json`, `promotion_decision.json` (`review: pending`). MLflow: its own `evaluation` run, metrics logged against `model_id` + a named `sealed_test` dataset (MLflow 3's native linkage), cost params as run params tagged with `costs_config_hash`, and the four gate criteria as model-version tags.
- `src/telco_churn/models/drift_reference.py` — pure builder, `build_reference(features_df, oos_proba, y, n_bins=10) -> dict` (per-feature bin edges/frequencies, reference score distribution, reference prevalence). `register.py` calls it with the **out-of-sample union of dev-OOF and sealed-test predictions** — never the champion's in-sample scores, which would bake in a permanent phantom drift signal.
- `src/telco_churn/models/register.py` — reads the gate's verdict; recomputes nothing; mints no new version (`CLAUDE.md` § *Run artifacts vs. registry versions*). Idempotency guard first, then: gate/review checks → manifest + `error_analysis.json` presence gates → the two-phase serving smoke check (environment-parity, then schema/output-range/golden-parity **by explicit version, before the alias moves** — only on a pass does it flip `champion` and re-confirm parity **through** the alias) → `drift_reference.json` + `model_card.json` logged onto the promoted run → tag `promoted`, log `model_promoted`. `rollback_champion()` (highest version tagged `promoted`, never `N−1`) is the single implementation both the post-flip failure path and any manual emergency rollback share. `model_card.json` is assembled strictly from artifacts already on record (`training_manifest.json`, `metrics.json`, `economics.json`, `error_analysis.json`) — baselined against `treat-none`/`treat-all` and the prevalence floor, never LogReg (a Phase 5 family-comparison candidate, not a champion rival) — never hand-transcribed.
- `src/telco_churn/models/diagnostics.py` — gains the shared V1/V2/V2b slicing surface (`build_segment_lookup`, `sliced_ranking_metrics`/`flag_segment_collapse`, `sliced_decision_rates` + the equal-opportunity/demographic-parity differences, `sliced_calibration`/`flag_calibration_collapse`), called by both Phase 6's dev-OOF screen (screened) and this phase's sealed-test slices (reported only — dev-OOF carries ~4× the churners per slice, so screening happens there, not here).
- `src/telco_churn/models/economics.py` — pure business-economics helpers (no I/O, no sklearn): `expected_value`, `campaign_cost`, `ev_by_k`, `break_even_retention_rate`, the one-/two-way sensitivity sweeps, `tornado`. Reuses Phase 6's `CostScenario`/EV formula rather than a second definition of `p·r·LTV − c`.
- `src/telco_churn/models/explain.py` — pure SHAP helpers (no I/O): `global_importance` (model documentation, not error analysis), `dependence_points` (V3's instrument — the only thing that shows *direction*, which mean |SHAP| cannot), `cohort_shap` (signed, not absolute), `binary_feature_effects`, `local_explanations` (illustrative only, never evidence).
- `src/telco_churn/models/error_analysis.py` — runs after `evaluate.py`; owns `reports/error_analysis.json`, which `register.py` aborts without. Error-concentration scan (all features, not just the seven pre-registered axes) on dev-OOF only — confirmatory, never generative, and never on the 1,409-row sealed test. Error-confidence (near-miss vs. confident-failure) and value-weighted (`MonthlyCharges` decile) profiles on the sealed test. Cohort-level signed SHAP beyond a single FN-vs-TP mean. **V3 — direction sanity** — the only veto criterion this module owns. Its own `error_analysis` MLflow run, persisting `shap_values.parquet` (not only the rendered charts) and tagging the model version `error_analysis_run_id`.

*Tests:* `tests/unit/test_evaluate.py`, `test_gate.py`, `test_diagnostics.py`, `test_stats.py`, `test_economics.py`, `test_explain.py`, `test_drift_reference.py`, `test_register.py` (17 tests), `test_error_analysis.py` + `tests/integration/test_error_analysis_subprocess.py` — the load-bearing cases: a challenger that ranks better but calibrates worse improves its Brier yet is vetoed by the calibration-slope guardrail; `sliced_decision_rates` catches a fairness gap that identical per-group PR-AUC hides; `ev_by_k`'s peak lands exactly at `t* = c/(r×LTV)`, a free cross-check of the whole cost model; `register.py`'s emergency-rollback fixture (a legitimate `promoted` champion at version M vs. a later, higher-numbered `rejected` challenger) proves `rollback_champion()` resolves by tag, never by version arithmetic; and a perturbed `golden_predictions.json` makes registration abort, proving the round trip is genuine rather than circular. Structural guards: `grep -r "data.split" src/telco_churn/models/error_analysis.py src/telco_churn/models/explain.py notebooks/05-evaluation-and-error-analysis.ipynb` returns nothing, and the notebook contains no `shap.` call and no model `.fit(`/`.predict`.

*Notebook:* `notebooks/05-evaluation-and-error-analysis.ipynb` — a renderer and the human-review interface; computes nothing. Renders `metrics.json` and `error_analysis.json`, including Phase 6's dev-OOF V1/V2/V2b panels (not recomputed). Its closing cell calls `gate.py::record_review(...)`, stamping `review: approved | rejected` onto `promotion_decision.json` — veto-only: it may reject the model, never iterate on it (seeing the test set's FNs and responding by engineering a feature would retroactively contaminate the seal). Phase 10's unattended weekly retrain runs `error_analysis.py` and skips this notebook (`register.require_review=false`) — only the review is human, and only it is skippable.

**Within-phase order: `evaluate` → `error_analysis` → human review → `register`**, enforced structurally, not by convention: `evaluate.py` persists the gate's verdict to `promotion_decision.json`; the notebook stamps the review onto that same file; `register.py` aborts unless it reads both a pass and an approval, and never recomputes either. For this first promotion the review is a genuine gate, not just confirmation — subgroup FN/FP breakdown and SHAP sanity can veto a model the automated gate already admitted.

**Verification:** `uv run python -m telco_churn.models.evaluate evaluate.model_version=1` produces `metrics.json`/`test_predictions.parquet`/`promotion_decision.json` (`review: pending`), with sealed-test PR-AUC/ROC-AUC CIs overlapping `README.md` and BSS/ECE comparable to Phase 6's dev-set figures. `uv run python -m telco_churn.models.register register.model_version=1` exits 1 until the notebook records an approval, then runs the two-phase serving check (golden parity by version, then through the alias) before flipping `champion`, and logs `drift_reference.json`/`model_card.json` onto the promoted run — confirm both are retrievable via the alias. After registration, `telco-churn-pipeline` holds one version tagged `promotion_status: promoted`, carrying `champion` and its sealed-test metrics as `test_*` tags. Full `pytest` green. See `ANALYSIS.md` §6–§8 for the recorded verdict and `CHANGELOG.md` `[0.7.0]`–`[0.7.3]` for delivery history.

---

### Phase 8 — DVC Pipeline Wrap *(1 day)*

**What this achieves:** The five-stage pipeline (ingest → validate → features → train → evaluate) becomes a content-hashed DAG. Changing a hyperparameter re-runs only the training and evaluation stages — not the full pipeline. This is the reproducibility guarantee that separates a real MLOps workflow from ad-hoc notebooks.

**Deliverables:**
- `dvc init` — **the raw CSV stays git-committed as a stage `dep`, not `dvc add`-ed.** `configs/config.yaml` reads `datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (the authentic Kaggle/IBM filename); it is 977 KB, static, and read-only, so git owns it and DVC only watches it — see the raw-data note below. (`dvc add` would make DVC *own* the file, contradicting that note.)
- `dvc.yaml` — six stages with explicit `deps` (code + configs) and `outs` (data artifacts, models, metrics):

  | Stage | Deps | Outs |
  |---|---|---|
  | `ingest` | `data/ingest.py`, `datasets/raw/*.csv` (git-tracked), `sql/schema/` | `reports/ingest_receipt.json` (row count, per-column null counts, frame checksum) |
  | `validate` | `data/validate.py`, `data/schema.py` | `reports/validation/` |
  | `split` | `data/split.py`, `reports/validation/` (validate output — split blocks on a passing validation run), `configs/config.yaml` (`test_size`, `random_seed`) | `datasets/processed/split_manifest.parquet` |
  | `features` | `features/build.py`, `sql/features/` | `datasets/processed/telco_churn_features.parquet` (Parquet — static snapshot for downstream stages), `preprocessing.pkl` |
  | `train` | `models/train/`, `configs/`, `datasets/processed/split_manifest.parquet` | `reports/train_run_id.txt` (the MLflow run-id receipt — a file, since outs must be files), `feature_space.txt`, `feature_columns.txt` |
  | `evaluate` | `models/evaluate.py`, `datasets/processed/split_manifest.parquet`, `reports/train_run_id.txt` | `reports/metrics.json` |

> **⚠ Two stages' real MLflow edge is not captured by their DVC `deps`/`outs` — document it rather than imply a pure file DAG.** The `train` and `evaluate` rows hide a coupling to MLflow that DVC does not track:
> - **`train`'s real output is an MLflow run, not a file.** "MLflow run ID" cannot literally be a DVC `out` — outs are files. Surface it as a **receipt file** the stage writes (`reports/train_run_id.txt`), the same pattern `ingest` uses to sequence around Postgres. That file is the concrete handle downstream stages and human cross-references resolve through; without it the `train → evaluate` edge is implicit. (Hence it is now listed as an `evaluate` dep above.)
> - **`evaluate`'s real *input* is that MLflow run, not a DVC dep.** `evaluate.py` resolves its model by explicit `run_id`/version from MLflow — never by alias (the contamination guard) — and the model it scores is the **calibrated challenger**, which `calibrate.py` produced *outside* the DAG (calibration is not a DVC stage). So the `train → evaluate` edge silently skips calibrate/threshold: `evaluate` reads `split_manifest.parquet` as a DVC dep, but its actual model comes from MLflow. `reports/metrics.json` is therefore a DVC-tracked output computed against a **non-DVC-tracked model** — consistent with the "DVC tracks the reproducible transform, MLflow holds the model" split, but it means **`evaluate` is not a pure function of its DVC deps.** The resolved `run_id`/version is logged into `metrics.json` (the *test-touched-once* invariant already requires this) so the stage's true input stays auditable even though DVC cannot hash it.
> - **The consequence to accept, not fix:** `dvc repro` cannot detect that a *new* calibrated challenger exists in MLflow — nothing in `evaluate`'s deps changed — so re-evaluating a fresh challenger is a Phase 10 Prefect responsibility, not a DVC-staleness trigger. This is the same boundary that keeps `register` out of the DAG, seen from the read side.

> **📌 Required in this phase — replace `_dvc_hash` with a content hash, or every model registered from here on carries empty provenance.** `models/train/common.py:73` resolves `<processed_data>/telco_churn_processed.csv.dvc` and falls back to `"unknown"` on `FileNotFoundError`, logged at **debug**. That fallback is correct today (DVC isn't initialised yet) and becomes a **silent failure the moment this phase lands**, for two compounding reasons. (1) The filename changes — the `features` stage outputs `telco_churn_features.parquet`. (2) More fundamentally, **`.dvc` sidecar files exist only for `dvc add`-ed files**; pipeline *stage outputs* are tracked in **`dvc.lock`**, under `stages.features.outs`. So no `.dvc` file for the parquet will ever exist, whatever it is named. The lookup misses, the handler shrugs, the provenance hash is `"unknown"` forever, and nothing warns — the fallback for *"not tracked yet"* is indistinguishable from *"tracked, but I'm reading the wrong file."*
>
> **Fix — do not read DVC internals at all; reuse the on-disk content hash the repo already computes.** `evaluate.py` already `sha256`s the features artifact it read, for provenance (Phase 7), and deliberately uses that direct content hash rather than `_dvc_hash` *precisely because the recorded-vs-on-disk gap makes the DVC hash unsuitable*. Rather than teach a second mechanism to parse `dvc.lock` — coupling provenance to DVC's file format to record a hash that gates nothing — **retire `_dvc_hash` and populate the manifest's data-lineage field from that same features-parquet `sha256`.** This buys three things: one hashing mechanism instead of two; no coupling to DVC's lockfile format; and a field that populates **unconditionally** — it reflects the bytes actually trained on, with no dependence on whether `dvc repro` re-recorded the lock. Rename the manifest field `dvc_data_hash` → `data_content_hash` and delete the now-dead `_dvc_hash` helper; the old name actively misleads once the value no longer comes from DVC. Add a test asserting a stable, non-`"unknown"` hash for a fixed features artifact. DVC keeps doing what it is actually good at here — the pipeline DAG and content-hashed staleness — without any code reading its lockfile.
>
> **Add `mlflow.log_input` for first-class, UI-visible lineage.** Alongside the manifest hash, log the training set as an MLflow dataset in the `train` stage — `mlflow.log_input(mlflow.data.from_pandas(df, source=...), context="training")`. MLflow computes its own digest, surfaces it in the run UI, and it pairs natively with the `LoggedModel` + `(model, dataset)` metric model this project already uses (Phase 7 evaluation logging). This is the idiomatic MLflow 3 lineage story and the more demonstrable one for a portfolio than a bespoke hash field; the `data_content_hash` remains the machine-checkable audit value, `log_input` the human-facing one.
>
> **The Optuna study-freshness consumer resolves the same way — and more strictly than the `dvc.lock` route would.** The Phase 5→6 Bridge discovered empirically that `tuning.py::_study_name()` content-addresses the Optuna study by hashing the data hash alongside `committed_features`, `search_space`, `cv_folds`, and the other tuning-config knobs, and `optuna.create_study(..., load_if_exists=True)` **resumes** any existing study sharing that name. While the data hash is a fixed `"unknown"` string, a genuine data change does *not* change the study name, so the tuning step silently resumes a study built on the *old* data, mixing incompatible trials from two datasets into one 1-SE selection pool — with nothing in the logs to flag it. Point `_study_name()` at the new `data_content_hash`: a content `sha256` changes the instant the features bytes change, independent of whether `dvc repro` re-recorded the lock, so it is a *stricter* freshness signal than a DVC-recorded hash would be. (`load_if_exists=True` itself is correct and intentional, for genuine crash recovery mid-optimization; the only defect was that `"unknown"` couldn't distinguish "same run resumed" from "different data entirely.")
>
> **This means the first post-Phase-8 training run is expected to produce different tuning numbers than whatever is currently recorded in `ANALYSIS.md` §4c — that is correct, not a regression.** The moment the data hash becomes real, `_study_name()`'s SHA256 digest changes, Optuna finds no existing study under the new name, and starts a genuinely fresh 50-trial search on the next `python -m telco_churn.models.train` (or `dvc repro`) run. The old `"unknown"`-keyed study rows are left behind, harmless, in the `optuna` Postgres schema (orphaned, not corrupted — `DROP SCHEMA optuna CASCADE` clears them if desired, but nothing requires it). Add a regression test: two `run_tuning_step` calls with different `data_content_hash` inputs (everything else held fixed) must produce different study names.
>
> **Ripple — the rename has one other touch-point, not a Phase 7 reopening.** Phase 5's `training_manifest.json` records this slot as "best-effort metadata that gates nothing, `unknown` by design" (DVC is not yet initialised when Phase 7 first runs); under this change it becomes `data_content_hash` carrying a real hash unconditionally. The **initial Phase-7 champion's manifest** keeps `"unknown"` — it was written before this fix: honest historical record, not something to rewrite. Update `CLAUDE.md`'s `training_manifest.json` field description to name `data_content_hash` and its `sha256` source.

> **Why `split` is its own stage, not folded into `validate` or `features`:** `data/split.py` depends only on `(customerid, churn)` from the validated raw table — never on an engineered feature — so it belongs strictly after `validate` and strictly before `features`, mirroring the module's own docstring. Hashing it separately means changing `features/build.py` never invalidates the split (the `features` stage doesn't depend on it), and changing `test_size`/`random_seed` reruns `split` → `train` → `evaluate` without re-triggering `features`. Both `train` and `evaluate` declare `split_manifest.parquet` as a dep — they are its two consumers (`data.split.partition()`/`dev_ids()` and `test_ids()` respectively) — so a manifest change correctly invalidates both, giving an explicit provenance chain from raw-data version → split manifest → model that was previously missing.

> **Feature versioning and lineage — what Phase 8 closes:** the `features` stage deps (`features/build.py` + `sql/features/*.sql` + the DVC-tracked raw CSV) are content-hashed by DVC. This is the *provenance* half of feature lineage — it records exactly which feature-engineering code ran on which data version to produce the processed dataset. The *membership* half is already covered by Phase 5 MLflow artifacts (`feature_space.txt` — what was available; `feature_columns.txt` — what was selected). Cross-referencing the DVC `features` stage cache entry with the MLflow run ID (logged as a `train` stage out) gives a complete, reproducible lineage chain: raw data version → feature code version → feature space → selection decision → model. This combination — DVC for provenance, MLflow for membership — is the standard approach for projects without a dedicated feature store.

- **No DVC remote is required, in this phase or any phase before 12.** A local cache is the whole configuration. Phase 12 adds an S3 remote as a genuine capability (shared cache, artifact storage), but nothing depends on it — see the two notes below.

> **⚠ Postgres is a materialised cache of the CSV, not a data source — and it cannot be a DVC `out`.** Earlier drafts of the stage table declared `ingest`'s output as "`customers_raw` table hash." **DVC `outs` must be files or directories in the workspace.** There is no table-hash out; `dvc repro` cannot cache it, restore it, or detect drift in it. The stage as previously specified could not be written, and pointing DVC at the database would not fix it — a DB is mutable external state and DVC is a content-addressed file store. They do not compose.
>
> The resolution is to see that the table's contents are *fully determined* by three things DVC **can** hash: the raw CSV, `sql/schema/001_create_raw.sql`, and `data/ingest.py`. The guarantee this phase ships — raw CSV → features Parquet — therefore holds with Postgres living **inside** the implementation rather than as a node in the graph. `ingest` emits a **receipt file** (`reports/ingest_receipt.json`) as its out: the hashable artifact standing in for the side effect, giving `dvc repro` something to cache and `validate` something to depend on. The `validate` stage already does exactly this — `reports/validation/` is a receipt in all but name; `ingest` is simply the stage that never got one.
>
> **The limitation, stated rather than discovered:** if someone hand-edits `customers_raw`, the receipt and every dep are unchanged, so DVC will not notice. That is inherent and acceptable — **the CSV is the source of truth, the table never is**, and `dvc repro --force` rebuilds it. Any workflow that treats the DB as authoritative is outside the guarantee. (The SQL layer exists because Phase 4's feature engineering is deliberately done in SQL views as a portfolio demonstration; drop Postgres and you drop that. Keep it, and the DAG needs a receipt to sequence around it.)

> **⚠ The raw CSV is committed to git — DVC lists it as a `dep`, it does not own it.** `deps` and `outs` are different contracts. `outs` are what DVC *owns*: cached, gitignored, replaced by a pointer. `deps` are what DVC *watches*: hashed into `dvc.lock` so it knows when to re-run. **A dep may be any path in the repo** — it need not be `dvc add`-ed, gitignored, or in a remote. Change the CSV and `dvc repro` re-runs everything downstream; lineage is fully preserved.
>
> DVC should own raw data when it is too large for git, changes over time, or needs its own version history. **None of those hold here:** the file is 977 KB, static, and `CLAUDE.md` marks `datasets/raw/` read-only. What DVC actually buys this project is the pipeline DAG — `dvc repro`, content-hashed staleness, cross-stage lineage — and that works with a local cache and no remote at all. The derived artifacts (`telco_churn_features.parquet`, `split_manifest.parquet`) are *regenerable by definition*; a fresh clone rebuilds them rather than pulling them, so a remote for them is a compute-saving cache, never a correctness requirement.
>
> **This is what makes the fresh-clone reproducibility claim (End-to-End Verification §1) true.** Gitignoring a one-megabyte static CSV behind a local cache would make the repo un-clonable in exchange for nothing, and would leave Phase 11's CI unable to fetch data from a remote that does not exist until Phase 12. The competence a reviewer looks for is in `dvc.yaml`, `dvc.lock`, and `dvc repro` — not in whether a small static file is gitignored. *(Licensing: the dataset is IBM's Cognos Analytics sample, redistributed in thousands of public repositories. Low risk, but it is a judgment call.)*

> **Deliberate scope — why the DAG stops at `evaluate`:** the six DVC stages cover the *data-transform* pipeline (raw → reproducible metrics). Calibration + thresholding (Phase 6) and registry promotion (Phase 7) are intentionally **not** DVC stages. They are *decision* steps, not deterministic data transforms: calibration depends on a held-out fold, the threshold encodes a business cost choice (owned outside the pipeline — see `summary.md` §4.5), and promotion compares against the live `champion` in the MLflow registry, which is mutable state DVC cannot content-hash. Folding them in would make `dvc repro` non-deterministic (its output would depend on whatever `champion` currently exists).
>
> There is a sharper, operational reason too: **DVC stages must be safe to re-run.** A stage's `outs` are files, and `dvc repro --force` is listed in `CLAUDE.md`'s key commands as a routine thing to type. `calibrate.py` mints an MLflow registry version and `register.py` moves the `champion` alias — neither a file nor idempotent. If `register` were a stage, asking DVC to rebuild would re-point a live service's alias as a side effect.
>
> Instead, those steps are driven by the Phase 10 Prefect `retrain` flow, which calls `train → evaluate` (reproducible, DVC-tracked) and then `calibrate → threshold → register` (decision layer) as explicit flow tasks. If full champion reproducibility is ever required, the fix is to pin the comparison baseline to a specific run ID rather than the `champion` alias — not to add these as DVC stages.

> **❓ OPEN QUESTION — between Phase 8 and Phase 10, what actually runs `calibrate → threshold → register` after `dvc repro` stops at `evaluate`, and where does the human-approval stamp happen?** The block above settles *why* calibration and registry promotion stay outside `dvc.yaml`; it names the Phase 10 Prefect flow as what eventually drives them — but Phase 10 lands several phases later. Phase 7's `register.py` already hard-blocks on `promotion_decision.json`'s `review` field being `"approved"` (`configs/register/default.yaml`'s `require_review: true`), stamped via `gate.py::record_review` — whose own docstring names "the notebook's closing cell" as the sole owner of loading and re-saving that file. So once `dvc repro` exists and becomes the primary reproducibility path, does a full re-run go `dvc repro` (ingest→validate→split→features→train→evaluate) → open `notebooks/05-evaluation-and-error-analysis.ipynb` to stamp review → then `calibrate`/`threshold`/`register` by hand? That gap is not hypothetical: this project's most recent clean-slate rerun hit it directly, and stamped review via a one-off inline script calling `record_review` instead of the notebook — workable, but not a documented, repeatable path. Candidate resolutions, none yet chosen:
>
> 1. **Notebook-only, by design, until Phase 10.** The closing cell of `notebooks/05-evaluation-and-error-analysis.ipynb` is the one blessed place a human stamps review pre-Phase-10; any CLI/script route is an unsupported shortcut that happens to work because `record_review` is a pure function, not a sanctioned alternative.
> 2. **A small dedicated CLI** (e.g. `python -m telco_churn.models.review register.model_version=<v> review.verdict=approved review.notes="..."`) wrapping `gate.py::record_review` and re-logging `promotion_decision.json` onto the eval run — a first-class, reproducible entry point independent of Jupyter, and a natural precursor to Phase 10's own open question of whether `require_review` ever flips back to `true` (below).
> 3. **Defer entirely to Phase 10.** Between Phase 7 and Phase 10, `calibrate → threshold → register` stays an undocumented manual sequence run ad hoc (notebook or scratch script, reviewer's choice), formalized only when the Prefect flow is built.
>
> Resolve before the next full pipeline rerun that needs a promotion decision, or record it as deliberately deferred to Phase 10 (option 3).

- **SQL view materialisation (required):** The Phase 4 SQL views recompute on every read — acceptable for development but not for a DVC pipeline. The `features` stage entry point must call `build_feature_df(engine)` to execute the SQL graph **once**, then immediately write the result to `datasets/processed/telco_churn_features.parquet` before exiting. The `train` stage lists that Parquet file as its sole data dependency, not Postgres. This gives three guarantees: (1) every training run reads a static, content-hashed snapshot; (2) `dvc repro` never blocks on the DB when the features hash is unchanged; (3) if Postgres is unavailable, all downstream stages still run from the cached Parquet. The DB is only contacted during the `features` stage, which DVC skips if its deps (raw CSV + `build.py` + SQL files) are unchanged.

- **Retire `build.py __main__` block:** The `if __name__ == "__main__"` block in `src/telco_churn/features/build.py` is a Phase 4 development scaffold — it wires together the full feature pipeline (config load → DB connect → SQL views → `build_feature_df` → write CSV) so the pipeline could be verified manually before DVC existed. In Phase 8 the DVC `features` stage entry point takes over that responsibility with two changes: output is Parquet instead of CSV, and DVC manages invocation. The `__main__` block should be removed from `build.py` at this point — it becomes dead code once the stage entry point exists. The core logic (`build_feature_df`, `_add_python_features`, column constants) stays in `build.py` permanently; only the CLI scaffold is retired. Also delete `datasets/processed/telco_churn_processed.csv` from the repo — it is superseded by the DVC-tracked `datasets/processed/telco_churn_features.parquet`. **The new stage entry point must assert `df_out.shape[0] > 0` before writing the Parquet output** — a zero-row result from a broken SQL view should be a hard failure, not a silent empty artifact (eighth-pass QA item 4).
- **Repoint the features accessor — one function, not four call sites.** Phase 6 introduced `features/accessor.py::load_features()` as the single owner of the processed artifact's path, format, and `sha256`. In Phase 8, change `pd.read_csv` → `pd.read_parquet` and the filename → `telco_churn_features.parquet` **inside that one function body**. The `train` stage in `dvc.yaml` declares the file as a dep, so DVC re-runs training whenever the features hash changes.
  > `models/calibrate.py` and `models/threshold.py` already read through `load_features()`; confirm `models/train/common.py`'s `_load_processed`/`_load_dev_features` do the same before this phase edits the accessor, so the CSV→Parquet swap stays a one-function change rather than a multi-file migration discovered mid-phase.

- **Update the Phase 5 notebooks' processed-data reads:** `03a-model-selection.ipynb:1104`, `03b-feature-selection.ipynb:553`, and `03c-hyperparameter-tuning.ipynb:709` each independently build `processed_path = get_project_root() / cfg.paths.processed_data / "telco_churn_processed.csv"` and `pd.read_csv` it — the same scaffold as the `src/` loader, repeated three times. Replace all three with a `load_features()` call, so the format is owned in one place and the notebooks stop encoding the storage decision. Until this phase lands, none of these notebooks are reproducible from a fresh clone: the raw CSV is present (git-tracked), but `datasets/processed/` is gitignored and there is no DAG to rebuild it, so `telco_churn_processed.csv` only exists after manually running the Phase 4 feature pipeline (`02a`/`02b` or `features/build.py`). Wiring the `features` stage in this phase is what closes that gap — `dvc repro` regenerates it from the committed CSV.

- **No manual retraining flags (replaces the notebook's `RETRAIN_BEST` / `LOG_ARTIFACTS` booleans):** the hand-set flags that decided what to recompute do **not** migrate into `src/`. DVC's content-hashed DAG determines staleness — changing a dep reruns exactly the affected stages and nothing else. This is the engineering replacement for the manual flags (former Group B item).
- **Phase 2 cleanup:** Remove `clean_dataframe()` from `validate.py` — imputation now belongs to the `features` stage's fitted `SimpleImputer`. Update `validate_clean()` to expect the features stage output directly. Remove the associated tests.
- **Phase 2 cleanup:** In the DVC `validate` stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event, and call `sys.exit(1)`.
- **SQL migration strategy:** `CREATE TABLE IF NOT EXISTS` is idempotent for creation but blind to changes — adding a column, renaming one, or tightening a constraint will be silently skipped on re-run. Adopt Alembic (the natural fit for a SQLAlchemy-backed project) before Phase 8 ships so that schema changes are applied reproducibly across local, CI, and AWS environments. Existing DDL in `sql/schema/001_create_raw.sql` becomes the initial migration version.

**Verification:** Change a hyperparameter in `configs/training/lightgbm.yaml`; `dvc repro` re-runs `train` and `evaluate` only, not `ingest`, `validate`, or `features`.

---

### Phase 9 — FastAPI Serving + Streamlit UI *(3 days)*

**What this achieves:** The champion model is accessible via a REST API with a health endpoint, batch prediction support, and built-in Prometheus metrics. A Streamlit UI lets a non-technical user pull up an existing customer by ID and see the churn probability alongside the top reasons, or manually edit a profile for a what-if scenario or a prospective customer with no account yet — both paths call the same API.

**Deliverables:**
- **Capacity-constrained contact policy (`top-K by expected value`) — resolved here, not in Phase 7.** Phase 6's `threshold.py` flags it: *"when the implied contact rate exceeds capacity, the correct policy is not a higher threshold but top-K by expected value `p·r·LTV − c`."* Raising `t*` until the flagged count fits the retention team's capacity abandons the Bayes-optimal cut for an arbitrary one; ranking by EV and taking as many as the budget allows is the correct response. **This is a serving-time decision mechanism, which is why it lives here** — the evaluation layer has no knowledge of operational capacity and must not invent one. Compare `threshold.json`'s `implied_contact_rate` against the configured capacity; when it fits, ship the fixed `t*` unchanged; when it does not, generate the contact list by top-K EV.
  > **Under the current cost model this is rank-identical to top-K-by-score, and that is expected — not a bug.** `c`, `r`, and `LTV` are scenario-level constants (`ANALYSIS.md` §0), so `EV_i = p_i · (r × LTV) − c` is a strictly increasing function of `p_i` and preserves rank order exactly. So implement the *policy* (the capacity constraint is real and the K-cut is real) but do not expect a different customer ordering than the score ranking gives. The two diverge only under a **per-customer `LTV`** — recorded in §0 as the leading v2 item, and the change that would make this mechanism earn its name.
- `src/telco_churn/serving/schemas.py` — Pydantic v2 request/response models; field constraints aligned with the Pandera schema (single source of truth)
- `src/telco_churn/serving/predict.py` — loads the `champion` model and preprocessor from MLflow at startup; exposes `predict_single` and `predict_batch`
  - **⚠ Leave the shadow/canary seam, even though no live traffic ever flows through it.** Real deployments do not point production at a new model on an offline gate alone — a challenger scores live traffic in **shadow** (logged, not served) or a **canary** slice before the pointer moves (`ANALYSIS.md` §9 #13). This project has no feed, so none of that can *run* — but the serving layer must be *structured* so it could: resolve the served model through a single indirection (a `resolve_serving_model()` that returns `champion` today) rather than hard-coding `models:/…@champion` at the call site, so a `challenger` shadow route plugs in without touching the request path. **Do not synthesize traffic to make a shadow comparison appear to execute** — a canary whose evidence you authored is a prop, not a demonstration, and it breaks the same honesty line as the drift stack (build the machinery, be explicit about what has never fired). The seam is the deliverable; the traffic is not.
  - **Serving-parity test against Phase 6's golden fixture.** `golden_predictions.json` (written by **`calibrate.py`** from the in-memory fitted pipeline, logged on the registered run, mirrored to `reports/`, and already verified through the `champion` alias by `register.py`) holds a row sample pinned by `customerID` and the scores the champion produced for them at promotion time. Assert the API returns those same scores through the full request path — Pydantic parse → `ColumnTransformer` → `predict_proba`. This is what catches a serving-layer bug that silently changes predictions (a column reordered, a dtype coerced, a category mapped to the wrong level) — the class of failure that produces plausible numbers and no error, and would otherwise be invisible until someone reconciled the API against the offline metrics. Phase 11's CI runs the same assertion.
  - **⚠ The `Dockerfile` must build from the champion's own locked environment, not from `pyproject.toml`'s open version ranges — this is the precondition the golden-parity tolerance above (and Phase 7's) actually rests on.** `register.py`'s smoke check compares reloaded scores against `golden_predictions.json` at `atol=1e-9`, and that tolerance is only well-posed if the scoring environment never drifts from the one `calibrate.py` logged the model under: a `numpy`/`scikit-learn`/`lightgbm` point-release bump shifts floating-point output in the same 1e-7–1e-9 range as the tolerance itself, with no actual bug present. `register.py` enforces this at the moment of promotion (comparing the registered version's own logged `requirements.txt`, via `mlflow.pyfunc.get_model_dependencies()`, against the environment running the check) — but that check only covers registration, not every later container build. So the serving image must install from **`uv.lock`** at the commit the champion was trained under, or — more robust to a stale lock — from the champion's own MLflow-logged `requirements.txt`/`conda.yaml`, fetched at build time; never from `pyproject.toml`'s ranges resolved fresh at image-build time. A container built that way, run months after promotion, would fail this same test for an environment reason that presents as a serialization bug, sending the reader to debug the wrong layer. Phase 11's CD pipeline builds this image exactly once per merge and redeploys the same SHA to staging and production, so getting the base image right here is a one-time discipline, not a per-deploy one.
- **Threshold as policy, not model state:** `/predict` returns the **calibrated `P(churn)`**; the decision rule (operating threshold, any per-segment cuts, EV formula) lives in a **separate versioned config / policy layer** loaded at startup, changeable without redeploying the model artifact. This keeps the business-owned operating point decoupled from the model — see `summary.md` §4.5. The response includes both the probability and the decision so callers can apply their own threshold if they prefer.
- `src/telco_churn/serving/app.py` — FastAPI app:
  - `POST /predict` — single customer prediction
  - `POST /predict/batch` — batch scoring (array of `CustomerFeatures` objects); same model and preprocessor as the single endpoint — batch is a delivery mode, not a model change; the prediction unit (one customer per score) is identical in both modes
  - `GET /customer/{customerid}` — read-only lookup of an existing customer's raw feature values from Postgres (`customerid` is already the Phase 1 primary key), used to prefill the UI form instead of requiring all fields typed from scratch; not a model endpoint, no write path
  - `GET /health` — liveness probe
  - `GET /ready` — readiness probe (model loaded)
  - `GET /metrics` — Prometheus metrics via `prometheus_fastapi_instrumentator`
  - Structured log per prediction: request ID, features, probability, threshold, decision

> **Note — batch as the operational backbone:** In production, batch is typically the primary delivery mode. A nightly/weekly job scores the entire active customer base, writes results to a `churn_scores` table, and the CRM reads from there. Real-time (`/predict`) is the supplement — used for event-triggered interventions (e.g., a customer calls to cancel). The `pipelines/batch_predict.py` flow in Phase 10 is the scheduled incarnation of this pattern. The prediction unit (`a single customer per score`) is identical in both modes.

- `src/telco_churn/ui/streamlit_app.py` — two entry modes over the same 19 raw-input fields (`build_feature_df` derives the 20th, `charge_per_service`, automatically — nothing to enter for it): **(1) Lookup** — enter a `customerid`, `GET /customer/{customerid}` prefills every field, review and predict. This is the default tab and the realistic operational flow: a retention agent is working an existing account, not fabricating one field-by-field. **(2) Manual / what-if** — fields are directly editable, blank for a prospective customer with no DB record, or pre-filled from a lookup and then tweaked to test a scenario (e.g. "if we cut `monthlycharges` by $10 as a retention offer, does the probability drop enough to justify it?"). Both modes → `POST /predict` → probability gauge + top-5 SHAP contributions.
- `Dockerfile` (multi-stage, FastAPI) + `Dockerfile.ui` (Streamlit); both added to `docker-compose.yml`
- `tests/integration/test_api.py` — FastAPI test client: `/predict` returns valid schema; `/health` returns 200; batch endpoint accepts arrays; `/customer/{customerid}` returns 404 for an unknown ID

> **Note — lookup-first is a serving-layer fix, not a feature-selection one.** A blank 20-field form is the wrong default for a churn tool: real users work existing accounts, they don't manually reconstruct a customer profile from memory. Solving that friction by shrinking the model's input space would have meant re-litigating the §3/§5 `03b-feature-selection.ipynb` decision — where the paired-bootstrap test materially favoured the full 20-feature set (Δ = 0.0173, CI [0.0104, 0.0246]) — against a second, competing objective (form usability), which the project's single-metric selection invariant (`CLAUDE.md`) explicitly rules out. Fixing it here instead, at the UI/serving layer, keeps the frozen feature set untouched and leaves manual entry available for the cases that actually need it (what-if scenarios, prospective customers).

> **❓ OPEN QUESTION — where does `model_card.json` actually get shown to a stakeholder?** `CLAUDE.md` calls `model_card.json` (Phase 7 `register.py`) the model's "stakeholder-facing narrative," but no phase in this plan renders it anywhere a stakeholder would actually look — today it is an MLflow artifact on the promoted run, mirrored to `reports/`, readable only by opening the JSON directly. Candidate resolutions, none yet chosen:
>
> 1. **A third Streamlit tab** (e.g. "About this model") that fetches `model_card.json` via the `champion`-aliased run and renders it as readable sections — consistent with this phase's existing lookup/manual tabs, and the cheapest place to add it since the Streamlit app already exists here.
> 2. **A static rendered doc regenerated at promotion** (e.g. `reports/model_card.md`, templated from the JSON) — no new UI surface, but no live "current champion" view either; goes stale after a rollback unless regenerated on every alias flip too.
> 3. **Leave it as a JSON artifact, by design.** Treat "stakeholder-facing" as describing the *format* (structured, no raw hyperparameters/hashes, plain-language fields) rather than committing to a rendered surface, and defer actual display to whatever future consumer needs it.
>
> Resolve before this phase's Streamlit app is scoped in detail — cheap to add as a third tab if decided early, awkward to bolt on after the two-mode UI is already built and tested.

**Verification:** `docker compose up && curl -X POST http://localhost:8000/predict -d @example_payload.json` returns a churn probability; `curl http://localhost:8000/customer/<id>` returns that customer's raw fields; Streamlit at `:8501` displays a prediction with SHAP contributions in both lookup and manual mode.

---

### Phase 10 — Prefect Orchestration (Continuous Training) *(2–3 days)*

**What this achieves:** The model retrains automatically every week and checks for data drift every day — without manual intervention. The Prefect UI provides a full audit trail of every run, including failures.

**Deliverables:**
- Prefect 3 server added to `docker-compose.yml` (UI at `:4200`)
- `pipelines/retrain.py` — weekly Sunday 02:00 schedule; runs `ingest → validate → features → train → calibrate → evaluate → register`; promotes `challenger` to `champion` via `register.py`'s gate — the same gate, in its **comparative regime** (`ANALYSIS.md` §0): PR-AUC must improve on the incumbent champion (selection), while recall@`t*` ≥ 0.65 and Brier non-inferiority hold as vetoes. Phase 7 exercised the same gate's cold-start regime; nothing is re-specified here. As in Phase 7, `calibrate.py` performs the cycle's only registration and `register.py` mints nothing new — a pass simply flips `champion` onto the same version `challenger` already points at, logged as `model_promoted`. Re-calibrated with the **pinned method** (`calibration.method` from config — never `auto`, which would let the method flip on Brier noise between weekly cycles). Inherits the operating threshold rather than re-deriving it (the closed-form cut is a function of costs, not of the model). **Dummy-floor guardrail:** each cycle recomputes the prevalence / `DummyClassifier(strategy='prior')` floor on the current data and asserts the retrained model clears it by a margin — a near-dummy result signals a broken feature/label pipeline (stale join, dropped column), not a modelling regression, and blocks promotion with a structured alarm. The floor is recomputed each cycle, never hardcoded, because churn prevalence drifts.
  > **📌 The alias flip is a distinct final step, behind a `require_traffic_validation` gate — because the weekly retrain is where the missing shadow/canary stage actually bites.** Phase 7's cold-start flip has no incumbent to shadow against; this one replaces a *serving* champion with a challenger, and in a real deployment that flip would follow a shadow or canary evaluation on live traffic, never an offline gate alone (`ANALYSIS.md` §9 #13). This dataset has no feed, so structure the flow to make the absence *legible and pluggable* rather than invisible: the promotion decision and the `set_registered_model_alias` call are **separate steps**, with a `require_traffic_validation` config gate between them that **no-ops on a static dataset** and carries a one-line comment marking it as the shadow/canary insertion point. **Do not fabricate traffic to make the gate appear to run** — a green light you wired to always-green is a prop, and it breaks the same honesty line as the demonstrative promotion gate two notes below. The seam is real; the traffic validation is explicitly stubbed, not faked.
  > **📌 Feature selection must not silently re-run every cycle.** The `train` step above currently maps to the full `python -m telco_churn.models.train` script (`__main__.py`), which chains Steps 1–5 — including Step 3's feature-selection freeze (Phase 5, `ANALYSIS.md` §4). That freeze is explicitly designed to run *once*, not on a schedule: a borderline feature (e.g. `paymentmethod`, 49/100 fold stability) could flip in or out between cycles from data-split noise alone, which would confound tuning (Step 4 assumes a fixed input space) and make model versions hard to compare week to week. Before this phase ships, either give `__main__.py` a flag to skip Steps 1–3 and load the already-committed `feature_columns.txt`, or have this flow call a narrower re-tuning-only path — the weekly cycle should reuse the frozen feature set, not re-derive it.
  > **📌 Persist the dev-OOF probability vector every cycle — do not rely on Phase 7's recomputation trick.** Phase 6's `calibrate.py` logs only `calibration_summary.json` and a figure: it persists its *conclusions* (method selected, BSS, `t*`) but not the **evidence they rest on**, discarding the calibrated OOF probability vector. Phase 7 works around this by recomputing the vector deterministically and writing `reports/dev_oof_predictions.parquet` — which is sound *only because the dataset is static and the folds are seeded*. **That workaround stops being reproducible the moment retraining is real.** Under a weekly cadence each cycle has its own data, its own folds, and its own calibration map, and "just recompute it" silently reconstructs a *different* vector than the one that cycle's decisions were made on. From this phase forward, `calibrate.py` should log the OOF vector as a first-class run artifact — alongside `calibration_summary.json`, on the same run — so every cycle's drift reference, veto surface, and calibration slope are computed from the numbers that cycle actually decided on. This is a one-line `log_artifact`, and it retires the workaround rather than inheriting it.
  > **📌 Warm-starting Optuna across retrain cycles.** Phase 5's `configs/tuning/optuna.yaml` `warm_start_params` is a one-time, hand-set prior (the archived notebook's own Optuna best) with no equivalent once that notebook is out of the loop. From this phase on, the `train` step should populate `warm_start_params` dynamically from the current `champion`-aliased MLflow run's logged params (`mlflow.get_run(champion_run_id).data.params`) before calling `run_tuning_step`, instead of reading a hardcoded config block — each retrain's winner becomes the next cycle's informed prior. `tuning.py`'s `study.enqueue_trial(...)` mechanism needs no change for this; only the source of the dict does.
  > **Clarification — what "rolling holdout" means here, since `customers_raw` doesn't grow.** The raw Kaggle CSV is a static, read-only snapshot (`CLAUDE.md`) — this project has no live customer feed appending new rows, so the DVC `split`/`ingest`/`validate` stages have no changed deps to react to, and `retrain.py`'s `ingest → validate → features` steps are cache hits every cycle. `data.split.make_split()` therefore runs once, ever, for this project; `split_manifest.parquet` is never regenerated, and there is no new-customer-driven reshuffling problem to solve. The "do not reuse the sealed test set — use a rolling/time-based holdout instead" guidance (§ Phase 7 evaluate.py, § Deliberate scope above) is about a different risk: scoring the *same* frozen `evaluate.py` test partition against every weekly promotion decision implicitly "spends" that sealed set dozens of times a year, eroding it as an unbiased estimate (repeated-peeking, not data-freshness). The data that genuinely grows over time in this system is `prediction_outcomes` (below, `performance_check.py`) — live predictions joined to realised labels as they mature — not `customers_raw`. Any future rolling-holdout construction for promotion decisions should be carved from `prediction_outcomes`/logged-prediction history, not from a re-partitioned `customers_raw`; the mechanism itself is not yet decided and is deferred to this phase's implementation.
  > - **⚠ A time-based cut is not available, and no split definition can create one.** IBM Telco is a **cross-sectional snapshot**: one row per customer, one moment, churn already labelled. The raw CSV has **no date column** — `tenure` is a *duration* (months as a customer), not a timestamp — so there is no time axis to cut along. `data.split.make_split()` is a random stratified `train_test_split` (`split.py:80`) not by preference but because it is the only split the data supports. This matters beyond terminology: in a real deployment the holdout is defined *temporally* and renews itself every cycle, which is precisely why "the test set is spent" is a non-problem in production. **That mechanism is unavailable here, and it is the dataset's shape — not the method — that forecloses it.** Any candidate resolution below that presumes a temporal cut is not viable; deterministic hashing over an existing partition is the only splitting mechanism this data admits.
  > **❓ OPEN QUESTION — what does the weekly promotion gate actually score, on a static dataset?** `customers_raw` is a fixed, non-growing snapshot: there is no way to produce a genuinely fresh evaluation surface for each retrain cycle, and `CLAUDE.md`'s own invariant already forbids reusing the original sealed test set repeatedly for challenger-vs-champion comparisons — repeated peeking erodes it, even though nothing ever trains on it. So `register.py`'s gate in its comparative regime (`ANALYSIS.md` §0 — `challenger` improves on `champion`'s PR-AUC, subject to the recall and Brier vetoes) has, as of Phase 10, **no defined evaluation set**. Note this is a question about the gate's *input data*, not its *rule* — and the rule is **fully settled**, so this is the only piece left. §0 fixes what is measured, what each metric may do with the answer, and how uncertainty binds (paired bootstrap on Δ, materiality Δ\* = 0.005, CI excludes 0 in the challenger's favour); Phase 7's `register.py` and `utils/stats.py::paired_bootstrap_metric_ci` implement it, and Phase 7 already exercises the guardrails and the decision function under test. What §0 deliberately does not name is the *rows* those metrics are computed over once the sealed set is spent. That is this phase's decision to make, and it is all that stands between the comparative regime and going live. Three candidate resolutions, none yet chosen:
  >
  > 1. **Score on `prediction_outcomes`.** Correct in principle, and it is what a real deployment would do. **⚠ But on this dataset it is not merely delayed — it never arrives, and the plan previously described it as a cold-start problem, which is wrong.** `prediction_outcomes` is *"live predictions joined to realised labels as they mature"* — and there are no maturing labels: churn is already fixed in the frozen CSV, for every one of the 7,043 customers, all of whom the champion trained on. No observation window ever closes, because no outcome is ever pending. `ANALYSIS.md` §9 #13 states this outright (*"the realised-performance loop has no labels to mature"*); this bullet contradicted it and is corrected here. **Option 1 is viable for the design and dead for this instance** — retain it as the documented production answer, not as something Phase 10 can wait for.
  > 2. **Nested CV on the full dataset per cycle.** Statistically defensible on static data and available immediately, but it measures *the spec*, not *the artifact*, and it re-fits every cycle — expensive, and the resulting number is not comparable to the Phase 7 sealed-test metrics of record.
  > 3. **Declare the gate demonstrative.** On a fixed 7,043-row snapshot with no live feed, Phase 10's weekly retrain is a *mechanism demonstration* — the plumbing, alerting, and alias-flip are real; the statistical claim behind the promotion decision is not. Honest, cheap, and arguably the right answer for a portfolio project. **`ANALYSIS.md` §9 limitation #13 now states this outright** (for drift, for the realised-performance loop, and for this gate), which makes option 3 a matter of *accepting* a documented boundary rather than newly conceding one.
  >
  > **⚠ Once option 1 is seen to be dead on this dataset, the question is far less open than "three candidates" suggests — and the honest reading is that option 3 is forced.** Option 1 never arrives (no labels mature). Option 2 is available but measures *the spec*, not *the artifact*, and yields a number not comparable to the Phase 7 metrics of record — so it cannot answer the question the gate actually asks (*"is this champion better than that one?"*). **That leaves option 3, and it is a conclusion rather than a concession:** the sealed test set was always scaffolding for a one-shot evaluation, not a renewable resource; in production the holdout is defined temporally and renews itself; this dataset has no time axis, so that mechanism is structurally unavailable. The weekly retrain is therefore exercised as **machinery** — the plumbing, gate, alias flip, and alerting are real and correct — and not as live learning. Being able to say precisely *why*, and to show it is the dataset's shape rather than a gap in the method, is worth more than a rolling holdout that merely looks live.
  >
  > This remains a **Phase 10 decision** — it does not block Phases 6–9 — but it should be **recorded as resolved in favour of option 3 before `retrain.py` is written**, rather than re-litigated then. `ANALYSIS.md` §9 #13 already carries the substance; the plan should simply stop describing the choice as open. Keep option 1 documented as the answer a real deployment would use, so the design's correctness is legible even though this instance cannot exercise it. The one outcome to avoid is shipping a gate that silently re-scores the spent test set every Sunday.
  > **❓ OPEN QUESTION — does the direction sanity check recur every retrain cycle, or was it a cold-start-only gate?** Phase 7's promotion gate has two structurally different parts: four automated numeric criteria (PR-AUC, recall, Brier, calibration slope) computed in `gate.py::decide_promotion`, and one human-executed veto — the direction sanity check (does any top-ranked SHAP feature's learned effect contradict the established EDA relationship?) — exercised once, in the Phase 7 notebook's closing human-review cell. The `retrain.py` bullet above re-specifies the automated criteria for the comparative regime (PR-AUC improvement, recall/Brier vetoes) but never mentions the direction sanity check. Segment collapse, fairness disparity, and per-group calibration collapse get the opposite, explicit treatment two bullets down — a deliberate promotion from "one-time reported diagnostic" to "continuously-enforced," via `performance_check.py`. The direction sanity check gets no equivalent statement either way: nothing carries it forward as a recurring automated check, and nothing retires it as cold-start-only either. Two candidate resolutions, neither chosen:
  >
  > 1. **Cold-start-only, by design.** Once the frozen feature spec's top-ranked features are verified directionally sane at launch, a weekly retrain on the *same* spec is unlikely to relearn a backwards relationship absent a pipeline bug — and a pipeline bug (a stale join, a flipped sign) is exactly what the dummy-floor guardrail and PSI/drift monitoring already exist to catch, from a different angle. Requires no new code; just say so explicitly, the same way the sealed-test-erosion question above was resolved by *naming* the boundary rather than building around it.
  > 2. **Recompute automatically, page a human only on failure.** `direction_sanity_check` is already a pure function over a persisted correlation, not a subjective judgment — `retrain.py` could re-run it every cycle against the same `_EXPECTED_EDA_DIRECTIONS` dictionary and treat a fresh contradiction as a structured alarm (mirroring the dummy-floor guardrail above), rather than requiring a human to open a notebook every Sunday. Keeps the veto's teeth without the manual step, at the cost of a dictionary that itself needs maintaining as features change.
  >
  > Resolve before `retrain.py` is written, same as the two open questions above.
  > **❓ OPEN QUESTION — should "inherits the operating threshold" be unconditional, or gated on `within_ci`?** Phase 6's `threshold.py` already computes an agreement diagnostic every time it runs: `within_ci` — whether the closed-form `t*` falls inside the bootstrap CI of the empirical argmax-EV threshold. Today that check is passive — a `logger.warning("threshold_argmax_disagreement", ...)` a human reads after a manual run, nothing more (see `run_threshold_step`, `threshold.py:380-390`). As written, Phase 10's `retrain.py` inherits the shipped threshold **unconditionally** every cycle, on the reasoning that `t*` is a pure function of `configs/costs.yaml` and doesn't move when the model does. But `within_ci` exists precisely to catch the case where that reasoning breaks: a re-calibration cycle producing a materially different probability distribution near the operating point, such that the closed-form cut no longer agrees empirically with where expected value is actually maximised. Candidate resolutions, none yet chosen:
  >
  > 1. **Recompute `within_ci` each cycle, unconditionally inherit `t*` regardless.** Cheapest — the diagnostic becomes a monitored metric (logged, alertable) but never blocks or auto-triggers anything. Consistent with "one metric drives selection" (`CLAUDE.md`), since `t*` isn't a selection criterion, but risks the disagreement sitting unnoticed in logs for cycles at a time.
  > 2. **`within_ci == False` triggers an automatic re-derivation** (`run_threshold_step` called instead of inherited) as part of that cycle's `retrain.py` run. Keeps the threshold self-correcting, but blurs the "cost-driven, not model-driven" boundary the inherit-by-default design rests on, and needs a decision on whether re-derivation can silently change the shipped threshold or must itself be gated (e.g. behind the same manual sign-off as a `costs.yaml` edit).
  > 3. **`within_ci == False` blocks promotion with a structured alarm**, mirroring the dummy-floor guardrail in the bullet above — treat sustained disagreement as a signal that calibration (not the threshold) needs attention, and let a human decide whether to re-derive, re-calibrate, or investigate further.
  >
  > **Deferred deliberately**, same as the promotion-gate question above — resolve before `retrain.py` is written and record the choice in `ANALYSIS.md`.
  > **❓ OPEN QUESTION — does `require_review` ever flip back to `true` after Phase 10 sets it `false`?** Phase 7 defaults `require_review: true` (`configs/register/default.yaml`) because a human stamps `promotion_decision.json`'s `review` field via notebook 05 each cycle; Phase 10's weekly `retrain.py` sets it `false` for the no-human-in-the-loop automated cycle. Nothing in this plan states whether `false` is the permanent Phase 10+ state or whether some condition re-inserts a human. Candidate resolutions, none yet chosen:
  >
  > 1. **Never re-escalate.** `false` is permanent from Phase 10 on — the automated gate's four numeric criteria plus vetoes are what a no-human cycle is defined to trust; re-escalating on top of that would concede the automation was never really trusted.
  > 2. **Escalate on a gate-boundary or rejected outcome.** A cycle that fails the gate, or passes within some margin of a veto threshold, routes to a human review before the alias flip instead of auto-rejecting/auto-promoting — the same "alarm on ambiguity" pattern the dummy-floor and `within_ci` guardrails already use elsewhere in this phase.
  > 3. **Periodic audit sample.** Every Nth cycle (or a fixed cadence) forces `require_review: true` regardless of outcome, as a spot-check that the automated gate hasn't drifted from what a human would actually approve.
  >
  > Resolve before `retrain.py` is written, same as the other open questions above.
- `pipelines/drift_check.py` — daily 06:00; pulls the last 24 hours of predictions from the API structured logs; runs an Evidently data drift report **against `drift_reference.json` fetched from the currently-`champion`-aliased run** (Phase 7 `register.py`) — never against a fixed path or a re-derived baseline, so a rollback automatically restores the matching reference; alerts (Prefect notification) when PSI > 0.2 on any top-5 feature. **Prevalence is monitored alongside feature PSI and routes differently:** a sustained prevalence shift is a *re-calibration* trigger, not a retrain trigger (§9 limitation #12 — they are different remedies).
  - **⚠ Open detail: what makes a prevalence shift "sustained" is not yet specified.** Feature drift has a concrete number (PSI > 0.2); prevalence does not — the routing rule above names the concept but not the test. Prevalence is a single proportion, not a multi-bin distribution, so a PSI-style magnitude heuristic isn't the natural fit here the way it is for features; a two-proportion comparison (a confidence interval on the current-window churn rate against `drift_reference.json`'s stored prevalence, required to persist across multiple measurement windows rather than firing on one noisy reading) is a reasonable candidate when this module is actually built, but it is not yet a committed design.
  - **⚠ Open gap: no recalibration pipeline exists that is actually distinct from a full retrain.** The rule above says a prevalence shift should trigger recalibration rather than retraining, but nowhere in this plan is there a `recalibrate.py`-style flow analogous to `retrain.py` or the `rollback_champion()` helper — the only place `calibrate.py` runs at all is as one step inside `retrain.py`'s full cycle (`train → calibrate → evaluate → register`). As specified today, "recalibrate" and "retrain" resolve to the same pipeline, which defeats the reason for naming them separate remedies in the first place — recalibration is supposed to be the cheaper, faster fix, not a relabelled full cycle. Closing this gap needs a lightweight flow that reuses `calibrate.py` against the frozen, already-trained champion spec (no `train.py` refit, no feature or hyperparameter changes) and re-derives the threshold's contact-rate figure against the new calibration map — but that flow is not yet designed.
- `pipelines/performance_check.py` — **realized-performance feedback loop (closes the ML loop — `input` drift is only a proxy for what we actually care about).** PSI and prediction-distribution drift detect that *inputs* moved; they do not measure whether the model is still *right*. For churn the ground truth does arrive — observed weeks/months later — so this flow joins **logged predictions → realised outcomes** once labels mature and computes **realised PR-AUC** (plus Brier and the decile-lift table) over the matured cohort — **both aggregate and sliced by the same Phase 7 robustness axes** (`contract_type`, `tenure_cohort`, `internetservice`), so segment-concentrated decay surfaces before it drags the aggregate down (the production analogue of Phase 7's V1 segment-collapse diagnostic — the continuous guard against subgroup collapse that a single sealed-test snapshot could never power) — comparing each to the Phase 7 offline estimate. **This is also where V2/V2b's enforcement actually lives**: the same matured cohorts are sliced by the **fairness axes** (`gender`, `seniorcitizen`, `has_partner`, `dependents`), computing realised equal-opportunity and demographic-parity gaps at the shipped `t*` (V2) and realised per-group calibration slope (V2b), each compared to its Phase 7 test-set baseline using the same segment-CI-derived alert-band approach as the robustness axes below — turning V1/V2/V2b from a one-time reported diagnostic (`ANALYSIS.md` §0) into a continuously-enforced one, once production volume supplies the power a single snapshot can't. **The baseline is Phase 7's *test-set* slice metrics, not its dev-OOF ones** — Phase 7 produces both, and they serve different purposes: the dev-OOF slices are non-gating diagnostics reported at promotion time (chosen for statistical power), while the test-set slices are the published metrics of record and what the model card told stakeholders. Production monitoring compares against what was published, so one number means one thing. **The thin-support caveat propagates:** a segment whose test baseline carries a wide CI (the two-year `contract_type` tier, ~10 churners) cannot support a tight degradation alert — derive that segment's alert band from its baseline CI rather than a fixed global threshold, or it will page on noise indefinitely. Mechanics: predictions are logged with `customerid` + `request_id` + timestamp (Phase 9); a `prediction_outcomes` table in Postgres records the eventual churn label per `customerid`; the flow runs on a **label-maturity cadence** (e.g. monthly, after the observation window closes), scoring only the cohort whose outcomes are now known. **Delayed-label discipline:** never score a cohort before its outcome window matures (premature labels bias the metric optimistic); the maturity lag is a configured constant. This realised number — not PSI — is the **authoritative health signal** and the primary, *performance-based* retrain trigger (see below).
  > **📌 Retraining is trigger-based, not only scheduled.** The weekly cron is the *floor*; a sustained realised-PR-AUC drop below a configured fraction of the Phase 7 baseline (or a sustained PSI breach) should *trigger* `retrain.py` off-cycle. Schedule-only retraining either wastes compute when nothing moved or reacts too slowly when something did. Connect `performance_check.py` / `drift_check.py` alerts to an event-triggered run of the retrain flow, with the weekly schedule as the backstop.
  > **📌 Two breach→response edges, not one — a fresh promotion and a stale champion need different remedies.** The trigger above wires drift/performance breaches to `retrain.py`; that is the correct *steady-state* remedy (the world moved, learn from it), but it is the wrong tool in the window right after a `champion` flip. There the previous champion is still in the registry tagged `promotion_status: promoted`, known-good, and one `rollback_champion()` away (Phase 7 `register.py`) — so the remedy for *"the model we promoted an hour ago is misbehaving"* is **revert to the incumbent**, not retrain, which takes a full cycle and, if the promotion itself was the fault (a silently-failed calibration, a data-quality regression), may just reproduce it. So promotion **opens a monitored bake-in window** (a configured N batches / T hours, keyed off `register.py`'s `promoted_at` marker) rather than ending at the smoke check: a breach *inside* the window routes to `rollback_champion()`; a breach *outside* it routes to the retrain trigger above. This closes the lifecycle loop — **promote → watch → (trust | revert)** — and reuses the rollback helper Phase 7 already builds rather than inventing a second revert path. **⚠ On this static dataset the window's live watch is a legible stub, not fabricated** — same discipline as `require_traffic_validation`: the routing distinction and the rollback path are real and testable; the anomaly-triggered auto-revert on live traffic is explicitly *not* wired to fake data (§9 #13). Auto-reverting on a monitoring blip also risks promote↔rollback flapping, which is why even in production this edge is gated rather than fully automatic — the demonstrable value recorded here is the *design*: that the two remedies are distinct and which one the post-promotion window selects.
- `pipelines/batch_predict.py` (optional) — nightly scoring of all customers; writes results to a `predictions` table in Postgres
  > **📌 This is the vehicle that makes the `require_traffic_validation` seam (above) actually runnable — batch shadow scoring.** The traffic-validation gate between the promotion decision and the `champion` flip is a no-op stub without something to run in it; batch scoring is where it runs, because scoring a whole cohort offline lets you compare two models directly with zero traffic-splitting infrastructure. Concretely: score each batch with **both** the incumbent `champion` **and** the staged candidate (resolved via the `shadow` alias the parameterized `promote_to_alias` from Phase 7 `register.py` sets). **Only `champion`'s outputs drive real actions; the candidate's are recorded, never acted on** — this is shadow mode, so the blast radius is exactly zero. Compare the two on **score-distribution PSI** (built on the same quantile-binning machinery `models/drift_reference.py` already uses for the champion's score baseline) and **decision-set parity** — the Jaccard overlap of who crosses `t*`, i.e. the actual contact list, computed via `economics.py` and the shipped threshold. A large unexpected swing in *who gets contacted* is the batch analog of a canary going red, and it is the failure that costs campaign budget — which no offline gate or latency check would surface. **No labels are needed**, so unlike the realised-performance loop this runs immediately.
  > - **⚠ Same honesty discipline as the retrain gate — this is a mechanism demonstration, not live traffic.** The dataset is the static 7,043-row snapshot with no live feed (`CLAUDE.md`; the open-question notes above), so "each batch" is the same frozen rows every cycle. The shadow *plumbing* — dual scoring, PSI, decision-set diff, alerting — is real and correct; the *statistical claim* that it is watching production traffic is not. Structure it so the comparison is genuine and the absence of a live feed is legible, exactly as line 966 requires of the `require_traffic_validation` gate: **do not fabricate traffic to make the shadow diff look live.** Uplift measurement (A/B on realised churn) stays out of scope — it needs maturing labels this dataset never produces (§9 #13) and is noted as a next step only.

**Verification:** Trigger each flow from the Prefect UI; inject synthetic rows with shifted `MonthlyCharges`; confirm the drift alert fires within the next scheduled drift check run. **Rollback-fidelity check (the reason the reference is version-scoped):** with two promoted versions, re-point `champion` at the older one and confirm `drift_check.py` now measures against *that* version's `drift_reference.json` and its alert state returns to baseline — a run that still reports drift after a rollback is reading a fixed path, which is the bug this design exists to prevent. `grep -rn "reports/drift_reference" pipelines/ src/telco_churn/` returns nothing: no code reads the mirrored copy, it is for humans only. Seed `prediction_outcomes` with a matured cohort of known labels and confirm `performance_check.py` computes a realised PR-AUC, compares it to the Phase 7 baseline, and (when forced below the configured fraction) fires the performance-based retrain trigger.

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
- `.github/workflows/data-quality.yml` — weekly cron: run Pandera validation against the git-tracked raw CSV, alert on failure. **No `dvc pull`, no AWS credentials** — the S3 remote does not exist until Phase 12, and this workflow must not depend on it. (An earlier draft specified `dvc pull` from S3 here, inverting the phase order.)
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
- DVC remote added: local cache → S3 (`dvc remote add -d storage s3://telco-churn-data`). **Additive, not a migration.** Nothing before this phase depends on a remote (see Phase 8), so this adds shared caching and off-machine artifact storage without any earlier phase's CI or fresh-clone path changing behaviour. The raw CSV stays in git; the remote holds regenerable stage outputs.
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
  - **Prediction distribution** — rolling histogram vs. the champion's score reference from `drift_reference.json` (Phase 7). That baseline is built from **out-of-sample** probabilities by construction; a panel baselined on in-sample scores would show a permanent phantom drift signal, since in-sample scores are systematically sharper than anything production will produce.
  - **Feature drift** — PSI per top-5 feature over time (sourced from Evidently reports)
    - **⚠ Open gap: how a daily batch job's Evidently output actually becomes a Prometheus time series is not specified.** Prometheus is pull-based — it scrapes metrics from a live endpoint on its own schedule — but `drift_check.py` (Phase 10) is a one-shot Prefect flow, not a long-running service, so there is no obvious endpoint for Prometheus to scrape between runs. The usual fix is a **Pushgateway** (a small intermediary Prometheus pulls from, that batch jobs push their metric values to after each run) or having `drift_check.py` write to a file-based exporter Prometheus is configured to scrape — but neither is named anywhere in this plan. Both named endpoints of the chain (Evidently computes the number; Grafana visualizes it) are specified; the connecting mechanism between a cron-scheduled batch job and Prometheus's pull model is not.
  - **Feature-importance stability** — `shap_importance_<feature>` per retrain cycle (logged by Phase 7's `error_analysis.py`, and by every Phase 10 cycle since it runs the same module). **This catches a failure PSI cannot:** the input distribution can be perfectly stable while the *model's reliance* on a feature shifts sharply — an upstream schema change, a silently broken join, a leaked column. A champion whose top feature moves from `contract_type` to `paymentmethod` between two Sundays has had something happen to it, and no distributional check would flag it.
  - **Realised performance** — realised PR-AUC / Brier over each matured cohort vs. the Phase 7 offline baseline — **aggregate plus one series per Phase 7 robustness segment, plus the V2/V2b fairness-gap and per-group-calibration series** (sourced from `performance_check.py`, which is where V1/V2/V2b's continuous enforcement actually lives — see Phase 10); the panel that distinguishes "inputs moved" from "the model is actually degrading." Updates on the label-maturity cadence, not in real time — annotate it as such so a flat line between updates is not misread as staleness.
  - **Threshold agreement** — `within_ci` (closed-form `t*` vs. the bootstrap CI of the empirical argmax-EV threshold) per retrain cycle, sourced from Phase 6 `threshold.py`'s diagnostic (today logged only, per `run_threshold_step`) via Phase 10's weekly `retrain.py` run. Updates on the retrain cadence, not in real time — same staleness caveat as the realised-performance panel above.
- Alert rules: p95 latency > 500 ms; error rate > 1 %; PSI > 0.2 sustained over 24 hours; **realised PR-AUC below a configured fraction of the Phase 7 baseline for one matured cohort — aggregate or any monitored segment** (the performance-based retrain trigger — see Phase 10 `performance_check.py`); **a realised V2 fairness-gap or V2b per-group-calibration series breaching its Phase 7 baseline-CI-derived band** (V1/V2/V2b's continuous enforcement, replacing the one-time reported diagnostic — see Phase 10 `performance_check.py`); **`within_ci` false for N consecutive retrain cycles** (candidate resolution 3 of the Phase 10 open question on threshold inheritance — see `pipelines/retrain.py` bullet — surfacing disagreement here is the "compute in Phase 10, surface in Phase 13" split already used for realised PR-AUC, and doesn't itself resolve whether disagreement should also auto-trigger or block within `retrain.py`)
- `src/telco_churn/utils/logging.py` updated so `log_level` is read from `LOG_LEVEL` environment variable (fallback `"INFO"`), enabling temporary debug logging in production without a redeploy

**Verification:** Grafana dashboards populate within minutes of the API receiving traffic; simulated drift raises a PSI alert; load test shows p95 latency panel updating live.

---

### Phase 14 — Documentation Polish *(1 day)*

**What this achieves:** A recruiter or hiring manager can understand the project's scope, results, and architecture within 90 seconds of landing on the repo.

**Deliverables:**
- `README.md` — top of file: 1-paragraph elevator pitch, architecture diagram, "Quick demo" GIF, tech-stack table with phase links, headline metrics table. **The business-impact figure is quoted as the three-scenario range with its driver named** (`r`, the retention rate — a benchmark guess, not a model output; `ANALYSIS.md` §0), never as a bare base-scenario point estimate. A confident single dollar figure that dissolves under one question is a weaker artifact than an honest bracket you can explain.
- `ANALYSIS.md` — full modelling narrative (already written; verify it references `src/` functions, not notebook code)
- `docs/runbook.md` — how to retrain, roll back to the last promoted champion (`python -m telco_churn.models.rollback`, which selects the highest `promotion_status: promoted` version — never a hand-typed version number; wraps Phase 7 `register.py`'s `rollback_champion()`), debug a drift alert, **prune stale `pending` registry orphans** (a manual utility that reaps only versions tagged `promotion_status: pending` *and* older than one training cycle — the crash residue the fail-safe mint-time default deposits by design; never touches `promoted`/`rejected`/`dev` versions, never selects on version number — see CLAUDE.md § MLflow Model Registry), and **triage a bad-prediction incident** (forensic loop: pull the offending `request_id` / `customerid` from the structured logs, re-run SHAP-local on the case, and classify root cause as relabel vs. feature-gap vs. drift → feed the verdict to the retrain backlog)
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
5. `notebooks/_archive/EDA-original.ipynb` is frozen and not retroactively split or edited — kept only as a comparison notebook for the original exploratory pass. `src/` and `ANALYSIS.md` are the source of truth wherever later phases diverge from it.

| Notebook | Phase | What It Demonstrates |
|---|---|---|
| `00-data-ingestion.ipynb` | 2 | Raw CSV → 5 Pandera gates → Postgres ingest; example violations |
| `01-eda.ipynb` | 3 | Statistical tests, distributions, churn-rate breakdowns |
| `02a-feature-discovery.ipynb` | 4a | Structured feature search: domain hypotheses + OOF blind-spot profiling → adoption gate, incl. a rejected decoy |
| `02b-feature-engineering.ipynb` | 4b | Distribution / justification view of the 4a-adopted feature set |
| `03a-model-selection.ipynb` | 5 | Model-family selection: baseline comparison + paired-bootstrap decision + 2c/2d loop |
| `03b-feature-selection.ipynb` | 5 | Permutation-importance selection experiment: full-set CI → decoy-referenced ranking → reduced refit → SHAP audit → keep/drop |
| `03c-hyperparameter-tuning.ipynb` | 5 | Optuna study summary + bias/variance progression + tuned-model logging confirmation |
| `04-calibration-and-threshold.ipynb` | 6 | Reliability diagrams + 3-scenario cost curves |
| `05-evaluation-and-error-analysis.ipynb` | 7 | Renders `metrics.json` + `error_analysis.json` (sealed-test evaluation, error concentration/confidence, value-weighted errors, SHAP, reported dev-OOF diagnostics V1/V2/V2b); hosts the V3 human review and stamps the verdict |

---

## Critical Files and Their Roles

| File | Role |
|---|---|
| `notebooks/_archive/EDA-original.ipynb` | **Comparison notebook for the original exploratory pass** — not a source of truth; `src/` and `ANALYSIS.md` are authoritative wherever they diverge. Frozen; do not alter it. |
| `README.md` | Master narrative and project landing page; updated each phase |
| `ANALYSIS.md` | Full modelling rationale, hyperparameters, error analysis, and business impact |
| `pyproject.toml` | Dependencies + all tool configuration (single source of truth) |
| `dvc.yaml` | Reproducible pipeline graph |
| `docker-compose.yml` | One-command local stack for development |
| `configs/config.yaml` | Hydra root; controls data paths, thresholds, MLflow URI, random seed |
| `src/telco_churn/data/schema.py` | Pandera schema — referenced by validation, training, and FastAPI request models |
| `src/telco_churn/models/train/` | Optuna + LightGBM + MLflow; called by both CLI and Prefect |
| `src/telco_churn/serving/app.py` | FastAPI app; loads `champion` model from MLflow Registry at startup |
| `pipelines/retrain.py` | Continuous-training DAG; ties together every `src/` module |
| `.github/workflows/cd.yml` | Deploy automation; ties code changes to AWS |

---

## Reuse from the Original Notebook

This table maps early-phase artifacts from `notebooks/_archive/EDA-original.ipynb` to their `src/` destination — a migration record, not a constraint. Later phases have diverged where a stated reason justified it (see `ANALYSIS.md` and `CLAUDE.md`'s Source of Truth section):

| Notebook artifact | Destination in `src/` |
|---|---|
| 5 data quality gates | `data/schema.py` + `data/checks.py` |
| `ColumnTransformer` definition | `features/preprocessing.py` → `models/train/` (Phase 5) — fitted per CV fold on development; production path is `tree_preprocessor` + LightGBM in an sklearn `Pipeline` (linear baselines use a separate `linear_preprocessor`, Step 1) |
| Optuna best hyperparameters | Default values in `configs/tuning/optuna.yaml` (still searchable; warm-start from these) |
| Cost-sensitive threshold logic (3 scenarios) | `models/threshold.py` |
| Bootstrap CI evaluation routine | `models/evaluate.py` |
| Lift / gains curves + decile lift table (§15, §16.2) | `models/evaluate.py` (Phase 7) → `reports/metrics.json` + `reports/figures/` |
| Business-impact / EV figure (§16.3) | `models/evaluate.py` (Phase 7) → README headline metric — reported as the **three-scenario range**, not a single base-scenario point estimate; parameter spread and bootstrap CI kept as distinct error bars |
| Bias/variance + McNemar diagnostics (§10.6, §10.8, §11.1) | `notebooks/03a-model-selection.ipynb` (Phase 5) — notebook-only, gates nothing |
| 8 documented limitations | README "known limitations" section; surfaced as Grafana alert thresholds where relevant |

---

## End-to-End Verification (post Phase 14)

1. **Reproducibility:** Fresh clone → `uv sync` → `docker compose up` → `dvc repro` → metrics within bootstrap CI of `README.md`. **No `dvc pull` step, and none needed:** the raw CSV is git-tracked and every downstream artifact is a regenerable stage output (Phase 8). This is the claim that fails silently if `datasets/raw/` is ever gitignored — verify it on a clone into a clean directory, not on the working tree.
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
