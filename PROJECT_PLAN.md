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
│   ├── policy/threshold.yaml   # shipped operating point + provenance (written by Phase 6)
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
│   │   └── split.py            # canonical dev/test partition, sealed before Phase 4a
│   ├── features/
│   │   ├── sql_features.py     # runs sql/features/*.sql via SQLAlchemy
│   │   ├── build.py            # column group exports (raw IBM columns + charge_per_service)
│   │   ├── preprocessing.py    # shared ColumnTransformer builder (Phase 4a; reused by train.py)
│   │   ├── generate.py         # error-driven discovery machinery: OOF profiler, gate, bootstrap CI (Phase 4a)
│   │   └── io.py               # load_features() + sha256 content hash; owns path & format (Phase 6)
│   ├── models/
│   │   ├── train.py            # Optuna + LightGBM + MLflow
│   │   ├── calibrate.py        # CalibratedClassifierCV
│   │   ├── threshold.py        # closed-form t* = c/(r*LTV) + argmax-EV agreement check
│   │   ├── evaluate.py         # test-set metrics + bootstrap CIs
│   │   ├── plots.py            # pure helpers: reliability bins, EV curves, r-sensitivity (Phase 6)
│   │   ├── refit.py            # full-data production refit (Phase 7, after evaluate)
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
│   ├── 03a-model-selection.ipynb     # Phase 5 — model-family selection: baseline comparison + paired-bootstrap decision
│   ├── 03b-feature-selection.ipynb  # Phase 5 — permutation-importance selection experiment
│   ├── 03c-hyperparameter-tuning.ipynb # Phase 5 — Optuna study (history, parallel-coords, importance) + model logging
│   ├── 04-calibration-and-threshold.ipynb  # Phase 6 — reliability diagrams + cost curves
│   └── 05-error-analysis.ipynb      # Phase 7 — SHAP + FN/FP analysis
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
| 6 | Calibration + cost-sensitive threshold | `models/calibrate.py`, `models/threshold.py` | Not started |
| 7 | Evaluation + error analysis + full-data refit + registry promotion | `models/evaluate.py`, `models/refit.py`, `models/register.py`, `notebooks/05-error-analysis.ipynb` | Not started |
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
5. **Tune LightGBM with Optuna** — 50 TPE trials on PR-AUC after the freeze; `n_estimators` resolved per fold by early stopping, not searched; a 1-SE selection rule over raw argmax picks the most-regularized trial within noise of the best.
6. **Log the tuned pipeline to the run — do not register it** — the full `[preprocessor → model]` `Pipeline` logged as pyfunc with `pyfunc_predict_fn="predict_proba"`, a signature inferred from that same probability output, and a log→reload→`predict_proba` parity check, retrievable at `runs:/<run_id>/model`; `training_manifest.json` records the engineering audit trail (git SHA, DVC hash, hyperparameters, feature space/columns, CV PR-AUC, paired-Δ vs. LogReg). Uncalibrated and un-thresholded — not serving-ready, therefore **not a registry version**. See `CLAUDE.md` § *Run artifacts vs. registry versions*: the registry holds only valid rollback targets, and an uncalibrated pipeline is a stage of construction, not a deployable model. Phase 6 `calibrate.py` performs the single registration of this training cycle, on the calibrated artifact, and points `challenger` at it.

**Deliverables:**

*Configs:* `configs/training/lightgbm.yaml` (searched ranges + fixed determinism/imbalance knobs), `configs/training/logreg.yaml`, `configs/training/selection.yaml`, `configs/tuning/optuna.yaml` (`n_trials`, sampler, pruner, `selection_rule`).

*Source:*
- `src/telco_churn/features/preprocessing.py` — adds `build_linear_preprocessor` alongside the existing tree-family `build_preprocessor`
- `src/telco_churn/features/schema.py` — adds a frozen `FeatureSchema` dataclass owning the binary/multi_cat/numeric column groups
- `src/telco_churn/data/split.py` — canonical dev/test split (sealed once, order-invariant), imported by every downstream consumer
- `src/telco_churn/models/train/` — one module per step (`candidates.py`, `comparison.py`, `feature_freeze.py`, `tuning.py`, `registration.py`) plus shared `common.py`; Optuna trials nested under a `tuning_study` parent run. `registration.py` logs the pipeline and manifest onto the `tuning_study` run but passes no `registered_model_name=` and sets no alias — those two lines move to Phase 6 `calibrate.py`
- `src/telco_churn/features/select.py` — the permutation-importance selector, wrapped in a `Pipeline` so it refits leak-free per fold
- `src/telco_churn/models/diagnostics.py` — pure helpers for the 2c/2d loop (`generalization_gap`, `learning_curve_points`, `segment_oof_errors`) and Step 2's fairness/robustness diagnostics
- `docker/mlflow/Dockerfile`, `sql/schema/000_create_mlflow_db.sql` — Postgres-backed MLflow tracking server

*Tests:* `tests/unit/test_train_*.py`, `test_select.py`, `test_diagnostics.py`, `test_split.py` — leakage-canary assertion, run-twice determinism, leak-free-refit proofs, synthetic planted-signal/noise fixtures; `tests/integration/test_train_subprocess.py` — full CLI composition path via Hydra fast-path config overrides.

*Notebooks:* `03a-model-selection.ipynb` (candidate comparison, bias/variance loop, fairness/robustness panel), `03b-feature-selection.ipynb` (selection experiment), `03c-hyperparameter-tuning.ipynb` (Optuna study, full→reduced→tuned progression, model-logging confirmation).

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and logs a tuned pipeline at `runs:/<run_id>/model` with the reload-parity assertion passing and `training_manifest.json` attached; `telco-churn-pipeline` has **no new registry version** at the end of this phase (the registry stays empty until Phase 6 registers the calibrated artifact). LightGBM clears the Dummy floor and beats/ties LogReg under the decision rule; full `pytest` green. See `ANALYSIS.md` §4 for the recorded results and rationale, `CHANGELOG.md` `[0.5.0]`–`[0.5.3]` for delivery history.

---

### Phase 6 — Calibration + Cost-Sensitive Threshold *(2 days)*

**What this achieves:** The model outputs calibrated probabilities (not raw scores), and the decision threshold is set to reflect the actual business cost of a missed churner versus a wasted retention call — not the default 0.5. Three cost scenarios are evaluated so the business can choose the level of risk they are comfortable with.

**Deliverables:**
- `src/telco_churn/models/calibrate.py` — `CalibratedClassifierCV(ensemble=False)` wrapping the **unfitted** `[preprocessor → LightGBM]` `Pipeline` (not the fitted model, and not pre-transformed data — the `ColumnTransformer` must refit inside every calibration fold or the imputer/scaler statistics leak across them), **cross-fit on the development set** — no static val holdout (see `src/telco_churn/data/split.py`): the calibrator is trained on **out-of-fold predictions** (each row calibrated by a model that did not train on it) while the base LightGBM is refit on all of development. Uses 100 % of development for both base fitting and calibration via rotation, so nothing is spent on a static val set. `cv=StratifiedKFold(5, shuffle=True, random_state=42)` passed explicitly — the bare `cv=5` default is unshuffled and violates the project's determinism invariant. Logs the calibrated pipeline as a **new registered model version** with `pyfunc_predict_fn="predict_proba"`, a matching probability signature, and a log→reload→`predict_proba` parity check (the pattern established in the bridge's Step 2), and points the `challenger` alias at it (see the registration flag below).

  **Where the unfitted `Pipeline` comes from — say it here, because this is where a reader looks.** `sklearn.clone(mlflow.sklearn.load_model(manifest["logged_model_uri"]))`. Resolve by the `models:/m-<id>` URI recorded in Phase 5's `training_manifest.json`, **never** by `runs:/<run_id>/model` — that URI is ambiguous once this module logs its own artifact onto the same run. The `clone()` contract (bridge, *Two design contracts*) exists so the pipeline spec has a single construction path; rebuilding it from `best_params` is the failure it forbids.

  **`serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE` is mandatory, not inherited style.** mlflow's skops default refuses to serialize a `CalibratedClassifierCV`, naming `sklearn.calibration._CalibratedClassifier` and `_SigmoidCalibration` as untrusted types. Phase 5 already pins cloudpickle for LightGBM's `Booster`; Phase 6 hits the same wall for a second reason.

  **Method selection — both methods implemented, sigmoid is the incumbent and isotonic must earn the switch.** Sigmoid is strictly monotone (PR-AUC-preserving *within a fold*), two-parameter, and immune to the sparsity artifact; isotonic is more flexible but collapses scores into ties, which can degrade ranking. The comparison is worth running rather than short-circuiting: the archived reliability curve shows over-prediction in 0.22–0.45 and *under*-prediction above 0.58 (`ANALYSIS.md` §7) — a shape a two-parameter sigmoid structurally cannot fix and isotonic can — so which method wins is a genuinely open empirical question. Cost: each `CalibratedClassifierCV(ensemble=False)` fit is 5 inner fits **plus a refit on all rows** (6, not 5), so 6 × 5 outer × 2 methods = 60 pipeline fits, plus 5 for the uncalibrated baseline = **65**. Minutes of compute on ~5,600 rows.
  - **Brier is scored on outer-CV OOF probabilities, never on the calibrator's own fitting rows.** `cross_val_predict(CalibratedClassifierCV(pipeline, method=m, cv=inner), X_dev, y_dev, cv=outer, method='predict_proba')` per method. Scoring Brier on the rows the calibrator was fit on would select isotonic by construction — a flexible step function with ~5,600 fitting points always beats a two-parameter sigmoid on its own training data. The winning method's pooled outer-OOF vector is the exact input `threshold.py` consumes, so the honest comparison is nearly free. The uncalibrated baseline the gate compares against is `cross_val_predict(pipeline, X_dev, y_dev, cv=outer)` on the **same `outer` object** — the comparison is paired or it is nothing.
  - **⚠ Hard gate, applied first — and it must be scored per fold, not on the pooled vector.** Disqualify isotonic outright if its PR-AUC falls more than `training_setup.delta_threshold` (0.005) below the uncalibrated one. Ranking degradation is a one-metric-invariant violation and no Brier improvement buys it back.

    **PR-AUC for this gate is the mean of per-fold average precision, `mean_k AP(y[test_k], p[test_k])`.** Sigmoid's monotonicity guarantee holds *inside* a fold; the pooled outer-OOF vector concatenates five folds each mapped by a **different** sigmoid, and a rank order preserved under each of five monotone maps is not preserved under their mixture. Pooling therefore (a) voids the sigmoid guarantee and (b) — much worse — *dilutes* isotonic's ranking damage, because the pooled AP averages over five different tie structures. Measured on a synthetic fixture matched to this problem's shape (n = 5,600, ~30 % prevalence, `class_weight='balanced'` LightGBM, 5×5 nested folds):

    | ΔPR-AUC vs. uncalibrated | pooled outer-OOF | per-fold mean |
    |---|---|---|
    | sigmoid | −0.0005 | **+0.000000** |
    | isotonic | −0.0046 → **passes** | −0.0172 → **fails** |

    Pooled, isotonic loses 0.0046, slips under Δ\* = 0.005, and is admitted. Per fold it loses 0.0172 — 3.4× the materiality threshold. **The pooled gate would admit precisely the model it exists to reject**, and would do so while reporting that sigmoid is not structurally safe either. Per-fold scoring restores both properties exactly (within-fold Spearman of sigmoid against raw scores is 1.0 in every fold). Magnitudes are fixture-specific; the mechanism is structural. Brier stays pooled — it is a per-row proper score and pooling is legitimate there.
  - **Then Brier decides, with a CI:** switch to isotonic only if a **paired bootstrap on the per-row Brier difference** (reuse `models/train/comparison.py`'s Step 2 machinery) yields a 95 % CI excluding zero in isotonic's favour. A point-estimate comparison at this sample size flips on noise. A paired CI is used rather than a fixed Brier materiality delta because Brier's scale (~0.13 here) has no natural analogue to Δ\*=0.005 on PR-AUC — inventing one would need its own justification, whereas the paired-CI idiom is already the repo's family-decision rule.
    - **⚠ Resample outer folds, not rows.** Rows within an outer fold share a calibrator, so their Brier differences are correlated and an iid row bootstrap understates the variance — biasing the CI toward excluding zero, i.e. toward switching to isotonic, which is the direction the gate is trying to make hard. Block-bootstrap by outer fold. *(The same objection applies to `comparison.py`'s Step 2 bootstrap, which is being reused here; worth knowing before it is inherited.)*
  - **Why Brier is the right second metric, not an arbitrary one.** Murphy's decomposition: `Brier = reliability − resolution + uncertainty`. The PR-AUC gate above bounds how far *resolution* can move, and *uncertainty* is a property of the data. Under the gate, ΔBrier therefore isolates the **reliability** term — which is exactly the quantity calibration exists to improve. The gate and the decision metric are not two independent hurdles; the first is what makes the second interpretable.
  - **Log both methods regardless** — per-fold mean PR-AUC, pooled outer-OOF Brier, ECE, and the ΔBrier CI for each. **Pin ECE's binning in config** (`n_bins`, `strategy="quantile"`): ECE is binning-sensitive and biased, and since it gates nothing it only needs to be *comparable across runs*. The loser's numbers are what make the winner's selection legible in `ANALYSIS.md` §7.

  **Config knob: `calibration.method: sigmoid | isotonic | auto`. Run `auto` exactly once, here, as a human-reviewed decision; pin the winner in `configs/` afterwards.** Under the Phase 10 weekly retrain, `auto` must **not** stay on: the method would flip between runs on Brier noise, silently changing serving behaviour and invalidating the previously-derived threshold in a way the promotion gate cannot catch — sigmoid and isotonic can map the same rank order to materially different probability *shapes* while both passing a PR-AUC + Brier comparison. The comparison code stays live in the retrain flow as a **diagnostic that flags** when the pinned method is now materially beaten, routed to a human; it never auto-switches. Same boundary the plan draws elsewhere: content-hashed transforms are automated, decisions with business-visible consequences are pinned and reviewed. Calibration method is a decision, not a transform.

  > **⚠ Flagged deviation from the archived notebook (per CLAUDE.md):** the notebook uses a *fixed* `method='sigmoid'` (Platt) calibrator (§14.1). Selecting sigmoid-vs-isotonic by an outer-CV Brier comparison under a pre-registered gate is a deliberate methodological change — not a transcription. Reproducing the notebook's assertion is not a deviation worth documenting; producing evidence for it is. **Document the method actually chosen**, both methods' outer-OOF Brier/PR-AUC/ECE, and the ΔBrier CI in `ANALYSIS.md` §7. If the result is sigmoid, the outcome matches the notebook and the deviation is moot — but it is now an empirical finding rather than an inherited prior. The notebook's threshold logic and evaluation math are otherwise preserved verbatim.

  > **📎 Class-weight interaction (state in the module docstring and `ANALYSIS.md` §7):** the base LightGBM ships with `class_weight='balanced'` (`models/train/common.py`), which systematically inflates raw scores toward the positive class. Calibration corrects this because sklearn fits the calibrator on **unweighted** OOF data, mapping the reweighted scores back to true posteriors against the real ~26.5 % prevalence. This is the mechanism that makes the calibrated probabilities valid inputs to the Phase 6 threshold search and the Phase 7 expected-value calculation at all. Corollary for `ANALYSIS.md` §11: the probabilities are calibrated *to development-set prevalence* — prevalence drift in production invalidates both the calibration and the derived threshold. That is the hook for Phase 13 drift monitoring.

  > **🔧 SHAP access path (supersedes `ANALYSIS.md` §7's "fold 0 of `cal_pipe`" note):** with `ensemble=False` there is exactly one internal base estimator, reachable as `calibrated.calibrated_classifiers_[0].estimator` — the uncalibrated `Pipeline` that Phase 7's error analysis needs for `TreeExplainer`. This replaces the archived notebook's "extract fold 0 of a 5-clone ensemble" workaround; update `ANALYSIS.md` §7 when this phase lands.

  > **📌 This phase owns the training cycle's single registration** (see `CLAUDE.md` § *Run artifacts vs. registry versions*). Phase 5 logs its tuned pipeline to the `tuning_study` run and stops there; the uncalibrated artifact is never a registry version, because a registry version must be a valid rollback target and an uncalibrated pipeline is not one — Phase 9's FastAPI service loads by alias, so an uncalibrated version reachable by `set_registered_model_alias(champion, version=N-1)` would restart cleanly and silently serve raw `class_weight='balanced'` scores against a threshold derived from calibrated probabilities. Nothing errors; nothing alerts. `calibrate.py` therefore calls `mlflow.sklearn.log_model(..., name="calibrated_model", registered_model_name="telco-churn-pipeline")` on the fitted `CalibratedClassifierCV` — onto the same `tuning_study` parent run, with the same signature + reload-parity assertion `log_model.py` performs — creating **version 1**, tags it `refit_scope: dev`, and points `challenger` at it. Phase 7's `register.py` evaluates and promotes *that* version to `champion`. An alias is a unique pointer per model name: re-aliasing never creates a second challenger, it silently orphans the previous version, which is precisely why the registration happens once and happens here.

  > **📌 Gate registration on `tuning_summary.trial_count_below_threshold` (Phase 5→6 Bridge finding).** Phase 5's `tuning.py::run_tuning_step` logs a warning and persists `trial_count_below_threshold` (bool) + `min_completed_trials` into `training_manifest.json`'s `tuning_summary` when fewer than `cfg.tuning.min_completed_trials` Optuna trials complete — but deliberately does not block anything there. Step 5 (`log_model.py`) logs every training cycle regardless of quality, per `CLAUDE.md`'s "log everything for audit, register nothing yet" boundary (§ *Run artifacts vs. registry versions*), so an untrustworthy tuning result must still make it into the run for inspection. `calibrate.py` is the first step that turns a tuned pipeline into a **registered** (`challenger`) model, which makes it the correct chokepoint: **read `manifest["tuning_summary"]["trial_count_below_threshold"]` before registering, and abort (or require an explicit override flag) if it is `true`.** Log the decision as its own structured event — distinct from Phase 7's `model_promoted`/`model_rejected` — since this is a data-quality gate on the tuning result, not a performance comparison against the incumbent champion.
  >
  > **This is the same class of gap `data/checks.py`'s `Severity.WARNING`/`Severity.ERROR` split already solves elsewhere.** `validate.py`'s `can_proceed` distinguishes checks that block the pipeline from checks that only inform it. `tuning.py`'s own quality checks (too-few-completed-trials, boundary-hit) have no equivalent tiered mechanism — just a bare `logger.warning()` call with nothing downstream reading it back. Wiring `trial_count_below_threshold` into `calibrate.py`'s registration path is what gives that warning actual teeth, one phase after it fires rather than never.

  > **⚠ `name="calibrated_model"` is load-bearing — do not reuse `name="model"`.** MLflow 3 permits two logged models with the same name on one run and resolves `runs:/<run_id>/model` to **the most recently logged**. Reusing the name silently rebinds that URI from Phase 5's uncalibrated pipeline to this calibrated one, with three consequences: the bridge's verification (`runs:/<run_id>/model` returns a raw pipeline) becomes false the moment this module runs; Phase 7's `TreeExplainer` path resolves a `CalibratedClassifierCV` where it expects a `Pipeline`; and — sharpest — `calibrate.py`'s own `clone()` source becomes its own output, so **a second run of `calibrate.py` wraps a `CalibratedClassifierCV` inside another one**, then fits, logs, and registers it without raising. Distinct names plus resolution via `manifest["logged_model_uri"]` close all three.
- `configs/calibration/default.yaml` — `method: sigmoid | isotonic | auto`, the `outer`/`inner` fold counts, and ECE's `n_bins` / `strategy`. Registered under Hydra `defaults` → `cfg.calibration`.
- `configs/costs.yaml` — the three cost scenarios as data, not code. Ports §14.3's **parameter derivation**, which is sound and should be kept verbatim: ARPU from churner `MonthlyCharges` quantiles (P25 / median / P75, development set only), `LTV = ARPU × gross_margin(0.60) × horizon(12 mo)`, `c = outreach + ARPU × discount_rate × 3 mo`, and a retention rate `r` from the telecom benchmark range (0.20 / 0.30 / 0.40 — not observable in this dataset, and flagged as an assumption, not a measurement).

- `src/telco_churn/models/threshold.py` — **the operating threshold is `t* = c / (r × LTV)`, in closed form.** Contact a customer iff the expected value of contacting exceeds doing nothing: `q·r·LTV − c > 0`. Derivation, with utilities taken relative to "customer leaves" = 0 — the `(1−q)·LTV` terms cancel, which is why `LTV` appears only through `r·LTV`:

  | | contact | do nothing |
  |---|---|---|
  | churner (`q`) | `−c + r·LTV` | `0` |
  | non-churner (`1−q`) | `−c + LTV` | `LTV` |

  - **⚠ This supersedes the archived §14.3 threshold, which is derived from a different — and incorrect — objective.** The notebook minimises `Cost = FP·c + FN·r·LTV`. In that formula a **true positive contributes zero**: contacting a churner is free, contacting a non-churner costs `c`. But you do not know which you have when you dial the phone — the agent and the three-month discount are paid either way. The recovery benefit *is* counted (turning an FN into a TP saves `r·LTV`); only the **charge on true positives is missing**, understating the cost of contacting by exactly `q·c`. Cheaper contacts → lower threshold → systematic over-contacting. The general rule: **in the classical cost matrix, cost attaches to errors; in retention, cost attaches to actions.** `t* = C_FP/(C_FP + C_FN)` presumes correct decisions are free, and here they are not.
  - **The archived notebook already contains the correct objective — it just doesn't select from it.** Cell 121's second panel plots `P&L = TP·(r·LTV − c) − FP·c`, which per customer reduces to `q·r·LTV − c` and yields exactly the closed form above. Selection is `np.argmin(oof_costs_by_scenario[...])` — the cost curve. The two panels of one figure encode inconsistent economics. Its own diagnostics show the consequence: §14.3 reports P&L "small but positive" at the chosen threshold (i.e. left of the P&L maximum), and §15 reports the implied 40.1 % contact budget buying marginal lift of **0.99×** — indistinguishable from random — against a KS-optimal cut-off of 34.2 %.

  | Scenario | `c` | `LTV` | `r·LTV` | archived (cost curve) | **this project (`c/(r·LTV)`)** |
  |---|---|---|---|---|---|
  | Conservative — highest *relative* FN penalty (3.65:1) | 22 | 402 | 80.4 | 0.2464 | **0.2736** |
  | Base | 68 | 575 | 172.5 | 0.2956 | **0.3942** |
  | Optimistic — costliest intervention (2.01:1) | 135 | 680 | 272.0 | 0.3596 | **0.4963** |

  - **⚠ Horizon alignment — pin this in `ANALYSIS.md` §0 before defending the number to anyone.** `q` is `P(churn)` over the *label's* window (IBM Telco: churn observed in the last month). `LTV` is defined over a 12-month horizon and `c` includes a three-month discount. `q·r·LTV` is coherent only if the event `q` predicts and the horizon `LTV` values are the same event. This is reconcilable, but it is currently nowhere stated, and §0 is the section that owns prediction unit and label definition.
  - **Empirical argmax-EV over the calibrated OOF probabilities runs as a diagnostic, not as the selector.** `cross_val_predict(method='predict_proba')` over the calibrated pipeline gives leak-free out-of-sample probabilities (*not* raw scores; the sealed test stays untouched for Phase 7). Sweep expected value over a threshold grid and confirm the argmax lands on `t*`. **Under correct calibration they coincide — a material gap is evidence the probabilities are not calibrated**, which is precisely the failure Phase 6 exists to catch. This makes the two halves of the phase check each other, and it costs one `cross_val_predict` you are already running.
    - **⚠ It is a *local* calibration check, at the operating point — not a global one.** The empirical EV curve peaks where `E[y | p ≈ t] = c/(r·LTV)`. It probes calibration in a neighbourhood of `t*` and is structurally blind to miscalibration elsewhere. That is arguably the right place to probe — it is the only region the decision depends on — but it has a direct test-design consequence, recorded under `test_threshold.py` below: a synthetic "miscalibrated" fixture that mangles the tails while leaving `E[y|p] = p` near 0.39 will **pass**, and the test meant to prove the check can fail is then itself broken. Brier and ECE own the global claim; this owns the operating point. State the distinction in `threshold.py`'s docstring.
  - **⚠ `t*` is the unconstrained EV optimum; retention teams have a fixed number of agent-hours.** Log the **implied contact rate** at `t*` — the fraction of development rows with `p > t*` — into `threshold.json`. When that rate exceeds capacity, the correct policy is **not** a higher threshold but **top-K by expected value** `p·r·LTV − c`, which orders customers differently from ranking on `p` alone whenever `c` or `LTV` varies per customer. This phase already faults the archived threshold for implying a 40.1 % contact budget; one logged number is what turns that criticism into an operational guardrail.
  - **⚠ Threshold stability — the 1-SE rule moves from *selection* to *verification*.** The EV curve is flat near its maximum, so a single empirical argmax is noisy: it moves by roughly ±0.03–0.05 across resamples. Under an empirical selector that noise would land directly in the shipped threshold, and the standard remedy (`tuning.py`'s 1-SE rule) would apply. **It no longer does.** `t*` is a closed-form function of `c`, `r`, and `LTV` — it has *zero sampling variance*, so there is nothing to stabilise and nothing to shrink. Adopting "ship the smallest threshold within one SE of the optimum" would now mean deliberately contacting customers whose expected value is negative; the conservatism runs the wrong way. **Ship `t*` exactly.**

    Repurpose the machinery instead. Bootstrap the OOF rows (B = 1,000, resample with replacement, recompute the argmax on each draw) to obtain a **percentile CI for the empirical argmax**, and state the agreement check as: **`t*` must fall inside that CI.** A point-vs-point comparison would fail spuriously on a flat curve; a point-vs-interval comparison has the right null. Log the CI, and report it in `ANALYSIS.md` §8 in place of a bare four-decimal estimate. The CI's *width* is itself the deliverable — it is the business's tolerance on the operating point, and on a flat EV curve a threshold error of ±0.03 costs very little, which is worth saying to whoever has to defend `r = 0.30`.
  - **⚠ Emit a sensitivity sweep over `r`, and treat it as a headline result.** `t*` is inversely proportional to the retention rate, and `r` is the one parameter **not estimable from this dataset** — measuring it requires intervening on customers and observing outcomes. §14.3 says so outright (*"industry benchmark — not observable in dataset"*) and takes 0.30 from a 0.15–0.40 literature range.

    | `r` | 0.15 | 0.20 | **0.30** | 0.40 | 0.45 |
    |---|---|---|---|---|---|
    | `t*` | 0.7884 | 0.5913 | **0.3942** | 0.2957 | 0.2628 |

    Moving `r` from 0.30 → 0.20 moves `t*` by **0.20** — more than any plausible PR-AUC gain would move the operating point. **The most consequential number in the deployment decision is a benchmark guess, not a model output.** Log the sweep to MLflow and plot it to `reports/figures/threshold_sensitivity.png`. State in `ANALYSIS.md` that the threshold is **provisional by construction**, correct *given* `r = 0.30`. *(A curiosity worth noting: `r = 0.40` yields `t* = 0.2957`, essentially the archived 0.2956 — that threshold is defensible only under optimistic retention paired with base-case costs, a combination nobody chose deliberately.)*

    - **⚠ The sweep treats `r` as an unknown scalar. The deeper limitation is that it is an unknown *function*** — belongs in `ANALYSIS.md` §11 alongside the caveat above, not instead of it. The utility table assumes every contacted churner is retained with the same probability. The decision-theoretically correct rule is not `q > c/(r·LTV)` but `τ(x)·LTV > c`, where `τ(x) = P(stay | contact, x) − P(stay | no contact, x)` is an **uplift**, not a churn probability. Thresholding on `q` is optimal only if `τ(x) ∝ q(x)`, and in retention it characteristically is not: the customers likeliest to churn are often the least persuadable (*lost causes*), and some are actively annoyed into leaving by the contact itself (*sleeping dogs*). This is the standard critique of churn-model-drives-retention, and this project is squarely inside it.
    - **⚠ `r` is NOT recoverable from Phase 10's outcome loop as currently designed — and that has to be fixed in Phase 9, not Phase 10.** If everyone above `t*` is contacted, outcomes are observed only for the contacted, and the contacted are selected on `q`. Joining logged predictions to matured labels then recovers a confounded average, not `r`. Identifying `r` — let alone `τ(x)` — requires **randomly withholding contact from a small slice of customers above `t*` and logging that assignment**. The decision must be made when Phase 9 starts serving, because it cannot be applied retroactively to traffic already scored. It is cheap, and it is the only path to the number the entire threshold rests on. Reserve the holdout flag in the Phase 9 prediction log schema.
  - **`t*` is a function of the cost parameters alone — not of the model.** That is the payoff of calibration, and it is why the threshold transfers to the Phase 7 full-data refit with no re-derivation (`models/refit.py`). An empirical argmax threshold would be model-dependent and would not transfer. The dependency direction is still load-bearing: `t*` must be applied to *calibrated* scores, so **calibration precedes thresholding**, and a change of calibration method invalidates the agreement check (though not `t*` itself).
  - **No conflation results:** the operating point is *derived* from cost parameters and *validated* on development OOF probabilities; the final cost/recall is *reported* on the sealed test (Phase 7), which the threshold never saw.
  - **📎 State the decision rule in `threshold.py`'s module docstring** — the utility derivation, the `q·r·LTV > c` inequality, and one sentence on why the classical `C_FP/(C_FP + C_FN)` form is *not* used here (correct decisions are not free; the intervention is paid on every contact). Two defensible formulations exist and they differ by ~0.11 in probability on the base scenario; a reader editing this function must not have to reconstruct which one is implemented, or reach for `ANALYSIS.md` §0 to find out. The docstring is where that decision is enforced against a future "simplification."
- **`threshold.json` — the versioned threshold policy artifact. Phase 6 is the only place it is written.** Phase 7 states *"the operating threshold is shipped as a separate versioned config artifact"* and Phase 9 promises *"a separate versioned config / policy layer loaded at startup"*; neither creates it. Logged to the MLflow run and mirrored to `configs/policy/threshold.yaml`. Fields:

  ```json
  {"threshold": 0.3942, "scenario": "base", "rule": "closed_form_c_over_r_ltv",
   "costs": {"c": 68, "r": 0.30, "ltv": 575}, "calibration_method": "sigmoid",
   "argmax_ev_bootstrap_ci": [0.36, 0.43], "implied_contact_rate": 0.21,
   "scenarios": {"conservative": 0.2736, "base": 0.3942, "optimistic": 0.4963},
   "model_run_id": "<run>", "model_version": "<n>"}
  ```

  - **All three scenarios' `t*` ship in the artifact, not just the selected one.** This file calls itself the versioned threshold *policy*; a policy that omits two of its three options forces Phase 9 to re-derive them to switch risk posture. They are three divisions — store them.

  - **`model_run_id` / `model_version` are the point of the artifact, not metadata.** They let `evaluate.py` assert that the threshold it applies was derived against the model it is evaluating, and let Phase 9 refuse to start if `champion` does not match. The plan already claims *"a re-calibration invalidates a previously-derived threshold"* — today that is prose. This makes it a check that fails loudly instead of a sentence nobody reads. A calibrated model and a threshold from different runs is exactly the silent-wrong-answer class this project keeps finding.
  - Hard-coding `$68` / `$172` in `threshold.py` would contradict Phase 8's own rationale that *"the threshold encodes a business cost choice owned outside the pipeline."* Costs come from `configs/costs.yaml`; the resolved values are echoed into `threshold.json` so the artifact is self-describing without a config lookup.

- `src/telco_churn/models/plots.py` — **pure helpers returning point data, no rendering.** Reliability-diagram bins, cost/EV curves, and the `r`-sensitivity series. Per `CLAUDE.md` § Code Style, notebooks are thin wrappers: `04-calibration-and-threshold.ipynb` imports these and renders. Heavy logic does not live in the notebook, and a chart the notebook draws must be reproducible from `src/` alone — Phase 7's `evaluate.py` and Phase 13's monitoring both need the same curve data without importing a notebook.

- `tests/integration/test_calibrate_subprocess.py`, `tests/integration/test_threshold_subprocess.py` — **mandated, not optional.** `CLAUDE.md` § Testing: *every* module with a `__main__` CLI entry point requires a subprocess integration test covering the exit-0 and exit-1 paths, because direct function calls miss argparse, OmegaConf resolution, dotenv loading, and env-var-to-engine joints that only surface at the subprocess boundary. Both modules have CLIs. The exit-1 paths worth covering: a missing/invalid `configs/costs.yaml`; `r = 0` (division by zero in `t* = c/(r × LTV)` — a config typo, not a code bug, and it must fail loudly rather than emit `inf`); and **`c ≥ r·LTV`, which yields `t* ≥ 1`** — "never contact anyone," an equally plausible `costs.yaml` typo that would otherwise ship a model that never fires. Assert `t* ∈ (0, 1)`.

- `tests/unit/test_calibrate.py` — **the leak-free claim must be verified, not asserted.** The plan says `CalibratedClassifierCV` receives the *unfitted* `Pipeline` so the `ColumnTransformer` refits inside every calibration fold; nothing currently proves it. Tests: (1) **leak canary — count preprocessor refits**; (2) `calibrated_classifiers_` has length 1 under `ensemble=False`, and `calibrated_classifiers_[0].estimator` is the `Pipeline` (the SHAP access path Phase 7 depends on); (3) run-twice determinism under the explicit `StratifiedKFold`; (4) outer-OOF calibrated Brier ≤ uncalibrated Brier on synthetic miscalibrated scores, both scored on the same pooled outer-OOF vector; (5) per-fold mean calibrated PR-AUC within Δ\* of uncalibrated (the sigmoid/isotonic gate). The same leak class bites `threshold.py`, so the canary belongs here where the wrapping happens.

  > **⚠ The canary cannot be "assert `statistics_` differ across `calibrated_classifiers_`."** Under `ensemble=False` that list has **length 1** — as test (2) two lines up correctly asserts. The per-fold models exist only long enough to produce OOF predictions and are then discarded; there is nothing left to compare. Nor can you spy on `ColumnTransformer.fit`: `Pipeline` calls `fit_transform` on intermediate steps, so a `.fit` counter reads **zero** regardless.
  >
  > **Assert `ColumnTransformer.fit_transform` is called exactly `n_folds + 1` times** (5 inner folds + the final refit on all of development). Verified against sklearn 1.8: `ensemble=False` → 6 calls, one `calibrated_classifiers_`; `ensemble=True` → 5 calls, five `calibrated_classifiers_`. That count is the only observable distinguishing a per-fold refit from a preprocessor fitted once outside the loop, and it doubles as a regression test on the `ensemble=False` semantics the whole phase rests on.
- `tests/unit/test_threshold.py` — closed form against hand-computed values for all three scenarios; the argmax-EV diagnostic agrees with `t*` on synthetic perfectly-calibrated probabilities and *disagrees* on deliberately miscalibrated ones (the check must be able to fail); the bootstrap CI of the empirical argmax contains `t*` on calibrated synthetic data and excludes it on miscalibrated data; degenerate cases (zero cost, `r = 0`, `t* ≥ 1`, single class).
  - **⚠ The miscalibrated fixture must be miscalibrated *near* `t\**`.** Because the agreement check is local (see the diagnostic bullet above), a distortion that mangles the tails while preserving `E[y|p] = p` around 0.39 leaves the argmax where it was and the test passes — meaning the test asserting "the check can fail" would itself be silently broken. Distort the probability–outcome relationship in a band around `t*` specifically.
  - **Leak-free proof — structural, because `threshold.py` cannot leak the way `calibrate.py` can.** It fits nothing. `calibrate.py` produces the winning method's outer-OOF vector; `threshold.py` *consumes* it. Its leak risk is therefore not "did the preprocessor refit inside the fold" but **"were these probabilities in-sample?"** — a contamination it inherits rather than commits. Two tests, mirroring the `feature_freeze.py` canary pattern but adapted to that difference:
    1. **Leak-free by construction, enforced by an AST scan.** Assert `models/threshold.py` contains no `.fit(` call, imports no sklearn estimator, and does not import `telco_churn.data.split`. Its public entry point takes `(oof_proba: NDArray, y_dev: NDArray, costs: CostScenario)` — arrays and parameters, never an estimator. A module that cannot fit cannot fit on the wrong rows. This is the same discipline `evaluate.py` gets from the touched-once invariant, made executable rather than trusted.
    2. **Inherited-contamination canary.** Fit a deliberately overfit model on synthetic data and derive two probability vectors: in-sample `predict_proba(X)` and `cross_val_predict(...)` OOF. Assert the agreement check **passes on the OOF vector and fails on the in-sample one** — `t*` inside the bootstrap CI for the former, outside it for the latter. In-sample probabilities are over-confident, pushed toward 0 and 1, which hollows out the score mass near `t*` and drags the empirical argmax away from it. This proves the Phase 6 agreement check is *itself* a leak detector, not merely a calibration check — the one guard that would catch someone passing `calibrated.predict_proba(X_dev)` in a future refactor, which is the single most plausible way this module's contract gets broken.
- Both modules log artifacts to the same MLflow run as Phase 5
- `notebooks/04-calibration-and-threshold.ipynb` — reliability diagrams before and after calibration; cost curve annotated with each scenario threshold

**Verification:** Five falsifiable checks, none of them circular. (1) Calibrated Brier ≤ uncalibrated Brier, both on the **pooled** outer-OOF vector over the same `outer` folds. (2) Calibrated **per-fold mean** PR-AUC within Δ\* (0.005) of uncalibrated — the hard gate that protects the ranking invariant; scored per fold, never pooled. (3) `t*` falls inside the bootstrap CI of the empirical argmax-EV over calibrated OOF probabilities. (4) `reports/figures/threshold_sensitivity.png` exists and its `r = 0.30` row equals the shipped `t*`. (5) `runs:/<tuning_study_run_id>/model` still resolves to the **uncalibrated** `Pipeline` after `calibrate.py` has run — the check that `name="calibrated_model"` held, and that `clone()` has an unambiguous source for the next cycle.

> **⚠ The old criterion was circular** — *"base-scenario threshold matches the value documented in `README.md`"* — and `README.md` currently carries the archived **0.2956**, produced from a different feature set, different hyperparameters, a different calibration method, and (per §8's superseding note) a different and incorrect objective. Per `CLAUDE.md` § Source of Truth, the archive is not a bar current work must clear: **Phase 6 derives the threshold and `README.md` is updated from it, never the reverse.** Updating `README.md`'s headline number is a Phase 6 deliverable, not a Phase 6 check.

> **📌 Cross-reference with Phase 4a (`notebooks/02a-feature-discovery.ipynb`):** The discovery notebook explicitly names `models/calibrate.py` and `models/threshold.py` as the home of the production decision threshold — telling readers that `DISCOVERY_THRESHOLD` (the prevalence-based reference used for lap-to-lap delta comparisons) is *not* the production threshold. When this phase lands: (1) confirm both modules exist at the paths named in the notebook (`src/telco_churn/models/calibrate.py` and `src/telco_churn/models/threshold.py`); (2) verify the notebook's cross-reference text is still accurate (function names, file paths); (3) update the notebook's threshold documentation block if the calibration/threshold API has changed from what was described.

> **⚠ Read the features table through a single accessor, starting here.** `CLAUDE.md` already flags `datasets/processed/telco_churn_processed.csv` as an intermediate artifact that Phase 8's `features` stage replaces with `telco_churn_features.parquet`. Phase 6 `calibrate.py` is the first module written after that decision was recorded; Phase 7 adds `evaluate.py` and `refit.py`. If each opens the CSV directly, Phase 8 rewrites three modules and their tests. Instead, add `features/io.py::load_features()` — one function that owns the path, the format, and the `sha256` content hash the Phase 7 consistency assertion needs. The Phase 8 CSV→Parquet swap then edits **one** function body. This is not premature abstraction: the replacement is already scheduled in writing, and the hash has two consumers before Phase 8 even lands.

> **⚠ Layout flag (from Phase 3):** Add `reports/figures/` to the repository layout in the "Repository Layout (target)" section when this phase lands. `reports/` is already used implicitly (Phase 2 writes `reports/validation/`, Phase 7 writes `reports/metrics.json`) but was omitted from the top-level layout. Phase 6 is the first phase to save evaluation charts (reliability diagrams, cost curves) to disk, making the omission visible. Update the layout table and add `reports/figures/` as the destination for saved charts.

---

### Phase 7 — Evaluation + Error Analysis + Full-Data Refit + Registry Promotion *(2–3 days)*

**What this achieves:** A sealed test-set evaluation (the test set has never been touched until this point) produces bootstrap-confidence-interval-bounded metrics that are honest estimates of production performance. A structured promotion gate, decided on those metrics, admits the model only when it improves on both **ranking (PR-AUC)** and **calibration (Brier)**; recall at the operating threshold is reported but does not gate the decision. The model that *passes* the gate and the model that *serves* are then two different artifacts: once the metrics are recorded, the frozen spec is refit on 100 % of the data and it is that refit which receives the `champion` alias.

**Deliverables:**
- `src/telco_churn/models/evaluate.py`:
  - Loads the sealed test set via the canonical split module (`telco_churn.data.split.test_ids()` / `partition()`, reading `datasets/processed/split_manifest.parquet`) — the same manifest Phase 4a discovery and Phase 5 `train.py` read for the `dev` partition. `evaluate.py` is the sole consumer of the `test` partition.
  - **⚠ Assert threshold provenance before applying it.** Load `threshold.json` (Phase 6) and check its `model_run_id` / `model_version` against the model being evaluated; abort on mismatch. Phase 6 states that *"a re-calibration invalidates a previously-derived threshold"* — this is where that becomes enforceable rather than aspirational. Applying a threshold derived against a different calibration map produces plausible, wrong numbers with nothing raised.
  - **⚠ Resolution rule — `evaluate.py` loads its model by explicit `run_id` or version number, never by alias.** `models:/telco-churn-pipeline@challenger` is a *moving pointer*. Once `refit.py` exists, a mis-ordered or re-run pipeline can leave that alias on the `full`-scope refit — a model trained on the test set — and `evaluate.py` would then score it, report excellent sealed-test metrics, and fail silently at exactly the step whose entire purpose is honesty. Nothing raises; the numbers just get better. This is the real contamination guard; within-phase ordering is not (see the ordering note below). Resolve the version to evaluate once, explicitly, and log it into `reports/metrics.json` so the metrics are attributable to a specific artifact rather than to whatever the registry happened to point at. Treat any refactor that "simplifies" this to an alias lookup as a correctness regression.
  - Sealed test-set metrics: ROC-AUC, PR-AUC, recall, precision, F1, Brier score
  - 1,000-iteration bootstrap 95 % CIs (routine lifted verbatim from the original notebook)
  - **Lift, gains & ranking diagnostics** (notebook §15 + §16.2): cumulative gains curve, lift curve, and a per-decile lift table (decile → count, churn rate, cumulative captured churn, lift vs. base rate). These answer the project's actual business question — *"which customers should we call this week?"* — by quantifying how much churn the top-k% capture; threshold-free, so they complement PR-AUC rather than competing with it.
  - **Business-impact summary** (notebook §16.3): the expected-value calculation at the base-scenario threshold — retained-revenue vs. retention-call cost — emitted as a single headline figure for `README.md`. The cost inputs come from the Phase 6 threshold scenarios; this step turns the chosen operating point into a dollar number, it does not re-select the threshold. The EV is reported **relative to two policy baselines** — *treat all* (`DummyClassifier(strategy='constant', constant=1)`: call everyone — maximum recall, maximum spend) and *treat none* (`constant=0`: do nothing — zero spend, full churn loss). The model must beat both on expected value; these are the cost-curve endpoints, bracket the achievable value, and persuade stakeholders more directly than a statistical floor.
  - Writes `reports/metrics.json` (metrics + bootstrap CIs + decile lift table + business-impact figure); saves gains/lift charts to `reports/figures/`
  - Logs all metrics and the report to the MLflow run
- `src/telco_churn/models/refit.py` — **the full-data production refit** (notebook §17.1). Runs *after* `evaluate.py` has recorded the sealed-test metrics and after the error-analysis review gate. Refits the frozen feature set and tuned hyperparameters on **100 % of the data**, re-calibrates with the **pinned** `calibration.method` (never `auto` — see Phase 6), registers a new version tagged `refit_scope: full`, and hands it to `register.py`.
  - **Reads the full feature table, never the test split.** `refit.py` loads `datasets/processed/telco_churn_features.parquet` whole and **must not import `data.split`**. The invariant in `CLAUDE.md` — *"`X_test`/`y_test` are imported and used in exactly one place: `models/evaluate.py`"* — is structural and survives literally: `refit.py` consumes the full dataset, not the `test` partition. The semantic consequence is stated separately below.
  - **It inherits the operating threshold; it does not re-derive one.** Under the closed-form rule (`t* = c / (r × LTV)`, Phase 6 `threshold.py`) the Bayes-optimal cut is a function of the cost parameters alone — *not* of the model. Any correctly calibrated model takes the same threshold. This is the payoff of calibration, and it is the reason the threshold transfers to the refit for free. **It would not transfer under an empirical argmax-of-EV threshold** (the archived notebook's 0.2464 / 0.2956 / 0.3596), which is model-dependent and noisy — an independent argument for the closed form.
  - **No metric is computed here, and none can be.** The refit has seen the test set; its test-set scores exist, look excellent, and mean nothing. `refit.py` emits no evaluation numbers. The metrics of record belong permanently to the `dev`-scope version.
  - **Learning-curve diagnostic** (`reports/figures/learning_curve.png`): CV PR-AUC against training-set fraction at 40/60/80/100 % of development. This gates nothing — it answers, for `ANALYSIS.md`, *"what did the last 25 % of the data buy?"*, a question the refit makes permanently unmeasurable by direct comparison. **Do not expect a plateau.** `ANALYSIS.md` §11 limitation #10 records that CV PR-AUC was **still rising at the maximum training size** in the Phase 5 Steps 2c/2d diagnostic — 0.613 → 0.655 from 20 % → 100 % of the dev-training folds. On this project's own evidence the refit is not ritual: the data is still paying at n ≈ 5,600, so training on 100 % is expected to buy real ranking quality. Cross-check the new curve against those numbers; a *disagreement* is the finding.
  - **Provenance triple, recorded at registration:** git SHA, the features-artifact `sha256`, and the `run_id` of the `dev`-scope version this refit was derived from. That triple *is* the champion's lineage — DVC never needs to own the model artifact. `dvc_data_hash` is recorded alongside it as **best-effort metadata that gates nothing**: Phase 7 precedes Phase 8, so DVC is not yet initialised and `_dvc_hash` returns `"unknown"` by design (`models/train/common.py:84`). The initial champion carries a real content hash and an `"unknown"` DVC hash; it acquires the latter on the first post-Phase-8 retrain. **Nothing in this phase depends on DVC.**
  - **⚠ Data-consistency assertion — on a direct content hash, not the DVC hash.** `evaluate.py` measures a model against dataset version *D*. `refit.py` trains the shipped model on whatever is on disk when it runs. If `features/build.py` changed in between, you evaluate model A on *D₁* and ship model B trained on *D₂*, and the metrics of record describe an artifact that never existed. So: `evaluate.py` records a `sha256` of the features artifact it read; `refit.py` recomputes it and **aborts on mismatch**. Use a direct content hash — **not** `_dvc_hash`, which reports the hash DVC *recorded* rather than the bytes on disk, so a parquet edited without `dvc repro` would pass the assertion vacuously. This also makes the guard **independent of DVC entirely**, so it works identically before and after Phase 8.
  - **`dvc pull` is a prerequisite in every non-local context** (CI, Prefect worker, Docker) — *from Phase 8 onward*. The features parquet becomes a cache pointer, not a file in git, and `refit.py` is not a DVC stage, so nothing pulls it for you.
  - **Its own `__main__` CLI, and therefore its own subprocess integration test** (`CLAUDE.md` § Testing). It is *not* a DVC stage — it registers a model version, and the DAG deliberately stops at `evaluate` (see Phase 8). The alternative considered was folding `refit_full()` into `register.py`'s `__main__` under the subprocess-test waiver, which would make it structurally impossible to promote without refitting. Rejected on failure isolation: a LightGBM fit and a calibration over 7,043 rows is expensive, restartable work whose failures should not surface as failures of the registry module. `refit.py` fits; `register.py` flips aliases. Phase 10's Prefect flow imports `refit_full(cfg)` directly as a task — Prefect tasks are Python functions and need no CLI.
- `src/telco_churn/models/register.py` — **the gate measures one artifact and promotes another.** The promotion decision is computed on the `dev`-scope version's sealed-test metrics: admit if and only if it beats the current `champion` on both **PR-AUC** (ranking quality; threshold-free) and **Brier score** (calibration; lower is better); no promotion otherwise; logs the decision with structured event `model_promoted` or `model_rejected`. The alias `champion` is then pointed at the **`full`-scope refit** from `refit.py`, logged as a distinct structured event `champion_refit` — *not* `model_promoted`, because nothing was compared. The full-data version can never pass the gate on its own: it was trained on the test set, so it has no honest metrics to submit. Do **not** express this by moving `challenger` onto the dev model and `champion` onto the refit — those aliases mean *incumbent versus contender for the same slot*, and a dev-fit and a full-fit of one frozen spec are not rivals but the same model at two data volumes. Phase 10's weekly retrain uses `challenger` for a genuinely new candidate; overloading the word here would make the registry's history unreadable. **On promotion, writes `model_card.json`** — the stakeholder-facing narrative (intended use, known limitations, performance summary vs. LogReg, the fixed-recall profile, fairness/robustness flags) — onto the promoted run. This is the **only** place the model card is written: when Phase 6 `calibrate.py` registers the challenger the model is calibrated but still un-thresholded and untested on the sealed set, so any "known limitations" section would describe an unfinished product; by promotion time calibration (Phase 6), the operating threshold (Phase 6), and sealed-test results (this phase's `evaluate.py`) are all real, so the card's claims are actually true of the artifact it describes. Phase 5's `training_manifest.json` remains the engineering audit trail across all stages. **The operating threshold is shipped as a separate versioned config artifact** alongside the model — *not* folded into the promotion comparison — so "is the new model better at ranking?" and "where do we cut?" stay independent, separately-auditable decisions. (Recall@threshold remains a *reported* metric; it does not gate promotion, because it inherits the fixed-threshold fragility discussed in `summary.md` §4.2.) **Model versioning is automatic:** MLflow auto-increments integer version numbers on every `log_model` call *that passes `registered_model_name=`* — there is nothing to manage manually, and since only Phase 6 `calibrate.py` passes it, version numbers advance once per training cycle and every one of them is a deployable artifact. What `register.py` manages is the *alias layer*: flipping the `champion` alias to point at the new version number when the promotion gate passes, and leaving `challenger` on the new version otherwise. The FastAPI service (Phase 9) always loads whichever run holds the `champion` alias at startup — it is decoupled from version numbers entirely. The Phase 10 Prefect retrain flow automates this promotion on every weekly retrain cycle; the only case requiring manual intervention is an emergency rollback, where `champion` is re-pointed at an older version number via `mlflow.MlflowClient().set_registered_model_alias()`. That rollback is safe *only because* every version in the registry is calibrated and serving-shaped — the invariant Phase 5's log-don't-register boundary exists to protect. Read the target version's `refit_scope` tag before rolling back: `dev` versions carry the sealed-test metrics of record, `full` versions are the full-data refits that actually served.
- `tests/unit/test_evaluate.py` — bootstrap CI math verified on a synthetic dataset with known population AUC
- `notebooks/05-error-analysis.ipynb` — SHAP global feature importance, SHAP local explanations for representative FN/FP cases, confusion matrix at each cost scenario threshold; plus three structured characterisation analyses run on the promoted champion:
  - **Fixed-recall profile** (recall ∈ {0.70, 0.80, 0.90}): precision and F1 at each target recall level, reported alongside the three cost-scenario thresholds logged by Phase 6 `threshold.py`. A threshold-planning tool — shows the business the precision trade-off at each campaign coverage level. **Values are read from the Phase 6 MLflow artifacts, not hard-coded here** — an earlier draft of this bullet inlined `0.22 / 0.30 / 0.38` while claiming not to, and those numbers were both stale and derived from the superseded cost-curve objective.
  - **Per-segment robustness check**: champion PR-AUC within three EDA-anchored axes — `contract_type` (Cramér's V 0.41), `tenure_cohort` (Pearson r −0.48 on continuous tenure; reuses `TENURE_COHORT_EDGES` from `preprocessing.py`), `internetservice` (V 0.32) — reported as per-segment PR-AUC and lift over the segment base rate. Checks whether the aggregate PR-AUC is uniform or concentrated in one cohort. A concentrated-edge flag routes to SHAP interaction-value drill-down on the `fiber-optic × month-to-month` compound cohort (54.6 % churn, the highest-risk documented interaction). Low-support caveat: the two-year `contract_type` tier churns <3 %; read its CI with caution.
  - **Fairness parity check**: per-group PR-AUC across four protected / quasi-protected axes — `gender` (sex; V 0.0086), `seniorcitizen` (age proxy), `has_partner` (marital status), `dependents` (familial status). `gender` is included precisely because its near-zero bivariate signal does not preclude proxy discrimination through correlated features; a clean parity result is the wanted, documentable outcome. Input policy: all four remain model inputs (benefit-allocation context — not a credit/employment denial); hard-excluding a protected attribute is a deliberate normative override, never a statistical-selector outcome. Flag any material gap for sign-off in `ANALYSIS.md`.

> **Within-phase order: `evaluate` → error-analysis review → `refit` → `register`.** Run `evaluate.py` first (sealed-test metrics + lift/gains + business impact), then review `05-error-analysis.ipynb`, then run `refit.py`, then `register.py`. This is a *within-phase* ordering constraint, not a reason to exile the refit to a later phase; earlier drafts of this plan conflated the two.
>
> **What the ordering does and does not protect.** It does **not** protect the sealed-test metrics arithmetically: `evaluate.py` scores the `dev`-scope version, `refit.py` produces a *different* version and mutates nothing the evaluation reads, so the numbers come out identical whichever runs first. What the ordering protects is (1) the **human veto** — the error-analysis review below can reject a model that passed the aggregate gate, and a refit that already registered and aliased has deployed before the review could fire; and (2) the audit trail, which should read in the order the decisions were actually made. The genuine contamination hazard is *not* addressed by ordering at all — see the resolution rule on `evaluate.py` above. Ordering discipline and resolution discipline are independent guards; the pipeline needs both. The deliverables are listed artifact-first (both `src/` modules together) but execute in this sequence. For the **first champion promotion this is a human review gate, not just confirmation**: subgroup FN/FP breakdown and SHAP sanity can veto a model that passes the aggregate PR-AUC + Brier gate but fails a protected/high-value segment — aggregate metrics hide subgroup collapse. The automated metric-only gate in `register.py` is what the **Phase 10** weekly retrain uses (no human in the loop every week); the initial Phase 7 promotion additionally requires the error-analysis review to pass. This is why error analysis is sequenced *before* `register.py`, not after.

> **Ordering & test-set discipline:** the error analysis here is *confirmatory of the metric decision* (notebook §12 / §16.4 — SHAP + FN/FP profiling of the final model), distinct from the *generative* error-feature loop already baked into Phase 4. The sealed test set is touched exactly **once**, here — `evaluate.py` is the *only* module permitted to import the test split, via `data.split.test_ids()`/`partition()` (the structural isolation set up in `data/split.py`, see `docs/canonical-split-refactor-tasks.md`). Under continuous retraining (Phase 10) do **not** re-use this same sealed test set for every challenger-vs-champion comparison — that erodes it; promote on a rolling/time-based holdout instead. This preserves the "test set touched once" invariant (Lifecycle & Framing Gaps, Group A). Turning the *generative* loop into reproducible production code (so the repo demonstrates the full lifecycle in code, not just the migration) is a **deliberately deferred v2** — see "What This Plan Deliberately Does Not Include" — sequenced after Phase 14 so the production spine ships first.

> **The evaluated model is not the deployed artifact (notebook §17.1 — full-data production refit):** the metrics produced here are honest estimates from a model fit, tuned, and calibrated on the development set (all via CV), with the test split held out. Before the model serves traffic it is refit on **100 % of the data** (development + test) using the frozen feature set and tuned hyperparameters — there is no metric left to estimate at that point, so withholding 1,409 labeled rows from the serving artifact only discards signal.
>
> **Why the refit costs nothing here.** The obvious objection is that it consumes the sealed test set. But `CLAUDE.md`'s invariant — *test set touched once* — already forbids ever using it for evaluation again; re-scoring it against future challengers is precisely the erosion the invariant exists to prevent. By the time `evaluate.py` returns, the test set is **already spent**. The refit destroys nothing that was not written off the moment the metrics were recorded. (The looser textbook advice — `GridSearchCV(refit=True)`, ESL — is only about refitting on the *training* data after CV folds; refitting on train **+** test is the stronger claim, and this is why it holds here.)
>
> **Why it belongs in Phase 7 and not Phase 10.** It must not run before the sealed-test metrics are recorded. That constrains *ordering within the phase*, and it is satisfied by `evaluate` → review → `refit` → `register` above. It says nothing about which phase. Placing it in Phase 10 would leave **Phase 9 shipping a service that serves the development-set model** — the very artifact this note calls "not the deployed artifact" — because the FastAPI service loads by `champion` alias and Phase 9 precedes Phase 10. Phase 10 retains a `refit-full` step because the *recurring* weekly retrain genuinely needs one; it is simply not the first one.
>
> **What Phase 7 leaves in the registry:** a `refit_scope: dev` version holding the sealed-test metrics of record, and a `refit_scope: full` version holding the `champion` alias and the traffic. Both are calibrated, thresholded, serving-shaped, and valid rollback targets. They are not interchangeable for *reporting* — only the `dev` one has honest numbers attached.

**Verification:** `uv run python -m telco_churn.models.evaluate` produces `reports/metrics.json` whose PR-AUC and ROC-AUC CIs overlap the CIs reported in `README.md`. After `refit.py` and `register.py`, `telco-churn-pipeline` holds two versions — one tagged `refit_scope: dev` carrying the metrics of record, one tagged `refit_scope: full` carrying the `champion` alias — and `refit.py` emits no test-set metrics of its own. `grep -r "data.split" src/telco_churn/models/refit.py` returns nothing.

---

### Phase 8 — DVC Pipeline Wrap *(1 day)*

**What this achieves:** The five-stage pipeline (ingest → validate → features → train → evaluate) becomes a content-hashed DAG. Changing a hyperparameter re-runs only the training and evaluation stages — not the full pipeline. This is the reproducibility guarantee that separates a real MLOps workflow from ad-hoc notebooks.

**Deliverables:**
- `dvc init`; `dvc add datasets/raw/Telco-Customer-Churn.csv`
- `dvc.yaml` — six stages with explicit `deps` (code + configs) and `outs` (data artifacts, models, metrics):

  | Stage | Deps | Outs |
  |---|---|---|
  | `ingest` | `data/ingest.py`, `datasets/raw/*.csv` (git-tracked), `sql/schema/` | `reports/ingest_receipt.json` (row count, per-column null counts, frame checksum) |
  | `validate` | `data/validate.py`, `data/schema.py` | `reports/validation/` |
  | `split` | `data/split.py`, `reports/validation/` (validate output — split blocks on a passing validation run), `configs/config.yaml` (`test_size`, `random_seed`) | `datasets/processed/split_manifest.parquet` |
  | `features` | `features/build.py`, `sql/features/` | `datasets/processed/telco_churn_features.parquet` (Parquet — static snapshot for downstream stages), `preprocessing.pkl` |
  | `train` | `models/train.py`, `configs/`, `datasets/processed/split_manifest.parquet` | MLflow run ID, `feature_space.txt`, `feature_columns.txt` |
  | `evaluate` | `models/evaluate.py`, `datasets/processed/split_manifest.parquet` | `reports/metrics.json` |

> **📌 Required in this phase — rewrite `_dvc_hash`, or every model registered from here on carries empty provenance.** `models/train/common.py:84` resolves `<processed_data>/telco_churn_processed.csv.dvc` and falls back to `"unknown"` on `FileNotFoundError`, logged at **debug**. That fallback is correct today (DVC isn't initialised yet) and becomes a **silent failure the moment this phase lands**, for two compounding reasons. (1) The filename changes — the `features` stage outputs `telco_churn_features.parquet`. (2) More fundamentally, **`.dvc` sidecar files exist only for `dvc add`-ed files**; pipeline *stage outputs* are tracked in **`dvc.lock`**, under `stages.features.outs`. So no `.dvc` file for the parquet will ever exist, whatever it is named. The lookup misses, the handler shrugs, `dvc_data_hash` is `"unknown"` forever, and nothing warns — the fallback for *"not tracked yet"* is indistinguishable from *"tracked, but I'm reading the wrong file."*
>
> **Fix:** point `_dvc_hash` at `dvc.lock` and resolve the `features` stage's out hash. Make the fallback **conditional**: if `dvc.lock` does not exist, `"unknown"` is still the right answer; if it exists and the stage output is absent from it, raise. Add a test asserting a non-`"unknown"` hash once `dvc.lock` is present. Note that **no correctness guard depends on this** — Phase 7's `refit.py` assertion deliberately uses a direct `sha256` of the features artifact, not `_dvc_hash`, precisely because the recorded-vs-on-disk gap makes the DVC hash unsuitable for it. What is at stake here is *provenance*: the audit trail answering "which data version produced this model."
>
> **A second, more serious consequence of the same `"unknown"` constant: `tuning.py::_study_name()` silently reuses stale Optuna studies across data changes.** The Phase 5→6 Bridge discovered this empirically — `_study_name()` content-addresses the Optuna study by hashing `dvc_hash` alongside `committed_features`, `search_space`, `cv_folds`, and the other tuning-config knobs, and `optuna.create_study(..., load_if_exists=True)` **resumes** any existing study sharing that hash rather than starting fresh. Every other hash input changes correctly when its underlying config changes; `dvc_hash` does not, because it is a fixed string until this phase lands. The practical risk: retrain on genuinely new or changed data today, and the tuning step silently resumes a study built on the *old* data, mixing incompatible trials from two different datasets into one 1-SE selection pool — with nothing in the logs to flag it. (`load_if_exists=True` itself is correct and intentional, for genuine crash recovery mid-optimization; the problem is only that `dvc_hash` can't currently distinguish "same run resumed" from "different data entirely.")
>
> **This resolves itself automatically once the fix above lands — no separate code change in `tuning.py`, and no manual migration step.** The moment `_dvc_hash` starts returning a real content hash instead of `"unknown"`, `_study_name()`'s SHA256 digest changes, Optuna finds no existing study under the new name, and starts a genuinely fresh 50-trial search on the next `python -m telco_churn.models.train` (or `dvc repro`) run. **This means the first post-Phase-8 training run is expected to produce different tuning numbers than whatever is currently recorded in `ANALYSIS.md` §4c — that is correct, not a regression.** The old `"unknown"`-keyed study rows are left behind, harmless, in the `optuna` Postgres schema (orphaned, not corrupted — `DROP SCHEMA optuna CASCADE` clears them if desired, but nothing requires it). Add a regression test: two `run_tuning_step` calls with different `dvc_hash` inputs (everything else held fixed) must produce different study names.

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

> **Deliberate scope — why the DAG stops at `evaluate`:** the six DVC stages cover the *data-transform* pipeline (raw → reproducible metrics). Calibration + thresholding (Phase 6) and the full-data refit + registry promotion (Phase 7) are intentionally **not** DVC stages. They are *decision* steps, not deterministic data transforms: calibration depends on a held-out fold, the threshold encodes a business cost choice (owned outside the pipeline — see `summary.md` §4.5), and promotion compares against the live `champion` in the MLflow registry, which is mutable state DVC cannot content-hash. Folding them in would make `dvc repro` non-deterministic (its output would depend on whatever `champion` currently exists).
>
> There is a sharper, operational reason too: **DVC stages must be safe to re-run.** A stage's `outs` are files, and `dvc repro --force` is listed in `CLAUDE.md`'s key commands as a routine thing to type. Those four modules mint MLflow registry versions and move the `champion` alias — neither a file nor idempotent. If `register` were a stage, asking DVC to rebuild would re-point a live service's alias as a side effect. This is why Phase 7's `refit.py` has a `__main__` but no stage.
>
> Instead, those steps are driven by the Phase 10 Prefect `retrain` flow, which calls `train → evaluate` (reproducible, DVC-tracked) and then `calibrate → threshold → refit → register` (decision layer) as explicit flow tasks. If full champion reproducibility is ever required, the fix is to pin the comparison baseline to a specific run ID rather than the `champion` alias — not to add these as DVC stages.

- **SQL view materialisation (required):** The Phase 4 SQL views recompute on every read — acceptable for development but not for a DVC pipeline. The `features` stage entry point must call `build_feature_df(engine)` to execute the SQL graph **once**, then immediately write the result to `datasets/processed/telco_churn_features.parquet` before exiting. The `train` stage lists that Parquet file as its sole data dependency, not Postgres. This gives three guarantees: (1) every training run reads a static, content-hashed snapshot; (2) `dvc repro` never blocks on the DB when the features hash is unchanged; (3) if Postgres is unavailable, all downstream stages still run from the cached Parquet. The DB is only contacted during the `features` stage, which DVC skips if its deps (raw CSV + `build.py` + SQL files) are unchanged.

- **Retire `build.py __main__` block:** The `if __name__ == "__main__"` block in `src/telco_churn/features/build.py` is a Phase 4 development scaffold — it wires together the full feature pipeline (config load → DB connect → SQL views → `build_feature_df` → write CSV) so the pipeline could be verified manually before DVC existed. In Phase 8 the DVC `features` stage entry point takes over that responsibility with two changes: output is Parquet instead of CSV, and DVC manages invocation. The `__main__` block should be removed from `build.py` at this point — it becomes dead code once the stage entry point exists. The core logic (`build_feature_df`, `_add_python_features`, column constants) stays in `build.py` permanently; only the CLI scaffold is retired. Also delete `datasets/processed/telco_churn_processed.csv` from the repo — it is superseded by the DVC-tracked `datasets/processed/telco_churn_features.parquet`. **The new stage entry point must assert `df_out.shape[0] > 0` before writing the Parquet output** — a zero-row result from a broken SQL view should be a hard failure, not a silent empty artifact (eighth-pass QA item 4).
- **Repoint the features accessor — one function, not four call sites.** Phase 6 introduces `features/io.py::load_features()` as the single owner of the processed artifact's path, format, and `sha256`. In Phase 8, change `pd.read_csv` → `pd.read_parquet` and the filename → `telco_churn_features.parquet` **inside that one function body**. The `train` stage in `dvc.yaml` declares the file as a dep, so DVC re-runs training whenever the features hash changes.
  > **If `load_features()` was not built in Phase 6, this task is four edits, not one.** The `telco_churn_processed.csv` filename is currently hardcoded at three `src/` sites — `models/train/common.py:48` (`_load_processed`), `models/train/common.py:100` (`_dvc_hash`, reading the `.dvc` sidecar), and `models/train/candidates.py:138` (which rebuilds the path independently rather than calling `_load_processed`) — plus `features/build.py:76` (the writer) and the three notebooks below. Earlier drafts of this task named only `_load_processed` and placed it at `models/train.py`, a path that has not existed since Phase 5 split the module. Migrate all of them, and route them through the accessor while you are there.

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
- `src/telco_churn/serving/schemas.py` — Pydantic v2 request/response models; field constraints aligned with the Pandera schema (single source of truth)
- `src/telco_churn/serving/predict.py` — loads the `champion` model and preprocessor from MLflow at startup; exposes `predict_single` and `predict_batch`
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

**Verification:** `docker compose up && curl -X POST http://localhost:8000/predict -d @example_payload.json` returns a churn probability; `curl http://localhost:8000/customer/<id>` returns that customer's raw fields; Streamlit at `:8501` displays a prediction with SHAP contributions in both lookup and manual mode.

---

### Phase 10 — Prefect Orchestration (Continuous Training) *(2–3 days)*

**What this achieves:** The model retrains automatically every week and checks for data drift every day — without manual intervention. The Prefect UI provides a full audit trail of every run, including failures.

**Deliverables:**
- Prefect 3 server added to `docker-compose.yml` (UI at `:4200`)
- `pipelines/retrain.py` — weekly Sunday 02:00 schedule; runs `ingest → validate → features → train → evaluate → refit-full → register`; promotes `challenger` to `champion` on PR-AUC AND Brier improvement (same gate as `register.py`). **`refit-full`:** the flow calls **Phase 7's `models/refit.py`** — this is the *recurring* incarnation of that step, not its first appearance. After `evaluate` records the cycle's held-out metrics, refit the frozen-feature, tuned-hyperparameter pipeline on **100 % of the data** — the deployed artifact uses all available signal, while the promotion gate is still decided on the held-out metrics. Same discipline as Phase 7: the gate is computed on the held-out-scope version, the `champion` alias is pointed at the `refit_scope: full` version, and the two events are logged distinctly (`model_promoted` vs. `champion_refit`). Re-calibrated with the **pinned method** (`calibration.method` from config — never `auto`, which would let the method flip on Brier noise between weekly cycles). Inherits the operating threshold rather than re-deriving it (the closed-form cut is a function of costs, not of the model). **Dummy-floor guardrail:** each cycle recomputes the prevalence / `DummyClassifier(strategy='prior')` floor on the current data and asserts the retrained model clears it by a margin — a near-dummy result signals a broken feature/label pipeline (stale join, dropped column), not a modelling regression, and blocks promotion with a structured alarm. The floor is recomputed each cycle, never hardcoded, because churn prevalence drifts.
  > **📌 Feature selection must not silently re-run every cycle.** The `train` step above currently maps to the full `python -m telco_churn.models.train` script (`__main__.py`), which chains Steps 1–5 — including Step 3's feature-selection freeze (Phase 5, `ANALYSIS.md` §4). That freeze is explicitly designed to run *once*, not on a schedule: a borderline feature (e.g. `paymentmethod`, 49/100 fold stability) could flip in or out between cycles from data-split noise alone, which would confound tuning (Step 4 assumes a fixed input space) and make model versions hard to compare week to week. Before this phase ships, either give `__main__.py` a flag to skip Steps 1–3 and load the already-committed `feature_columns.txt`, or have this flow call a narrower re-tuning-only path — the weekly cycle should reuse the frozen feature set, not re-derive it.
  > **📌 Warm-starting Optuna across retrain cycles.** Phase 5's `configs/tuning/optuna.yaml` `warm_start_params` is a one-time, hand-set prior (the archived notebook's own Optuna best) with no equivalent once that notebook is out of the loop. From this phase on, the `train` step should populate `warm_start_params` dynamically from the current `champion`-aliased MLflow run's logged params (`mlflow.get_run(champion_run_id).data.params`) before calling `run_tuning_step`, instead of reading a hardcoded config block — each retrain's winner becomes the next cycle's informed prior. `tuning.py`'s `study.enqueue_trial(...)` mechanism needs no change for this; only the source of the dict does.
  > **Clarification — what "rolling holdout" means here, since `customers_raw` doesn't grow.** The raw Kaggle CSV is a static, read-only snapshot (`CLAUDE.md`) — this project has no live customer feed appending new rows, so the DVC `split`/`ingest`/`validate` stages have no changed deps to react to, and `retrain.py`'s `ingest → validate → features` steps are cache hits every cycle. `data.split.make_split()` therefore runs once, ever, for this project; `split_manifest.parquet` is never regenerated, and there is no new-customer-driven reshuffling problem to solve. The "do not reuse the sealed test set — use a rolling/time-based holdout instead" guidance (§ Phase 7 evaluate.py, § Deliberate scope above) is about a different risk: scoring the *same* frozen `evaluate.py` test partition against every weekly promotion decision implicitly "spends" that sealed set dozens of times a year, eroding it as an unbiased estimate (repeated-peeking, not data-freshness). The data that genuinely grows over time in this system is `prediction_outcomes` (below, `performance_check.py`) — live predictions joined to realised labels as they mature — not `customers_raw`. Any future rolling-holdout construction for promotion decisions should be carved from `prediction_outcomes`/logged-prediction history, not from a re-partitioned `customers_raw`; the mechanism itself (deterministic hashing vs. time-based cut) is not yet decided and is deferred to this phase's implementation.
  > **❓ OPEN QUESTION — what does the weekly promotion gate actually score, on a static dataset?** Phase 7's full-data refit sharpens the problem the note above defers. After `refit.py` runs, the sealed test set is not merely *eroded by repeated peeking* — it is **inside the champion's training data**. There is then no clean holdout anywhere in the project, and `customers_raw` will never grow to supply another one. So `register.py`'s gate (`challenger` beats `champion` on PR-AUC **and** Brier) has, as of Phase 10, **no defined evaluation set**. Three candidate resolutions, none yet chosen:
  >
  > 1. **Score on `prediction_outcomes`.** The only genuinely fresh labeled data this system produces. Correct in principle, and it is what a real deployment would do — but it does not exist until Phase 9 has served traffic and the label-maturity window has closed, so the first several retrain cycles have nothing to gate on. Needs a defined cold-start behaviour (block promotion? auto-promote? require manual sign-off?).
  > 2. **Nested CV on the full dataset per cycle.** Statistically defensible on static data and available immediately, but it measures *the spec*, not *the artifact*, and it re-fits every cycle — expensive, and the resulting number is not comparable to the Phase 7 sealed-test metrics of record.
  > 3. **Declare the gate demonstrative.** State plainly in `ANALYSIS.md` that on a fixed 7,043-row snapshot with no live feed, Phase 10's weekly retrain is a *mechanism demonstration* — the plumbing, alerting, and alias-flip are real; the statistical claim behind the promotion decision is not. Honest, cheap, and arguably the right answer for a portfolio project.
  >
  > This is a **Phase 10 decision, deferred deliberately** — it does not block Phases 6–9, and `performance_check.py`'s realised-PR-AUC loop (below) is the piece that makes option 1 viable. Resolve it before `retrain.py` is written, and record the choice in `ANALYSIS.md`. The one outcome to avoid is shipping a gate that silently re-scores the spent test set every Sunday.
- `pipelines/drift_check.py` — daily 06:00; pulls the last 24 hours of predictions from the API structured logs; runs an Evidently data drift report; alerts (Prefect notification) when PSI > 0.2 on any top-5 feature
- `pipelines/performance_check.py` — **realized-performance feedback loop (closes the ML loop — `input` drift is only a proxy for what we actually care about).** PSI and prediction-distribution drift detect that *inputs* moved; they do not measure whether the model is still *right*. For churn the ground truth does arrive — observed weeks/months later — so this flow joins **logged predictions → realised outcomes** once labels mature and computes **realised PR-AUC** (plus Brier and the decile-lift table) over the matured cohort — **both aggregate and sliced by the same Phase 7 segment axes** (`contract_type`, `tenure_cohort`, `internetservice`), so segment-concentrated decay surfaces before it drags the aggregate down (the production analogue of the Phase 7 per-segment robustness check — the offline guard against subgroup collapse, now run continuously) — comparing each to the Phase 7 offline estimate. Mechanics: predictions are logged with `customerid` + `request_id` + timestamp (Phase 9); a `prediction_outcomes` table in Postgres records the eventual churn label per `customerid`; the flow runs on a **label-maturity cadence** (e.g. monthly, after the observation window closes), scoring only the cohort whose outcomes are now known. **Delayed-label discipline:** never score a cohort before its outcome window matures (premature labels bias the metric optimistic); the maturity lag is a configured constant. This realised number — not PSI — is the **authoritative health signal** and the primary, *performance-based* retrain trigger (see below).
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
  - **Prediction distribution** — rolling histogram vs. training reference
  - **Feature drift** — PSI per top-5 feature over time (sourced from Evidently reports)
  - **Realised performance** — realised PR-AUC / Brier over each matured cohort vs. the Phase 7 offline baseline — **aggregate plus one series per Phase 7 segment** (sourced from `performance_check.py`); the panel that distinguishes "inputs moved" from "the model is actually degrading." Updates on the label-maturity cadence, not in real time — annotate it as such so a flat line between updates is not misread as staleness.
- Alert rules: p95 latency > 500 ms; error rate > 1 %; PSI > 0.2 sustained over 24 hours; **realised PR-AUC below a configured fraction of the Phase 7 baseline for one matured cohort — aggregate or any monitored segment** (the performance-based retrain trigger — see Phase 10 `performance_check.py`)
- `src/telco_churn/utils/logging.py` updated so `log_level` is read from `LOG_LEVEL` environment variable (fallback `"INFO"`), enabling temporary debug logging in production without a redeploy

**Verification:** Grafana dashboards populate within minutes of the API receiving traffic; simulated drift raises a PSI alert; load test shows p95 latency panel updating live.

---

### Phase 14 — Documentation Polish *(1 day)*

**What this achieves:** A recruiter or hiring manager can understand the project's scope, results, and architecture within 90 seconds of landing on the repo.

**Deliverables:**
- `README.md` — top of file: 1-paragraph elevator pitch, architecture diagram, "Quick demo" GIF, tech-stack table with phase links, headline metrics table
- `ANALYSIS.md` — full modelling narrative (already written; verify it references `src/` functions, not notebook code)
- `docs/runbook.md` — how to retrain, roll back to the previous champion, debug a drift alert, and **triage a bad-prediction incident** (forensic loop: pull the offending `request_id` / `customerid` from the structured logs, re-run SHAP-local on the case, and classify root cause as relabel vs. feature-gap vs. drift → feed the verdict to the retrain backlog)
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
| `05-error-analysis.ipynb` | 7 | SHAP global/local plots + FN/FP analysis |

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
| `src/telco_churn/models/train.py` | Optuna + LightGBM + MLflow; called by both CLI and Prefect |
| `src/telco_churn/serving/app.py` | FastAPI app; loads `champion` model from MLflow Registry at startup |
| `pipelines/retrain.py` | Continuous-training DAG; ties together every `src/` module |
| `.github/workflows/cd.yml` | Deploy automation; ties code changes to AWS |

---

## Reuse from the Original Notebook

This table maps early-phase artifacts from `notebooks/_archive/EDA-original.ipynb` to their `src/` destination — a migration record, not a constraint. Later phases have diverged where a stated reason justified it (see `ANALYSIS.md` and `CLAUDE.md`'s Source of Truth section):

| Notebook artifact | Destination in `src/` |
|---|---|
| 5 data quality gates | `data/schema.py` + `data/checks.py` |
| `ColumnTransformer` definition | `features/preprocessing.py` → `models/train.py` (Phase 5) — fitted per CV fold on development; production path is `tree_preprocessor` + LightGBM in an sklearn `Pipeline` (linear baselines use a separate `linear_preprocessor`, Step 1) |
| Optuna best hyperparameters | Default values in `configs/tuning/optuna.yaml` (still searchable; warm-start from these) |
| Cost-sensitive threshold logic (3 scenarios) | `models/threshold.py` |
| Bootstrap CI evaluation routine | `models/evaluate.py` |
| Lift / gains curves + decile lift table (§15, §16.2) | `models/evaluate.py` (Phase 7) → `reports/metrics.json` + `reports/figures/` |
| Business-impact / EV figure (§16.3) | `models/evaluate.py` (Phase 7) → README headline metric |
| Bias/variance + McNemar diagnostics (§10.6, §10.8, §11.1) | `notebooks/03a-model-selection.ipynb` (Phase 5) — notebook-only, gates nothing |
| Full-data production refit (§17.1) | `src/telco_churn/models/refit.py` (Phase 7, after `evaluate.py`); re-invoked as `pipelines/retrain.py`'s `refit-full` step (Phase 10) |
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
