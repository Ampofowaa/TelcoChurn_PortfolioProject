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

**Verification:** Notebook executes end-to-end without errors (`jupyter nbconvert --execute notebooks/01-eda.ipynb`).

---

### Phase 4 — Feature Engineering (SQL + Python) *(2 days)*

**What this achieves:** Features are built in two layers — SQL views in Postgres (tenure bucketing, charge-per-service ratios) and a scikit-learn `ColumnTransformer` (scaling, encoding, imputation). This mirrors how features are built in a real warehouse-backed ML system. The ColumnTransformer definition is lifted verbatim from the original notebook to preserve the science exactly.

> **Ordering note (where the error-feature loop lives):** the *generative* error-driven feature loop (notebook §10.5 — scan false negatives, hypothesise features, re-measure) already ran during the original modelling; its surviving engineered features are part of the feature set lifted here. Phase 4 transcribes that converged feature set — it does **not** re-run the loop. The *confirmatory* error analysis on the final model is a separate, later step (Phase 7). This is why feature engineering (Phase 4) correctly precedes training (Phase 5) even though "errors drive features" — the loop was closed in the notebook, not re-opened in the migration.

> **Feature-selection note (diagnose, don't reflexively drop):** the engineered features above are **carried forward in full** at this phase. LightGBM is immune to multicollinearity and tolerates irrelevant inputs, and the post-encoding width (~30–45 columns) is not high-dimensional, so filter/wrapper selection buys little accuracy here. The notebook already does the industry-correct thing — it *measures* redundancy (VIF, §8) and relevance (Cramér's V, effect sizes) and drops nothing. The *decision* about which features to keep is made against the model in **Phase 5** (`notebooks/04-feature-selection.ipynb`), where a null-importance experiment selects inside CV, refits LightGBM on the survivors, and compares to the full set with a bootstrap CI — adopting a reduced set only if there is no significant PR-AUC loss. Feature selection is the *drop*-half of the feature loop (Phase 7 error analysis is the *add*-half) — see `summary.md` §4.4.

> **Where feature work happens across this project (generation vs. selection vs. confirmatory analysis).** These are three *distinct* activities, not three rounds of the same one — the project does **not** engineer features twice:
>
> | Activity | What it does | Phase | Iterated here? |
> |---|---|---|---|
> | Feature **engineering** (generate) | *creates* features | Phase 4 — transcribe the converged set | **No** — the generative loop already closed in `EDA-original.ipynb`; Phase 4 lifts the result |
> | Feature **selection** (prune) | *drops* weak features | Phase 5, step 3 | Once; set frozen before tuning |
> | Error analysis (diagnose) | *finds* where the model fails | Phase 7 | **Confirmatory** — validates the shipped model; does **not** reopen engineering |
>
> In a greenfield build, engineering is a *loop* (error analysis → new feature → re-model, 2–5× — "data-centric" iteration); the linear phase list shows only the first pass. This project migrates a loop that **already converged**, so it engineers once. The loop genuinely reopens only for the **next model version**, driven by post-deployment monitoring and drift (Phases 10 & 13) on the retrain cadence — not within this build.

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

### Phase 5 — Model Training (LightGBM + Optuna + MLflow) *(3–4 days)*

**What this achieves:** A reproducible, experiment-tracked training run that searches hyperparameters with Bayesian optimisation and logs every trial to MLflow. The best model is registered in the MLflow Model Registry as `challenger`, ready for evaluation and promotion.

**Order of operations within this phase (the modelling loop — do not reorder):**
1. **Build candidates** — the two reference baselines (`DummyClassifier`, `LogisticRegression`) *and* the tree models, all through the *identical* Phase 4 pipeline and CV splits. (Baselines must run through the same pipeline to be a fair measuring instrument — that is why they live here, after features exist, not earlier.)
2. **Compare on PR-AUC** → confirm LightGBM beats the baselines; document the margin (expect a near-tie with `LogisticRegression`).
3. **Select features (freeze the input space)** — run null-importance / target-permutation selection (inside CV, on train + val only) against a *default-config* LightGBM, refit on the survivors, and confirm the reduced model's CV PR-AUC sits within the bootstrap CI of the full-feature model. This is what *freezes* the feature set before tuning (§4.3: tune late, on a frozen input space). The expected, fully acceptable outcome on this dataset is "keep most/all" — documented honestly either way. Detail in `notebooks/04-feature-selection.ipynb`.
4. **Tune only the confirmed family** (LightGBM) with Optuna — *late*, after the feature set is frozen by step 3.
5. **Register the tuned model** as `challenger`. Calibration + thresholding (Phase 6) and sealed-test evaluation (Phase 7) are deliberately separate, later phases.

**Deliverables:**
- `configs/model/lightgbm.yaml` — LightGBM parameter ranges (lifted from the original notebook's Optuna study as warm-start defaults): `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- `configs/training/optuna.yaml` — Optuna config: `n_trials: 50`, `sampler: tpe`, `direction: maximize`, `metric: average_precision` (PR-AUC — see the flagged deviation below), `cv_folds: 5`, `random_state: 42`
- **Reference baselines (floor + linear control)** — `DummyClassifier(strategy="most_frequent")` and `LogisticRegression(class_weight="balanced", max_iter=1000)`, run through the *identical* pipeline, CV splits, and metric as the tree models. The notebook's existing "baselines" (RandomForest / LightGBM / XGBoost) are all non-linear candidates — there is currently no no-information floor and no linear reference. These two establish: (a) the prior-only floor (~73 % accuracy → exposes the accuracy trap), and (b) how much signal is *linear*. Report both as rows in the §10.3 / `03-model-selection.ipynb` comparison table on **PR-AUC (primary) and recall @ 0.35 (diagnostic)**; they are reference rows, not selection candidates.
  > **Expected finding to document:** Telco churn is near-linear — logistic regression typically reaches AUC ≈ 0.83, within the test-set CI of the tuned LightGBM. If so, state explicitly that LightGBM is chosen for calibration quality / SHAP interaction structure / error-analysis-driven features, **not** because it materially out-predicts the linear model. This is the honest justification for downstream complexity and ties to the significance discipline (Phase 7 bootstrap CIs).
- **Model-selection metric = PR-AUC (average precision).** Cross-family model selection and the §10.3 / `03-model-selection.ipynb` comparison rank on **PR-AUC**, not recall@0.35 — PR-AUC is threshold-free and imbalance-appropriate (ROC-AUC is optimistic at a ~27 % positive rate). Recall / precision / F1 at the operating threshold are **reported as diagnostics, not used to decide**. Sanity check: plot the candidate PR curves together and confirm they do not cross near the operating region (if the PR-AUC ranking agrees with recall@0.35, the old conclusion stands, now properly justified). Rationale: ranking quality and operating point are separate decisions; the threshold is set later (Phase 6) and is a business-owned knob — see `summary.md` §4.2 and §4.5.
  > **⚠ Flagged deviation from the archived notebook (per CLAUDE.md):** the notebook *selects and tunes* on CV recall@0.35 (`recall_scorer` is the Optuna objective; PR-AUC / ROC-AUC / average precision are logged there only as diagnostics). Standardising selection *and* the Optuna objective to PR-AUC is a deliberate methodological change (ranking-vs-operating-point separation), **not** a transcription — and note the plan's prior `metric: roc_auc` was itself an unflagged deviation. The notebook's preprocessing, hyperparameter ranges, calibration, threshold logic and evaluation math are otherwise preserved verbatim. Decision approved in design discussion (June 2026); record in `ANALYSIS.md` when Phase 5 lands.
- `src/telco_churn/models/train.py`:
  - Reads config via Hydra
  - Loads the feature matrix from `datasets/processed/`
  - Stratified 5-fold cross-validation; **optimisation objective is PR-AUC (average precision)** — imbalance-appropriate and consistent with the cross-family selection metric; ROC-AUC, recall, precision and F1 are also logged for diagnosis (see flagged deviation above)
  - Class imbalance: `scale_pos_weight` set to negative/positive ratio (~2.77)
  - Each Optuna trial is a nested MLflow child run; the study itself is the parent run
  - Best model logged as `pyfunc` artifact with `feature_columns.txt` and `preprocessing.pkl`
  - Best model registered in MLflow Model Registry as `telco-churn-pipeline` with alias `challenger`
- **Structural test-set isolation (no leakage by construction):** the train/val/test split lives in a dedicated module; `train.py` imports only the train + val splits, never test. The test split is importable *only* by `evaluate.py` (Phase 7). This makes the "test set touched once" invariant a structural guarantee, not a convention (former Group B item — see `summary.md` §4.6 for the sealed-test rationale and §4.4 for the profiling-holdout discipline).
- **Feature selection (`src/telco_churn/features/select.py`) — null-importance / target-permutation:** ranks features by comparing each feature's *real* gain importance against its importance distribution under repeated target shuffles; a feature is a drop candidate when its real importance is not meaningfully above its null distribution. Fit **inside CV on train + val only**, against a *default-config* LightGBM (not the tuned model — selection precedes HPO so the input space is frozen first). Returns the surviving feature list plus the importance/null table, and reports **selection stability** (how consistently each feature survives across folds). Mechanically, the selector is wrapped in an sklearn `Pipeline` with the model so `cross_val_score` re-fits selection on each fold's training portion only (leak-free — the standard safeguard); the single deployed feature set comes from one run on all training data. **Industry default, not nested CV:** the honest performance number is the Phase 7 sealed test reported with a bootstrap CI (`evaluate.py`) — the selection loop is deliberately *not* nested, matching production practice (a frozen feature set is required for serving/monitoring, and the CI already states the small-sample noise).
  > **⚠ Flagged deviation from the archived notebook (per CLAUDE.md):** the notebook performs **no** feature selection — it measures VIF / Cramér's V / permutation importance as *diagnostics* and deliberately keeps every feature. Adding an explicit selection step is a deliberate methodological addition for learning and portfolio completeness, **not** a transcription. It is constructed to be *honest*: the reduced set is adopted **only if** its CV PR-AUC is within the full set's bootstrap CI (no significant loss) *and* there is a parsimony / operational reason to prefer it; otherwise the full set stands. Record the actual keep/drop decision and its rationale in `ANALYSIS.md`. The notebook's preprocessing, hyperparameters, calibration, threshold and evaluation math are otherwise preserved verbatim.
- `notebooks/04-feature-selection.ipynb` — the experiment end to end: (1) full-set CV PR-AUC + bootstrap CI as the reference; (2) null-importance ranking; (3) drop features that do not beat noise; (4) **refit** LightGBM on the reduced set; (5) reduced-vs-full CV PR-AUC with an overlapping-CI check → documented decision. Imports from `select.py`; heavy logic stays out of the notebook.
- `tests/unit/test_select.py` — synthetic data with planted noise columns and known-informative columns; assert the selector drops the noise and keeps the signal; assert selection is fit inside the fold (no access to held-out rows); cover the empty-dataframe and all-noise edge cases.
- `tests/unit/test_train.py` — config loading, metric logging contract (mock MLflow client)
- `notebooks/03-model-selection.ipynb` — loads the Optuna study from MLflow; renders parallel-coordinates plot, hyperparameter importance, and validation AUC distribution across trials

**Verification:** `uv run python -m telco_churn.models.train` completes 50 trials and produces an MLflow run whose cross-validation **PR-AUC** falls within the bootstrap CI reported in `README.md`; the two reference baselines appear as rows in the comparison and LightGBM's PR-AUC is ≥ both. `notebooks/04-feature-selection.ipynb` runs end to end and records a keep/drop decision in `ANALYSIS.md` (a reduced set is adopted only if its CV PR-AUC stays within the full set's CI).

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

---

### Phase 7 — Evaluation + Error Analysis + Registry Promotion *(2 days)*

**What this achieves:** A sealed test-set evaluation (the test set has never been touched until this point) produces bootstrap-confidence-interval-bounded metrics that are honest estimates of production performance. A structured promotion decision replaces the `challenger` alias with `champion` only when the new model improves on both **ranking (PR-AUC)** and **calibration (Brier)**; recall at the operating threshold is reported but does not gate the decision.

**Deliverables:**
- `src/telco_churn/models/evaluate.py`:
  - Sealed test-set metrics: ROC-AUC, PR-AUC, recall, precision, F1, Brier score
  - 1,000-iteration bootstrap 95 % CIs (routine lifted verbatim from the original notebook)
  - Writes `reports/metrics.json`
  - Logs all metrics and the report to the MLflow run
- `src/telco_churn/models/register.py` — promotes `challenger` → `champion` if and only if it beats the current `champion` on both **PR-AUC** (ranking quality; threshold-free) and **Brier score** (calibration; lower is better); no promotion otherwise; logs the decision with structured event `model_promoted` or `model_rejected`. **The operating threshold is shipped as a separate versioned config artifact** alongside the model — *not* folded into the promotion comparison — so "is the new model better at ranking?" and "where do we cut?" stay independent, separately-auditable decisions. (Recall@threshold remains a *reported* metric; it does not gate promotion, because it inherits the fixed-threshold fragility discussed in `summary.md` §4.2.)
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
  | `features` | `features/build.py`, `sql/features/` | `datasets/processed/`, `preprocessing.pkl` |
  | `train` | `models/train.py`, `configs/` | MLflow run ID, `feature_columns.txt` |
  | `evaluate` | `models/evaluate.py` | `reports/metrics.json` |

- DVC local remote configured for now; swapped to S3 in Phase 12

> **Deliberate scope — why the DAG stops at `evaluate`:** the five DVC stages cover the *data-transform* pipeline (raw → reproducible metrics). Calibration + thresholding (Phase 6) and registry promotion (Phase 7) are intentionally **not** DVC stages. They are *decision* steps, not deterministic data transforms: calibration depends on a held-out fold, the threshold encodes a business cost choice (owned outside the pipeline — see `summary.md` §4.5), and promotion compares against the live `champion` in the MLflow registry, which is mutable state DVC cannot content-hash. Folding them in would make `dvc repro` non-deterministic (its output would depend on whatever `champion` currently exists). Instead, those steps are driven by the Phase 10 Prefect `retrain` flow, which calls `train → evaluate` (reproducible, DVC-tracked) and then `calibrate → threshold → register` (decision layer) as explicit flow tasks. If full champion reproducibility is ever required, the fix is to pin the comparison baseline to a specific run ID rather than the `champion` alias — not to add these as DVC stages.

- **No manual retraining flags (replaces the notebook's `RETRAIN_BEST` / `LOG_ARTIFACTS` booleans):** the hand-set flags that decided what to recompute do **not** migrate into `src/`. DVC's content-hashed DAG determines staleness — changing a dep reruns exactly the affected stages and nothing else. This is the engineering replacement for the manual flags (former Group B item).
- **Phase 2 cleanup:** Remove `clean_dataframe()` from `validate.py` — imputation now belongs to the `features` stage's fitted `SimpleImputer`. Update `validate_clean()` to expect the features stage output directly. Remove the associated tests.
- **Phase 2 cleanup:** In the DVC `validate` stage entry point, catch `ValidationError`, emit a `pipeline_blocked` structured log event, and call `sys.exit(1)`.

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
- **A live generative error-analysis → feature loop (v2)** — this project's story is *"productionised validated science"*: it ships the **converged** feature set from `EDA-original.ipynb` (which stays the source of truth), and Phase 7's error analysis is *confirmatory*. A natural **v2**, built *after* the production spine ships (post-Phase 14), turns the generative loop into reproducible code — error analysis on a profiling holdout proposes a **new** feature, the `select.py` + training harness re-measure it, and it is adopted only if it beats the frozen set on PR-AUC within CI. That would let the repo *also* claim "full DS lifecycle in code" without re-deriving work already done well. Deferred deliberately so the finishable migration artifact exists first; the v2 loop **extends**, not replaces, the notebook — see `summary.md` §4.4.

---

## Lifecycle & Framing Gaps (June 2026)

Items surfaced in a review of the project's workflow against the industry-standard data-science lifecycle (CRISP-DM and its MLOps descendants). The *modelling science* is already at or above standard — these are **documentation, framing, and engineering-discipline** gaps, not science gaps. Full reasoning is in `summary.md` → "The Industry Data Science Lifecycle".

**Group A — Documentation & framing (do before Phase 3):**

These four are pure documentation (no code) and lock the rules that govern Phases 5–7. Do them before starting the EDA notebook.

| Priority | Status | Item | Detail |
|---|---|---|---|
| High | [x] | Add "Step 0: Problem Framing & Cost Definition" as an explicit lifecycle step | Exists in `ANALYSIS.md` but is not a named step. Document: prediction unit (a customer), label definition + horizon, the decision the score feeds, FN-vs-FP cost structure, baseline-to-beat, and success criterion. This is the phase that makes the cost-sensitive threshold meaningful; reviewers look for it. |
| High | [x] | Reframe the workflow string as a loop, not a straight line | The linear `validation → EDA → … → registration` string hides the two feedback arrows. Show **error analysis in two places** (error-driven FE *before* tuning, deep error analysis *after*) and the Evaluation→Business / Modeling→Data-Prep loops. |
| Medium | [x] | Document the EDA-vs-validation ordering rationale | Phase 2 (validation) before Phase 3 (EDA) looks backwards without a note. Clarify: validation gates are *discovered* during EDA (notebook) and then *enforced* as automated checks (Phase 2) — the two orderings serve discover-vs-enforce purposes. Fold the note into the Phase 3 intro. |
| Medium | [x] | State the "test set touched once" and "one metric drives selection" invariants as written policy | Currently enforced by notebook convention only. Record in `CLAUDE.md` / `ANALYSIS.md` so they survive the migration to `src/` as explicit rules. |

**Former Groups B & C — now embedded in their phases (this is a completion index only; the phase deliverables are the source of truth):**

The engineering-discipline and metric/threshold items have been folded into the relevant phase deliverables so each spec lives in exactly one place and cannot drift. Use this table only to tick off completion as each phase lands.

| Status | Phase | Item | Specified in |
|---|---|---|---|
| [ ] | 5 | Reference baselines (floor + linear control) | Phase 5 deliverables; `summary.md` §4.1 |
| [ ] | 5 | PR-AUC for selection + Optuna objective (⚠ flagged deviation) | Phase 5 deliverables; `summary.md` §4.2 |
| [ ] | 5 & 7 | Test-set leakage structurally impossible | Phase 5 (split isolation bullet) + Phase 7 ordering note |
| [ ] | 7 | Promotion gate = PR-AUC + Brier; threshold as versioned artifact | Phase 7 `register.py` deliverable; `summary.md` §4.2 |
| [ ] | 8 | Drop notebook retraining flags (DVC DAG replaces them) | Phase 8 deliverables |
| [ ] | 9 | Threshold as a versioned policy layer | Phase 9 deliverables; `summary.md` §4.5 |
| [ ] | 3, 5–7 | Split the 7,600-line notebook into its three jobs | Phase 3 intro + roadmap |
