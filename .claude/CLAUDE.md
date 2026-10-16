# CLAUDE.md

## Project Overview

End-to-end ML portfolio project on the IBM Telco Customer Churn dataset. The goal is a production-grade system covering data ingestion → validation → feature engineering → model training → serving → monitoring. The full build plan is in `docs/PROJECT_PLAN.md` (15 phases, 0–14, ordered as a natural data-science lifecycle). Always consult it before starting work on a new phase or feature.

## Source of Truth

- **Modelling science:** `notebooks/_archive/EDA-original.ipynb` is a **comparison notebook, not a source of truth** — the original exploratory pass over data validation gates, preprocessing logic, feature engineering decisions, Optuna hyperparameters, calibration, cost-sensitive threshold derivation, and evaluation metrics. `src/` and `ANALYSIS.md` are authoritative wherever they diverge from it; the archive is useful only as a historical reference for *why* a decision changed, not as a bar current work must clear. Significant deviations (wider search ranges, different selection rules, different decision mechanisms entirely, etc.) are expected at this point in the project — document each new decision's rationale in `ANALYSIS.md`, or inline (config comments, docstring, commit/PR body) for smaller ones, rather than re-justifying current work against the archived notebook.
- **Modelling rationale:** `ANALYSIS.md` documents every modelling decision, analytical results and findings, and known limitations — the full narrative from data insights through to model results. **§0 (Problem Framing & Cost Definition)** is the entry point — it defines the prediction unit, label definition, cost structure, success criteria, and the two modelling invariants (test set touched once; one metric drives selection) that govern all phases. Do not contradict it.
- **System & workflow diagrams:** `docs/architecture.md` — system architecture diagram (infrastructure flow) and ML workflow diagram (modelling lifecycle with the two feedback loops).
- **Project landing page:** `README.md` is the recruiter-facing overview (headline metrics, pipeline diagram, quick start, project status). It does not contain modelling rationale.
- **Build roadmap:** `docs/PROJECT_PLAN.md` defines the 15-phase lifecycle-ordered execution plan. Do not implement features that belong to a later phase.

## Key Commands

These commands are set up in Phase 0. Run them from the project root.

```bash
uv sync --all-extras              # install / sync dependencies (serving + ui extras included)
uv run pytest                    # run test suite
uv run pytest -k "test_name"     # run a single test by name
uv run ruff check src/           # lint
uv run mypy src/                 # type-check (strict on src/ only)
uv run pre-commit run --all-files  # run all pre-commit hooks

dvc repro                        # re-run only changed pipeline stages
dvc repro train                  # re-run a single stage (and anything it invalidates downstream)
dvc repro --force                # re-run all stages regardless of cache
dvc dag                          # show the pipeline DAG

mlflow ui                        # open experiment tracking UI (localhost:5000)

make format                      # ruff format src/ then black src/ — always in this order
make lint                        # shortcut: ruff check + mypy
make test                        # shortcut: pytest --cov=src
make calibrate                   # shortcut: uv run python -m telco_churn.models.calibrate (optional RUN_ID= override)

docker compose up -d postgres mlflow    # start Postgres + MLflow only, no api/ui build (Phase 1+ / Phase 5+)
docker compose up -d                    # start full local stack (Phase 9+)
```

## Detailed Conventions — read before changing the code they cover

The full rules, with their reasoning, live in `docs/conventions/`. This file keeps only the summary. **Before editing code in an area below, read its file.** A refactor that breaks one of these rules is a correctness regression, not a simplification.

## MLflow Model Registry

Full rules: [`docs/conventions/mlflow-registry.md`](../docs/conventions/mlflow-registry.md). Read it before touching `models/` registration, evaluation logging, `drift_reference.json` or `model_card.json`.

- Tracking backend is Postgres via the MLflow server; `mlruns/` is the artifact store only. The `tracking_uri` fallback must stay `sqlite:///mlflow.db`, never a bare `mlruns` path.
- Registered model `telco-churn-pipeline`; aliases `champion` (served) and `challenger`.
- Every version is tagged `promotion_status: pending` at mint time; only `promoted` versions are rollback targets. Roll back by tag, never by version arithmetic. Stale `pending` versions may be reaped (age + tag filter, never version number).
- Validate, then promote: smoke-check the explicit version URI before the alias flip; re-check through the alias and roll back on failure.
- One registration per training cycle, of the calibrated pipeline. Phase 5 and `calibrate.py` log only; `register.py` is the only module that writes to the registry or sets a model-version tag.
- The gate is the pure `models/gate.py::decide_promotion`, called by `evaluate.py` and logged as `promotion_decision.json`. The human verdict is the append-only `promotion_review.json` written by `models/review.py`. `register.py` reads both, never recomputes.
- `register.py` resolves cross-cycle artifacts by tag/`runs:/` URI, never by a local `reports/` path (receipts are a one-time bootstrap only).
- Log the evidence arrays, not just conclusions (`dev_oof_predictions.parquet`, `shap_values.parquet`).
- Evaluation gets its own run; metrics are `test_*`-prefixed and attached to `(model_id, dataset)`. The four gate criteria are model-version tags; EV is never one. Log cost inputs as params plus a `costs_config_hash` tag.
- `drift_reference.json` and `model_card.json` are written at champion promotion, logged on the promoted run, and read back via the `champion` alias. The score baseline uses out-of-sample probabilities only. Model-card baselines are treat-none/treat-all/Dummy, never LogReg.

## Data Handling and DVC Pipeline

Full rules: [`docs/conventions/data-and-pipeline.md`](../docs/conventions/data-and-pipeline.md). Read it before touching `data/`, `features/`, `dvc.yaml`, receipts or `customers_crm`.

- `datasets/raw/` is read-only. The raw CSV is committed to git and is a DVC dep, not an out.
- Postgres is a materialised cache of the CSV, never a source of truth. A hand-edited `customers_raw` needs `dvc repro --force`. Per-cycle receipts are `cache: false` and gitignored.
- `customers_crm` is serving-only (`make crm-data`), never read by training.
- Read processed features only through `features/accessor.py::load_features()`.
- TotalCharges: coerce to numeric, then median-impute; never drop. Target is `churn` (0/1). Class imbalance handling is required.
- Random state is always 42, read from each stage's own config key, never hardcoded.
- DVC tracks what the DAG consumes; MLflow owns terminal artifacts; figures are never DVC outs. Registration stays outside the DAG — a full cycle is `dvc repro calibrate` → `make register-challenger` → `dvc repro error_analysis` → `make review` → `make register`.
- `params:` deliberately omits env-resolved keys, identity tokens, `paths.*` and `mlflow.experiment_name`; `test_architecture.py` enforces read ⊆ declared.

## Modelling Invariants and Serving

Full rules: [`docs/conventions/modelling-and-serving.md`](../docs/conventions/modelling-and-serving.md). Read it before touching evaluation, the gate, or anything under `serving/`.

- Test set touched once: only `models/evaluate.py` binds the test partition. Routine retrain cycles gate on the `reserve` cohort, not the sealed set.
- `evaluate.py` resolves its model by explicit run_id/version, never by alias.
- PR-AUC is the sole selection metric. Recall at `t*`, Brier and calibration slope may veto, never promote. The gate is specified in `ANALYSIS.md` §0 only.
- Serving resolves models only via `resolve_champion_model`/`resolve_challenger_model` (TTL hot-reload; fail-open on refresh, fail-closed on initial load). The SHAP explainer and threshold travel in the same `ModelBundle`.
- Shadow and canary are independent toggles, both off by default; canary hashes `customerid` salted with the challenger version. `customerid` is never a model feature.
- `decision` and `contact` are separate bool fields. Probe routes never need an API key. Adopt `/v1` routes by Phase 11. The API image builds from `uv.lock`.

## Project Structure (target — being built phase by phase)

```
src/telco_churn/        # importable Python package
configs/                # Hydra YAML configs (model params, paths, thresholds)
sql/                    # Postgres schema + feature-engineering SQL views
tests/unit/             # pytest unit tests (≥80% coverage target on src/)
tests/integration/      # integration tests (Postgres, API, pipeline smoke)
pipelines/              # Prefect flows (training_cycle, drift_check, performance_check, performance_monitor, nightly_batch_predict)
monitoring/             # Prometheus config + Grafana dashboard JSON
.github/workflows/      # CI (ci.yml), CD (cd.yml), data quality (data-quality.yml)
```

## Git Conventions

- **Branch prefix:** `feat/`, `fix/`, `chore/`, `docs/`, `test/`, `refactor/`, `data/`, `model/` — e.g. `feat/phase-0-tooling`, `fix/schema-null-handling`, `data/phase-1-ingest`.
- **Commit style:** Conventional Commits — `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `data:`, `model:`.
  - `data:` — dataset changes, schema updates, DVC pipeline edits, SQL changes.
  - `model:` — hyperparameter tuning, training logic, calibration, threshold changes.
- **Base branch:** `main`. Open a PR for each phase or significant feature; do not commit directly to `main`.
- **Flag any schema-breaking changes** before implementing them — raise it explicitly rather than proceeding silently.

## Changelog, Code Style, Notebooks and Testing

Full rules: [`docs/conventions/engineering.md`](../docs/conventions/engineering.md). Read it before editing `CHANGELOG.md`, a notebook, or adding a module, CLI entry point or test.

- **Changelog:** Keep a Changelog; update at the end of every phase or significant fix. Bullets 1–2 lines with file paths; keep test counts; no data statistics or deep rationale.
- **Formatting:** always `make format` (ruff format, then black — black has final say). `mypy --strict` on `src/`.
- **Every public `src/` module** has `__all__` after its imports; every function has a docstring. Comment only the non-obvious *why*.
- No error handling inside pure functions; handle errors at I/O, DB, model-load and API boundaries. `logger.error` inside `except` always passes `exc_info=True`.
- Never bare relative paths in `src/` — anchor to `get_project_root()`. Every `compose_config()` call is followed by `activate_config(cfg)`.
- No `src/` module imports from a `__main__`-bearing module except the named allowances; shared logic goes in a plain module.
- Notebooks are thin wrappers over `src/`; strict `#`→`####` heading hierarchy; cross-reference with `§N`.
- **Testing:** ≥80% coverage on `src/`. Integration tests need Docker (testcontainers), run with `--run-integration --no-cov` standalone. Every `__main__` module needs a subprocess test covering exit 0 and exit 1. New packages get a scoped `make test-<package>` target. Run `pytest` before calling a task complete.

## Environment Variables

Copy `.env.example` to `.env` for local development. Never commit `.env`. Required vars (set up per phase):

```
POSTGRES_URL=postgresql://user:pass@localhost:5432/telco_churn  # pragma: allowlist secret
MLFLOW_TRACKING_URI=http://localhost:5000
PREFECT_API_URL=http://localhost:4200/api
AWS_REGION=us-east-1
```

## Phase Checklist (quick reference)

Lifecycle-ordered. Tests are written alongside each module — there is no dedicated "tests phase."

| Phase | Goal | Key deliverable |
|---|---|---|
| 0 | Project foundation | `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, skeleton dirs, Hydra root, structlog |
| 1 | Data ingestion (CSV → Postgres) | `docker-compose.yml` w/ Postgres + `sql/schema/001_create_raw.sql` + `data/ingest.py` |
| 2 | Data validation (Pandera + 5 gates) | `data/schema.py` + `data/checks.py` + `data/validate.py` + `tests/unit/test_checks.py` + `tests/unit/test_validate.py` |
| 3 | EDA notebook (slim rewrite) | `notebooks/01-eda.ipynb` importing from `src/`; original archived |
| 4 | Feature engineering (SQL + Python) | `sql/features/*.sql` + `features/sql_features.py` + `features/build.py` |
| 5 | Model training (LightGBM + Optuna + MLflow) | `models/train/` + `configs/{model,training}/*.yaml`; logged only, no registration |
| 6 | Calibration + cost-sensitive threshold | `models/calibrate.py` + `models/threshold.py` |
| 7 | Evaluation + error analysis + registry promotion | `models/evaluate.py` + `models/gate.py` + `models/economics.py` + `models/explain.py` + `models/error_analysis.py` + `models/register.py` + `models/drift_reference.py` + `notebooks/05-evaluation-and-error-analysis.ipynb` |
| 8 | DVC pipeline wrap | `dvc.yaml` with 9 stages; reproducible end-to-end |
| 9 | Serving + UI | FastAPI (`/predict`, `/health`, `/metrics`) + Streamlit + Dockerfiles |
| 10 | Orchestration | Prefect retrain flow (monthly) + drift check flow (daily) |
| 11 | CI/CD | GitHub Actions ci.yml + cd.yml + data-quality.yml |
| 12 | AWS deployment | ECR + App Runner + RDS + S3 |
| 13 | Monitoring | Prometheus + Grafana dashboards + Evidently drift |
| 14 | Docs polish | README update + runbook + architecture diagram |
