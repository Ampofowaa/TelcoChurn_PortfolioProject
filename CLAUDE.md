# CLAUDE.md

## Project Overview

End-to-end ML portfolio project on the IBM Telco Customer Churn dataset. The goal is a production-grade system covering data ingestion → validation → feature engineering → model training → serving → monitoring. The full build plan is in `PROJECT_PLAN.md` (15 phases, 0–14, ordered as a natural data-science lifecycle). Always consult it before starting work on a new phase or feature.

## Source of Truth

- **Modelling science:** `notebooks/_archive/EDA-original.ipynb` is a **comparison notebook, not a source of truth** — the original exploratory pass over data validation gates, preprocessing logic, feature engineering decisions, Optuna hyperparameters, calibration, cost-sensitive threshold derivation, and evaluation metrics. `src/` and `ANALYSIS.md` are authoritative wherever they diverge from it; the archive is useful only as a historical reference for *why* a decision changed, not as a bar current work must clear. Significant deviations (wider search ranges, different selection rules, different decision mechanisms entirely, etc.) are expected at this point in the project — document each new decision's rationale in `ANALYSIS.md`, or inline (config comments, docstring, commit/PR body) for smaller ones, rather than re-justifying current work against the archived notebook.
- **Modelling rationale:** `ANALYSIS.md` documents every modelling decision, analytical results and findings, and known limitations — the full narrative from data insights through to model results. **§0 (Problem Framing & Cost Definition)** is the entry point — it defines the prediction unit, label definition, cost structure, success criteria, and the two modelling invariants (test set touched once; one metric drives selection) that govern all phases. Do not contradict it.
- **System & workflow diagrams:** `docs/architecture.md` — system architecture diagram (infrastructure flow) and ML workflow diagram (modelling lifecycle with the two feedback loops).
- **Project landing page:** `README.md` is the recruiter-facing overview (headline metrics, pipeline diagram, quick start, project status). It does not contain modelling rationale.
- **Build roadmap:** `PROJECT_PLAN.md` defines the 15-phase lifecycle-ordered execution plan. Do not implement features that belong to a later phase.

## Key Commands

These commands are set up in Phase 0. Run them from the project root.

```bash
uv sync                          # install / sync dependencies
uv run pytest                    # run test suite
uv run pytest -k "test_name"     # run a single test by name
uv run ruff check src/           # lint
uv run ruff format src/          # format
uv run mypy src/                 # type-check (strict on src/ only)
uv run pre-commit run --all-files  # run all pre-commit hooks

dvc repro                        # re-run only changed pipeline stages
dvc repro --force                # re-run all stages regardless of cache

mlflow ui                        # open experiment tracking UI (localhost:5000)

make lint                        # shortcut: ruff check + mypy
make test                        # shortcut: pytest --cov=src
make validate                    # shortcut: uv run python -m telco_churn.data.validate
make train                       # shortcut: dvc repro (train + evaluate stages)

docker compose --profile infra up -d    # start Postgres + MLflow (Phase 1+ / Phase 5+)
docker compose up -d                    # start full local stack (Phase 9+)
```

## MLflow Model Registry

- **Tracking backend:** local `mlruns/` directory (Phase 5–11); migrated to RDS + S3 in Phase 12.
- **Registered model name:** `telco-churn-pipeline`
- **Production alias:** `champion` — the FastAPI service loads whichever run carries this alias at startup.
- **Challenger alias:** `challenger` — used during evaluation before promotion.

### Run artifacts vs. registry versions

The **run** is the experiment record; the **registry** is the deployment surface. Everything fitted during a training cycle is logged to the run and is retrievable there via `runs:/<run_id>/model` — nothing needs registering to be loadable. A registered model version answers exactly one question: *could this be serving traffic right now?*

- **Every version in the registry must be a valid rollback target.** Emergency rollback is `MlflowClient().set_registered_model_alias(name, "champion", version=N-1)`, typed under pressure. If undeployable intermediates occupy version numbers, that command succeeds, the service restarts cleanly, and it silently serves the wrong thing. Nothing errors and nothing alerts.
- **One registration per training cycle, at the point the artifact becomes serving-shaped.** Calibration is part of the model's inference graph, not a downstream consumer of it — `CalibratedClassifierCV(Pipeline([preprocessor, lgbm]))` is one sklearn estimator with one `predict_proba`, logged with one `log_model` call. The uncalibrated pipeline is a *stage of construction*, not a model that lost a comparison: it is logged to the run for auditability and **never registered**.
- Concretely: **Phase 5 logs, does not register** (no `registered_model_name=`, no alias). **Phase 6 `calibrate.py` performs the single registration** of the calibrated pipeline and points `challenger` at it. **Phase 7 `evaluate.py` → `refit.py` → `register.py`** then closes the cycle. An alias is a unique pointer per model name — re-aliasing does not create a second challenger, it silently orphans the previous version.
- **Model-version tag `refit_scope: dev | full`**, set at registration. The registry legitimately holds two shapes of deployable artifact: the dev-set calibrated model (sealed-test metrics of record) and the full-data refit (what actually serves). Both are valid rollback targets; only the `dev` one has honest test metrics attached, so they are not interchangeable for reporting.
- **The gate measures one artifact and promotes another.** Phase 7's promotion decision is computed on the `dev`-scope version's sealed-test PR-AUC and Brier; the `champion` alias is then pointed at the `full`-scope refit that `refit.py` produced from the same frozen spec. The refit can never pass the gate on its own — it was trained on the test set, so its test metrics exist, look excellent, and mean nothing. Log the two events distinctly: `model_promoted` (a comparison happened) vs. `champion_refit` (none did).
- **Do not overload `challenger` to mean "the dev-set model."** `challenger`/`champion` denote *incumbent versus contender for the same slot*. A dev-fit and a full-fit of one frozen spec are not rivals — they are the same model at two data volumes. Phase 10's weekly retrain uses `challenger` for a genuinely new candidate; reusing the word for the dev model would make the registry's history unreadable.
- **`models/refit.py` must not import `data.split`.** It reads `datasets/processed/features.parquet` in full, so the *test set touched once* invariant holds literally. The consequence is real, though: after the refit, the test set's information lives inside the champion and no clean holdout remains anywhere in the project. This is why Phase 10's promotion gate cannot re-score it — see the open question in `PROJECT_PLAN.md` Phase 10.

- **Logged artifacts per run (every training cycle, Phase 5+):** model (pyfunc), `feature_space.txt`, `feature_columns.txt`, `preprocessing.pkl`, `training_manifest.json`.
  - `feature_space.txt` — full **feature space**: every column `build_feature_df` produced before selection (the complete output of the Phase 4 feature pipeline).
  - `feature_columns.txt` — **model input space**: the subset that survived `select.py` and entered the `ColumnTransformer`. The diff between `feature_space.txt` and `feature_columns.txt` is what selection dropped for that run — a per-run audit trail that requires no git lookup.
  - `preprocessing.pkl` — fitted `ColumnTransformer`; encodes the exact transformations applied to the model input space at training time.
  - `training_manifest.json` — **engineering audit trail**: git SHA, DVC data hash, model family, full hyperparameters, `feature_space`/`feature_columns`, CV PR-AUC, the paired-Δ vs LogReg with its full bootstrap CI, and `tuning_summary` (trial counts, selected vs. raw-best trial number/score, 1-SE standard error and band floor — the diagnostics behind the `selection_rule` pick).
- **`model_card.json` — written once, at champion promotion (Phase 7 `register.py`), not at Phase 6 challenger registration.** Stakeholder-facing narrative: intended use, known limitations, performance summary vs. LogReg, fixed-recall profile, fairness/robustness flags. It is not populated when `calibrate.py` registers the challenger — at that point the model is calibrated but un-thresholded and untested on the sealed set, so a "known limitations" section would still be a disclaimer about an unfinished product. No hyperparameters, hashes, or raw statistical detail either way — that lives in `training_manifest.json`.

## Data Handling

- **`datasets/raw/` is read-only.** Never modify, overwrite, or delete any file there.
- **The raw CSV is committed to Git; DVC owns only derived artifacts.** DVC lists `datasets/raw/*.csv` as a stage **`dep`** (hashed into `dvc.lock`, re-runs the DAG on change), not an **`out`** (cached, gitignored, remote-backed). A dep may be any path in the repo. The file is 977 KB, static, and read-only — none of DVC's reasons to own raw data apply, and gitignoring it would break fresh-clone reproducibility and Phase 11's CI. `datasets/processed/` is gitignored; DVC owns it as stage output.
- **No DVC remote is required before Phase 12.** A local cache is the full configuration. Phase 12 *adds* an S3 remote for shared caching; it is not a migration, and nothing depends on it.
- **Postgres is a materialised cache of the CSV, never a source of truth.** Its contents are fully determined by the raw CSV + `sql/schema/` + `data/ingest.py`. It cannot be a DVC `out` (outs are files); the `ingest` stage emits `reports/ingest_receipt.json` instead. A hand-edited `customers_raw` is invisible to `dvc repro` — rebuild with `dvc repro --force`.
- **`datasets/processed/telco_churn_processed.csv`** is an intermediate artifact — it will be replaced by the DVC `features` stage output (`telco_churn_features.parquet`). Read it through `features/accessor.py::load_features()`, never by constructing the path inline.
- **TotalCharges gotcha:** the raw CSV has `TotalCharges` as a string with whitespace for 11 zero-tenure customers. Coerce to numeric first, then impute with median — never drop.
- **Target column** is always named `churn` (binary: 0/1).
- **Class imbalance handling is required** — never train a model without it.
- **Random state is always 42** for reproducibility across all splits, samplers, and models.

## Modelling Invariants

These are written policy — not notebook conventions — and must be preserved in all `src/` code:

- **Test set touched once.** `X_test` / `y_test` are imported and used in exactly one place: `models/evaluate.py`. No other module may access the test split. Under continuous retraining (Phase 10), the sealed test set is **not** reused for challenger-vs-champion comparisons — that erodes it; use a rolling holdout instead.
- **`evaluate.py` resolves its model by explicit `run_id` or version number, never by alias.** An alias is a moving pointer. Once `models/refit.py` exists, a re-run or mis-ordered pipeline can leave `challenger` on the `full`-scope refit — a model trained on the test set — and evaluating *that* reports excellent sealed-test metrics with nothing raised and nothing logged. This is the contamination guard; module ordering is not. Log the resolved version into `reports/metrics.json` so metrics are attributable to a specific artifact. A refactor to `models:/telco-churn-pipeline@challenger` is a correctness regression, not a simplification.
- **One metric drives selection.** PR-AUC (average precision) is the sole criterion for model selection, Optuna tuning, and champion promotion. All other metrics (recall, precision, F1, ROC-AUC) are reported as diagnostics only. If two metrics point in different directions, PR-AUC wins — mixing selection signals introduces cherry-picking risk.

## Project Structure (target — being built phase by phase)

```
src/telco_churn/        # importable Python package
configs/                # Hydra YAML configs (model params, paths, thresholds)
sql/                    # Postgres schema + feature-engineering SQL views
tests/unit/             # pytest unit tests (≥80% coverage target on src/)
tests/integration/      # integration tests (Postgres, API, pipeline smoke)
pipelines/              # Prefect flows (retrain, drift_check, batch_predict)
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

## Changelog Conventions

Update `CHANGELOG.md` at the end of every phase or significant fix. Follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Audience:** engineers consuming the project. Recruiters are a secondary audience — the italic summary per version handles them; the bullets do not need to.

**Structure per version:**
- Version header: `## [X.Y.Z] - YYYY-MM-DD — Phase N: Short Title`
- Italic summary (1–3 sentences): what the phase accomplishes and any key architectural boundary. Keep it concise.
- `### Added / ### Changed / ### Fixed` sections as needed.

**Bullet rules:**
- 1–2 lines max per bullet. State what changed; add a short *why* clause only if non-obvious.
- Include file paths and function names so the entry is navigable (`src/telco_churn/features/build.py`).
- Keep test counts and coverage numbers — they are concrete signals worth preserving.
- **Do not embed data statistics** (churn rates, median values, percentage gaps) — those belong in `ANALYSIS.md`.
- **Do not explain implementation rationale** in depth — that belongs in commit messages or PR descriptions.
- Fixed items: one clause on what was wrong, one clause on what was done. Stop there.

## Code Style

- **Formatter:** `black` (line length 88). `ruff` for linting (replaces flake8/isort).
- **Type hints:** required on all public functions in `src/`. `mypy --strict` on `src/` only.
- **`__all__` required in every public module under `src/`.** List every public function, class, and constant. Omit underscore-prefixed internals. Place it after the imports block, before the first definition.
- **All functions must have docstrings.**
- **No comments explaining what the code does** — use clear names instead. Only comment the non-obvious *why* (hidden constraints, workarounds, subtle invariants).
- **No error handling inside pure functions.** Handle errors at system boundaries only: file I/O, DB calls, model load, external API calls.
- **Never use bare relative paths for file I/O in `src/`.** `"configs/config.yaml"` resolves from the shell's CWD and breaks in DVC stages, Prefect workers, CI, and Docker containers. Always anchor to `get_project_root() / "configs" / "config.yaml"` (`utils/paths.py`). This applies everywhere in `src/` — `__main__` blocks, module-level constants, and function bodies.
- **All `logger.error(...)` calls inside `except` blocks must include `exc_info=True`.** `str(e)` logs only the message; `exc_info=True` attaches the full traceback so failures are diagnosable in CI and pipeline logs.
- **Notebooks** are thin wrappers: import from `src/`, call functions, render outputs. Heavy logic lives in `src/`.

## Notebook Conventions

### Heading hierarchy

Four levels, strictly nested — never skip a level:

| Level | Markdown | Usage |
|---|---|---|
| `#` | `# Title` | Notebook title — one per notebook |
| `##` | `## N. Section name` | Top-level numbered sections; unnumbered only for closing structural sections (e.g. `## Summary`, `## Provenance Output`) |
| `###` | `### Topic name` | Named topics within a section — LAPs, deep-dives, structural conclusions, analysis sub-groups |
| `####` | `#### Sub-step name` | Protocol or analysis sub-steps — `#### Construct`, `#### Evaluate`, `#### Record`, `#### Demographics` |

**No alphanumeric sub-labels on `###` in workflow/protocol notebooks** (e.g. `02a`, `02b`, future modelling notebooks). Topic headers are named descriptively; numbering adds no information and rots when sections are restructured.

**Exception — EDA and reference notebooks** (`01-eda.ipynb` and similar): `###` sections may carry alphanumeric labels (e.g. `### 6d. Numeric features vs churn`) when those labels serve as stable cross-reference addresses cited by downstream notebooks. Do not strip them.

### Cross-references between notebooks

Use `§N` notation when referring to a section in another notebook:

- Top-level section: `§6 \`01-eda.ipynb\`` → `## 6. Bivariate analysis`
- Subsection: `§6d \`01-eda.ipynb\`` → `### 6d. Numeric features vs churn`
- `ANALYSIS.md` sections: `§2` or a Markdown link `[§2 EDA](../ANALYSIS.md#2-eda--statistical-testing)`

Always use the `§N` format consistently — not "Section N", not "chapter N", not bare numbers.

## Testing

- Target ≥80% coverage on `src/`. Do not chase 100%.
- Run a single test: `uv run pytest -k "test_name"` or `uv run pytest tests/unit/test_checks.py`.
- Integration tests require Docker services to be running (`docker compose --profile infra up -d`).
- The integration smoke test trains on a 500-row stratified sample and asserts ROC-AUC ≥ 0.75.
- When writing data, schema, or validation tests, cover: normal case, missing values, wrong dtypes, and empty dataframe.
- Every module with a `__main__` CLI entry point requires a **subprocess** integration test — invoked via `subprocess.run([sys.executable, "-m", "<module>"], env={...}, capture_output=True)` — covering the full composition path (argparse/config load → I/O → exit code). Direct function calls do not qualify: they miss argparse, OmegaConf resolution, dotenv loading, and env-var-to-engine joints that only surface at the subprocess boundary. Cover both the exit 0 (success) and exit 1 (error) paths. Exception: waived only when the module's entire `__main__` body is exercised as a named subroutine call inside another module's subprocess integration test — direct calls to shared helper functions do not satisfy the exception.
- Running integration tests in isolation will fail the `fail_under=80` gate because coverage is measured across all `src/` modules. Append `--no-cov` for standalone integration runs: `uv run pytest tests/integration/ --run-integration --no-cov`.
- Every new package added under `src/telco_churn/` must have a scoped `make test-<package>` Makefile target using `--override-ini="addopts=" --cov=src/telco_churn/<package>`. The global `make test` remains the CI gate; the scoped target lets phase tests run in isolation without a false `fail_under=80` failure from other packages.
- Run `pytest` before marking any task complete; if the phase has no tests yet, note it explicitly instead of skipping.
- When closing a QA backlog item (`[x]`), verify the fix is present in the code — read the relevant file or run the relevant test. Never mark `[x]` based on intent; only mark it after the change is confirmed in the working tree.

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
| 7 | Evaluation + error analysis + full-data refit + registry promotion | `models/evaluate.py` + `models/refit.py` + `models/register.py` + `notebooks/05-error-analysis.ipynb` |
| 8 | DVC pipeline wrap | `dvc.yaml` with 5 stages; reproducible end-to-end |
| 9 | Serving + UI | FastAPI (`/predict`, `/health`, `/metrics`) + Streamlit + Dockerfiles |
| 10 | Orchestration | Prefect retrain flow (weekly) + drift check flow (daily) |
| 11 | CI/CD | GitHub Actions ci.yml + cd.yml + data-quality.yml |
| 12 | AWS deployment | ECR + App Runner + RDS + S3 |
| 13 | Monitoring | Prometheus + Grafana dashboards + Evidently drift |
| 14 | Docs polish | README update + runbook + architecture diagram |
