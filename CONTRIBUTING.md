# Contributing

See [README.md](README.md) for the project overview and pipeline, and [CLAUDE.md](CLAUDE.md) for the governing conventions (Git branch/commit rules, code style, testing philosophy) — this file covers setup and day-to-day commands only, and defers to `CLAUDE.md` rather than duplicating its rules, to avoid the two drifting apart.

## Setup

```bash
uv sync
pre-commit install
```

Copy `.env.example` to `.env` and fill in the required values. For Phases 0–4 only `POSTGRES_URL` (and the `POSTGRES_*` vars) are needed; the MLflow, Prefect, and AWS vars are filled in as those phases are completed.

`make data` (below) requires a [Kaggle API token](https://www.kaggle.com/settings/api) configured for the Kaggle CLI.

## Available make commands

Run `make help` for the full, current list — it's generated directly from the Makefile, so it can't go stale the way a hand-maintained table here would. The typical first-time pipeline run, in order:

```bash
make data                                       # download the raw CSV (skips if already present)
make db-up                                      # start Postgres + MLflow in Docker
make ingest                                     # load the CSV into Postgres
make validate                                   # run the 5 Pandera quality gates
make features                                   # build SQL feature views
make train                                      # LightGBM + Optuna tuning; logs to MLflow
make calibrate RUN_ID=<run_id>                  # sigmoid calibration; registers the challenger
make threshold MODEL_VERSION=<version>          # closed-form cost-sensitive threshold
make evaluate MODEL_VERSION=<version>           # one-time sealed-test evaluation + gate
make error-analysis MODEL_VERSION=<version>     # SHAP explainability + error diagnosis
```

Each step's console output (structlog JSON) prints the `run_id` / `model_version` the next command needs.

## Testing

- `uv run pytest` — full unit suite, no Docker required.
- `uv run pytest -k "test_name"` — a single test by name.
- `make test-integration` — integration tests; requires Docker (`make db-up` first).
- `make test-data` / `make test-features` / `make test-models` — scoped coverage per package, for running one phase's tests in isolation without a false `fail_under=80` failure from unrelated packages.
- Every new package under `src/telco_churn/` needs its own scoped `make test-<package>` target (see `CLAUDE.md`'s Testing section for the required pattern and coverage conventions).

## Pre-commit hooks

The following hooks run automatically on every `git commit`:

- **ruff** — lint with auto-fix
- **black** — format
- **mypy** — type-check `src/`
- **nbstripout** — strips notebook outputs before committing, *except* `notebooks/_archive/` and the numbered presentation notebooks (`00-`, `01-`, `02a-`, … `05-`), which keep their outputs so they render on GitHub
- **fix-notebook-outputs** — coalesces fragmented stream outputs in executed notebooks (a Windows/ipykernel quirk — see `scripts/fix_notebook_outputs.py`'s docstring) so diffs stay clean
- **upgrade-notebooks** — upgrades any notebook to nbformat 4.5
- **detect-secrets** — blocks accidental credential commits
- **end-of-file-fixer**, **trailing-whitespace**, **check-yaml**, **check-toml**

If a hook fails, fix the issue, re-stage the affected files, and commit again.

## Branch and commit conventions

Follow `CLAUDE.md`'s **Git Conventions** section (branch prefixes, Conventional Commits style) — not restated here so the two files can't disagree.

## Submitting a PR

1. Ensure `make lint` and `make test` both pass locally.
2. Push your branch and open a PR against `main`.
3. CI must be green before merge.
4. Flag any schema-breaking changes in the PR description before implementing them.
