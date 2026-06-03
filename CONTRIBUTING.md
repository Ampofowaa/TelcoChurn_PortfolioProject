# Contributing

## Setup

```bash
uv sync
pre-commit install
```

Copy `.env.example` to `.env` and fill in the required values. For Phases 0–2 only `POSTGRES_URL` (and the `POSTGRES_*` vars) are needed; the MLflow, Prefect, and AWS vars are filled in as those phases are completed.

## Available make commands

| Command | What it does |
|---|---|
| `make lint` | `ruff check src/` + `mypy src/` |
| `make format` | `ruff format src/` + `black src/` |
| `make test` | `pytest` (unit tests — no Docker required) |
| `make test-integration` | `pytest -m integration` (requires Docker — run `make db-up` first) |
| `make data` | Download raw dataset via Kaggle CLI into `datasets/raw/` |
| `make db-up` | Start Postgres in Docker (`docker compose --profile infra up -d`) |
| `make db-down` | Stop and remove the Postgres container |
| `make ingest` | Load raw CSV into Postgres (`python -m telco_churn.data.ingest`) |
| `make validate` | Run the 5 Pandera validation gates (`python -m telco_churn.data.validate`) |
| `make train` | Re-run the DVC pipeline (`dvc repro`) |

## Pre-commit hooks

The following hooks run automatically on every `git commit`:

- **ruff** — lint with auto-fix
- **black** — format
- **mypy** — type-check `src/`
- **nbstripout** — strips notebook outputs before committing (keeps cell IDs)
- **upgrade-notebooks** — upgrades any notebook to nbformat 4.5
- **detect-secrets** — blocks accidental credential commits
- **end-of-file-fixer**, **trailing-whitespace**, **check-yaml**, **check-toml**

If a hook fails, fix the issue, re-stage the affected files, and commit again.

## Branch and commit conventions

- Branch from `main` using the appropriate prefix:
  `feat/`, `fix/`, `chore/`, `docs/`, `test/`, `refactor/`, `data/`, `model/`
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `chore:`, `data:`, `model:`, etc.
- One logical change per PR; keep PRs small and reviewable.

## Submitting a PR

1. Ensure `make lint` and `make test` both pass locally.
2. Push your branch and open a PR against `main`.
3. CI must be green before merge.
4. Flag any schema-breaking changes in the PR description before implementing them.
