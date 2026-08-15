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
make data                                                 # download the raw CSV (skips if already present)
make db-up                                                # start Postgres + MLflow in Docker
dvc repro calibrate                                       # ingest -> validate -> split -> features -> train -> calibrate; exits 0 (see below)
make register-challenger                                  # mints calibrate's output as a new registry version, tagged pending
dvc repro error_analysis                                  # threshold -> evaluate -> error_analysis (calibrate and upstream stay cached)
make review VERDICT=approved APPROVER="..." NOTES="..."  # human verdict, stamped on top of evaluate's gate verdict
make register MODEL_VERSION=<version>                     # reads the gate verdict + review: promote or reject
```

`ingest`/`validate`/`split`/`features`/`train`/`calibrate`/`threshold`/`evaluate`/`error_analysis` are all DVC stages (`dvc.yaml`), not Makefile targets — `dvc.yaml` is their single definition, and every one of them resolves its input by default from whatever the previous stage's receipt says, so `dvc repro <stage>` is enough to run any of them with zero arguments. A Makefile target here would just retype `dvc.yaml`'s own `cmd:` a second time with none of DVC's caching.

**`dvc repro` genuinely cannot go straight through in one call, and that split is structural, not a workflow preference — but naming the target you actually want avoids ever seeing an error for it.** `threshold`/`evaluate` resolve their model via a *registered version*, and `register.py` (never `calibrate.py`) is the sole place that mints one, so nothing downstream of `calibrate` can succeed until `register-challenger` runs. `dvc repro <stage>` reproduces exactly that stage's upstream dependency chain and stops — never anything downstream — so `dvc repro calibrate` runs `ingest` through `calibrate` and exits 0 cleanly; it never attempts `threshold` at all, because `threshold` isn't upstream of `calibrate`. (A bare `dvc repro` would attempt the whole graph in one call and genuinely fail at `threshold` with a message naming `register-challenger` as the fix — not wrong, just a scarier first-run experience than naming the target.) `register-challenger` needs no `RUN_ID` for this default path either — it reads `calibrate`'s own receipt automatically, the same way `dvc repro`'s stages do. Once it has run, `dvc repro error_analysis` reproduces its own upstream chain — `calibrate` and everything before it are already cached and skipped, so what actually executes is `threshold -> evaluate -> error_analysis`, each resolving the version `register-challenger` just minted, automatically. `review` is the actual human-in-the-loop step, and it belongs after `evaluate` (inside the `error_analysis` target's upstream chain), not before `threshold`: `evaluate` computes the automated gate verdict that `review` then stamps a human decision on top of. `register` is the one step with no automatic default — it requires an explicit `MODEL_VERSION`, deliberately, since it is the step that can put a model into production.

Every command above also accepts an explicit override for re-running against a specific historical run/version instead of "whatever the previous step just produced" — `make calibrate RUN_ID=<run_id>` / `make threshold MODEL_VERSION=<version>` / `make evaluate MODEL_VERSION=<version>` / `make error-analysis MODEL_VERSION=<version>` / `make register-challenger RUN_ID=<run_id>` for a manual/debugging path `dvc repro` itself cannot express. Not needed for the sequence above. Run `make dag` to see the full stage graph, `make repro` as a synonym for a bare `dvc repro` (every stage, not just one target's upstream chain).

Each step's console output (structlog JSON) prints the `run_id` / `model_version` the next command needs.

## Testing

- `uv run pytest` — full unit suite, no Docker required.
- `uv run pytest -k "test_name"` — a single test by name.
- `make test-integration` — integration tests; requires a running Docker daemon. Each test spins up its own ephemeral Postgres via `testcontainers` and points MLflow at a tmp-scoped SQLite file, so `make db-up` is not a prerequisite.
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
3. CI must be green before merge. Unit tests, lint, and type-check run on every push and PR; the Docker-dependent integration suite only runs on the daily schedule, manual dispatch, or PRs targeting `main` — so a PR opened against `main` will also need the integration job green, even if you only ran unit tests locally.
4. Flag any schema-breaking changes in the PR description before implementing them.
