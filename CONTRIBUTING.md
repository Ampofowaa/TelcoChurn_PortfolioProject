# Contributing

See [README.md](README.md) for the project overview and pipeline, and [CLAUDE.md](.claude/CLAUDE.md) for the governing conventions (Git branch/commit rules, code style, testing philosophy) — this file covers setup and day-to-day commands only, and defers to `CLAUDE.md` rather than duplicating its rules, to avoid the two drifting apart.

## Setup

```bash
uv sync --all-extras
pre-commit install
```

Copy `.env.example` to `.env` and fill in the required values. For Phases 0–4 only `POSTGRES_URL` (and the `POSTGRES_*` vars) are needed; the MLflow, Prefect, and AWS vars are filled in as those phases are completed.

`make data` (below) requires a [Kaggle API token](https://www.kaggle.com/settings/api) configured for the Kaggle CLI.

## Available make commands

Run `make help` for the full, current list — it's generated directly from the Makefile, so it can't go stale the way a hand-maintained table here would. The typical first-time pipeline run, in order:

```bash
make data                                                 # download the raw CSV (skips if already present)
make db-up                                                # start Postgres + MLflow in Docker
make repro STAGE=calibrate                                # ingest -> validate -> split -> features -> train -> calibrate; exits 0 (see below)
make register-challenger                                  # mints calibrate's output as a new registry version, tagged pending
make repro STAGE=error_analysis                           # threshold -> evaluate -> error_analysis (calibrate and upstream stay cached)
make review VERDICT=approved APPROVER="..." NOTES="..."  # human verdict, stamped on top of evaluate's gate verdict
make register MODEL_VERSION=<version>                     # reads the gate verdict + review: promote or reject
```

`ingest`/`validate`/`split`/`features`/`train`/`calibrate`/`threshold`/`evaluate`/`error_analysis` are all DVC stages (`dvc.yaml`), not Makefile targets — `dvc.yaml` is their single definition, and every one of them resolves its input by default from whatever the previous stage's receipt says, so `dvc repro <stage>` is enough to run any of them with zero arguments. A per-stage Makefile target would just retype `dvc.yaml`'s own `cmd:` a second time with none of DVC's caching; `repro`'s `STAGE=` parameter avoids that duplication by staying a generic pass-through (`dvc repro $(STAGE)`) rather than a named target per stage — see **Pre-commit hooks** below for why `make repro STAGE=<stage>` is preferred over calling `dvc repro <stage>` directly.

**`dvc repro` genuinely cannot go straight through in one call, and that split is structural, not a workflow preference — but naming the target you actually want avoids ever seeing an error for it.** `threshold`/`evaluate` resolve their model via a *registered version*, and `register.py` (never `calibrate.py`) is the sole place that mints one, so nothing downstream of `calibrate` can succeed until `register-challenger` runs. `dvc repro <stage>` reproduces exactly that stage's upstream dependency chain and stops — never anything downstream — so `dvc repro calibrate` runs `ingest` through `calibrate` and exits 0 cleanly; it never attempts `threshold` at all, because `threshold` isn't upstream of `calibrate`. (A bare `dvc repro` would attempt the whole graph in one call and genuinely fail at `threshold` with a message naming `register-challenger` as the fix — not wrong, just a scarier first-run experience than naming the target.) `register-challenger` needs no `RUN_ID` for this default path either — it reads `calibrate`'s own receipt automatically, the same way `dvc repro`'s stages do. Once it has run, `dvc repro error_analysis` reproduces its own upstream chain — `calibrate` and everything before it are already cached and skipped, so what actually executes is `threshold -> evaluate -> error_analysis`, each resolving the version `register-challenger` just minted, automatically. `review` is the actual human-in-the-loop step, and it belongs after `evaluate` (inside the `error_analysis` target's upstream chain), not before `threshold`: `evaluate` computes the automated gate verdict that `review` then stamps a human decision on top of. `register` is the one step with no automatic default — it requires an explicit `MODEL_VERSION`, deliberately, since it is the step that can put a model into production.

Every command above also accepts an explicit override for re-running against a specific historical run/version instead of "whatever the previous step just produced" — `make calibrate RUN_ID=<run_id>` / `make threshold MODEL_VERSION=<version>` / `make evaluate MODEL_VERSION=<version>` / `make error-analysis MODEL_VERSION=<version>` / `make register-challenger RUN_ID=<run_id>` for a manual/debugging path `dvc repro` itself cannot express. Not needed for the sequence above. Run `make dag` to see the full stage graph. `make repro` (bare) runs the whole graph like a bare `dvc repro`; `make repro STAGE=<stage>` runs just that stage's upstream chain like `dvc repro <stage>` — either way `make repro` formats `src/` first, which calling `dvc repro` directly does not. See **Pre-commit hooks** below for why that matters.

Each step's console output (structlog JSON) prints the `run_id` / `model_version` the next command needs.

## Running the full stack locally

The complete first-time training cycle plus serving, step by step. (Nothing to install? The live demo is at **[telco-churn.duckdns.org](https://telco-churn.duckdns.org)**.)

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/), and [Docker Desktop](https://www.docker.com/products/docker-desktop/).

**1 — Clone and install**

```bash
git clone https://github.com/Ampofowaa/churn-retention-mlops.git
cd churn-retention-mlops
uv sync --all-extras
pre-commit install
```

**2 — The dataset is already here**

The raw CSV is committed at `datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (977 KB, read-only) — nothing to download. `make data` re-fetches it via the Kaggle CLI only if it's missing, which needs a [Kaggle API token](https://www.kaggle.com/settings/api).

**3 — Run tests**

```bash
uv run pytest          # unit tests — needs nothing but the install above, no Docker or pipeline run required
make test-integration  # integration tests — spins up its own ephemeral Postgres, requires only a running Docker daemon
```

A sanity check that the install worked, not a pipeline step — neither command depends on anything below, which is why it's not sitting in the middle of the pipeline run.

**4 — Start Postgres**

```bash
cp .env.example .env   # defaults work with Docker out of the box
make db-up             # start postgres:16 in Docker
```

**5 — Run the pipeline through calibration**

```bash
make repro STAGE=calibrate   # ingest -> validate -> split -> features -> train -> calibrate
```

Exits 0 without touching `threshold` — that stage needs a registered model version, minted next. See [Available make commands](#available-make-commands) above for why this is deliberate, not a partial failure.

**6 — Mint the challenger, then finish the pipeline**

```bash
make register-challenger        # mints calibrate's output as a new registry version, tagged pending
make repro STAGE=error_analysis # threshold -> evaluate -> error_analysis
```

Review the gate result in MLflow (`mlflow ui` → this cycle's `evaluation` run) rather than the notebook, which only reflects this cycle if someone re-executes it.

**7 — Record human sign-off, then promote the champion**

```bash
make review VERDICT=approved APPROVER="Your Name" NOTES="..."
make register MODEL_VERSION=<version>  # flips champion on a pass, tags rejected on a fail
```

`MODEL_VERSION` (printed by step 6, or `mlflow ui`) has no default — deliberately, since this is the step that can put a model into production.

<details>
<summary>Re-running one stage against a specific historical run instead</summary>

Every command above works with no arguments because it always resolves "whatever the previous step just produced." `calibrate`/`threshold`/`evaluate`/`error-analysis` also accept an explicit override — `make calibrate RUN_ID=<run_id>` or `make threshold MODEL_VERSION=<version>` — to target a different run/version than the default chain, and `make repro STAGE=<stage>` (e.g. `make repro STAGE=train`) re-runs one DVC stage alone. Neither is needed for a first run.

</details>

**8 — Browse experiment runs**

`make db-up` already started the MLflow tracking server alongside Postgres —
open [http://localhost:5000](http://localhost:5000) to explore the logged runs, the registered `telco-churn-pipeline` model, and the `challenger` alias.

**9 — Serve the champion and try it**

```bash
docker compose up -d --build   # full stack: postgres, mlflow, fastapi, streamlit
```

Once `/ready` returns `200` (`curl http://localhost:8000/ready`), open [http://localhost:8501](http://localhost:8501) for the Streamlit UI, or hit the API directly:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"customerid":"0002-ORFBO"},{"customerid":"0003-MKNFE"}]'
```

In the Batch Prediction tab, click **Score the sample** to score [`examples/sample_batch_predictions.csv`](examples/sample_batch_predictions.csv) (50 real customers, resolved live from `customers_crm`, regenerated deterministically by [`scripts/generate_sample_batch_predictions.py`](scripts/generate_sample_batch_predictions.py)) — see [docs/architecture.md § Serving Components](docs/architecture.md#serving-components) for what it demonstrates. In the Score a Customer tab, click any of the example IDs under the Fetch button (`0094-OIFMO`, `0096-BXERS`, `0096-FCPUF`, `0098-BOWSO`, `0100-DUVFC`) to load that customer in one click, or type your own and hit "Fetch customer". `make smoke-test-serving` runs the same checks as a script; `docker compose down` tears the stack back down.

## Project Structure

```
src/telco_churn/
  data/                          # ingest, Pandera validation/schema, dev/test split
  features/                      # SQL feature views, 9-lap discovery, permutation-importance selection
  models/
    train/                       # candidate comparison, feature freeze, Optuna tuning, model logging
    calibrate.py, threshold.py, evaluate.py, gate.py    # calibration, threshold, sealed-test evaluation, promotion gate
    economics.py, explain.py, error_analysis.py, shap_values.py  # business impact, SHAP, error diagnosis
    register.py, review.py       # registry writes, human review verdict
    drift_reference.py, environment_parity.py  # drift baseline, hot-reload dependency diff
    artifacts.py, policy_config.py, dev_features.py, calibration_metrics.py  # shared loaders/helpers
    plots.py, diagnostics.py
  serving/                       # FastAPI app + shadow/canary-aware scoring
    predict.py, app.py           # hot-reloading model bundle, /predict endpoints
    contact_policy.py, customer_lookup.py, schemas.py  # contact ranking, Postgres lookup, request/response models
  ui/                            # Streamlit demo app
    streamlit_app.py             # Score a Customer, Batch Prediction, Model Info tabs
  utils/                         # paths, logging, db, mlflow, stats, hashing helpers
configs/                 # Hydra YAML — training/, tuning/, calibration/, threshold/, evaluate/, error_analysis/, review/, register/, serving/, costs.yaml, model_promotion.yaml
sql/                     # Postgres schema + feature SQL views
tests/unit/              # pytest unit tests (≥80 % coverage target)
tests/integration/       # Postgres-backed tests (ingest, split, sql_features, validate) + subprocess CLI tests (train, calibrate, threshold, evaluate, error_analysis, predict) + FastAPI TestClient suite
tests/streamlit/         # headless Streamlit `AppTest` smoke suite (Phase 9)
pipelines/               # Prefect flows (retrain, drift check, batch predict) — Phase 10
monitoring/              # Prometheus config + Grafana dashboard JSON — Phase 13
docker/api/, docker/ui/  # multi-stage Dockerfiles (FastAPI, Streamlit), built from uv.lock (Phase 9)
examples/                # sample_batch_predictions.csv — real dev-partition customer IDs for /predict/batch and the Batch Prediction tab
                         #   regenerate: uv run python scripts/generate_sample_batch_predictions.py
dvc.yaml, .dvc/, .dvcignore  # DVC pipeline — 9 stages, ingest through error_analysis (Phase 8)
datasets/raw/             # committed to git — the source CSV DVC hashes as a stage dep, never modified
datasets/processed/, datasets/interim/  # gitignored; DVC-cached stage outputs
mlruns/                  # gitignored; MLflow artifact store (run metadata lives in Postgres)
notebooks/               # 00–05: ingestion → EDA → feature discovery/engineering →
                         #        model selection/feature selection/hyperparameter tuning →
                         #        calibration & threshold → evaluation & error analysis (Phase 7)
```


## Testing

- `uv run pytest` — full unit suite, no Docker required.
- `uv run pytest -k "test_name"` — a single test by name.
- `make test-integration` — integration tests; requires a running Docker daemon. Each test spins up its own ephemeral Postgres via `testcontainers` and points MLflow at a tmp-scoped SQLite file, so `make db-up` is not a prerequisite.
- `make test-data` / `make test-features` / `make test-models` / `make test-serving` — scoped coverage per package, for running one phase's tests in isolation without a false `fail_under=80` failure from unrelated packages.
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

**Pre-commit reformatting can make `dvc.lock` stale.** `dvc.lock` hashes each stage's deps by file content, not semantics — if you run the pipeline first and only let `black`/`end-of-file-fixer` reformat files at commit time, `dvc.lock` no longer matches what actually got committed, and the next `dvc repro` reruns every stage whose dep changed, for a pure whitespace/formatting diff rather than a real code or data change (it's still correct, just wasted compute — a full retrain for nothing). `make repro` depends on `make format` for exactly this reason, so `black` has nothing left to touch by the time you commit. Every `reports/*.json` receipt writer also emits a trailing newline for the same reason (`end-of-file-fixer` was otherwise rewriting them on every single commit). Calling `uv run dvc repro <stage>` directly, bypassing `make`, skips this guard — run `make format` by hand first if you do.

## Branch and commit conventions

Follow `CLAUDE.md`'s **Git Conventions** section (branch prefixes, Conventional Commits style) — not restated here so the two files can't disagree.

## Submitting a PR

1. Ensure `make lint` and `make test` both pass locally.
2. Push your branch and open a PR against `main`.
3. CI must be green before merge. Unit tests, lint, and type-check run on every push and PR; the Docker-dependent integration suite only runs on the daily schedule, manual dispatch, or PRs targeting `main` — so a PR opened against `main` will also need the integration job green, even if you only ran unit tests locally.
4. Flag any schema-breaking changes in the PR description before implementing them.
