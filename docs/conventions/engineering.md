# Engineering Conventions: Changelog, Code Style, Notebooks, Testing

Moved verbatim from CLAUDE.md, which keeps a one-line summary of each rule and links here. These rules are binding: read this file before changing the code it covers.

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

- **Formatter:** `black` (line length 88) has final say — `.pre-commit-config.yaml` only gates `black`, not `ruff format`. `ruff format` and `black` disagree on a handful of edge cases (e.g. chained ternaries) that neither project's config can fully reconcile, so running `ruff format` alone can leave a file that fails the `black` pre-commit hook. Always run `make format` (`ruff format` then `black`, in that order) rather than `ruff format` standalone. `ruff` itself is for linting (replaces flake8/isort).
- **Type hints:** required on all public functions in `src/`. `mypy --strict` on `src/` only.
- **`__all__` required in every public module under `src/`.** List every public function, class, and constant. Omit underscore-prefixed internals. Place it after the imports block, before the first definition. **"Public" excludes two shapes:** any `__main__.py` entry point (declares no importable surface — it exists only to be run via `python -m`), and a package `__init__.py` that imports nothing from `telco_churn` (an empty placeholder — today, `serving/`, `ui/`, `monitoring/`, `utils/__init__.py`, and the root `__init__.py`). An `__init__.py` that *does* re-export project symbols, as `data/__init__.py` does, is a real API surface and is held to the rule — enforced by `tests/unit/test_architecture.py::test_every_public_module_defines_dunder_all`.
- **All functions must have docstrings.**
- **No comments explaining what the code does** — use clear names instead. Only comment the non-obvious *why* (hidden constraints, workarounds, subtle invariants).
- **No error handling inside pure functions.** Handle errors at system boundaries only: file I/O, DB calls, model load, external API calls.
- **Never use bare relative paths for file I/O in `src/`.** `"configs/config.yaml"` resolves from the shell's CWD and breaks in DVC stages, Prefect workers, CI, and Docker containers. Always anchor to `get_project_root() / "configs" / "config.yaml"` (`utils/paths.py`). This applies everywhere in `src/` — `__main__` blocks, module-level constants, and function bodies.
- **Every `src/` entry point that calls `compose_config()` must also call `activate_config(cfg)`** immediately after, so modules that resolve their own config via `load_config()` internally (rather than taking `cfg` as a parameter — e.g. `features/accessor.py`) see the same composed, override-applied tree the entry point resolved, not `config.yaml`'s on-disk defaults. Pair with `reset_active_config()` between in-process test runs (`tests/unit/conftest.py`'s autouse fixture) — enforced by `tests/unit/test_architecture.py::test_every_compose_config_call_site_also_calls_activate_config`.
- **No module under `src/` may import a name from another module that contains a `__main__` block, except a named allowance.** A `__main__` block marks a stage entry point — the module a human or `make` target runs directly, with its own argparse/`compose_config`/exit-code contract — and importing library code back out of one couples two entry points' internals together. Shared logic that needs to cross that boundary belongs in a plain (non-`__main__`-bearing) module instead — `models/calibration_metrics.py`, `models/artifacts.py`, `models/dev_features.py`, `models/policy_config.py`, `models/gate.py`, and `utils/` all exist because something calibration/threshold/evaluate/error_analysis/register needed shared without importing one another. A short allow-list captures the pre-existing legitimate exceptions (the data/features stage modules' own dev/test-partition and raw-data-validation cross-imports) — enforced by `tests/unit/test_architecture.py::test_no_module_imports_from_a_dunder_main_bearing_module`. `calibrate.py` and `register.py` no longer cross-import at all: `register.py`'s mint step (`register_challenger`) resolves everything it needs — `run_id`/`model_uri`/`logged_model_id` — from an explicit CLI override or `calibrate.py`'s own `reports/calibrate_receipt.json`, never from an in-process import, since it runs as a separate CLI invocation after `calibrate.py` has already exited.
- **All `logger.error(...)` calls inside `except` blocks must include `exc_info=True`.** `str(e)` logs only the message; `exc_info=True` attaches the full traceback so failures are diagnosable in CI and pipeline logs — enforced by `tests/unit/test_architecture.py::test_logger_error_inside_except_passes_exc_info`.
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
- Integration tests require a running Docker daemon, not the `docker compose up -d postgres mlflow` stack — each test spins up its own ephemeral Postgres via `testcontainers` (`data.ingest`/`split`/`sql_features`/`validate` suites) and points MLflow at a tmp-scoped SQLite file (the `*_subprocess.py` suites), so `make db-up` is not a prerequisite.
- `test_train_subprocess.py` runs the full Steps 1–5 pipeline (candidates → selection → tuning → logging) against a synthetic ~150-row fixture under relaxed Hydra overrides (`tuning.n_trials=2`, etc.) — a composition-path smoke test, not a benchmark; it asserts the pipeline completes and logs the expected artifacts, not a ROC-AUC/PR-AUC bar.
- When writing data, schema, or validation tests, cover: normal case, missing values, wrong dtypes, and empty dataframe.
- Every module with a `__main__` CLI entry point requires a **subprocess** integration test — invoked via `subprocess.run([sys.executable, "-m", "<module>"], env={...}, capture_output=True)` — covering the full composition path (argparse/config load → I/O → exit code). Direct function calls do not qualify: they miss argparse, OmegaConf resolution, dotenv loading, and env-var-to-engine joints that only surface at the subprocess boundary. Cover both the exit 0 (success) and exit 1 (error) paths. Exception: waived only when the module's entire `__main__` body is exercised as a named subroutine call inside another module's subprocess integration test — direct calls to shared helper functions do not satisfy the exception. Enforced by `tests/unit/test_architecture.py::test_every_main_module_has_a_subprocess_test` (parametrized per `__main__`-bearing module; a named waiver dict holds the exception case, empty today).
- Running integration tests in isolation will fail the `fail_under=80` gate because coverage is measured across all `src/` modules. Append `--no-cov` for standalone integration runs: `uv run pytest tests/integration/ --run-integration --no-cov`.
- Every new package added under `src/telco_churn/` must have a scoped `make test-<package>` Makefile target using `--override-ini="addopts=" --cov=src/telco_churn/<package>`. The global `make test` remains the CI gate; the scoped target lets phase tests run in isolation without a false `fail_under=80` failure from other packages.
- Run `pytest` before marking any task complete; if the phase has no tests yet, note it explicitly instead of skipping.
- When closing a QA backlog item (`[x]`), verify the fix is present in the code — read the relevant file or run the relevant test. Never mark `[x]` based on intent; only mark it after the change is confirmed in the working tree.
