# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions map to project phases; see [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

---

## [0.4.1] - 2026-06-12 — Post-Phase 4 QA Hardening

*Addresses correctness and code-quality gaps identified across Phases 1–4. Highlights: ingest
rewritten to the staging-table upsert pattern; `RawSchema` nullability corrected for 19 columns;
`clean_dataframe()` placeholder removed; Gate 5 null-check set made schema-derived. No modelling
logic changed. 169 unit tests; 94.97% coverage.*

### Added
- **`src/telco_churn/utils/paths.py`** — `get_project_root()` walks upward from `__file__` to
  the first directory containing `pyproject.toml`. Canonical path anchor for all `src/` modules;
  replaces three separate inline copies scattered across `__main__` blocks.
- **`data/__init__.py`** — was empty; now re-exports the `data` package's public API with a
  matching `__all__`.
- **`ingest.setup_schema(engine)`** (`data/ingest.py`) — new public function; executes
  `001_create_raw.sql` via `CREATE TABLE IF NOT EXISTS` so the PRIMARY KEY is never silently
  dropped between runs.
- **`eda.inspect_missing(df)`** (`data/eda.py`) — returns missingness-context rows for each null
  column; supports MCAR/MAR/MNAR analysis in the EDA notebook.
- **`__all__`** defined in every public module under `src/` — `data/schema.py`, `data/checks.py`,
  `data/validate.py`, `data/ingest.py`, `data/eda.py`, `features/schema.py`, `features/build.py`.
  Controls the star-import surface and documents the intended public API per module.
- **Module-load assertion guard** (`features/build.py`) — asserts `PYTHON_ENGINEERED_COLS ⊆` the
  union of the four typed feature lists at import time; raises immediately if the sets diverge.
- **Integration tests** (`tests/integration/test_validate_postgres.py`) — subprocess tests
  covering `validate.py __main__` exit 0 (valid data) and exit 1 (blocking gate failure).
- **Unit tests** (`tests/unit/test_schema.py`) — 5 tests: DDL↔`RawSchema` column parity (no
  Docker required) and `CleanedSchema`↔`FeatureOutputSchema` cross-schema invariant consistency.
- **Unit tests** (`tests/unit/test_ingest.py`) — 2 tests added: `test_missing_column_raises` and
  `test_extra_column_raises` covering `load_raw_csv` column-validation.
- **Unit tests** (`tests/unit/test_checks.py`) — 1 test:
  `test_null_rate_check_covers_all_non_nullable_schema_columns` verifying Gate 5 monitors all 19
  non-nullable columns, not just the original hardcoded 3.
- **Unit tests** (`tests/unit/test_eda.py`) — 7 tests added for `inspect_missing`.

### Changed
- **`ingest.py` rewritten** (`data/ingest.py`) — `to_sql(if_exists='replace')` replaced with the
  staging-table upsert pattern (`INSERT … ON CONFLICT DO UPDATE`); merge is atomic. `ingest()` now
  calls `validate_raw(strict=True)` before writing; `_DTYPE_MAP` removed — column types live in
  DDL only.
- **`001_create_raw.sql`** — 19 columns upgraded from nullable to `NOT NULL`; `seniorcitizen`,
  `tenure`, `monthlycharges`, and `churn` additionally get `CHECK` constraints matching
  `RawSchema` field rules.
- **`RawSchema` nullability** (`data/schema.py`) — 19 fields corrected from `nullable=True` to
  `nullable=False`; only `totalcharges` remains nullable. `REQUIRED_COLUMNS` hardcoded frozenset
  removed; `ingest.py` now derives the required column set from `RawSchema.to_schema().columns`.
- **`_NULL_CHECKED_COLS`** (`data/checks.py`) — changed from a hardcoded 3-column frozenset to
  schema-derived from `RawSchema` non-nullable columns; `customerid` excluded (already covered
  by Gates 1 and 2).
- **`eda.py` column constants** — `CAT_FEATURES`, `NUM_FEATURES`, `BINARY_INT_FEATURES` changed
  from `list[str]` to `Final[tuple[str, ...]]`; `TARGET` from `str` to `Final[str]`.
- **`eda.compute_vif()`** — `sklearn.LinearRegression` replaced with `numpy.linalg.lstsq`
  + centred OLS; eliminates a sklearn dependency for a single function. Adds `warnings.warn`
  when any VIF is `inf`.
- **`eda.detect_outliers()`, `compute_chi2_tests()`, `compute_mann_whitney()`, `compute_vif()`**
  — mutable list default arguments replaced with `None` + in-body defaults.
- **`CLAUDE.md`** Code Style — three rules added: `__all__` required in every public module;
  `get_project_root()` required instead of bare relative paths; `exc_info=True` required on all
  `logger.error()` calls inside `except` blocks.
- **`PROJECT_PLAN.md`** — Phase 8 deliverables: Alembic note added; existing DDL becomes the
  initial migration version.

### Fixed
- **`clean_dataframe()` removed** (`data/validate.py`) — was a Phase 2 placeholder that
  duplicated the Phase 5 `ColumnTransformer`/`SimpleImputer` responsibility. 6 unit tests
  removed alongside it.
- **`validate_raw` / `validate_clean` duplication** — identical gate-assembly blocks extracted
  into a private `_run_gates()` helper.
- **`validate.py __main__`** — `validate_raw(strict=False)` replaced with `strict=True` +
  `except ValidationError`; `pragma: no cover` removed (now covered by integration test).
- **Bare `OmegaConf.load` paths** — fixed in `ingest.py __main__`, `validate.py __main__`, and
  `sql_features.py __main__`; all now use `get_project_root() / "configs" / "config.yaml"`.
  Inline `_project_root()` copy in `sql_features.py` removed.
- **`_REPORTS_DIR`** (`validate.py`) — was `Path("reports/validation")` (CWD-relative); replaced
  with `get_project_root() / "reports" / "validation"`.
- **Typed exception handlers** — bare `except Exception` replaced with typed handlers across
  `ingest.py`, `validate.py`, and `features/build.py __main__` blocks; all include `exc_info=True`.
- **`eda_df` fixture** (`tests/unit/test_eda.py`) — was generating structurally invalid data
  (`multiplelines` independent of `phoneservice`); now respects IBM Telco structural constraints.
- **Integration test imports** — `test_ingest_postgres.py`, `test_sql_features_postgres.py`,
  `test_validate_postgres.py` imported `get_project_root` from `tests/unit/helpers.py` via pytest
  sys.path manipulation; replaced with `from telco_churn.utils.paths import get_project_root`.
  Duplicate removed from `helpers.py`.
- **`sql/features/tenure_buckets.sql`** — stale `pd.cut` reference removed; implementation uses
  `CASE WHEN`.
- **`sql/features/charge_per_service.sql`** — comment added documenting the intentional
  `phoneservice`/`multiplelines` double-count in `service_count`.

---

## [0.4.0] - 2026-06-08 — Phase 4: Feature Engineering (SQL + Python)

*Builds features in two layers: SQL views in Postgres and four hypothesis-driven Python columns.
Returns a raw, untransformed feature DataFrame — ColumnTransformer fitting and train/val/test
split are Phase 5 responsibilities.*

### Added
- **SQL feature views** (`sql/features/`) — `tenure_buckets.sql` (four tenure cohorts),
  `charge_per_service.sql` (monthly charges ÷ active service count), and `customer_features.sql`
  (join view read by the Python feature builder). Built idempotently via `build_sql_features(engine)`.
- **Python feature engineering** (`src/telco_churn/features/build.py`) — `build_feature_df(df)`
  adds four columns: `is_long_month_to_month` (H1), `monthly_to_total_ratio` (H2),
  `fiber_contract` and `dsl_contract` (H3a/b — contract × internet-service interactions).
  Returns the raw augmented DataFrame; y extraction is Phase 5's responsibility. NaN preserved
  for 11 zero-tenure rows for Phase 5 imputation.
- **Column group exports** — `BINARY_STR_COLS`, `BINARY_INT_COLS`, `MULTI_CAT_COLS`, `NUMERIC_COLS`,
  `PYTHON_ENGINEERED_COLS` exported from `build.py` and surfaced via `features/__init__.py` as the
  public API for Phase 5.
- **SQL feature runner** (`src/telco_churn/features/sql_features.py`) — executes the three SQL
  files in dependency order via SQLAlchemy; idempotent (`CREATE OR REPLACE VIEW`).
- **Feature engineering notebook** (`notebooks/02-feature-engineering.ipynb`) — thin wrapper
  rendering SQL feature distributions and Python feature validation outputs.
- **Unit tests** (`tests/unit/test_build.py`) — 37 tests covering H1, H2, H3a, H3b correctness,
  NaN propagation, column count invariant, target/ID exclusion, and no input mutation. Includes
  `hypothesis` property-based tests.
- **Integration tests** (`tests/integration/test_sql_features_postgres.py`) — verifies SQL views
  are created correctly against a live Postgres instance; skips when Docker is not running.

### Changed
- `ANALYSIS.md §3` rewritten with Phase 4 results: SQL feature evidence tables, Python feature
  hypothesis → evidence → result for H1–H3, and the 25-column feature inventory.
- `PROJECT_PLAN.md` Phase 4 spec cleaned up; Phase 5 updated to own `ColumnTransformer`
  definition and fitting; `FeatureSchema` frozen dataclass added to Phase 5 deliverables.
- `CLAUDE.md` Testing section — two rules added: `__main__` CLI entry points require an
  integration test covering the full pipeline path; each new package requires a scoped
  `make test-<package>` Makefile target.

### Fixed
- `make test-features` added to `Makefile` — scoped to `--cov=src/telco_churn/features`; Phase 4
  tests run in isolation without a false `fail_under=80` failure from uncovered Phase 2 modules.
- `_PYTHON_ENGINEERED` renamed to `PYTHON_ENGINEERED_COLS` in `features/build.py`, exported from
  `features/__init__.py`, and import in integration test updated to the public path.
- `CustomerFeaturesSchema.monthlycharges` now declares `lt=np.inf`; `FeatureOutputSchema.monthly_to_total_ratio`
  drops `lt=np.inf` — `inf` is now unreachable by construction rather than caught on output.
- `customer_features.sql` LEFT JOIN intent undocumented; comment added noting `customers_raw` is
  the authoritative row source and Pandera `nullable=False` is the downstream guard.
- `_make_feature_row` fixture hardcoded `charge_per_service` as `/ 2`; corrected to derive from
  service flags via the SQL formula (correct for default configuration is `/ 3`).
- `test_build_feature_df_invalid_tenure_cohort_raises` added — invalid `tenure_cohort` category
  was not covered by existing input schema guard tests.
- Hypothesis `max_examples` raised from 40 → 100 on both property tests.
- `build_feature_df` docstring corrected from "H1–H3" to "H1, H2, H3a, and H3b".
- `exclude_lines` was under `[tool.coverage.run]` instead of `[tool.coverage.report]`; coverage.py
  silently ignored it. Moved to the correct section — suite coverage corrected from 86.61% to 97.64%.
- H1 boundary (`tenure > 24`) had no fencepost tests at `tenure=24` or `tenure=25`; two tests added.
- Four queries in `test_sql_features_postgres.py` used f-string interpolation into raw SQL;
  replaced with SQLAlchemy bound parameters.
- `build_feature_df` previously raised an opaque `KeyError` when SQL view columns were missing;
  now raises a named `SchemaError` via `CustomerFeaturesSchema` input validation.
- `SELECT *` in `build.py __main__` replaced with an explicit column list from
  `BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS`.
- `customerid` was excluded from `telco_churn_processed.csv`; now written as the first column.
- `make test-integration` was a silent no-op — ran without `--run-integration` so all integration
  tests were skipped. Flag added to the target.
- `make features` target was missing from the `Makefile`; added to complete the
  `ingest → validate → features → train` chain.

---

## [0.3.0] - 2026-06-06 — Phase 3: EDA Notebook

*Promotes all EDA logic from the original research notebook into a testable, importable `src/`
module. The EDA notebook becomes a thin rendering wrapper; statistical helpers become reusable
production code.*

### Added
- **EDA helper library** (`src/telco_churn/data/eda.py`) — seven public functions: IQR outlier
  detection, per-group churn rates, chi-squared + Cramér's V, Mann-Whitney U + rank-biserial r,
  Pearson correlation matrix, top-N target correlations, and VIF. All functions handle edge cases
  (empty DataFrame, NaN, constant columns, perfect collinearity) via `warnings.warn` rather than
  raising.
- **VIF without `statsmodels`** — derived from `sklearn.LinearRegression` R², eliminating a
  50 MB+ dependency.
- **Column constants** — `CAT_FEATURES`, `NUM_FEATURES`, `BINARY_INT_FEATURES`, `TARGET` exported
  as the single source of truth for column lists across EDA, validation, and feature engineering.
- **EDA notebook** (`notebooks/01-eda.ipynb`) — thin wrapper covering class imbalance, univariate
  distributions, bivariate churn-rate analysis, statistical tests with effect sizes, correlation
  heatmap, VIF table, and a contract × internet-service interaction that motivates Phase 4
  engineering. Original archived at `notebooks/_archive/EDA-original.ipynb`.
- **Unit tests** (`tests/unit/test_eda.py`) — 50+ tests covering all seven functions under normal
  inputs, missing values, wrong dtype, and empty DataFrame. Warning emission verified with
  `pytest.warns`.

---

## [0.2.2] - 2026-06-03 — Pre-Phase 3 Cleanup

*Fixes identified in a post-QA audit. No modelling logic changed.*

### Added
- Two missing schema constraint tests added to `tests/unit/test_checks.py`: invalid
  `contract_type` value and unexpected extra column. Test count: 49 → 51; coverage holds at
  80.73 %.

### Changed
- `CLAUDE.md` corrected: source-of-truth pointers, test file references, and Phase 7 notebook
  name fixed.

### Fixed
- `mirrors-mypy` pre-commit hook bumped from `v1.13.0` → `v2.1.0` to match `mypy>=2.1.0` in
  `pyproject.toml`; eliminates "passes locally, fails in CI" type errors.
- Pre-commit mypy hook now runs `mypy src/` with `pass_filenames: false` — matches CI exactly
  and avoids duplicate-module errors from the two `conftest.py` files.
- `tests/` added to mypy `exclude` list in `pyproject.toml`; type-checking scoped to `src/` only.

---

## [0.2.1] - 2026-06-03 — QA & Standards Hardening

*Addresses code quality and documentation gaps identified in a review against industry DS
standards. No modelling logic changed.*

### Added
- `CHANGELOG.md`, `CONTRIBUTING.md`, and `ANALYSIS.md` added.
- `src/telco_churn/py.typed` added (PEP 561) so downstream type checkers pick up inline types.
- `fail_under = 80` added to `pyproject.toml` — failing coverage is caught locally before CI.
- Minimal GitHub Actions CI pipeline (`.github/workflows/ci.yml`): lint, type-check, and unit
  tests with coverage on every push and PR.

### Changed
- `PROJECT_PLAN.md` restored and rewritten — Phases 3, 4, and the body of Phase 5 were missing
  due to file corruption; all 15 phases now present. Phase status table added.
- README rewritten as a project landing page; modelling rationale moved to `ANALYSIS.md`.
- Validation thresholds (min row count, max null rate) moved from source code to
  `configs/config.yaml`.
- Data source path now configurable via `--csv-path` CLI flag.

### Fixed
- Integration tests now skip automatically when Docker is not running rather than failing with a
  connection error. Pass `--run-integration` to run them explicitly.

---

## [0.2.0] - 2026-06-01 — Phase 2: Data Validation

*Establishes automated data quality checks that run before any modelling begins.*

### Added
- **Five validation gates** with two severity levels — ERROR (blocking) and WARNING (non-blocking,
  logged). All five pass on the IBM Telco dataset:
  - Gate 1 — Schema (ERROR): Pandera validates column presence, types, ranges, and allowed
    categorical values.
  - Gate 2 — Duplicate IDs (ERROR): asserts `customerid` is unique.
  - Gate 3 — Churn labels (ERROR): asserts `churn` is binary with no nulls.
  - Gate 4 — Unexpected TotalCharges nulls (WARNING): flags nulls only where `tenure > 0`; the
    11 zero-tenure nulls are expected and ignored.
  - Gate 5 — Distribution sanity (WARNING): row count below 1,000 or null rate above 5 % on
    key columns.
- **Schema inheritance** (`RawSchema` → `CleanedSchema`) — `CleanedSchema` additionally requires
  `totalcharges` to be non-null, verifying imputation ran before downstream stages.
- **`clean_dataframe()`** — median imputation for the 11 known NULL `totalcharges` rows;
  preserves all 7,043 customers.
- **Validation reports** written to `reports/validation/<timestamp>/` on failure — `summary.csv`
  per gate and `<gate>_failures.csv` for offending rows.

### Fixed
- Analysis notebooks now render correctly on GitHub. Notebook format upgraded and cell identifiers
  standardised; pre-commit hook added to prevent regression.

---

## [0.1.0] - 2026-05-31 — Phase 1: Data Ingestion

*Moves the raw CSV into Postgres, establishing the foundation for all downstream SQL-based
feature engineering.*

### Added
- **Raw data ingestion** (`src/telco_churn/data/ingest.py`) — loads CSV into `customers_raw`
  table. Retains all 7,043 rows including the 11 zero-tenure customers with null `TotalCharges`.
- **Idempotent ingestion** (`if_exists="replace"`) — re-running never creates duplicates.
- **SQL schema** (`sql/schema/001_create_raw.sql`) — explicit column types; `customerid` as
  primary key. Applied automatically on container first start.
- **Column name normalisation** — names lowercased and SQL reserved words renamed at ingest time
  (`partner` → `has_partner`, `contract` → `contract_type`).
- **Structured logging** — records row counts and table names on every run.
- **Postgres in Docker** (`docker-compose.yml`, `infra` profile) — reproducible local database
  with a healthcheck.
- **Unit and integration tests** — unit tests cover parsing; integration tests use
  `testcontainers` to verify the full CSV → Postgres path including idempotency.

---

## [0.0.1] - 2026-05-28 — Phase 0: Project Foundation

*Establishes the development environment, tooling, and project structure that all subsequent
phases build on.*

### Added
- **Reproducible environment** via `uv` + `pyproject.toml` + `uv.lock`.
- **Pre-commit hooks** — `ruff`, `black`, `mypy --strict` (src/ only), `detect-secrets`, and
  standard file checks.
- **Configuration-driven design** via Hydra — paths, random seed, MLflow settings, and tunable
  parameters in `configs/config.yaml`.
- **Structured JSON logging** via `structlog` — machine-readable, compatible with CloudWatch
  and Grafana.
- **Project directory skeleton** — `src/`, `tests/`, `configs/`, `sql/`, `pipelines/`, `docs/`,
  `notebooks/`, `datasets/`.
- **`Makefile` shortcuts** — `make lint`, `make test`, `make validate`, `make train`.
- **Architecture diagram** (`docs/architecture.md`) documenting the intended end-to-end system
  design.

---

<!-- Version comparison links (added in Phase 11 when the GitHub remote is wired into CI/CD):
[Unreleased]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/releases/tag/v0.0.1
-->
