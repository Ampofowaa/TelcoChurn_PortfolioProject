# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions map to project phases; see [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

---

## [0.2.1] - 2026-06-03 — QA & Standards Hardening

*Addresses code quality and documentation gaps identified in a review against industry DS
standards. No modelling logic changed.*

### Documentation
- `PROJECT_PLAN.md` restored and rewritten: Phases 3, 4, and the body of Phase 5 were
  entirely missing due to a file corruption; all 15 phases are now present and complete.
  Language improved throughout for both technical and non-technical readers; a phase status
  table and plain-English goal summaries added to each phase.
- README rewritten as a concise project landing page with headline results, pipeline overview,
  quick-start instructions, CI badge, and project status table. Full modelling rationale
  moved to `ANALYSIS.md`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, and `ANALYSIS.md` added — standard collateral expected
  in a professionally maintained DS project.

### Code quality
- Package declared as fully typed (PEP 561) via `src/telco_churn/py.typed` so downstream
  type checkers pick up inline types correctly.
- Coverage threshold enforced locally: `fail_under = 80` in `pyproject.toml` so a dropping
  test suite is caught before CI, not after.
- Minimal GitHub Actions CI pipeline added (`.github/workflows/ci.yml`): lint, type-check,
  and unit tests with coverage run on every push and pull request.

### Testing
- Integration tests (those requiring a live Postgres database) now skip automatically when
  Docker is not running, rather than failing with a confusing connection error. Pass
  `--run-integration` to run them explicitly.

### Configuration
- The two key validation thresholds — minimum acceptable row count (1,000) and maximum
  acceptable null rate (5 %) — are now defined in `configs/config.yaml` rather than
  hardcoded in the validation module. Changing a threshold no longer requires editing source code.
- Data source path is now configurable via a `--csv-path` flag when running ingestion from
  the command line, with the config file value as the default.

---

## [0.2.0] - 2026-06-01 — Phase 2: Data Validation

*Establishes automated data quality checks that run before any modelling begins, ensuring
the raw data meets expected structure, completeness, and domain constraints.*

### What changed
- **Five automated pass/fail validation gates** run on the raw data on every pipeline
  execution. All five currently pass on the IBM Telco dataset:
  - No duplicate customer IDs — each customer appears exactly once
  - Null rate below 5 % per column — catches upstream data feed failures
  - No negative tenure values — a business-impossible value that signals data corruption
  - Monthly charges are always positive — zero or negative charges indicate a billing error
  - Churn column contains only valid values (0 or 1) — rejects unexpected encodings

- **Schema enforcement** — every column is validated for correct data type and allowed
  categorical values (e.g. contract type must be one of "Month-to-month", "One year",
  "Two year"). Unexpected columns or value categories fail loudly rather than silently
  propagating through the pipeline.

- **Validation reports** are written to `reports/validation/` on every run, creating an
  audit trail of data quality over time.

- **Analysis notebooks now render correctly on GitHub.** Notebook format upgraded and
  cell identifiers standardised; a pre-commit hook prevents regression.

---

## [0.1.0] - 2026-05-31 — Phase 1: Data Ingestion

*Moves the raw CSV off the filesystem and into a structured database, establishing the
foundation for all downstream SQL-based feature engineering.*

### What changed
- **Raw data is loaded into a Postgres database** via a reproducible ingestion script.
  The pipeline handles the known data quality issue in the source file: 11 zero-tenure
  customers have a blank TotalCharges field (no first bill issued yet). These rows are
  retained with a null value rather than dropped, preserving the full 7,043-customer dataset.

- **The ingestion is idempotent** — running the pipeline multiple times always produces
  the same result. Re-running never creates duplicates or corrupts existing data.

- **Database schema mirrors the validated column structure** with appropriate data types
  (e.g. monetary values stored as fixed-precision decimals, binary flags as integers).
  `customerid` is the primary key, enforcing uniqueness at the database level.

- **Column names are standardised to lowercase** at ingest time, avoiding quoting
  friction in all downstream SQL queries.

- **Structured logging** records row counts and table names on every run, giving operators
  a clear audit trail without needing to query the database.

---

## [0.0.1] - 2026-05-28 — Phase 0: Project Foundation

*Establishes the development environment, tooling, and project structure that all subsequent
phases build on.*

### What changed
- **Reproducible environment** via `uv` — any contributor can recreate the exact dependency
  set with a single command (`uv sync`).

- **Automated code quality gates** run on every commit via pre-commit hooks: linting,
  formatting, type checking, and a secrets scanner that blocks accidental credential commits.

- **Configuration-driven design** — paths, environment variables, and tunable parameters
  live in `configs/config.yaml` rather than scattered across source files.

- **Makefile shortcuts** for the most common workflows: `make lint`, `make test`,
  `make validate`, `make train`.

- **Architecture diagram** documents the intended end-to-end system design
  (`docs/architecture.md`) before implementation begins.
