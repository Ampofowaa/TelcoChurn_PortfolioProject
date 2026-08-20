.PHONY: help setup lint format pre-commit test test-data test-features test-models test-serving test-integration data db-up db-down crm-data serve-up serve-down smoke-test-serving mlflow-ui repro dag metrics params calibrate threshold evaluate error-analysis review register-challenger register clean

.DEFAULT_GOAL := help

RUN := uv run

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies (incl. serving/ui extras) and pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Run ruff check + mypy on src/
	$(RUN) ruff check src/
	$(RUN) mypy src/

format: ## Auto-format src/ with ruff format + black
	$(RUN) ruff format src/
	$(RUN) black src/

pre-commit: ## Run all pre-commit hooks against every file
	$(RUN) pre-commit run --all-files

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run the full pytest suite with coverage (CI gate, fail_under=80)
	$(RUN) pytest

test-data: ## Run data package tests with scoped coverage
	$(RUN) pytest tests/unit/test_split.py tests/unit/test_checks.py tests/unit/test_eda.py tests/unit/test_ingest.py tests/unit/test_schema.py tests/unit/test_validate.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/data --cov-report=term-missing

test-features: ## Run features package tests with scoped coverage
	$(RUN) pytest tests/unit/test_build.py tests/unit/test_sql_features.py tests/unit/test_generate.py tests/unit/test_preprocessing.py tests/unit/test_select.py tests/unit/test_accessor.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/features --cov-report=term-missing

test-models: ## Run models package tests with scoped coverage
	$(RUN) pytest tests/unit/test_train_common.py tests/unit/test_train_candidates.py tests/unit/test_train_comparison.py tests/unit/test_train_feature_audit.py tests/unit/test_train_feature_selection.py tests/unit/test_train_tuning.py tests/unit/test_train_log_model.py tests/unit/test_diagnostics.py tests/unit/test_calibrate.py tests/unit/test_calibration_metrics.py tests/unit/test_threshold.py tests/unit/test_economics.py tests/unit/test_error_analysis.py tests/unit/test_evaluate.py tests/unit/test_explain.py tests/unit/test_gate.py tests/unit/test_plots.py tests/unit/test_register.py tests/unit/test_review.py tests/unit/test_drift_reference.py tests/unit/test_artifacts.py tests/unit/test_policy_config.py tests/unit/test_shap_values.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/models --cov-report=term-missing

test-serving: ## Run serving package tests with scoped coverage
	$(RUN) pytest tests/unit/test_predict.py tests/unit/test_contact_policy.py tests/unit/test_schemas.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/serving --cov-report=term-missing

test-integration: ## Run integration tests (requires Docker; run `make db-up` first)
	$(RUN) pytest -m integration --run-integration

# ---------------------------------------------------------------------------
# Data & infra (Postgres + MLflow)
# ---------------------------------------------------------------------------

data: ## Download the raw dataset via Kaggle CLI (skips if already present)
	@if [ -f datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv ]; then \
		echo "datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv already exists — skipping download."; \
	else \
		$(RUN) kaggle datasets download -d blastchar/telco-customer-churn -p datasets/raw/ --unzip; \
	fi

db-up: ## Start Postgres + MLflow in Docker (infra services only, no api/ui build)
	docker compose up -d postgres mlflow

db-down: ## Stop and remove the infra containers
	docker compose stop postgres mlflow
	docker compose rm -f postgres mlflow

crm-data: ## Populate customers_crm from customers_raw (run after db-up + ingest; safe to re-run)
	$(RUN) python -m telco_churn.serving.crm_data

serve-up: ## Start the full local stack — postgres, mlflow, fastapi, streamlit (Phase 9+)
	docker compose up -d

serve-down: ## Stop and remove the full stack, including postgres/mlflow
	docker compose down

smoke-test-serving: ## docker compose up -d --build, then curl /ready, /predict, /customer/<id> (Phase 9 §8)
	./scripts/smoke_test_serving.sh

mlflow-ui: ## Open the MLflow tracking UI at localhost:5000
	$(RUN) mlflow ui

# ---------------------------------------------------------------------------
# DVC pipeline (ingest -> validate -> split -> features -> train -> calibrate
# -> threshold -> evaluate -> error_analysis)
#
# dvc.lock conflict policy: it is a machine-generated lockfile (stage hashes,
# not human decisions) — on a merge/rebase conflict, never hand-edit it. Take
# the other side wholesale and let `dvc repro` regenerate the true state from
# source: `git checkout --theirs dvc.lock && dvc repro`. No merge driver is
# configured for it deliberately — a driver that "merges" two stage-hash sets
# would produce a dvc.lock that matches neither branch's actual pipeline
# output, which is worse than forcing a manual regeneration.
#
# The ingest/validate/split/features/train stages have no per-run overrides —
# `dvc.yaml`'s `cmd:` already is `uv run python -m telco_churn.<stage>`, so a
# Makefile target here would just retype it a second time with no caching or
# dependency-checking, and the two definitions can silently drift apart. Run
# them via `dvc repro <stage>` (or plain `make repro` for the whole graph)
# instead. calibrate/threshold/evaluate/error-analysis, below, keep their own
# targets because RUN_ID/MODEL_VERSION overrides are a one-off manual/
# debugging path DVC itself cannot express (dvc repro always uses the current
# deps/params, never an explicit historical run/version override).
#
# `repro` depends on `format` deliberately: dvc.lock hashes deps by content,
# so running the pipeline against unformatted source and only then committing
# (letting pre-commit's black hook reformat at commit time) leaves dvc.lock
# stale the moment the hook runs — the next `dvc repro` reruns every stage
# whose dep it touched, for a pure whitespace change. Formatting first means
# black finds nothing left to do at commit time, so dvc.lock and the
# committed source agree from the start.
#
# STAGE is what makes this guard reach the actual documented workflow —
# `uv run dvc repro calibrate` / `error_analysis` bypass `make` (and its
# `format` prerequisite) entirely, which is the command a first-time run
# genuinely needs (a bare `dvc repro` fails at `threshold` before anything is
# registered — see CONTRIBUTING.md). `make repro STAGE=calibrate` runs the
# same `dvc repro calibrate` but formats first, so use it instead of calling
# `dvc repro <stage>` directly.
# ---------------------------------------------------------------------------

repro: format ## Re-run DVC pipeline stages whose deps/params changed (optional STAGE=<name>, e.g. `make repro STAGE=calibrate`)
	$(RUN) dvc repro $(STAGE)

dag: ## Show the DVC pipeline DAG
	$(RUN) dvc dag

metrics: ## Show tracked DVC metrics (reports/metrics_summary.json, threshold's dev-OOF diagnostics)
	$(RUN) dvc metrics show

params: ## Show DVC params changes against the last commit
	$(RUN) dvc params diff

# ---------------------------------------------------------------------------
# Pipeline stage overrides (calibrate -> threshold -> evaluate ->
# error-analysis)
#
# These already run automatically inside `dvc repro` — every target below
# exists only so you can re-run one by hand against a specific past run or
# model version, instead of whatever `dvc repro` would pick automatically.
# Useful for debugging or an audit; not needed for a normal pipeline run.
# ---------------------------------------------------------------------------

calibrate: ## Calibrate probabilities (fit + log only, no registry write; optional RUN_ID=<run_id>; defaults to train.py's receipt)
	$(RUN) python -m telco_churn.models.calibrate $(if $(RUN_ID),calibration.run_id=$(RUN_ID),)

threshold: ## Derive the cost-sensitive threshold (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.threshold $(if $(MODEL_VERSION),threshold.model_version=$(MODEL_VERSION),)

evaluate: ## One-time sealed-test evaluation + promotion gate (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.evaluate $(if $(MODEL_VERSION),evaluate.model_version=$(MODEL_VERSION),)

error-analysis: ## SHAP explainability + error diagnosis (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.error_analysis $(if $(MODEL_VERSION),error_analysis.model_version=$(MODEL_VERSION),)

# ---------------------------------------------------------------------------
# Registry actions (register-challenger, review, register)
#
# None of these are DVC stages — a registry write isn't a file DVC can
# track. Order matters and is NOT top-to-bottom: register-challenger runs
# right after `calibrate`, before `threshold` can even start (it needs a
# registered version to resolve); review and register run at the very end,
# after error-analysis.
# ---------------------------------------------------------------------------

register-challenger: ## Mint the calibrated pipeline as a new challenger version (optional RUN_ID=<run_id>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.register $(if $(RUN_ID),register.run_id=$(RUN_ID),)

review: ## Stamp a human promotion-review verdict (VERDICT=approved|rejected APPROVER="..." NOTES="..." required; optional EVAL_RUN_ID=<run_id>, defaults to evaluate.py's receipt)
	@if [ -z "$(VERDICT)" ] || [ -z "$(APPROVER)" ] || [ -z "$(NOTES)" ]; then \
		echo 'Error: VERDICT, APPROVER, and NOTES are all required.'; \
		echo 'Usage: make review VERDICT=approved APPROVER="J. Doe" NOTES="reason for the verdict"'; \
		exit 1; \
	fi
	$(RUN) python -m telco_churn.models.review $(if $(EVAL_RUN_ID),review.eval_run_id=$(EVAL_RUN_ID),) review.verdict=$(VERDICT) review.approver="'$(APPROVER)'" review.notes="'$(NOTES)'"

register: ## Act on the promotion gate verdict: flip champion, or reject (MODEL_VERSION=<version> required — no receipt fallback, deliberately explicit)
	@if [ -z "$(MODEL_VERSION)" ]; then echo "Error: MODEL_VERSION is required. Usage: make register MODEL_VERSION=<version>"; exit 1; fi
	$(RUN) python -m telco_churn.models.register register.model_version=$(MODEL_VERSION)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove caches, coverage artifacts, and __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
