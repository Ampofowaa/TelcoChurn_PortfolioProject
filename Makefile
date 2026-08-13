.PHONY: help lint format test test-data test-features test-models test-integration data db-up db-down ingest validate split features train calibrate threshold evaluate error-analysis review register-challenger register pre-commit mlflow-ui clean

.DEFAULT_GOAL := help

RUN := uv run

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff check + mypy on src/
	$(RUN) ruff check src/
	$(RUN) mypy src/

format: ## Auto-format src/ with ruff format + black
	$(RUN) ruff format src/
	$(RUN) black src/

pre-commit: ## Run all pre-commit hooks against every file
	$(RUN) pre-commit run --all-files

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

test-integration: ## Run integration tests (requires Docker; run `make db-up` first)
	$(RUN) pytest -m integration --run-integration

data: ## Download the raw dataset via Kaggle CLI (skips if already present)
	@if [ -f datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv ]; then \
		echo "datasets/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv already exists — skipping download."; \
	else \
		$(RUN) kaggle datasets download -d blastchar/telco-customer-churn -p datasets/raw/ --unzip; \
	fi

db-up: ## Start Postgres + MLflow in Docker (infra profile)
	docker compose --profile infra up -d

db-down: ## Stop and remove the infra containers
	docker compose --profile infra down

mlflow-ui: ## Open the MLflow tracking UI at localhost:5000
	$(RUN) mlflow ui

ingest: ## Load the raw CSV into Postgres
	$(RUN) python -m telco_churn.data.ingest

validate: ## Run the 5 Pandera data-quality gates
	$(RUN) python -m telco_churn.data.validate

split: ## Create the dev/test split
	$(RUN) python -m telco_churn.data.split

features: ## Build SQL feature views -> write the processed dataset
	$(RUN) python -m telco_churn.features.build

train: ## Train LightGBM + Optuna tuning; logs to MLflow
	$(RUN) python -m telco_churn.models.train

calibrate: ## Calibrate probabilities (fit + log only, no registry write; optional RUN_ID=<run_id>; defaults to train.py's receipt)
	$(RUN) python -m telco_churn.models.calibrate $(if $(RUN_ID),calibration.run_id=$(RUN_ID),)

threshold: ## Derive the cost-sensitive threshold (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.threshold $(if $(MODEL_VERSION),threshold.model_version=$(MODEL_VERSION),)

evaluate: ## One-time sealed-test evaluation + promotion gate (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.evaluate $(if $(MODEL_VERSION),evaluate.model_version=$(MODEL_VERSION),)

error-analysis: ## SHAP explainability + error diagnosis (optional MODEL_VERSION=<version>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.error_analysis $(if $(MODEL_VERSION),error_analysis.model_version=$(MODEL_VERSION),)

review: ## Stamp a human promotion-review verdict (VERDICT=approved|rejected APPROVER="..." NOTES="..." required; optional EVAL_RUN_ID=<run_id>, defaults to evaluate.py's receipt)
	@if [ -z "$(VERDICT)" ] || [ -z "$(APPROVER)" ] || [ -z "$(NOTES)" ]; then \
		echo 'Error: VERDICT, APPROVER, and NOTES are all required.'; \
		echo 'Usage: make review VERDICT=approved APPROVER="J. Doe" NOTES="reason for the verdict"'; \
		exit 1; \
	fi
	$(RUN) python -m telco_churn.models.review $(if $(EVAL_RUN_ID),review.eval_run_id=$(EVAL_RUN_ID),) review.verdict=$(VERDICT) review.approver="'$(APPROVER)'" review.notes="'$(NOTES)'"

register-challenger: ## Mint the calibrated pipeline as a new challenger version (optional RUN_ID=<run_id>; defaults to calibrate.py's receipt)
	$(RUN) python -m telco_churn.models.register $(if $(RUN_ID),register.run_id=$(RUN_ID),)

register: ## Act on the promotion gate verdict: flip champion, or reject (MODEL_VERSION=<version> required — no receipt fallback, deliberately explicit)
	@if [ -z "$(MODEL_VERSION)" ]; then echo "Error: MODEL_VERSION is required. Usage: make register MODEL_VERSION=<version>"; exit 1; fi
	$(RUN) python -m telco_churn.models.register register.model_version=$(MODEL_VERSION)

clean: ## Remove caches, coverage artifacts, and __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
