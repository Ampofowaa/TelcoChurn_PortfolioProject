.PHONY: lint format test test-data test-features test-models test-integration data db-up db-down ingest validate split features train calibrate threshold

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/
	uv run black src/

test:
	uv run pytest

test-data:
	uv run pytest tests/unit/test_split.py tests/unit/test_checks.py tests/unit/test_eda.py tests/unit/test_ingest.py tests/unit/test_schema.py tests/unit/test_validate.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/data --cov-report=term-missing

test-features:
	uv run pytest tests/unit/test_build.py tests/unit/test_sql_features.py tests/unit/test_generate.py tests/unit/test_preprocessing.py tests/unit/test_select.py tests/unit/test_accessor.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/features --cov-report=term-missing

test-models:
	uv run pytest tests/unit/test_train_common.py tests/unit/test_train_candidates.py tests/unit/test_train_comparison.py tests/unit/test_train_feature_freeze.py tests/unit/test_train_tuning.py tests/unit/test_train_log_model.py tests/unit/test_diagnostics.py tests/unit/test_calibrate.py tests/unit/test_threshold.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/models --cov-report=term-missing

test-integration:
	uv run pytest -m integration --run-integration

data:
	uv run kaggle datasets download -d blastchar/telco-customer-churn -p datasets/raw/ --unzip

db-up:
	docker compose --profile infra up -d

db-down:
	docker compose --profile infra down

ingest:
	uv run python -m telco_churn.data.ingest

validate:
	uv run python -m telco_churn.data.validate

split:
	uv run python -m telco_churn.data.split

features:
	uv run python -m telco_churn.features.build

train:
	uv run python -m telco_churn.models.train

calibrate:
	uv run python -m telco_churn.models.calibrate calibration.run_id=$(RUN_ID)

threshold:
	uv run python -m telco_churn.models.threshold threshold.model_version=$(MODEL_VERSION)
