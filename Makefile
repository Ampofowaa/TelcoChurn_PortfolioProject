.PHONY: lint format test test-features test-integration data db-up db-down ingest validate features train

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/
	uv run black src/

test:
	uv run pytest

test-features:
	uv run pytest tests/unit/test_build.py tests/unit/test_sql_features.py tests/unit/test_generate.py tests/unit/test_preprocessing.py \
		--override-ini="addopts=" \
		--cov=src/telco_churn/features --cov-report=term-missing

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

features:
	uv run python -m telco_churn.features.build

train:
	dvc repro
