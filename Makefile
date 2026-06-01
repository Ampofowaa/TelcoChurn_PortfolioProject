.PHONY: lint format test test-integration data db-up db-down ingest validate train

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/
	uv run black src/

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

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

train:
	dvc repro
