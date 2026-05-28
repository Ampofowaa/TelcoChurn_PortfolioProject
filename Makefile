.PHONY: lint format test train

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/
	uv run black src/

test:
	uv run pytest --cov=src --cov-report=term-missing

train:
	dvc repro
