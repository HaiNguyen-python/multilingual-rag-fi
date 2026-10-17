.PHONY: install test lint typecheck format index eval report serve check

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest --cov-fail-under=80

lint:
	python3 -m ruff check .
	python3 -m ruff format --check .

typecheck:
	python3 -m mypy src

format:
	python3 -m ruff format .
	python3 -m ruff check --fix .

# All local quality gates, same as CI
check: lint typecheck test

index:
	mlrag index

# Fails (exit 1) if the shipped setup drops below evals/thresholds.yaml
eval:
	mlrag eval

# Regenerates docs/results.md and docs/img/*.png
report:
	mlrag report

serve:
	mlrag serve --port 8000
