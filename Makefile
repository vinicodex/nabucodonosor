.PHONY: lint format type-check test check

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy src/

test:
	uv run pytest --tb=short -q

check: lint format type-check test
