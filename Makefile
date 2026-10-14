.PHONY: install test lint suite compare view

install:
	uv pip install -e ".[dev,openai,anthropic]" pyyaml

test:
	pytest -q -p no:logging

lint:
	ruff check aisi_evals tests && ruff format --check aisi_evals tests

suite:
	aisi-evals run --log-dir logs/suite

compare:
	aisi-evals compare logs/suite --output reports/leaderboard.md

view:
	inspect view --log-dir logs/suite
