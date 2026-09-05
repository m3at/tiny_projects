.PHONY: setup data check train ppo evaluate demo validate build
CHARS ?= 永
STEPS ?= 32768

setup:
	uv sync --locked
data:
	uv run shodo data
check:
	uv run ruff check .
	uv run pytest -q
train: data
	uv run shodo train
ppo: data
	uv run shodo ppo --steps $(STEPS)
evaluate:
	uv run shodo evaluate
demo:
	uv run shodo demo --chars '$(CHARS)'
validate: data
	uv run shodo validate
build:
	uv build --wheel
