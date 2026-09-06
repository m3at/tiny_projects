.PHONY: setup data check train ppo residual evaluate demo validate benchmark mechanics figures build
CHARS ?= 永
STEPS ?= 32768
RUN_DIR ?= runs/v2
SEED ?= 7
EPISODES ?= 28
EPOCHS ?= 35
POLICY ?= learned
BASE_POLICY ?= runs/v2/bc.pt
REPEATS ?= 3
COMMON = --run-dir "$(RUN_DIR)" --seed $(SEED) $(if $(CONFIG),--config "$(CONFIG)",)

setup:
	@if test -f uv.lock; then uv sync --locked; else uv sync; fi
data:
	uv run shodo data
check:
	uv run ruff check .
	uv run ruff format --check src tests scripts
	uv run pytest -q
train: data
	uv run shodo train $(COMMON) --episodes $(EPISODES) --epochs $(EPOCHS) $(if $(TRAIN_CHARS),--chars "$(TRAIN_CHARS)",)
ppo: data
	uv run shodo ppo $(COMMON) --steps $(STEPS)
residual: data
	uv run shodo ppo $(COMMON) --steps $(STEPS) --residual --base-policy "$(BASE_POLICY)"
evaluate:
	uv run shodo evaluate $(COMMON) --policy $(POLICY) $(if $(EVAL_CHARS),--chars "$(EVAL_CHARS)",)
demo:
	uv run shodo demo $(COMMON) --policy $(POLICY) --chars '$(CHARS)'
validate: data
	uv run shodo validate $(COMMON) --policy $(POLICY) --episodes $(EPISODES) --epochs $(EPOCHS)
benchmark:
	uv run shodo benchmark $(COMMON) --chars '$(CHARS)' --repeats $(REPEATS)
mechanics:
	uv run python -m shodo.mechanics
figures:
	uv run python scripts/plot_audit.py
build:
	uv build --wheel
