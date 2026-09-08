.PHONY: setup data prepare check train ppo residual evaluate demo validate benchmark mechanics figures build record robustness
.PHONY: smolvla-setup smolvla-prepare smolvla-train smolvla-verify smolvla-evaluate smolvla-demo
CHARS ?= 永
STEPS ?= 32768
RUN_DIR ?= runs
SEED ?= 7
EPISODES ?= 28
EPOCHS ?= 35
POLICY ?= learned
BASE_POLICY ?= runs/bc.pt
REPEATS ?= 3
DEVICE ?= cpu
CAMERA_EVERY ?= 0
EXPERT_NOISE ?=
# Use the project environment even when the caller has activated another venv.
unexport VIRTUAL_ENV
COMMON = --run-dir "$(RUN_DIR)" --seed $(SEED) --device $(DEVICE) $(if $(CONFIG),--config "$(CONFIG)",) $(if $(OBSERVATION),--observation $(OBSERVATION),) $(if $(SENSOR_CONFIG),--sensor-config "$(SENSOR_CONFIG)",)
VLA_DEVICE ?= auto
VLA_STEPS ?= 100
VLA_EPISODES ?= runs/smolvla-recordings/episodes
VLA_DATASET ?= runs/smolvla-data
VLA_OUTPUT ?= runs/smolvla-adapter
VLA_DEMO_OUTPUT ?= runs/smolvla-demo
VLA_ARGS ?=
VLA = uv run --project integrations/smolvla --locked python -m shodo.smolvla

setup:
	@if test -f uv.lock; then uv sync --locked; else uv sync; fi
data:
	uv run shodo data
prepare: data
	uv run shodo prepare $(COMMON)
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
record:
	uv run shodo record $(COMMON) --policy $(POLICY) --chars '$(CHARS)' --episodes $(EPISODES) --camera-every $(CAMERA_EVERY) $(if $(EXPERT_NOISE),--expert-noise $(EXPERT_NOISE),)
robustness:
	uv run shodo robustness $(COMMON) --policy $(POLICY) $(if $(EVAL_CHARS),--chars "$(EVAL_CHARS)",) $(if $(SEEDS),--seeds $(SEEDS),)
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
smolvla-setup:
	uv sync --project integrations/smolvla --locked
smolvla-prepare:
	$(VLA) prepare --episodes "$(VLA_EPISODES)" --dataset "$(VLA_DATASET)" $(VLA_ARGS)
smolvla-train:
	$(VLA) train --dataset "$(VLA_DATASET)" --output "$(VLA_OUTPUT)" --steps $(VLA_STEPS) --device $(VLA_DEVICE) --seed $(SEED) $(VLA_ARGS)
smolvla-verify:
	$(VLA) verify --output "$(VLA_OUTPUT)" --device $(VLA_DEVICE) $(VLA_ARGS)
smolvla-evaluate:
	$(VLA) evaluate --output "$(VLA_OUTPUT)" --device $(VLA_DEVICE) --chars '$(CHARS)' --seed $(SEED) $(VLA_ARGS)
smolvla-demo:
	uv run --project integrations/smolvla --locked python -m shodo.vla_demo --adapter "$(VLA_OUTPUT)" --output-directory "$(VLA_DEMO_OUTPUT)" --device $(VLA_DEVICE) --chars '$(CHARS)' --seed $(SEED) $(VLA_ARGS)
