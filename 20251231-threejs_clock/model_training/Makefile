.PHONY: help all \
	setup lint precommit \
	debug \
	train


all: setup

help:  ## Show commands
	@echo "Default to \`setup\` and \`download-embedding-model\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


# Make sure the current make version is modern enough
ifneq ($(firstword $(sort $(MAKE_VERSION) 4.3)),4.3)
	$(error Needs GNU Make 4.3 or higher. In MacOS, try using gmake, and maybe add it to the path)
endif

MAKEFILE_PARENT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Override the externally defined value, if any
override VIRTUAL_ENV := $(PWD)/.venv

# export PYTHONPATH := $(PWD):$(PYTHONPATH)

# Pull some vars from .env
ifneq (,$(wildcard ./.env))
include .env
# Error out if .env is missing
else
$(error A .env is required. Please copy .env.example into .env and fill up the variables)
endif


# ┌──────────────────────────────────────────────────────────┐
# │                      Initial setup                       │
# └──────────────────────────────────────────────────────────┘

OS := $(shell uname -s)

# Install uv if it does not exist, and create the env
.venv:
ifeq (, $(shell which uv))
	$(info uv is not installed yet, fetching it now)
	$(shell curl -LsSf https://astral.sh/uv/install.sh | sh)
else
	$(info uv exist (use sunscreen!))
endif
	uv venv --allow-python-downloads --allow-existing

setup: .venv  ## Initial virtual setup through uv
	uv sync --all-groups

# ┌──────────────────────────────────────────────────────────┐
# │                        Utilities                         │
# └──────────────────────────────────────────────────────────┘

UV_FROZEN_DEV = uv run --only-group dev --frozen --preview-features extra-build-dependencies

lint:  ## Lint and format python code
	@$(UV_FROZEN_DEV) ruff format
	@$(UV_FROZEN_DEV) ruff check --fix --extend-select=I
	@# Type checking
	$(UV_FROZEN_DEV) ty check --exit-zero --respect-ignore-files --color always 2>&1 | grep -v 'ty is pre-release software'


# ┌──────────────────────────────────────────────────────────┐
# │                         Training                         │
# └──────────────────────────────────────────────────────────┘

train:  ## Model training
	uv run scripts/model_training.py
