.PHONY: help all \
	lint precommit \
	up down pg_ping psql script \
	debug


# Most shell should do this automatically but who doesn't enjoy more awk?
help:  ## Show commands
	@echo "Default to \`build\` and \`run\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


# Make sure the current make version is modern enough
ifneq ($(firstword $(sort $(MAKE_VERSION) 4.3)),4.3)
	$(error Needs GNU Make 4.3 or higher. In MacOS, try using gmake, and maybe add it to the path)
endif

# Override the externally defined value, if any
override VIRTUAL_ENV := $(PWD)/.venv


# ┌──────────────────────────────────────────────────────────┐
# │                      Initial setup                       │
# └──────────────────────────────────────────────────────────┘

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
	uv sync


# ┌──────────────────────────────────────────────────────────┐
# │                            DB                            │
# └──────────────────────────────────────────────────────────┘

PG_DB_NAME=vecdemo
PG_USERNAME=alice
PG_HOST=localhost
PG_PORT=5432

up:  ## Docker compose up
	docker compose up -d

down:  ## Docker compose down
	docker compose down -v

pg_ping:  ## Sanity check pg connection
	pg_isready --dbname=${PG_DB_NAME} --host=${PG_HOST} --port=${PG_PORT} --username=${PG_USERNAME}
	psql --host=${PG_HOST} --port=${PG_PORT} --dbname=${PG_DB_NAME} --username=${PG_USERNAME} -c 'SELECT 1;'

psql:  ## psql repl
	psql --dbname=${PG_DB_NAME} --host=${PG_HOST} --port=${PG_PORT} --username=${PG_USERNAME}


# ┌──────────────────────────────────────────────────────────┐
# │                  Interract with the db                   │
# └──────────────────────────────────────────────────────────┘

script:  ## Interract with the db
	@UV_ENV_FILE=.env uv run --frozen script.py

scale:  ## Small scaling test on 100k random vectors
	@UV_ENV_FILE=.env uv run --frozen scale.py


# ┌──────────────────────────────────────────────────────────┐
# │                        Utilities                         │
# └──────────────────────────────────────────────────────────┘


lint:  ## Lint and format python code
	@uv run --frozen ruff format
	@uv run --frozen ruff check --fix --extend-select=I
