.PHONY: help all \
	setup npmtailwindsetup lint precommit \
	preparecss deletedb migrate resetmigrate  \
	runserver runworker runflower \
	synctoremote install restart


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

# Allow overriding db dir path
DB_DIR := $(PWD)/db_mount

# Variables that I want to make sure are pulled from the .env
reqenvs = DJANGO_ADMIN_USERNAME DJANGO_ADMIN_PASSWORD SERVER_REMOTE_PORT SERVER_REMOTE_USER SERVER_REMOTE_HOST SERVER_REMOTE_PATH

# Pull some vars from .env
ifneq (,$(wildcard ./.env))
# Clear them up. If you want to allow setting those env differently, delete the loop below
$(foreach var,$(reqenvs),\
  $(if $(value $(var)),,$(undefine $(var)))\
)
# Fetch those
include .env
# Error out if .env is missing
else
$(error A .env is required. Please copy .env.example into .env and fill up the variables)
endif

# Make sure the required variables are set
$(foreach var,$(reqenvs),\
  $(if $(value $(var)),,$(error $(var) was not set in the .env))\
)


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
	@# The prerelease is until celery reaches 5.5 stable
	uv sync --prerelease=allow
	@# Setup the commit hook, but only if .git/ exists
ifneq (,$(wildcard .git/index))
	@# Link if the file does not exist already
	[ ! -e .git/hooks/pre-commit ] && \
		ln -s $(PWD)/pre-commit-checks .git/hooks/pre-commit
endif


npmtailwindsetup:  ## Install tailwindcss setup
	cd kitsune/tasks/templates && \
		npm install -D tailwindcss && \
		npx tailwindcss init


# ┌──────────────────────────────────────────────────────────┐
# │                        Utilities                         │
# └──────────────────────────────────────────────────────────┘


lint:  ## Lint and format python code
	@uv run --frozen ruff format
	@uv run --frozen ruff check --fix --extend-select=I


precommit:  ## Manually run the precommit hook
	@./pre-commit-checks


# ┌──────────────────────────────────────────────────────────┐
# │                  Web app preparation                     │
# └──────────────────────────────────────────────────────────┘


preparecss:  ## Prepare css from the templated html files
	@# This is NOT the way, but meh
	@cd kitsune/tasks/templates && \
		npx tailwindcss -i ./input.css --minify -o ../../static/tailwind_min.css


deletedb:
	rm -rf db_mount/db.* || true


migrate:  ## Run migrations
	@cd kitsune && \
		uv run --frozen python manage.py makemigrations --noinput && \
		uv run --frozen python manage.py migrate --noinput && \
		uv run --frozen python manage.py createuser && \
		uv run --frozen python manage.py populate && \
		uv run --frozen python manage.py collectstatic --noinput && \
		DJANGO_SUPERUSER_PASSWORD=$(DJANGO_ADMIN_PASSWORD) uv run python manage.py createsuperuser --noinput \
		    --username $(DJANGO_ADMIN_USERNAME) --email $(DJANGO_ADMIN_USERNAME)@test.example --verbosity 0 2>/dev/null || true


resetmigrate: deletedb migrate  ## Remove database then run migrations


# ── Run stuff ─────────────────────────────────────────────────────────

runserver:  ## Run django server in debug mode
	@# With uvicorn it'll look something like:
	@# uv run --frozen uvicorn --host 0.0.0.0 --port 4242 --workers 2 kitsune.asgi:application
	@cd kitsune && uv run --frozen python manage.py runserver

runworker:  ## Run celery worker
	@# cd kitsune && uv run celery -A tasks worker --loglevel=info --concurrency=1 --pool=threads --purge
	@cd kitsune && uv run --frozen python manage.py celery_worker

runflower:  ## Run flower for celery monitoring
	@# To port foward flower:
	@# ssh -NL 5555:localhost:5555 -p $(SERVER_REMOTE_PORT) $(SERVER_REMOTE_USER)@$(SERVER_REMOTE_HOST)
	@cd kitsune && uv run celery -A tasks flower


# ┌──────────────────────────────────────────────────────────┐
# │                        Deployment                        │
# └──────────────────────────────────────────────────────────┘

OS := $(shell uname -s)
USER_ID := $(shell id -u)


# Copy files to remote server (assume that ssh is already setup)
# rsync options:
# --recursive   see --recursive
# --perms       preserve permissions
# --whole-file  don't try to be smart and sync part of files (make things faster)
# --partial     keep partial files, nice when interrupted to restart from there
# --compress    compress before transfer, the network is slower than the cpu (max level is 22 for zstd)
synctoremote:  ## Sync local files to remote, except git, venv, and db files
	@rsync --recursive --perms --whole-file --partial \
		--compress --compress-choice=zstd --compress-level=18 \
		--info=progress2 \
		-e "ssh -p $(SERVER_REMOTE_PORT)" \
		--exclude=.git \
		--exclude=.gitignore \
		--exclude=.venv \
		--exclude=.env \
		--exclude='db_mount/db.sqlite3*' \
		--exclude='*/.ruff_cache' --exclude='*/__pycache__' \
		--exclude='*/_tmp_*' --exclude='_tmp_*' \
		--exclude='kitsune/tasks/templates/node_modules' \
		--exclude='kitsune/static/collected' \
		./ \
		$(SERVER_REMOTE_USER)@$(SERVER_REMOTE_HOST):$(SERVER_REMOTE_PATH)


# ┌──────────────────────────────────────────────────────────┐
# │                  Deployment server side                  │
# └──────────────────────────────────────────────────────────┘

install:  ## Install systemd service files for django and celery
ifeq ($(OS),Linux)
ifeq ($(USER_ID),0)
	$(info Copying service files)
	mkdir -p /etc/conf.d /var/run/celery /var/log/celery
	chown -R $(SERVER_REMOTE_USER):$(SERVER_REMOTE_USER) /var/run/celery
	chown -R $(SERVER_REMOTE_USER):$(SERVER_REMOTE_USER) /var/log/celery
	cp systemd_services/etc_confd_celery /etc/conf.d/celery
	cp systemd_services/django.service /etc/systemd/system/django.service
	cp systemd_services/celery.service /etc/systemd/system/celery.service
	systemctl daemon-reload
	systemctl enable django
	systemctl enable celery
	systemctl start django
	systemctl start celery
else
	$(warning Can only run this as root)
endif
else
	$(warning Only applicable to Linux)
endif


restart:  ## Restart django and celery systemd services
ifeq ($(OS),Linux)
ifeq ($(USER_ID),0)
	$(info Restarting services)
	systemctl restart django
	systemctl restart celery
else
	$(warning Can only run this as root)
endif
else
	$(warning Only applicable to Linux)
endif
