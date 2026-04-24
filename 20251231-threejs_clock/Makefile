.PHONY: help all \
	serve dataset

all: help

help:  ## Show commands
	@echo "Default to \`help\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


# Make sure the current make version is modern enough
ifneq ($(firstword $(sort $(MAKE_VERSION) 4.3)),4.3)
	$(error Needs GNU Make 4.3 or higher. In MacOS, try using gmake, and maybe add it to the path)
endif


static/hdri:
	./download_hdri.sh

serve: | static/hdri  ## Start FastAPI server
	./serve.py

# TODO: manage in uv. For now:
# uv pip install playwright tqdm && uv run python -m playwright install chromium
dataset:  ## Generate dataset images (N=16 by default, e.g., make dataset N=100)
	uv run python generate_dataset.py -n $(or $(N),64)
