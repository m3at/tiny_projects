.PHONY: help lint embedding-server query-augmentation-server

# Most shell should do this automatically but who doesn't enjoy more awk?
help:  ## Show commands
	@echo "Default to \`build\` and \`run\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint:  ## Lint
	@uv run ruff check --fix --select=I && uv run ruff format

# TODO: add download off onnx models


# Require compiling llamafile: https://github.com/Mozilla-Ocho/llamafile
embedding-server:
	llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/gte-base-Q8_0.gguf

query-augmentation-server:
	llamafile -m $$DIR_MLMODELS/qwen2.5-1.5b-instruct-q8_0.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
