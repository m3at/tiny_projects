.PHONY: help lint serve-embedding serve-query-augmentation models

# Most shell should do this automatically but who doesn't enjoy more awk?
help:  ## Show commands
	@echo "Default to \`build\` and \`run\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint:  ## Lint
	@uv run ruff check --fix --select=I && uv run ruff format

# TODO: add download of onnx models (reranker), embedding (gte-base) and qwen2.5

models/jina-reranker-v2-base-multilingual_quantized.onnx:
	@mkdir -p models
	@wget -q --show-progress -O models/jina-reranker-v2-base-multilingual_quantized.onnx "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/onnx/model_quantized.onnx?download=true"

# TODO
models/gte-multilingual-reranker-base-quantized.onnx:
	@mkdir -p models
	@wget -q --show-progress -O models/gte-multilingual-reranker-base-quantized.onnx "https://huggingface.co/onnx-community/gte-multilingual-reranker-base/resolve/main/onnx/model_quantized.onnx?download=true"

# models/mxbai-rerank-base-v1.onnx:
# 	@mkdir -p models
# 	@wget -q --show-progress -O models/mxbai-rerank-base-v1.onnx "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1/resolve/main/onnx/model_quantized.onnx?download=true"

# models/mxbai-rerank-large-v1.onnx:
# 	@mkdir -p models
# 	@wget -q --show-progress -O models/mxbai-rerank-large-v1.onnx "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/onnx/model_quantized.onnx?download=true"

models: models/jina-reranker-v2-base-multilingual_quantized.onnx models/gte-multilingual-reranker-base-quantized.onnx

_EMBEDDING_PORT:=8989

# Require compiling llamafile: https://github.com/Mozilla-Ocho/llamafile
serve-embedding:
	@# All the quality tests are on a micro useless benchmark
	@# 113Mb. Sucks. ~55 rq/s
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/gte-base-Q8_0.gguf
	@
	@# 669Mb. Slightly better? . ~16 rq/s
	@# https://huggingface.co/vadimbadikov/gte-large-en/resolve/main/gte-large-en.gguf?download=true
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/gte-large-en.gguf
	@
	@# 112Mb. Fine. ~55 rq/s
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/snowflake-arctic-embed-m-v1.5-q8_0.gguf
	@
	@# 342Mb. Exactly the same as gte-large-en!? ~16 rq/s
	@# https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5-q8_0.gguf?download=true
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/bge-large-en-v1.5-q8_0.gguf
	@
	@# 139Mb. Sucks. ~45 rq/s
	@# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf?download=true
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/nomic-embed-text-v1.5.Q8_0.gguf
	@
	@# 341Mb. Better. ~16 rq/s (Q8) ~11 rq/s (Q6)
	@# https://huggingface.co/ChristianAzinn/mxbai-embed-large-v1-gguf/resolve/main/mxbai-embed-large-v1.Q6_K.gguf?download=true
	@#llamafiler --unsecure --trust 127.0.0.1/32 --listen 8989 --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/mxbai-embed-large-v1.Q8_0.gguf
	@
	@# 96Mb. Better. ~32 rq/s
	@# https://huggingface.co/djuna/jina-embeddings-v2-base-en-Q5_K_M-GGUF/resolve/main/jina-embeddings-v2-base-en-q5_k_m.gguf?download=true
	llamafiler --unsecure --trust 127.0.0.1/32 --listen 127.0.0.1:${_EMBEDDING_PORT} --gpu DISABLE --fast -m $$DIR_MLMODELS/embeddings/jina-embeddings-v2-base-en-q5_k_m.gguf

# serve-query-augmentation-qwen:
# 	@# 2.8 rq/s
# 	llamafile -m $$DIR_MLMODELS/qwen2.5-1.5b-instruct-q8_0.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
#
# serve-query-augmentation-gemma:
# 	@# 1.0 rq/s
# 	llamafile -m $$DIR_MLMODELS/gemma-2-2b-it-Q8_0.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082

serve-query-augmentation:
	@# 3.26 rq/s
	@# llamafile -m $$DIR_MLMODELS/Llama-3.2-1B-Instruct-Q8_0.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
	@# 4.4 rq/s
	@# llamafile -m $$DIR_MLMODELS/Llama-3.2-1B-Instruct-Q4_K_M.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
	@# 2.0 rq/s
	@# llamafile -m $$DIR_MLMODELS/Llama-3.2-1B-Instruct-f16.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
	@# 1.67 rq/s
	@# llamafile -m $$DIR_MLMODELS/Llama-3.2-3B-Instruct-Q6_K.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 2048 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
	@# 1.97 rq/s on APPLE, 1.65 rq/s on CPU
	llamafile -m $$DIR_MLMODELS/Llama-3.2-3B-Instruct-Q4_K_M.gguf --server --nobrowser --n-gpu-layers 999 --temp 0.8 -c 512 --gpu APPLE --n-predict 128 --log-disable --parallel 1 --fast --flash-attn --port 8082
