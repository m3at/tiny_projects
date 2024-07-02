#!/usr/bin/env bash

set -eo pipefail

### Get the llamafile server (<30Mb):
# curl -L -o ~/.local/bin/llamafile "https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.9/llamafile-0.8.9"
### Get a model (1.2Gb):
# wget -O Qwen2-1.5B-Instruct.Q6_K.gguf "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q6_k.gguf?download=true"
### Start the server with (for example):
# llamafile --model Qwen2-1.5B-Instruct.Q6_K.gguf --server --nobrowser --n-gpu-layers 999 --gpu APPLE -c 0 --parallel 1 --port 8080 --fast --flash-attn

QUERY="${1:-bradwurst}"

# OpenAI style, with grammar constraints (https://github.com/ggerganov/llama.cpp/tree/master/grammars)
# Run:
# ./post.sh <ARGUMENT> | jq '.choices[0].message.content'
curl --silent --request POST \
    --url http://localhost:8080/v1/chat/completions \
    --header "Content-Type: application/json" \
    --data @<(sed "s/tobereplaced/$QUERY/g" sample_request_grammar.json)
