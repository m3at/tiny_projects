#!/usr/bin/env bash


set -eo pipefail

TOKEN_JINA=$(awk '/TOKEN_JINA=/ {gsub(/.*=/,""); print}' .env)

if [ -z "$TOKEN_JINA" ]; then
  echo "Error: TOKEN_JINA is not set in .env"
  exit 1
fi

curl --silent https://api.jina.ai/v1/embeddings -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN_JINA" -d @sample.json > embeddings.json
# | jq '.data | map(.embedding[:8])'

exit 0
