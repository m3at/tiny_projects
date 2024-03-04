#!/usr/bin/env bash

set -eo pipefail

curl --silent --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data @prompt_chatml.json
