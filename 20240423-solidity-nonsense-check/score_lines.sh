#!/usr/bin/env bash

set -eo pipefail

solc --ast-compact-json "$1" | tail -n +5 | jq -r '.nodes[] | select(.nodes) | .nodes[] | select(.kind == "function") | .name' | python check_stdin.py
