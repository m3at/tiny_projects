#!/usr/bin/env zsh

set -eo pipefail

CONTENT="${1:-java dev wfh}"

curl http://localhost:8082/completion --silent -H "Content-Type: application/json" --data '{
  "temp": 1.2,
  "n_predict": 32,
  "cache_prompt": true,
  "stop": ["<|im_end|>"],
  "prompt": "<|im_start|>system\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|im_end|>\n<|im_start|>user\nSoftware engineer jobs in New York<|im_end|>\n<|im_start|>assistant\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|im_end|>\n<|im_start|>user\nRemote marketing positions<|im_end|>\n<|im_start|>assistant\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|im_end|>\n<|im_start|>user\ndata sciens job entry level<|im_end|>\n<|im_start|>assistant\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|im_end|>\n<|im_start|>user\nsenior devops engineer san francisco<|im_end|>\n<|im_start|>assistant\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|im_end|>\n<|im_start|>user\npart time IT jobs near me<|im_end|>\n<|im_start|>assistant\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|im_end|>\n<|im_start|>user\n'$CONTENT'<|im_end|>\n<|im_start|>assistant\n"
}'| jq '.content'
