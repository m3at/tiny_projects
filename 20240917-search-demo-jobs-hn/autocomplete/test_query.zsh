#!/usr/bin/env zsh

set -eo pipefail

# echo "$(./test_query.zsh | sed 's/^.//;s/.$//')"

CONTENT="${1:-java dev wfh}"

# Qwen 2.5
# curl http://localhost:8082/completion --silent -H "Content-Type: application/json" --data '{
#   "temp": 1.2,
#   "n_predict": 32,
#   "cache_prompt": true,
#   "stop": ["<|im_end|>"],
#   "prompt": "<|im_start|>system\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|im_end|>\n<|im_start|>user\nSoftware engineer jobs in New York<|im_end|>\n<|im_start|>assistant\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|im_end|>\n<|im_start|>user\nRemote marketing positions<|im_end|>\n<|im_start|>assistant\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|im_end|>\n<|im_start|>user\ndata sciens job entry level<|im_end|>\n<|im_start|>assistant\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|im_end|>\n<|im_start|>user\nsenior devops engineer san francisco<|im_end|>\n<|im_start|>assistant\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|im_end|>\n<|im_start|>user\npart time IT jobs near me<|im_end|>\n<|im_start|>assistant\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|im_end|>\n<|im_start|>user\n'$CONTENT'<|im_end|>\n<|im_start|>assistant\n"
# }'| jq '.content'

# Gemma 2
# curl http://localhost:8082/completion --silent -H "Content-Type: application/json" --data '{
#   "temp": 1.2,
#   "n_predict": 32,
#   "cache_prompt": true,
#   "stop": ["<|eot_id|>"],
#   "prompt": "<bos><start_of_turn>user\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<end_of_turn>\n<start_of_turn>model\n<end_of_turn>\n<start_of_turn>user\nSoftware engineer jobs in New York<end_of_turn>\n<start_of_turn>model\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<end_of_turn>\n<start_of_turn>user\nRemote marketing positions<end_of_turn>\n<start_of_turn>model\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<end_of_turn>\n<start_of_turn>user\ndata sciens job entry level<end_of_turn>\n<start_of_turn>model\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<end_of_turn>\n<start_of_turn>user\nsenior devops engineer san francisco<end_of_turn>\n<start_of_turn>model\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<end_of_turn>\n<start_of_turn>user\npart time IT jobs near me<end_of_turn>\n<start_of_turn>model\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<end_of_turn>\n<start_of_turn>user\n'$CONTENT'<end_of_turn>\n<start_of_turn>model\n"
# }'| jq '.content'

# Llama 3.2 1B
curl http://localhost:8082/completion --silent -H "Content-Type: application/json" --data '{
  "temp": 1.2,
  "n_predict": 32,
  "cache_prompt": true,
  "stop": ["<|eot_id|>"],
  "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a query augmentation engine, part of a job board. You augment queries to increase the chances of a search match. You write exactly 3 augmentations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSoftware engineer jobs in New York<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNew York software engineer job openings\nsoftware development positions in NYC\nsoftware engineer roles in New York City<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRemote marketing positions<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nremote marketing job opportunities\nwork from home marketing roles\ndigital marketing positions remote<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ndata sciens job entry level<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nentry-level data science jobs\njunior data scientist positions\ndata analyst internships<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nsenior devops engineer san francisco<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nsenior DevOps positions in San Francisco\nSan Francisco DevOps engineer jobs\nlead DevOps roles in SF<|eot_id|><|start_header_id|>user<|end_header_id|>\n\npart time IT jobs near me<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\npart-time IT support roles nearby\nIT technician part-time jobs\nflexible IT jobs in my area<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'$CONTENT'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}'| jq '.content'
