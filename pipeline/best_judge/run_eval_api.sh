#!/bin/bash

# Configuration
input_dir="best_judge/data"
output_dir="best_judge/outputs_eval_gpt/"
model="openai/gpt-5.2"
base_url="https://openrouter.ai/api/v1"
max_tokens=80000
seed=42

# Enable OpenRouter reasoning
reasoning_enabled="1"

# Maximum number of entries to process per file (leave empty for all)
max_entries="500"

# Setup logging
timestamp=$(date +%Y%m%d_%H%M%S)
log_dir="best_judge/outputs_eval_gpt/logs"
mkdir -p "$log_dir"

OUT_FILE="$log_dir/${timestamp}.out"
ERR_FILE="$log_dir/${timestamp}.err"

echo "Logging output to: $OUT_FILE"
echo "Logging errors to: $ERR_FILE"

mkdir -p "$output_dir"

# Build command
cmd="python pipeline/best_judge/run_eval_api.py \
    --input-dir $input_dir \
    --output-dir $output_dir \
    --model $model \
    --base-url $base_url \
    --max-tokens $max_tokens \
    --seed $seed"

if [ -n "$reasoning_enabled" ]; then
    cmd="$cmd --reasoning-enabled"
fi

if [ -n "$max_entries" ]; then
    cmd="$cmd --max-entries $max_entries"
fi

# Execute the command and redirect output/errors while showing on terminal
eval "$cmd" > >(tee "$OUT_FILE") 2> >(tee "$ERR_FILE" >&2)
