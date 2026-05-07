#!/bin/bash

constraints_file="data/constraints/single.jsonl"
tasks_file="create_tuples/data/tasks.jsonl"
tuple_size=3
num_tuples=10000
output_dir="create_tuples/tuples/openai__gpt-oss-120b"
model_path="openai/gpt-oss-120b"
tensor_parallel_size=4
seed=42


# MODELS:
# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# zai-org/GLM-4.7-FP8


mkdir -p "$output_dir"

# Save run info
echo "constraints_file: ${constraints_file}"
echo "tasks_file: ${tasks_file}"
echo "tuple_size: ${tuple_size}"
echo "num_tuples: ${num_tuples}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "seed: ${seed}"
echo "run_date: $(date)"

python pipeline/create_tuples/task_tuples.py \
    --constraints-file "$constraints_file" \
    --tasks-file "$tasks_file" \
    --tuple-size "$tuple_size" \
    --num-tuples "$num_tuples" \
    --output-dir "$output_dir" \
    --model-path "$model_path" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --seed "$seed"
