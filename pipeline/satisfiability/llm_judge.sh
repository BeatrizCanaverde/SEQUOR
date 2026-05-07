#!/bin/bash

# Directory containing JSONL files
constraints_file="satisfiability/data/constraints.jsonl"
tasks_file="satisfiability/data/tasks.jsonl"
output_dir="satisfiability/outputs_judge/"
model_path="openai/gpt-oss-120b"
tensor_parallel_size=4
seed=42

# MODELS:
# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# zai-org/GLM-4.7-FP8


# Save run info
echo "constraints_file: ${constraints_file}"
echo "tasks_file: ${tasks_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "seed: ${seed}"
echo "run_date: $(date)"


python pipeline/satisfiability/llm_judge.py \
    --constraints-file $constraints_file \
    --tasks-file $tasks_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --seed $seed
