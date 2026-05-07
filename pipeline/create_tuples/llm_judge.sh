#!/bin/bash

# Configuration
data_file="create_tuples/outputs_eval/google__gemma-3-4b-it.jsonl"
output_dir="create_tuples/outputs_judge/Qwen__Qwen3-235B-A22B-Instruct-2507-FP8"
model_path="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
tensor_parallel_size=4
pipeline_parallel_size=1
seed=42

# Create output directory and logs directory if they don't exist
mkdir -p "$output_dir"

# MODELS
# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# zai-org/GLM-4.7-FP8

# Save run info
echo "data_file: ${data_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "run_date: $(date)"


python pipeline/create_tuples/llm_judge.py \
    --data-file $data_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size $pipeline_parallel_size \
    --seed $seed
