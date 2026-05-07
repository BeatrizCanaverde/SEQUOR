#!/bin/bash

# Directory containing JSONL files
constraints_file="triviality/data/constraints.jsonl"
data_file="triviality/outputs_eval/Qwen__Qwen3-4B-Instruct-2507.jsonl"
output_dir="triviality/outputs_judge/openai__gpt-oss-120b"
model_path="openai/gpt-oss-120b"
tensor_parallel_size=4
seed=42

mkdir -p "$output_dir"

# MODELS
# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# zai-org/GLM-4.7-FP8

# DATA FILES
# triviality/outputs_eval/allenai__Olmo-3-7B-Instruct.jsonl
# triviality/outputs_eval/google__gemma-3-4b-it.jsonl
# triviality/outputs_eval/meta-llama__Llama-3.2-3B-Instruct.jsonl
# triviality/outputs_eval/Qwen__Qwen3-4B-Instruct-2507.jsonl

# Save run info
echo "constraints_file: ${constraints_file}"
echo "data_file: ${data_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "seed: ${seed}"
echo "run_date: $(date)"


python pipeline/triviality/llm_judge.py \
    --constraints-file $constraints_file \
    --data-file $data_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --seed $seed
