#!/bin/bash

# Configuration
input_dir="best_judge/outputs_eval_gpt"
output_dir="best_judge/outputs_judge_gpt/zai-org__GLM-4.7-FP8"
model_path="zai-org/GLM-4.7-FP8"
tensor_parallel_size=4
seed=42

# Create output directory and logs directory if they don't exist
mkdir -p "$output_dir"

# MODELS
# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# Qwen/Qwen3-235B-A22B-Thinking-2507-FP8
# zai-org/GLM-4.7-FP8

# Save run info
echo "input_dir: ${input_dir}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "seed: ${seed}"
echo "run_date: $(date)"


python pipeline/best_judge/llm_judge.py \
    --input-dir $input_dir \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --seed $seed
