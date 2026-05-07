#!/bin/bash

# Configuration
data_file="subjectivity/outputs_eval/meta-llama__Llama-3.2-3B-Instruct.jsonl"
output_dir="subjectivity/outputs_judge/Qwen__Qwen3-235B-A22B-Instruct-2507-FP8"
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

# DATA FILES
# subjectivity/outputs_eval/allenai__Olmo-3-7B-Instruct.jsonl
# subjectivity/outputs_eval/google__gemma-3-4b-it.jsonl
# subjectivity/outputs_eval/meta-llama__Llama-3.2-3B-Instruct.jsonl
# subjectivity/outputs_eval/Qwen__Qwen3-4B-Instruct-2507.jsonl

# Save run info
echo "data_file: ${data_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "pipeline_parallel_size: ${pipeline_parallel_size}"
echo "run_date: $(date)"


python pipeline/subjectivity/llm_judge.py \
    --data-file $data_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size $pipeline_parallel_size \
    --seed $seed
