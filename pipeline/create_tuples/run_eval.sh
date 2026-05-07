#!/bin/bash

# Configuration
tuples_file="create_tuples/filtered_tuples/all_tuples_yes_counts_threshold_70.0.jsonl"
tasks_file="create_tuples/data/tasks.jsonl"
output_dir="create_tuples/outputs_eval/meta-llama__Llama-3.1-8B-Instruct"
model_path="meta-llama/Llama-3.1-8B-Instruct"

tensor_parallel_size=4
pipeline_parallel_size=1
seed=42
batch_size=1000  # Process prompts in batches to avoid memory overflow
checkpoint_interval=1000  # Save outputs every N items for crash recovery

# MODELS:
# openai/gpt-oss-20b
# allenai/Olmo-3.1-32B-Instruct
# google/gemma-3-27b-it
# Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
# meta-llama/Llama-3.1-8B-Instruct

# openai/gpt-oss-120b
# Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
# zai-org/GLM-4.7-FP8

mkdir -p "$output_dir"

# Save run info
echo "tuples_file: ${tuples_file}"
echo "tasks_file: ${tasks_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "batch_size: ${batch_size}"
echo "checkpoint_interval: ${checkpoint_interval}"
echo "run_date: $(date)"


python pipeline/create_tuples/run_eval.py \
    --tuples-file $tuples_file \
    --tasks-file $tasks_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size $pipeline_parallel_size \
    --seed $seed \
    --batch-size $batch_size \
    --checkpoint-interval $checkpoint_interval
