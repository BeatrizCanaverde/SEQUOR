#!/bin/bash

# Configuration
constraints_file="subjectivity/data/constraints.jsonl"
tasks_file="subjectivity/data/tasks_3.jsonl"
output_dir="subjectivity/outputs_eval/"
model_path="google/gemma-3-4b-it"
tensor_parallel_size=2
pipeline_parallel_size=1
seed=42

mkdir -p "$output_dir"


# tasks_1.jsonl - Qwen/Qwen3-4B-Instruct-2507
# tasks_2.jsonl - meta-llama/Llama-3.2-3B-Instruct
# tasks_3.jsonl - google/gemma-3-4b-it
# tasks_4.jsonl - allenai/Olmo-3-7B-Instruct


# Save run info
echo "constraints_file: ${constraints_file}"
echo "tasks_file: ${tasks_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "pipeline_parallel_size: ${pipeline_parallel_size}"
echo "run_date: $(date)"


python pipeline/subjectivity/run_eval.py \
    --constraints-file $constraints_file \
    --tasks-file $tasks_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size \
    --pipeline-parallel-size $pipeline_parallel_size \
    --seed $seed
