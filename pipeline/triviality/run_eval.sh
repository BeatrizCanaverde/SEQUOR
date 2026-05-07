#!/bin/bash

tasks_file="triviality/data/tasks_1.jsonl"
output_dir="triviality/outputs_eval/"
model_path="Qwen/Qwen3-4B-Instruct-2507"  #  "meta-llama/Llama-3.2-3B-Instruct"  "google/gemma-3-4b-it"  "Qwen/Qwen3-4B-Instruct-2507"
tensor_parallel_size=1

# tasks_1.jsonl - Qwen/Qwen3-4B-Instruct-2507
# tasks_2.jsonl - meta-llama/Llama-3.2-3B-Instruct
# tasks_3.jsonl - google/gemma-3-4b-it
# tasks_4.jsonl - allenai/Olmo-3-7B-Instruct

# Save run info
echo "tasks_file: ${tasks_file}"
echo "output_dir: ${output_dir}"
echo "model_path: ${model_path}"
echo "tensor_parallel_size: ${tensor_parallel_size}"
echo "run_date: $(date)"


python pipeline/triviality/run_eval.py \
    --tasks-file $tasks_file \
    --output-dir $output_dir \
    --model-path $model_path \
    --tensor-parallel-size $tensor_parallel_size